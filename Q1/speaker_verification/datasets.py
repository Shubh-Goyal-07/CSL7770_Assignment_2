import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import UniSpeechSatModel, UniSpeechSatConfig, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import torchaudio
import random
import math
from collections import defaultdict
from peft import get_peft_model, LoraConfig, TaskType
import re
import librosa


class VoxCeleb1Dataset(Dataset):
    def __init__(self, root_dir, speaker_ids=None, duration=3, sampling_rate=16000):
        self.root_dir = root_dir
        self.max_frames = duration * 100
        self.sampling_rate = sampling_rate

        self.speakers = []
        self.samples = []
        self.speaker_idxs = {}

        self.__read_root_dir(root_dir, speaker_ids)

    def __read_root_dir(self, root_dir, speaker_ids):
        # Walk through the directory to get all audio files
        if speaker_ids is not None:
            for speaker_id in os.listdir(root_dir):
                if speaker_id not in speaker_ids:
                    continue

                speaker_dir = os.path.join(root_dir, speaker_id)
                if not os.path.isdir(speaker_dir):
                    continue

                if speaker_id not in self.speaker_idxs:
                    self.speaker_idxs[speaker_id] = len(self.speaker_idxs)

                for session_id in os.listdir(speaker_dir):
                    session_dir = os.path.join(speaker_dir, session_id)
                    if not os.path.isdir(session_dir):
                        continue

                    for samples_file in os.listdir(session_dir):
                        if not (samples_file.endswith('.wav') or samples_file.endswith('.m4a')):
                            continue
                            
                        samples_path = os.path.join(session_dir, samples_file)
                        self.speakers.append(speaker_id)
                        self.samples.append(samples_path)
        
        print(f"Loaded {len(self.samples)} samples from {len(self.speaker_idxs)} speakers")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        samples_path = self.samples[idx]
        speaker_id = self.speakers[idx]
        speaker_idx = self.speaker_idxs[speaker_id]
        
        waveform, sample_rate = torchaudio.load(samples_path)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        # if sample_rate != self.sampling_rate:
        #     resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
        #     waveform = resampler(waveform)
        
        # Random crop if longer than max_frames
        if waveform.shape[1] > self.max_frames * sample_rate // 1000:
            max_start = waveform.shape[1] - self.max_frames * sample_rate // 1000
            start = random.randint(0, max_start)
            waveform = waveform[:, start:start + self.max_frames * sample_rate // 1000]
        
        # Pad if shorter than max_frames
        if waveform.shape[1] < self.max_frames * sample_rate // 1000:
            padding = self.max_frames * sample_rate // 1000 - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        return waveform, speaker_idx


class VoxCeleb2Dataset(Dataset):
    """Dataset for loading and processing VoxCeleb2 audio samples."""
    
    def __init__(self, audio_dir, text_dir, speaker_ids, sample_rate=16000, duration=3.0):
        """
        Initialize the VoxCeleb2 dataset.
        
        Args:
            root_dir: Base directory containing both audio and text folders
            speaker_ids: List of speaker IDs to include
            sample_rate: Audio sample rate (default: 16000)
            clip_duration: Duration of audio clips in seconds (default: 3.0)
        """
        self.audio_dir = audio_dir
        self.text_dir = text_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_samples = int(sample_rate * duration)
        self.allowed_speakers = set(speaker_ids)
        self.samples = self._build_dataset()
    
    def _build_dataset(self):
        """Build the dataset by scanning files and extracting metadata."""
        samples = []
        
        # Iterate through allowed speaker IDs
        for speaker_id in self.allowed_speakers:
            speaker_path = os.path.join(self.text_dir, speaker_id)
            if not os.path.isdir(speaker_path):
                continue
                
            # Iterate through session folders
            for session_id in os.listdir(speaker_path):
                session_path = os.path.join(speaker_path, session_id)
                if not os.path.isdir(session_path):
                    continue
                    
                # Process text files in session folder
                for filename in os.listdir(session_path):
                    if not filename.endswith(".txt"):
                        continue
                        
                    # Get corresponding file paths
                    txt_path = os.path.join(session_path, filename)
                    base_name = os.path.splitext(filename)[0]
                    m4a_path = os.path.join(self.audio_dir, speaker_id, session_id, f"{base_name}.m4a")
                    
                    # Extract offset from text file
                    offset = self._get_offset_from_txt(txt_path)
                    
                    # Add to dataset if audio file exists
                    if os.path.exists(m4a_path):
                        samples.append((m4a_path, offset, speaker_id))
        
        print(len(samples))
        return samples
    
    def _get_offset_from_txt(self, txt_path):
        """Extract the offset value from a text file."""
        try:
            with open(txt_path, "r") as file:
                for line in file:
                    line = line.strip()
                    if line.startswith("Offset:"):
                        try:
                            return float(line.split(":", 1)[1].strip())
                        except ValueError:
                            return 0.0
            return 0.0
        except Exception as e:
            print(f"Error reading {txt_path}: {e}")
            return 0.0
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, index):
        audio_path, offset, speaker_idx = self.samples[index]
        
        # Load audio segment starting at offset time
        waveform, _ = librosa.load(
            audio_path,
            sr=self.sample_rate,
            offset=offset,
            duration=self.duration
        )
        
        # Handle audio shorter than expected length
        if len(waveform) < self.target_samples:
            # Pad with zeros if audio is shorter than needed
            padding = np.zeros(self.target_samples - len(waveform))
            waveform = np.concatenate([waveform, padding])
        else:
            # Truncate if longer than needed
            waveform = waveform[:self.target_samples]
        
        # Convert to PyTorch tensor
        waveform = torch.tensor(waveform, dtype=torch.float)
        
        return waveform, speaker_idx
    

def collate_fn(batch):
    return torch.stack([x[0] for x in batch]), [x[1] for x in batch]