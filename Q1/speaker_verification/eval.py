import os
import torch
import librosa
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm
import logging

# Function to compute EER
def compute_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer * 100, thresh

# Function to compute TAR@1%FAR
def compute_tar_at_far(scores, labels, far_value=0.01):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = np.argmin(np.abs(fpr - far_value))
    return tpr[idx] * 100

# Function to evaluate the model
def evaluate_verification_task(model, device, root_dir, pair_txt_file):
    logging.info("Preparing evaluation dataset...")
    dataset = []
    with open(pair_txt_file, "r") as f:
        for line in f:
            label, audio1_path, audio2_path = line.strip().split()
            label = int(label)
            dataset.append((label, audio1_path, audio2_path))
    
    model.eval()

    scores = []
    labels = []

    logging.info("Starting evaluation...\n")
    with torch.no_grad():
        for label, audio1_path, audio2_path in tqdm(dataset, desc="Evaluating"):
            try:
                audio1, sr1 = librosa.load(os.path.join(root_dir, audio1_path), sr=16000)
                audio2, sr2 = librosa.load(os.path.join(root_dir, audio2_path), sr=16000)
            except Exception as e:
                print(f"Error loading audio files: {e}")
                continue

            audio1 = torch.tensor(audio1, dtype=torch.float32).unsqueeze(0).to(device)
            audio2 = torch.tensor(audio2, dtype=torch.float32).unsqueeze(0).to(device)

            embed1 = model(audio1).last_hidden_state.mean(dim=1)
            embed2 = model(audio2).last_hidden_state.mean(dim=1)

            score = torch.nn.functional.cosine_similarity(embed1, embed2).item()
            scores.append(score)
            labels.append(label)

            if len(scores) % 1000 == 0:
                logging.info(f"Processed {len(scores)}/{len(dataset)} pairs")

    logging.info("Evaluation completed.\n")
    eer, thresh = compute_eer(scores, labels)
    tar_at_far = compute_tar_at_far(scores, labels)

    logging.info("Results:")
    logging.info(f"EER: {eer:.2f}%")
    logging.info(f"TAR@1%FAR: {tar_at_far:.2f}%")

