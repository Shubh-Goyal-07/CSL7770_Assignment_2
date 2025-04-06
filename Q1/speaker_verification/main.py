import os
import torch
from torch.utils.data import DataLoader
from datasets import VoxCeleb1Dataset, VoxCeleb2Dataset, collate_fn
from utils import load_model, get_lora_model
from utils import train_and_test_model
from utils import ArcFaceLoss
from eval import evaluate_verification_task
import logging
import argparse
import torch.nn as nn

import warnings

warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa.core.audio")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths
VOX1_PATH = "../data/vox1_wav"
VOX2_PATH = "../data/vox2_aac"
VOX2_TEXT_PATH = "../data/vox2_txt"
TRIALS_PATH = "../data/veri_test2.txt"

BATCH_SIZE = 32
NUM_EPOCHS = 4
LR = 1e-4


def get_vox2_ids(vox2_path):
    ids = sorted([d for d in os.listdir(vox2_path) if d.startswith("id") and os.path.isdir(os.path.join(vox2_path, d))])
    return ids[:100], ids[100:118]

def finetune_model():
    logging.info("Loading VoxCeleb2 dataset...")
    vox2_train_ids, vox2_test_ids = get_vox2_ids(VOX2_PATH)
    vox2_train_dataset = VoxCeleb2Dataset(VOX2_PATH, VOX2_TEXT_PATH, vox2_train_ids)
    vox2_test_dataset = VoxCeleb2Dataset(VOX2_PATH, VOX2_TEXT_PATH, vox2_test_ids)
    
    train_vox2_loader = DataLoader(vox2_train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_vox2_loader = DataLoader(vox2_test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    id_train_map = {identity: idx for idx, identity in enumerate(sorted(vox2_train_ids))}
    id_test_map = {identity: idx + 100 for idx, identity in enumerate(sorted(vox2_test_ids))}

    logging.info("Loaded VoxCeleb2 dataset.\n")

    logging.info("Loading Pretrained Model...")
    model = load_model()
    model = get_lora_model(model)
    model.to(device)
    logging.info("Loaded Pretrained Model.\n")

    logging.info("Making Classifier...")
    embedding_dim = model.config.hidden_size
    num_classes = len(id_train_map) + len(id_test_map)

    classifier = nn.Linear(embedding_dim, num_classes)
    # if torch.cuda.device_count() > 1:
    #     classifier = torch.nn.DataParallel(classifier, device_ids=[0, 1, 2, 3, 4, 5])
    classifier.to(device)

    logging.info("Classifier made.\n")
    
    logging.info("Starting Training...\n")
    train_and_test_model(model, classifier, train_vox2_loader, test_vox2_loader, id_train_map, id_test_map, NUM_EPOCHS, LR, device)


def evaluate_model(pretrained=False):
    logging.info("Evaluating Model...")
    model = load_model()
    model = get_lora_model(model)
    model.to(device)

    if not pretrained:
        torch_state_dict = torch.load("models/best_model.pth")
        model.load_state_dict(torch_state_dict, strict=False)
    evaluate_verification_task(model, device, VOX1_PATH, TRIALS_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speaker Verification")
    parser.add_argument("--train", type=bool, default=False, help="Train the model")
    parser.add_argument("--eval_model", type=str, default="", help="finetuned or pre_trained")

    args = parser.parse_args()

    if not os.path.exists("logs"):
        os.mkdir("logs")

    if args.train:
        logging.basicConfig(level=logging.INFO, filename="logs/finetune.log", format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S", filemode="w")
        finetune_model()
    else:
        logging.basicConfig(level=logging.INFO, filename=f"logs/evaluate_{args.eval_model}.log", format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S", filemode="w")
        if args.eval_model == "finetuned":
            evaluate_model(pretrained=False)
        elif args.eval_model == "pre_trained":
            evaluate_model(pretrained=True)
        else:
            print("Invalid evaluation model specified.")

    


