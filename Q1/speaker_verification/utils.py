import torch
import torch.nn as nn
from transformers import UniSpeechSatModel
from peft import LoraConfig, get_peft_model
import os
from torch.optim import Adam
from torch.amp import GradScaler, autocast
import logging
from tqdm import tqdm
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    def __init__(self, scale_factor=30.0, angular_margin=0.5):
        super(ArcFaceLoss, self).__init__()
        self.scale = scale_factor
        self.margin = angular_margin
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, cosine_values, target_labels):
        angles = torch.acos(torch.clamp(cosine_values, -1.0 + 1e-6, 1.0 - 1e-6))
        
        target_mask = torch.zeros_like(cosine_values)
        target_mask.scatter_(1, target_labels.view(-1, 1).long(), 1)
        
        modified_angles = angles + (self.margin * target_mask)
        margin_cosine = torch.cos(modified_angles)
        
        final_logits = torch.where(target_mask.bool(), margin_cosine, cosine_values)
        
        scaled_logits = final_logits * self.scale
        loss = self.loss_fn(scaled_logits, target_labels)
        
        return loss
    

def load_model(model_path=None):
    model = UniSpeechSatModel.from_pretrained("microsoft/unispeech-sat-large")
    if model_path:
        checkpoint = torch.load(model_path)
        model.load_state_dict(torch.load(model_path))
        
    return model

def get_lora_model(model):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["attention.q_proj", "attention.k_proj", "attention.v_proj"]
    )

    return get_peft_model(model, lora_config)


def train_and_test_model(model, classifier, train_loader, test_loader, train_label_map, test_label_map, num_epochs, learning_rate, device):
    criterion = ArcFaceLoss()
    optimizer = Adam(list(model.parameters()) + list(classifier.parameters()), lr=learning_rate)
    grad_scaler = GradScaler()

    best_train_acc = 0.0
    best_test_acc = 0.0
    best_train_loss = float('inf')
    best_test_loss = float('inf')

    if not os.path.exists("models"):
        os.makedirs("models")

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        classifier.train()
        
        epoch_train_loss = 0.0
        epoch_train_acc = 0.0

        for audio, identities in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            audio = audio.to(device)
            labels = torch.tensor([train_label_map[id_] for id_ in identities], dtype=torch.long).to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type):
                embeddings = model(audio).last_hidden_state.mean(dim=1)
                embeddings_norm = F.normalize(embeddings, p=2, dim=1)
                # weight_norm = F.normalize(classifier.module.weight, p=2, dim=1)
                # logits = torch.matmul(embeddings_norm, weight_norm.t())
                logits = classifier(embeddings_norm)
                loss = criterion(logits, labels)

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            epoch_train_loss += loss.item()
            epoch_train_acc += (logits.argmax(dim=1) == labels).float().mean().item()
            torch.cuda.empty_cache()

        epoch_test_loss = 0.0
        epoch_test_acc = 0.0

        model.eval()
        classifier.eval()
        with torch.no_grad():
            for audio, identities in tqdm(test_loader, desc=f"Testing Epoch {epoch+1}"):
                audio = audio.to(device)
                labels = torch.tensor([test_label_map[id_] for id_ in identities], dtype=torch.long).to(device)

                embeddings = model(audio).last_hidden_state.mean(dim=1)
                embeddings_norm = F.normalize(embeddings, p=2, dim=1)
                # weight_norm = F.normalize(classifier.module.weight, p=2, dim=1)
                # logits = torch.matmul(embeddings_norm, weight_norm.t())
                logits = classifier(embeddings_norm)
                loss = criterion(logits, labels)

                epoch_test_loss += loss.item()
                epoch_test_acc += (logits.argmax(dim=1) == labels).float().mean().item()
                torch.cuda.empty_cache()

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_acc = epoch_train_acc / len(train_loader)
        avg_test_loss = epoch_test_loss / len(test_loader)
        avg_test_acc = epoch_test_acc / len(test_loader)

        logging.info(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_acc:.4f}")
        logging.info(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_acc:.4f}\n")

        if avg_train_acc >= best_train_acc:
            best_test_acc = avg_test_acc
            best_test_loss = avg_test_loss
            best_train_acc = avg_train_acc
            best_train_loss = avg_train_loss
            logging.info(f"New best model found at epoch {epoch+1}. Saving...\n")
            torch.save(model.state_dict(), f"models/best_model.pth")
            torch.save(classifier.state_dict(), f"models/best_classifier.pth")

    logging.info("Training complete.")
    logging.info(f"Best Test Accuracy: {best_test_acc:.4f}")
    logging.info(f"Best Test Loss: {best_test_loss:.4f}")
    logging.info(f"Best Train Accuracy: {best_train_acc:.4f}")
    logging.info(f"Best Train Loss: {best_train_loss:.4f}")
