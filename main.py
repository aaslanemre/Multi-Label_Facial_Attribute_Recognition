import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
import numpy as np
# --- W&B Import ---
import wandb 

# Import necessary components from project files
from config import (
    NUM_EPOCHS, BATCH_SIZE_TRAIN, LEARNING_RATE, WEIGHT_DECAY, DEVICE, 
    MLFLOW_RUN_NAME, MLFLOW_EXPERIMENT_NAME, NUM_CLASSES, MODEL_NAME, 
    FREEZE_BASE_LAYERS
)
from dataset_hf import initialize_data_loaders 
from model import ResNet18MultiLabel, ViTMultiLabel 

# --- Constants for Checkpointing ---
# W&B handles cloud logging, but we still need local paths for saving/resuming.
LOG_ROOT_DIR = os.path.join("runs", MLFLOW_EXPERIMENT_NAME, MLFLOW_RUN_NAME)
CHECKPOINT_PATH = os.path.join(LOG_ROOT_DIR, "checkpoint.pth")
BEST_MODEL_PATH = os.path.join(LOG_ROOT_DIR, "best_model.pth")


# --- Initialization Functions (omitted for brevity) ---
def initialize_model_and_training_components():
    # ... (content remains the same, but remove the TensorBoard writer initialization)
    print(f"Initializing Model: {MODEL_NAME} (Frozen Base: {FREEZE_BASE_LAYERS})")
    
    if MODEL_NAME == "ResNet50":
        model = ResNet18MultiLabel(num_classes=NUM_CLASSES, freeze_base=FREEZE_BASE_LAYERS).to(DEVICE)
    elif MODEL_NAME == "ViT":
        model = ViTMultiLabel(num_classes=NUM_CLASSES, freeze_base=FREEZE_BASE_LAYERS).to(DEVICE)
    else:
        raise ValueError(f"Unknown MODEL_NAME: {MODEL_NAME}")
    
    criterion = nn.BCEWithLogitsLoss()
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    optimizer = optim.Adam(
        trainable_params, 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    return model, criterion, optimizer

# --- Training, Validation, and Metric Functions (omitted for brevity) ---
def train_epoch(model, loader, criterion, optimizer):
    # ... (content remains the same)
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def calculate_metrics(all_labels, all_scores, attribute_names):
    # ... (content remains the same)
    all_labels = np.concatenate([t.numpy() for t in all_labels])
    all_scores = np.concatenate([t.numpy() for t in all_scores])
    all_preds = (all_scores > 0.5).astype(int)
    metrics = {}
    metrics['macro_avg_precision'] = average_precision_score(all_labels, all_scores, average='macro')
    metrics['macro_auc_roc'] = roc_auc_score(all_labels, all_scores, average='macro')
    metrics['macro_f1'] = f1_score(all_labels, all_preds, average='macro')
    metrics['overall_accuracy'] = (all_preds == all_labels).mean()
    return metrics

def validate_and_evaluate(model, loader, criterion, phase='Validation', attribute_names=None):
    # ... (content remains the same)
    model.eval()
    running_loss = 0.0
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"{phase}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            scores = torch.sigmoid(outputs)
            all_scores.append(scores.cpu())
            all_labels.append(labels.cpu())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = calculate_metrics(all_labels, all_scores, attribute_names)
    return epoch_loss, metrics


# --- Main Training Runner ---

def run_training():
    """Main function to run the training process with Weights & Biases and checkpointing."""
    
    # 0. Setup directories and Hyperparameters for W&B
    os.makedirs(LOG_ROOT_DIR, exist_ok=True)
    
    hparams = {
        "model_name": MODEL_NAME, "freeze_base": FREEZE_BASE_LAYERS,
        "num_epochs": NUM_EPOCHS, "batch_size_train": BATCH_SIZE_TRAIN,
        "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, 
        "device": str(DEVICE), "optimizer": "Adam",
        "loss_fn": "BCEWithLogitsLoss"
    }
    
    # --- W&B Initialization ---
    run = wandb.init(
        project=MLFLOW_EXPERIMENT_NAME, 
        name=MLFLOW_RUN_NAME,
        config=hparams, # Log hyperparameters
        resume="allow" # Allows resuming if run ID is found
    )
    print(f"W&B Run started: {run.url}")
    
    # Log model topology (gradients and parameters)
    # The model object is created in the next step, so we will call run.watch later.
        
    # 1. Initialize data, model, and training components
    train_loader, val_loader, test_loader, attribute_names = initialize_data_loaders(MODEL_NAME)
    model, criterion, optimizer = initialize_model_and_training_components()
    
    # Now that the model is created, call run.watch
    wandb.watch(model, criterion, log="all", log_freq=100)
    
    start_epoch = 0
    best_val_mAP = 0.0
    
    # --- Checkpoint Loading/Resuming Logic ---
    if os.path.exists(CHECKPOINT_PATH) and run.resumed:
        print(f"Found checkpoint: {CHECKPOINT_PATH}. Resuming training...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        
        # Load states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_mAP = checkpoint['best_val_mAP']
        
        print(f"Resumed from Epoch {start_epoch}. Best mAP so far: {best_val_mAP:.4f}")
    
    # --- 3. Start Training Loop ---
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\n[Epoch {epoch+1}/{NUM_EPOCHS}]")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_metrics = validate_and_evaluate(model, val_loader, criterion, phase='Validation', attribute_names=attribute_names)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mAP: {val_metrics['macro_avg_precision']:.4f}")
        
        # --- Log metrics to W&B ---
        wandb.log({
            "Loss/train": train_loss,
            "Loss/val": val_loss,
            "Metrics/val_mAP": val_metrics['macro_avg_precision'],
            "Metrics/val_auc": val_metrics['macro_auc_roc'],
            "Metrics/val_f1": val_metrics['macro_f1'],
            "epoch": epoch
        })
        
        # 1. Save Full Checkpoint (for immediate resume)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_mAP': best_val_mAP
        }
        torch.save(checkpoint, CHECKPOINT_PATH)
        

        # 2. Save Best Model Weights (for final evaluation)
        current_mAP = val_metrics['macro_avg_precision']
        if current_mAP > best_val_mAP:
            best_val_mAP = current_mAP
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            
            # Log the best model file as a W&B Artifact
            artifact = wandb.Artifact(f"{MODEL_NAME}-best-model", type="model")
            artifact.add_file(BEST_MODEL_PATH)
            wandb.log_artifact(artifact)
            
            print("Model saved (New best validation mAP) and logged to W&B Artifacts.")

    # --- 4. Final Evaluation on Test Set ---
    print("\n--- 4. Evaluating Final Best Model on Test Set ---")
    
    # Load the best model weights
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    
    test_loss, test_metrics = validate_and_evaluate(model, test_loader, criterion, phase='Test', attribute_names=attribute_names)
    
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test mAP: {test_metrics['macro_avg_precision']:.4f}")
    
    # Log final test results to W&B as summary metrics (run.summary)
    wandb.summary['final/test_loss'] = test_loss
    wandb.summary['final/test_mAP'] = test_metrics['macro_avg_precision']
    wandb.summary['final/test_auc'] = test_metrics['macro_auc_roc']
    wandb.summary['final/test_f1'] = test_metrics['macro_f1']
    
    # --- W&B Cleanup ---
    wandb.finish()
    print("\nTraining completed. W&B logs saved and synchronized.")


if __name__ == '__main__':
    run_training()