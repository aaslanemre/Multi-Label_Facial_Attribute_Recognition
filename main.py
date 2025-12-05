# --- main.py ---
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
import numpy as np

# Import necessary components from project files
from config import (
    NUM_EPOCHS, BATCH_SIZE_TRAIN, LEARNING_RATE, WEIGHT_DECAY, DEVICE, 
    MLFLOW_RUN_NAME, MLFLOW_EXPERIMENT_NAME, NUM_CLASSES, MODEL_NAME, 
    FREEZE_BASE_LAYERS
)
from dataset_hf import initialize_data_loaders
from model import ResNet18MultiLabel, ViTMultiLabel 

# --- Global Initialization Functions ---

def initialize_model_and_training_components():
    """Initializes model, criterion, and optimizer based on config."""
    
    print(f"Initializing Model: {MODEL_NAME} (Frozen Base: {FREEZE_BASE_LAYERS})")
    
    if MODEL_NAME == "ResNet50":
        model = ResNet18MultiLabel(num_classes=NUM_CLASSES, freeze_base=FREEZE_BASE_LAYERS).to(DEVICE)
    elif MODEL_NAME == "ViT":
        model = ViTMultiLabel(num_classes=NUM_CLASSES, freeze_base=FREEZE_BASE_LAYERS).to(DEVICE)
    else:
        raise ValueError(f"Unknown MODEL_NAME: {MODEL_NAME}")
    
    # Using Binary Cross-Entropy with Logits for multi-label classification (Numerically stable)
    criterion = nn.BCEWithLogitsLoss()
    
    # Only optimize parameters that require gradients (the new FC layer if frozen)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    optimizer = optim.Adam(
        trainable_params, 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    
    print("Model, Criterion, and Optimizer Initialized Successfully.")
    return model, criterion, optimizer

# --- Training Loop Functions ---

def train_epoch(model, loader, criterion, optimizer):
    """Performs one epoch of training."""
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
    """Calculates comprehensive multi-label metrics."""
    
    # Convert list of tensors to numpy arrays
    all_labels = np.concatenate([t.numpy() for t in all_labels])
    all_scores = np.concatenate([t.numpy() for t in all_scores])
    
    # Convert scores to binary predictions (threshold at 0.5)
    all_preds = (all_scores > 0.5).astype(int)

    metrics = {}
    
    # 1. Macro-Average Metrics (Average across all 40 attributes)
    metrics['macro_avg_precision'] = average_precision_score(all_labels, all_scores, average='macro')
    metrics['macro_auc_roc'] = roc_auc_score(all_labels, all_scores, average='macro')
    metrics['macro_f1'] = f1_score(all_labels, all_preds, average='macro')

    # 2. Overall Accuracy (Micro-level for the whole set)
    metrics['overall_accuracy'] = (all_preds == all_labels).mean()
    
    # 3. Per-Attribute mAP (for detailed logging)
    # ap_scores = average_precision_score(all_labels, all_scores, average=None)
    # for i, attr in enumerate(attribute_names):
    #     metrics[f'AP_{attr}'] = ap_scores[i]

    return metrics

def validate_and_evaluate(model, loader, criterion, phase='Validation', attribute_names=None):
    """Evaluates the model on validation or test set."""
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
            
            # Outputs are logits; we apply sigmoid to get probabilities (scores)
            scores = torch.sigmoid(outputs)
            
            all_scores.append(scores.cpu())
            all_labels.append(labels.cpu())

    epoch_loss = running_loss / len(loader.dataset)
    
    # Calculate comprehensive metrics
    metrics = calculate_metrics(all_labels, all_scores, attribute_names)
    
    return epoch_loss, metrics

# --- Main Training Runner ---

def run_training():
    """Main function to run the training process."""
    
    print(f"--- 0. Training Setup: Device={DEVICE} ---")

    # MLflow Setup
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=MLFLOW_RUN_NAME):
        
        # Log Hyperparameters
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "freeze_base": FREEZE_BASE_LAYERS,
            "num_epochs": NUM_EPOCHS,
            "batch_size_train": BATCH_SIZE_TRAIN,
            "learning_rate": LEARNING_RATE,
            "device": DEVICE,
            "num_classes": NUM_CLASSES
        })
        
        print("--- 1. Initializing Data Loaders ---")
        train_loader, val_loader, test_loader, attribute_names = initialize_data_loaders(MODEL_NAME)
        
        print("--- 2. Initializing Model, Criterion, and Optimizer ---")
        model, criterion, optimizer = initialize_model_and_training_components()
        
        best_val_mAP = 0.0
        
        # --- 3. Start Training Loop ---
        for epoch in range(NUM_EPOCHS):
            print(f"\n[Epoch {epoch+1}/{NUM_EPOCHS}]")
            
            train_loss = train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_metrics = validate_and_evaluate(model, val_loader, criterion, phase='Validation', attribute_names=attribute_names)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mAP: {val_metrics['macro_avg_precision']:.4f}")
            
            # Log metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_mAP", val_metrics['macro_avg_precision'], step=epoch)
            mlflow.log_metric("val_auc", val_metrics['macro_auc_roc'], step=epoch)
            mlflow.log_metric("val_f1", val_metrics['macro_f1'], step=epoch)
            
            # Save the best model based on macro Average Precision (mAP)
            current_mAP = val_metrics['macro_avg_precision']
            if current_mAP > best_val_mAP:
                best_val_mAP = current_mAP
                
                # Save model state dict and log to MLflow
                model_artifact_path = f"{MLFLOW_RUN_NAME}_best_model.pth"
                torch.save(model.state_dict(), model_artifact_path)
                mlflow.pytorch.log_model(
                    pytorch_model=model, 
                    artifact_path="best_model", 
                    metadata={"epoch": epoch, "mAP": best_val_mAP}
                )
                print("Model saved (New best validation mAP).")

        # --- 4. Final Evaluation on Test Set ---
        print("\n--- 4. Evaluating Final Best Model on Test Set ---")
        
        # Load the best model weights
        best_model_path = f"{MLFLOW_RUN_NAME}_best_model.pth"
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
        
        test_loss, test_metrics = validate_and_evaluate(model, test_loader, criterion, phase='Test', attribute_names=attribute_names)
        
        print(f"Final Test Loss: {test_loss:.4f}")
        print(f"Final Test mAP: {test_metrics['macro_avg_precision']:.4f}")
        
        # Log final results to MLflow
        mlflow.log_metric("final_test_loss", test_loss)
        mlflow.log_metric("final_test_mAP", test_metrics['macro_avg_precision'])
        mlflow.log_metric("final_test_auc", test_metrics['macro_auc_roc'])
        mlflow.log_metric("final_test_f1", test_metrics['macro_f1'])
        
        mlflow.end_run()
        print("\nMLflow run completed. Check results in the mlruns folder.")


if __name__ == '__main__':
    run_training()