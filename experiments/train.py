import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchmetrics.classification import F1Score
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from traitlets import default

from project_config import RESULTS_DIR, dataset_root, require_file, resolve_device
from model.CD_model import CrossAttention

from datasets.vl_cmu_cd import VL_CMU_CD, Diff_VL_CMU_CD
from datasets.pscd import PSCD, CroppedPSCD
from datasets.changesim import ChangeSimDataset
from datasets.SF_XL import SF_XL_Dataset
from datasets.St_lucia import St_lucia_Dataset
from datasets.nordland import Nordland_Dataset

def unnormalize(img, mean, std):
    mean = np.array(mean).reshape(1, 1, -1)  # Reshape for broadcasting
    std = np.array(std).reshape(1, 1, -1)
    return (img * std) + mean

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for t0, t1, mask in tqdm(dataloader, total=len(dataloader)):
        t0, t1, mask = t0.to(device), t1.to(device), mask.to(device)
        mask = torch.where(mask > 0, 1, 0)
        
        optimizer.zero_grad()
        outputs,_,_,_ = model(t0, t1)
        transform_outputs = outputs.permute(0, 3, 1, 2)
        
        class_weights = torch.tensor([0.025, 0.975]).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        loss = criterion(transform_outputs, mask.squeeze(1).long())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    predictions, ground_truths = [], []

    # Initialize Multiclass F1 metric
    f1_metric = F1Score(task="binary").to(device)
    f1_score = 0

    with torch.no_grad():
        for t0, t1, mask in tqdm(dataloader, total=len(dataloader)):
            t0, t1, mask = t0.to(device), t1.to(device), mask.to(device)
            mask = torch.where(mask > 0, 1, 0)

            outputs,_,_,_ = model(t0, t1)
            transform_outputs= outputs.permute(0, 3, 1, 2)
            
            class_weights = torch.tensor([0.025, 0.975]).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            loss = criterion(transform_outputs, mask.squeeze(1).long())
            total_loss += loss.item()
            
            # Convert logits to class indices
            preds = torch.argmax(outputs, dim=3)
            f1_score += f1_metric(preds, mask.squeeze(1)).item()

    f1_score = f1_score / len(dataloader)
    
    return total_loss / len(dataloader), f1_score

def test_model(model, dataloader, device, save_dir):
    """
    Test the model and save predictions grouped by sequence ID.

    Args:
        model: Trained PyTorch model.
        dataloader: DataLoader for the test dataset.
        device: Device to run the model on.
        save_dir: Directory to save the output images.
    """
    # model.eval()
    predictions, ground_truths = [], []

    os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists
    _, f1 = evaluate(model, dataloader, device)

    return f1

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Training script for CD model")
    parser.add_argument("--dataset", type=str, default="vl-cmu-cd", help="Select dataset")
    parser.add_argument("--data_root", type=str, default=None, help="Dataset root for VL-CMU-CD or selected dataset")
    parser.add_argument("--pscd_root", type=str, default=None, help="Root for Cropped PSCD train/val/test data")
    parser.add_argument("--nordland_root", type=str, default=None, help="Root for Nordland data")
    parser.add_argument("--sf_xl_root", type=str, default=None, help="Root for SF-XL data")
    parser.add_argument("--st_lucia_root", type=str, default=None, help="Root for St Lucia data")
    parser.add_argument("--changesim_train_root", type=str, default=None, help="Root for ChangeSim train/val data")
    parser.add_argument("--changesim_test_root", type=str, default=None, help="Root for ChangeSim test data")
    parser.add_argument("--output_dir", type=str, default=str(RESULTS_DIR), help="Directory for experiment outputs")
    parser.add_argument("--sam2_checkpoint", type=str, default=None, help="Path to SAM2.1 large checkpoint")
    parser.add_argument("--dry_run", action="store_true", help="Validate dataset setup without loading the model")
    parser.add_argument("--model", type=str, default="RobustViT", help="Select Model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Initial learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for training: auto, cuda, cuda:N, or cpu")
    parser.add_argument("--test", action="store_true", help="Test the model without training")
    parser.add_argument("--adj_dist", type=int, default=0, help="adjacent distance for diff-vl-cmu-cd")
    parser.add_argument("--ckpt", type=str, default="", help="Resume training from ckpt")

    # Parse arguments
    args = parser.parse_args()
    args.save_dir = os.path.join(args.output_dir, args.model + '_' + args.dataset)
    os.makedirs(args.save_dir, exist_ok=True)
    # Configurations from arguments
    root = args.data_root
    batch_size = args.batch_size
    epochs = args.num_epochs
    lr = args.lr
    adj_dist = args.adj_dist
    
    device = torch.device(resolve_device(args.device))

    # TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "logs"))

    # Dataset and Dataloader
    if args.dataset == "vl-cmu-cd":
        root = str(dataset_root("vl-cmu-cd", root))
        train_dataset = VL_CMU_CD(root=root, mode="train", augment_with_swaps=False, img_size=(512,512))
        val_dataset = VL_CMU_CD(root=root, mode="val", augment_with_swaps=False, img_size=(512,512))
        test_dataset = VL_CMU_CD(root=root, mode="test", img_size=(512,512))
    elif args.dataset == "diff-vl-cmu-cd":
        root = str(dataset_root("diff-vl-cmu-cd", root))
        train_dataset = Diff_VL_CMU_CD(root=root, mode="train", adjacent_distance=adj_dist)
        val_dataset = Diff_VL_CMU_CD(root=root, mode="val", adjacent_distance=adj_dist)
        test_dataset = Diff_VL_CMU_CD(root=root, mode="test", adjacent_distance=adj_dist)
    elif args.dataset == "pscd":
        root = str(dataset_root("pscd", args.pscd_root or root))
        train_dataset = CroppedPSCD(root=root, mode="train")
        val_dataset = CroppedPSCD(root=root, mode="val")
        test_dataset = CroppedPSCD(root=root, mode="test")
    elif args.dataset == 'Nordland':
        root = str(dataset_root("Nordland", args.nordland_root or root))
        train_dataset = Nordland_Dataset(file_root=root)
        val_dataset = Nordland_Dataset(file_root=root)
        test_dataset = Nordland_Dataset(file_root=root)
    elif args.dataset == 'SF-XL':
        root = str(dataset_root("SF-XL", args.sf_xl_root or root))
        train_dataset = SF_XL_Dataset(file_root=root)
        val_dataset = SF_XL_Dataset(file_root=root)
        test_dataset = SF_XL_Dataset(file_root=root)
    elif args.dataset == 'St-Lucia':
        root = str(dataset_root("St-Lucia", args.st_lucia_root or root))
        train_dataset = St_lucia_Dataset(file_root=root)
        val_dataset = St_lucia_Dataset(file_root=root)
        test_dataset = St_lucia_Dataset(file_root=root)
    elif args.dataset == 'changesim':
        train_root = str(dataset_root("changesim", args.changesim_train_root or root))
        test_root = str(dataset_root("ChangeSim", args.changesim_test_root))
        train_dataset = ChangeSimDataset(file_root=train_root, mode="train")
        val_dataset = ChangeSimDataset(file_root=train_root, mode="val")
        test_dataset = ChangeSimDataset(mode="test", file_root=test_root)
    if args.dataset == "mixed":
        vlcmu_root = str(dataset_root("vl-cmu-cd", root))
        changesim_train_root = str(dataset_root("changesim", args.changesim_train_root))
        changesim_test_root = str(dataset_root("ChangeSim", args.changesim_test_root))
        vlcmu_train = VL_CMU_CD(root=vlcmu_root, mode="train", augment_with_swaps=False, img_size=(512,512))
        changesim_train = ChangeSimDataset(file_root=changesim_train_root, mode="train")

        vlcmu_val = VL_CMU_CD(root=vlcmu_root, mode="val", augment_with_swaps=False, img_size=(512,512))
        changesim_val = ChangeSimDataset(file_root=changesim_train_root, mode="val")

        train_dataset = ConcatDataset([vlcmu_train, changesim_train])
        val_dataset = ConcatDataset([vlcmu_val, changesim_val])
        test_dataset = ChangeSimDataset(mode="test", file_root=changesim_test_root)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    if args.dry_run:
        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        print("Dry run complete.")
        return
    
    # Model, loss, and optimizer
    model = CrossAttention(target_shp=(512,512), device=str(device), sam2_checkpoint=args.sam2_checkpoint)
    model.to(device)
    if args.ckpt:
        model.load_state_dict(torch.load(require_file(args.ckpt, "training checkpoint"), map_location=device))
        
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.2*lr)

    # Training loop
    if not args.test:
        best_f1_score = 0
        for epoch in tqdm(range(epochs), desc="Epochs Progress", unit="epoch"):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_loss, val_f1 = evaluate(model, val_loader, device)
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val F1 Score: {val_f1:.4f}")

            # Log metrics to TensorBoard
            writer.add_scalar("Loss/Train", train_loss, epoch + 1)
            writer.add_scalar("Loss/Validation", val_loss, epoch + 1)
            writer.add_scalar("F1/Validation", val_f1, epoch + 1)

            # Save the model if validation F1 improves
            if val_f1 > best_f1_score:
                best_f1_score = val_f1
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))

            if epoch % 5 == 0:
                torch.save(model.state_dict(), os.path.join(args.save_dir, f"epoch_{epoch}.pth"))
                
            # Step the scheduler to adjust learning rate
            scheduler.step()
        print("Training complete!")

    # Load the best model and test
    print("Testing the model...")
    test_ckpt = args.ckpt or os.path.join(args.save_dir, "best_model.pth")
    model.load_state_dict(torch.load(require_file(test_ckpt, "best model checkpoint"), map_location=device))
    test_f1 = test_model(model, test_loader, device, args.save_dir+"/predictions")
    print("Testing F1: ", test_f1)

    writer.close()

if __name__ == "__main__":
    main()
