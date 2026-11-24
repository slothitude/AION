"""Training loop and utilities."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import yaml
from tqdm import tqdm
import time

from nnfs.config import TrainingConfig, ModelConfig
from nnfs.training.metrics import MetricsTracker, compute_metrics


class Trainer:
    """
    Training loop with support for different model types,
    learning rate scheduling, early stopping, and checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        device: str = "cpu",
        log_dir: str = "logs",
        save_checkpoints: bool = True,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device(device)
        self.log_dir = Path(log_dir)
        self.save_checkpoints = save_checkpoints
        
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.metrics = MetricsTracker()
        
        # State
        self.epoch = 0
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.start_time = time.time()
        
        # Create log dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.epochs, eta_min=1e-6
            )
        elif self.config.scheduler == "cosine_warmup":
            # Warmup + cosine
            def lr_lambda(epoch):
                if epoch < self.config.warmup_epochs:
                    return epoch / self.config.warmup_epochs
                else:
                    progress = (epoch - self.config.warmup_epochs) / (self.config.epochs - self.config.warmup_epochs)
                    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159))).item()
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        elif self.config.scheduler == "step":
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        else:
            return optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # Reset memory if model has it
        if hasattr(self.model, 'reset_memory'):
            self.model.reset_memory()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}/{self.config.epochs}", leave=False)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Flatten if needed (for MNIST etc.)
            if data.dim() > 2:
                data = data.view(data.size(0), -1)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if hasattr(self.model, 'forward') and 'target_next' in self.model.forward.__code__.co_varnames:
                # Specialized net with prediction
                output = self.model(data, target_next=None)
                logits = output["logits"] if isinstance(output, dict) else output
            else:
                logits = self.model(data)
            
            loss = self.criterion(logits, target)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * data.size(0)
            pred = logits.argmax(dim=1)
            total_correct += (pred == target).sum().item()
            total_samples += data.size(0)
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100*total_correct/total_samples:.2f}%"})
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        return {"train_loss": avg_loss, "train_acc": accuracy}
    
    def evaluate(self, loader: DataLoader, desc: str = "Val") -> Dict[str, float]:
        """Evaluate on a data loader."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_targets = []
        
        if hasattr(self.model, 'reset_memory'):
            self.model.reset_memory()
        
        with torch.no_grad():
            for data, target in tqdm(loader, desc=desc, leave=False):
                data, target = data.to(self.device), target.to(self.device)
                
                if data.dim() > 2:
                    data = data.view(data.size(0), -1)
                
                if hasattr(self.model, 'forward') and 'target_next' in self.model.forward.__code__.co_varnames:
                    output = self.model(data, target_next=None)
                    logits = output["logits"] if isinstance(output, dict) else output
                else:
                    logits = self.model(data)
                
                loss = self.criterion(logits, target)
                
                total_loss += loss.item() * data.size(0)
                pred = logits.argmax(dim=1)
                total_correct += (pred == target).sum().item()
                total_samples += data.size(0)
                
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        # Compute additional metrics
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        metrics = compute_metrics(all_preds, all_targets)
        metrics.update({f"{desc.lower()}_loss": avg_loss, f"{desc.lower()}_acc": accuracy})
        
        return metrics
    
    def train(self) -> Dict[str, List[float]]:
        """Full training loop."""
        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "lr": [],
        }
        
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.evaluate(self.val_loader, "Val")
            
            # Scheduler step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            history["train_loss"].append(train_metrics["train_loss"])
            history["train_acc"].append(train_metrics["train_acc"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_acc"].append(val_metrics["val_acc"])
            history["lr"].append(current_lr)
            
            # Log
            print(f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                  f"Train: {train_metrics['train_loss']:.4f} ({100*train_metrics['train_acc']:.2f}%) | "
                  f"Val: {val_metrics['val_loss']:.4f} ({100*val_metrics['val_acc']:.2f}%) | "
                  f"LR: {current_lr:.2e}")
            
            # Checkpoint
            if val_metrics["val_acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["val_acc"]
                self.patience_counter = 0
                if self.save_checkpoints:
                    self.save_checkpoint("best_model.pt")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Final test evaluation
        if self.test_loader is not None:
            test_metrics = self.evaluate(self.test_loader, "Test")
            print(f"Test Accuracy: {100*test_metrics['test_acc']:.2f}%")
            history["test_acc"] = test_metrics["test_acc"]
            history["test_loss"] = test_metrics["test_loss"]
        
        total_time = time.time() - self.start_time
        print(f"Training completed in {total_time:.1f}s")
        
        return history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.log_dir / filename
        # Only save serializable config dict
        config_dict = {}
        if hasattr(self.config, '__dict__'):
            for k, v in self.config.__dict__.items():
                if isinstance(v, (int, float, str, bool, list, dict)):
                    config_dict[k] = v
        torch.save({
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "config": config_dict,
        }, path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.log_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_val_acc = checkpoint["best_val_acc"]


def create_data_loaders(
    task: str,
    batch_size: int = 128,
    val_split: float = 0.1,
    num_workers: int = 0,
) -> tuple:
    """Create data loaders for standard tasks."""
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import random_split, DataLoader
    
    if task == "mnist":
        # ToTensor only: inputs stay in [0, 1] (the loader's documented contract)
        transform = transforms.ToTensor()
        train_dataset = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)
        input_dim = 784
        num_classes = 10
        
    elif task == "fashion_mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_dataset = torchvision.datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
        input_dim = 784
        num_classes = 10
        
    elif task == "cifar10_flattened":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        train_dataset = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)
        input_dim = 3072  # 32*32*3
        num_classes = 10
        
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Split train into train/val
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, input_dim, num_classes