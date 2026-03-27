import json
import matplotlib.pyplot as plt
import os
from pathlib import Path


def plot_training_history(history_path='logs/training_history.json', save_path='logs/training_curves.png'):
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {save_path}")


def plot_metrics_summary(history_path='logs/training_history.json'):
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    print("\n" + "=" * 50)
    print("Training Summary")
    print("=" * 50)
    
    best_train_acc = max(history['train_acc'])
    best_val_acc = max(history['val_acc'])
    best_train_loss = min(history['train_loss'])
    best_val_loss = min(history['val_loss'])
    
    print(f"Best Train Accuracy: {best_train_acc:.2f}%")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best Train Loss: {best_train_loss:.4f}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    plot_training_history()
    plot_metrics_summary()