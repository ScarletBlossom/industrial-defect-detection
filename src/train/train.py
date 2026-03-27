import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import Config
from src.data import get_dataloader
from src.models import get_model
from src.train.trainer import train_model
from src.data.generate_data import set_seed


def main():
    set_seed(Config.SEED)
    
    print("\n" + "=" * 60)
    print("Industrial Defect Detection - Training Pipeline")
    print("=" * 60)
    
    print(f"\nDevice: {Config.DEVICE}")
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
    
    print("\nLoading datasets...")
    train_loader = get_dataloader(
        Config.TRAIN_DIR, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        split='train'
    )
    val_loader = get_dataloader(
        Config.VAL_DIR, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        split='val'
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    print("\nCreating model...")
    model = get_model(
        num_classes=Config.NUM_CLASSES,
        pretrained=Config.PRETRAINED,
        device=Config.DEVICE
    )
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=Config,
        device=Config.DEVICE
    )
    
    print("\nTraining history saved to logs/")


if __name__ == '__main__':
    main()