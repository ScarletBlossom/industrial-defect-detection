import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
import torch
from torchvision import transforms

from src.config import Config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_normal_image(size=224):
    """Generate a normal (no defect) product surface image"""
    img = Image.new('RGB', (size, size), color=(
        random.randint(180, 220),
        random.randint(180, 220),
        random.randint(180, 220)
    ))
    
    draw = ImageDraw.Draw(img)
    
    for _ in range(random.randint(5, 15)):
        x1 = random.randint(0, size)
        y1 = random.randint(0, size)
        x2 = random.randint(0, size)
        y2 = random.randint(0, size)
        color = (
            random.randint(150, 250),
            random.randint(150, 250),
            random.randint(150, 250)
        )
        draw.line([(x1, y1), (x2, y2)], fill=color, width=1)
    
    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    
    return img


def generate_defect_image(size=224):
    """Generate a defective product surface image"""
    img = Image.new('RGB', (size, size), color=(
        random.randint(180, 220),
        random.randint(180, 220),
        random.randint(180, 220)
    ))
    
    draw = ImageDraw.Draw(img)
    
    defect_types = ['scratch', 'dent', 'spot', 'crack', 'discoloration']
    defect_type = random.choice(defect_types)
    
    if defect_type == 'scratch':
        for _ in range(random.randint(1, 3)):
            x1 = random.randint(0, size)
            y1 = random.randint(0, size)
            length = random.randint(30, 100)
            angle = random.uniform(0, 3.14159)
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            draw.line([(x1, y1), (x2, y2)], fill=(80, 80, 80), width=random.randint(2, 5))
    
    elif defect_type == 'dent':
        cx = random.randint(size // 4, 3 * size // 4)
        cy = random.randint(size // 4, 3 * size // 4)
        r = random.randint(10, 30)
        for i in range(r):
            color_val = random.randint(100, 150)
            draw.ellipse([cx - r + i, cy - r + i, cx + r - i, cy + r - i], 
                        outline=(color_val, color_val, color_val))
    
    elif defect_type == 'spot':
        for _ in range(random.randint(3, 8)):
            x = random.randint(0, size)
            y = random.randint(0, size)
            r = random.randint(3, 12)
            draw.ellipse([x - r, y - r, x + r, y + r], 
                        fill=(random.randint(50, 100), random.randint(50, 100), random.randint(50, 100)))
    
    elif defect_type == 'crack':
        x, y = random.randint(0, size // 2), random.randint(0, size)
        for _ in range(random.randint(5, 15)):
            nx = x + random.randint(-10, 10)
            ny = y + random.randint(-10, 10)
            draw.line([(x, y), (nx, ny)], fill=(60, 60, 60), width=1)
            x, y = nx, ny
    
    else:
        x1 = random.randint(0, size // 2)
        y1 = random.randint(0, size // 2)
        x2 = x1 + random.randint(size // 4, size // 2)
        y2 = y1 + random.randint(size // 4, size // 2)
        draw.rectangle([x1, y1, x2, y2], fill=(random.randint(100, 150), 
                                               random.randint(100, 150), 
                                               random.randint(100, 150)))
    
    return img


def generate_dataset(output_dir, num_samples, split='train'):
    """Generate synthetic dataset"""
    set_seed(Config.SEED + hash(split) % 1000)
    
    normal_dir = os.path.join(output_dir, 'normal')
    defect_dir = os.path.join(output_dir, 'defect')
    
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(defect_dir, exist_ok=True)
    
    num_normal = num_samples // 2
    num_defect = num_samples - num_normal
    
    print(f"Generating {num_normal} normal images and {num_defect} defect images for {split}...")
    
    for i in range(num_normal):
        img = generate_normal_image(Config.IMAGE_SIZE)
        img.save(os.path.join(normal_dir, f'normal_{i:04d}.jpg'))
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_normal} normal images")
    
    for i in range(num_defect):
        img = generate_defect_image(Config.IMAGE_SIZE)
        img.save(os.path.join(defect_dir, f'defect_{i:04d}.jpg'))
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_defect} defect images")
    
    print(f"Done generating {split} dataset!")


def main():
    print("=" * 50)
    print("Generating synthetic defect detection dataset")
    print("=" * 50)
    
    generate_dataset(Config.TRAIN_DIR, Config.NUM_TRAIN_SAMPLES, 'train')
    generate_dataset(Config.VAL_DIR, Config.NUM_VAL_SAMPLES, 'val')
    generate_dataset(Config.TEST_DIR, Config.NUM_TEST_SAMPLES, 'test')
    
    print("\nDataset generation complete!")
    print(f"Train: {Config.NUM_TRAIN_SAMPLES} samples")
    print(f"Val: {Config.NUM_VAL_SAMPLES} samples")
    print(f"Test: {Config.NUM_TEST_SAMPLES} samples")


if __name__ == '__main__':
    main()