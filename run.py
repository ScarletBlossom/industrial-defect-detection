#!/usr/bin/env python3
import sys
import subprocess


def main():
    print("=" * 60)
    print("Industrial Defect Detection System")
    print("=" * 60)
    print("\nSelect an option:")
    print("1. Generate synthetic dataset")
    print("2. Train model")
    print("3. Visualize training results")
    print("4. Launch Gradio web interface")
    print("5. Run full pipeline (generate + train + launch)")
    print("0. Exit")
    
    choice = input("\nEnter your choice: ").strip()
    
    if choice == '1':
        print("\nGenerating synthetic dataset...")
        subprocess.run([sys.executable, "src/data/generate_data.py"])
    
    elif choice == '2':
        print("\nStarting training...")
        subprocess.run([sys.executable, "src/train/train.py"])
    
    elif choice == '3':
        print("\nVisualizing training results...")
        subprocess.run([sys.executable, "src/utils/visualization.py"])
    
    elif choice == '4':
        print("\nLaunching Gradio interface...")
        subprocess.run([sys.executable, "src/inference/gradio_app.py"])
    
    elif choice == '5':
        print("\nRunning full pipeline...")
        print("\n[1/3] Generating dataset...")
        subprocess.run([sys.executable, "src/data/generate_data.py"])
        
        print("\n[2/3] Training model...")
        subprocess.run([sys.executable, "src/train/train.py"])
        
        print("\n[3/3] Visualizing results...")
        subprocess.run([sys.executable, "src/utils/visualization.py"])
        
        print("\n" + "=" * 60)
        print("Pipeline complete!")
        print("To launch Gradio interface, run option 4")
        print("=" * 60)
    
    else:
        print("Invalid choice!")


if __name__ == '__main__':
    main()