#!/usr/bin/env python3

import argparse
import os
import torch
from geta_opengait_integration import GETAOpenGaitTrainer

def main():
    parser = argparse.ArgumentParser(description='Train GaitBase with GETA compression')
    parser.add_argument('--config', '-c', 
                       default='./configs/gaitbase/gaitbase_da_casiab_geta.yaml',
                       help='Path to config file')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device')
    
    args = parser.parse_args()
    
            
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Initialize trainer
    trainer = GETAOpenGaitTrainer(args.config)



    if args.validate:
        print("Validating compression compatibility...")
        if not trainer.validate_compression_compatibility():
            print("❌ Compatibility issues detected. Please review model architecture.")
            return
        print("✅ Compatibility check passed. Proceeding with training.")


    
    print("Setting up model...")
    trainer.setup_model()
    
    print("Setting up data...")
    trainer.setup_data()
    
    print("Setting up GETA OTO...")
    trainer.setup_geta_oto()
    
    print("Setting up losses...")
    trainer.setup_losses()
    
    print("Starting training with compression...")
    trainer.train()
    
    print("Evaluating compression results...")
    results = trainer.evaluate_compression()
    
    print("Training and compression completed!")
    print(f"Final compression ratio: {results['compression_ratio']:.3f}")

if __name__ == '__main__':
    main()






"""
requirements
# Install GETA requirements
pip install torch torchvision torchaudio
pip install thop torchsummary

# Install OpenGait requirements (if not already done)
pip install pyyaml tensorboard opencv-python

Run

python train_gaitbase_with_geta.py --config gaitbase_geta.yaml --gpu 0
"""