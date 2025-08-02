#!/usr/bin/env python3

import argparse
import os
import sys
import torch

def main():
    parser = argparse.ArgumentParser(description='Train GaitBase with GETA compression')
    parser.add_argument('--config', '-c', 
                       default='gaitbase_geta.yaml',
                       help='Path to config file')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device')
    parser.add_argument('--validate', action='store_true', 
                       help='Run compatibility validation before training')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Add paths dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    geta_path = os.path.join(current_dir, '../../geta')
    opengait_path = os.path.join(current_dir, '../../OpenGait')
    
    if os.path.exists(geta_path):
        sys.path.insert(0, geta_path)
        print(f"✅ Added GETA path: {geta_path}")
    else:
        print(f"❌ GETA path not found: {geta_path}")
        return
    
    if os.path.exists(opengait_path):
        sys.path.insert(0, opengait_path)
        print(f"✅ Added OpenGait path: {opengait_path}")
    else:
        print(f"❌ OpenGait path not found: {opengait_path}")
        return

    # Import after path setup
    from geta_opengait_integration import GETAOpenGaitTrainer

    # Convert relative path to absolute path
    if not os.path.isabs(args.config):
        args.config = os.path.abspath(args.config)
    
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return

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