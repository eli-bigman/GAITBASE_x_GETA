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

    print("Setting up model...")
    trainer.setup_model()
    
    print("Setting up data...")
    trainer.setup_data()
    
    # Quick fix: Check for checkpoint restoration and get starting iteration
    restore_hint = trainer.cfg['trainer_cfg'].get('restore_hint', 0)
    starting_iteration = 0
    
    if restore_hint > 0:
        print(f"🔄 Attempting to resume from iteration {restore_hint}...")
        
        # Look for the checkpoint file
        save_name = trainer.cfg['trainer_cfg']['save_name']
        checkpoint_path = f'./checkpoints/{save_name}-{restore_hint:05d}.pt'
        
        try:
            if os.path.exists(checkpoint_path):
                print(f"📂 Found checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # Load model state
                trainer.model.load_state_dict(checkpoint['model'])
                
                # Get the actual iteration from checkpoint
                starting_iteration = checkpoint.get('iteration', restore_hint)
                print(f"✅ Model restored from iteration {starting_iteration}")
                
                # Load optimizer state if available and not reset
                if not trainer.cfg['trainer_cfg'].get('optimizer_reset', False):
                    if 'optimizer' in checkpoint and hasattr(trainer, 'optimizer'):
                        try:
                            trainer.optimizer.load_state_dict(checkpoint['optimizer'])
                            print("✅ Optimizer state restored")
                        except Exception as e:
                            print(f"⚠️ Could not restore optimizer state: {e}")
                
                print(f"🎯 Training will resume from iteration {starting_iteration}")
            else:
                print(f"⚠️ Checkpoint not found at {checkpoint_path}, starting from iteration 0")
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}")
            print("Starting fresh training from iteration 0")
    
    # Store starting iteration for the trainer
    trainer.starting_iteration = starting_iteration
    
    if args.validate:
        print("Validating compression compatibility...")
        if not trainer.validate_compression_compatibility():
            print("❌ Compatibility issues detected. Please review model architecture.")
            return
        print("✅ Compatibility check passed. Proceeding with training.")
    else:
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