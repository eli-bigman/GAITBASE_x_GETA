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
    
    # Validate paths
    print(f"✅ Found GETA at: {geta_path}")
    print(f"✅ Found OpenGait at: {opengait_path}")
    
    # ✅ FIX: Check if config file exists
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return
    else:
        print(f"✅ Config file found: {args.config}")
    
    # Import and initialize trainer
    try:
        from geta_opengait_integration import GETAOpenGaitTrainer
        
        # ✅ FIX: Pass absolute path to config
        config_path = os.path.abspath(args.config)
        trainer = GETAOpenGaitTrainer(config_path)
        
        if args.validate:
            print("✅ Validation successful - trainer initialized")
            print("🎯 Ready to start training with GETA compression")
        else:
            print("🚀 Starting GETA-compressed training...")
            trainer.train()
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()