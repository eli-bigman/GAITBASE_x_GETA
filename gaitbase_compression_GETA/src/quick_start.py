#!/usr/bin/env python3

"""
Quick Start Script for GETA-OpenGait Integration

This script handles all the setup and initialization needed to run 
GETA compression with OpenGait's GaitBase model.

Usage:
    python quick_start.py --action [test|train|evaluate]
"""

import os
import sys
import argparse

def setup_environment():
    """Setup all required paths and environment variables"""
    print("🔧 Setting up environment...")
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find GETA path
    geta_candidates = [
        '/kaggle/working/geta',
        '/kaggle/working/GAITBASE_x_GETA/geta',
        os.path.join(current_dir, '../../geta'),
    ]
    
    geta_path = None
    for path in geta_candidates:
        if os.path.exists(path):
            geta_path = os.path.abspath(path)
            sys.path.insert(0, geta_path)
            print(f"✅ Found GETA: {geta_path}")
            break
    
    if not geta_path:
        print("❌ GETA not found. Please check your installation.")
        return False, None, None
    
    # Find OpenGait path
    opengait_candidates = [
        '/kaggle/working/OpenGait',
        '/kaggle/working/GAITBASE_x_GETA/OpenGait',
        os.path.join(current_dir, '../../OpenGait'),
    ]
    
    opengait_path = None
    for path in opengait_candidates:
        if os.path.exists(path):
            opengait_path = os.path.abspath(path)
            sys.path.insert(0, opengait_path)
            print(f"✅ Found OpenGait: {opengait_path}")
            break
    
    if not opengait_path:
        print("❌ OpenGait not found. Please check your installation.")
        return False, None, None
    
    # Set environment variables
    os.environ['GETA_PATH'] = geta_path
    os.environ['OPENGAIT_PATH'] = opengait_path
    
    return True, geta_path, opengait_path

def init_distributed():
    """Initialize distributed training for single GPU"""
    print("🔧 Initializing distributed training...")
    
    try:
        import torch
        import torch.distributed as dist
        
        # Check if already initialized
        if dist.is_initialized():
            print("✅ Distributed training already initialized")
            return True
        
        # Set environment variables
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29500')
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '1')
        
        # Initialize
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(
            backend=backend,
            init_method='env://',
            world_size=1,
            rank=0
        )
        
        print(f"✅ Distributed training initialized with {backend}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize distributed training: {e}")
        return False

def test_setup():
    """Test if everything is working"""
    print("🧪 Testing setup...")
    
    try:
        # Test GETA import
        from only_train_once import OTO
        print("✅ GETA import successful")
        
        # Test OpenGait imports
        from opengait.utils import get_msg_mgr
        from opengait.modeling import models
        print("✅ OpenGait imports successful")
        
        # Test message manager
        msg_mgr = get_msg_mgr()
        msg_mgr.log_info("Test message from quick start script")
        print("✅ Message manager working")
        
        # Test GPU
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ GPU not available, will use CPU")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_training(config_path, gpu='0', validate=True):
    """Run GETA training"""
    print("🚀 Starting GETA training...")
    
    try:
        from geta_opengait_integration import GETAOpenGaitTrainer
        
        # Set GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        
        # Initialize trainer
        trainer = GETAOpenGaitTrainer(config_path)
        
        if validate:
            print("🔍 Running compatibility check...")
            if not trainer.validate_compression_compatibility():
                print("❌ Compatibility issues detected")
                return False
        
        # Setup components
        print("🏗️ Setting up model...")
        trainer.setup_model()
        
        print("📊 Setting up data...")
        trainer.setup_data()
        
        print("⚙️ Setting up GETA...")
        trainer.setup_geta_oto()
        
        print("🎯 Setting up losses...")
        trainer.setup_losses()
        
        # Start training
        print("🚀 Starting training...")
        trainer.train()
        
        print("✅ Training completed!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_evaluation(config_path, model_path=None):
    """Run model evaluation"""
    print("📊 Starting evaluation...")
    
    try:
        from geta_opengait_integration import GETAOpenGaitTrainer
        
        trainer = GETAOpenGaitTrainer(config_path)
        trainer.setup_model()
        trainer.setup_data()
        
        if model_path:
            results = trainer.evaluate_model(model_path)
        else:
            # Look for recent checkpoints
            checkpoint_dir = './checkpoints'
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
                if checkpoints:
                    latest_checkpoint = os.path.join(checkpoint_dir, sorted(checkpoints)[-1])
                    print(f"🔍 Using latest checkpoint: {latest_checkpoint}")
                    results = trainer.evaluate_model(latest_checkpoint)
                else:
                    print("❌ No checkpoints found")
                    return False
            else:
                print("❌ No checkpoint directory found")
                return False
        
        print("📊 Evaluation Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='GETA-OpenGait Quick Start Script')
    parser.add_argument('--action', choices=['test', 'train', 'evaluate'], 
                       default='test', help='Action to perform')
    parser.add_argument('--config', default='gaitbase_geta.yaml',
                       help='Config file path')
    parser.add_argument('--gpu', default='0', help='GPU device')
    parser.add_argument('--model', help='Model path for evaluation')
    parser.add_argument('--no-validate', action='store_true',
                       help='Skip validation before training')
    
    args = parser.parse_args()
    
    print("🚀 GETA-OpenGait Quick Start")
    print("=" * 50)
    
    # Setup environment
    success, geta_path, opengait_path = setup_environment()
    if not success:
        sys.exit(1)
    
    # Initialize distributed training
    if not init_distributed():
        print("⚠️ Distributed training failed, but continuing...")
    
    # Convert config to absolute path
    if not os.path.isabs(args.config):
        args.config = os.path.abspath(args.config)
    
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)
    
    # Perform requested action
    if args.action == 'test':
        success = test_setup()
    elif args.action == 'train':
        success = run_training(args.config, args.gpu, not args.no_validate)
    elif args.action == 'evaluate':
        success = run_evaluation(args.config, args.model)
    
    if success:
        print("\n🎉 Operation completed successfully!")
    else:
        print("\n❌ Operation failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()
