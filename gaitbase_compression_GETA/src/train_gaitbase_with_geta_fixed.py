#!/usr/bin/env python3

import os
import argparse
import sys

def main():
    """
    Fixed entry point for GETA-OpenGait training that handles working directory issues
    """
    parser = argparse.ArgumentParser(description='Train GaitBase with GETA compression - FIXED VERSION')
    parser.add_argument('--config', '-c', 
                       default='gaitbase_geta.yaml',
                       help='Path to config file')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device')
    parser.add_argument('--validate', action='store_true', 
                       help='Run compatibility validation before training')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with more verbose output')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Get absolute paths to avoid working directory issues
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # GETA path resolution
    geta_candidates = [
        '/kaggle/working/geta',
        '/kaggle/working/GAITBASE_x_GETA/geta',
        os.path.join(current_dir, '../../geta'),
        os.path.abspath(os.path.join(current_dir, '../../geta'))
    ]
    
    geta_path = None
    for path in geta_candidates:
        if os.path.exists(path):
            geta_path = os.path.abspath(path)
            sys.path.insert(0, geta_path)
            print(f"✅ Added GETA path: {geta_path}")
            break
    
    if not geta_path:
        print("❌ GETA directory not found in any of these locations:")
        for path in geta_candidates:
            print(f"  - {path}")
        return 1
    
    # OpenGait path resolution
    opengait_candidates = [
        '/kaggle/working/OpenGait',
        '/kaggle/working/GAITBASE_x_GETA/OpenGait',
        os.path.join(current_dir, '../../OpenGait'),
        os.path.abspath(os.path.join(current_dir, '../../OpenGait'))
    ]
    
    opengait_path = None
    for path in opengait_candidates:
        if os.path.exists(path):
            opengait_path = os.path.abspath(path)
            sys.path.insert(0, opengait_path)
            print(f"✅ Added OpenGait path: {opengait_path}")
            break
    
    if not opengait_path:
        print("❌ OpenGait directory not found in any of these locations:")
        for path in opengait_candidates:
            print(f"  - {path}")
        return 1
    
    # Convert config path to absolute path
    if not os.path.isabs(args.config):
        args.config = os.path.abspath(args.config)
    
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return 1
    
    print(f"📁 Using config: {args.config}")
    
    # Store paths globally for the integration module
    os.environ['OPENGAIT_PATH'] = opengait_path
    os.environ['GETA_PATH'] = geta_path
    
    try:
        # Import after path setup
        from geta_opengait_integration import GETAOpenGaitTrainer
        
        print("🚀 Initializing GETA-OpenGait trainer...")
        trainer = GETAOpenGaitTrainer(args.config)
        
        if args.validate:
            print("🔍 Running compatibility validation...")
            if not trainer.validate_compression_compatibility():
                print("❌ Compatibility issues detected. Please review model architecture.")
                return 1
            print("✅ Compatibility check passed.")
        
        print("🏗️ Setting up model...")
        trainer.setup_model()
        
        print("📊 Setting up data...")
        trainer.setup_data()
        
        print("⚙️ Setting up GETA OTO...")
        trainer.setup_geta_oto()
        
        print("🎯 Setting up losses...")
        trainer.setup_losses()
        
        print("🚀 Starting training with compression...")
        trainer.train()
        
        print("📊 Evaluating compression results...")
        results = trainer.evaluate_compression()
        
        print("✅ Training and compression completed!")
        print(f"📈 Final compression ratio: {results['compression_ratio']:.3f}")
        print(f"💾 Size reduction: {((1 - results['compression_ratio']) * 100):.1f}%")
        
        return 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔧 Try installing missing dependencies or check path configuration")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    except Exception as e:
        print(f"❌ Training failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
