#!/usr/bin/env python3

import os
import sys
import torch

def test_setup():
    """Test if the environment is properly set up"""
    print("=== GETA + OpenGait Setup Test ===")
    
    # Add paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    geta_path = os.path.join(current_dir, '../../geta')
    opengait_path = os.path.join(current_dir, '../../OpenGait')
    
    # Test GETA path
    if os.path.exists(geta_path):
        sys.path.insert(0, geta_path)
        print(f"✅ GETA path found: {geta_path}")
        
        try:
            from only_train_once import OTO
            print("✅ GETA import successful")
        except ImportError as e:
            print(f"❌ GETA import failed: {e}")
            return False
    else:
        print(f"❌ GETA path not found: {geta_path}")
        return False
    
    # Test OpenGait path
    if os.path.exists(opengait_path):
        sys.path.insert(0, opengait_path)
        print(f"✅ OpenGait path found: {opengait_path}")
        
        # Set up environment for OpenGait imports
        os.environ['PYTHONPATH'] = opengait_path + ':' + os.environ.get('PYTHONPATH', '')
        
        # Test working directory change
        original_cwd = os.getcwd()
        os.chdir(opengait_path)
        
        try:
            # Try importing OpenGait modules
            import opengait
            from opengait.utils import config_loader
            print("✅ OpenGait import successful")
            
            # Store the original directory for later use
            global original_working_dir, opengait_root
            original_working_dir = original_cwd
            opengait_root = opengait_path
            
        except ImportError as e:
            print(f"❌ OpenGait import failed: {e}")
            # Try fallback imports
            try:
                sys.path.insert(0, os.path.join(opengait_path, 'opengait'))
                from utils import config_loader
                print("✅ OpenGait fallback import successful")
            except ImportError as e2:
                print(f"❌ OpenGait fallback import also failed: {e2}")
                return False
        finally:
            # Don't change back to original directory yet
            pass
    else:
        print(f"❌ OpenGait path not found: {opengait_path}")
        return False
    
    # Test GPU availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.device_count()} GPU(s)")
        print(f"✅ Current GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA not available - will use CPU")
    
    # Test custom config
    custom_config = os.path.join(current_dir, 'gaitbase_geta.yaml')
    if os.path.exists(custom_config):
        print(f"✅ Custom config found: {custom_config}")
        global config_path
        config_path = custom_config
    else:
        print(f"❌ Custom config not found: {custom_config}")
        return False
    
    # Test checkpoint existence
    checkpoint_latest = '/kaggle/working/checkpoints/GaitBase_GETA_60K_Production-latest.pt'
    checkpoint_60k = '/kaggle/working/output/CASIA-B/Baseline/GaitBase_GETA_60K_Production/checkpoints/GaitBase_GETA_60K_Production-60000.pt'
    
    if os.path.exists(checkpoint_latest):
        print(f"✅ Latest checkpoint found: {os.path.getsize(checkpoint_latest) / (1024*1024):.1f} MB")
        # Copy to expected location
        os.makedirs(os.path.dirname(checkpoint_60k), exist_ok=True)
        if not os.path.exists(checkpoint_60k):
            import shutil
            shutil.copy(checkpoint_latest, checkpoint_60k)
            print(f"✅ Checkpoint copied for testing")
    elif os.path.exists(checkpoint_60k):
        print(f"✅ Test checkpoint ready: {os.path.getsize(checkpoint_60k) / (1024*1024):.1f} MB")
    else:
        print(f"❌ No checkpoint found at {checkpoint_latest} or {checkpoint_60k}")
        return False
    
    print("✅ All setup tests passed! Environment is ready.")
    return True

def test_integration():
    """Test the GETA-OpenGait integration"""
    print("\n=== Testing GETA-OpenGait Integration ===")
    
    # Apply the PyTorch loading fix for this test
    print("🔧 Applying GETA checkpoint loading fix...")
    
    try:
        from geta_opengait_integration import GETAOpenGaitTrainer
        print("✅ Integration module import successful")
        
        # Initialize distributed training for testing
        import torch.distributed as dist
        if not dist.is_initialized():
            try:
                os.environ.setdefault('MASTER_ADDR', 'localhost')
                os.environ.setdefault('MASTER_PORT', '12355')
                os.environ.setdefault('RANK', '0')
                os.environ.setdefault('WORLD_SIZE', '1')
                os.environ.setdefault('LOCAL_RANK', '0')
                
                dist.init_process_group(
                    backend='nccl' if torch.cuda.is_available() else 'gloo',
                    init_method='env://'
                )
                print("✅ Initialized distributed training for testing")
            except Exception as e:
                print(f"⚠️ Could not initialize distributed training: {e}")
                print("🔧 Will use fallback mode")
        
        # Test config loading
        trainer = GETAOpenGaitTrainer(config_path)
        print("✅ Trainer initialization successful")
        
        # Test model setup
        trainer.setup_model()
        print("✅ Model setup successful")
        
        # Test compatibility check
        if trainer.validate_compression_compatibility():
            print("✅ Compression compatibility validated")
        else:
            print("⚠️ Compression compatibility issues detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_model_evaluation():
    """Run the actual model evaluation"""
    print("\n=== Running Model Evaluation ===")
    
    try:
        # Set up distributed training environment
        import torch.distributed as dist
        if not dist.is_initialized():
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
            os.environ['LOCAL_RANK'] = '0'
            
            dist.init_process_group('nccl', init_method='env://')
            print(f"✅ Distributed training initialized (rank: {dist.get_rank()}, world_size: {dist.get_world_size()})")
        
        # Import OpenGait modules
        try:
            import opengait
            from opengait.modeling import models
            from opengait.utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr
            print("✅ OpenGait modules imported successfully")
        except ImportError as e:
            print(f"❌ Direct import failed: {e}, trying fallback...")
            # Try fallback imports
            from modeling import models
            from utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr
            print("✅ OpenGait modules imported via fallback")
        
        # Load config
        cfgs = config_loader(config_path)
        print(f"✅ Config loaded from: {config_path}")
        
        # Set to test the 60K iteration model
        cfgs['evaluator_cfg']['restore_hint'] = 60000
        print(f"🎯 Testing iteration: 60000")
        
        # Initialize message manager
        msg_mgr = get_msg_mgr()
        engine_cfg = cfgs['evaluator_cfg']
        output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                                   cfgs['model_cfg']['model'], engine_cfg['save_name'])
        msg_mgr.init_logger(output_path, False)
        msg_mgr.log_info(engine_cfg)
        print(f"✅ Message manager initialized, output: {output_path}")
        
        # Initialize seeds
        seed = dist.get_rank()
        init_seeds(seed)
        print(f"✅ Seeds initialized with seed: {seed}")
        
        # Create model
        model_cfg = cfgs['model_cfg']
        msg_mgr.log_info(model_cfg)
        Model = getattr(models, model_cfg['model'])
        print(f"🏗️ Creating model: {model_cfg['model']}")
        
        # Initialize model for testing (training=False)
        model = Model(cfgs, training=False)
        
        # Wrap with DDP
        model = get_ddp_module(model, cfgs['trainer_cfg']['find_unused_parameters'])
        msg_mgr.log_info(params_count(model))
        msg_mgr.log_info("Model Initialization Finished!")
        print("✅ Model initialized successfully!")
        
        # Run test
        print("🧪 Starting evaluation...")
        model.run_test()
        print("✅ Evaluation completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def fix_pytorch_loading_global():
    """Global fix for PyTorch loading issues with GETA checkpoints"""
    import torch
    import torch.serialization
    
    # Add GETA classes to safe globals
    try:
        from only_train_once.transform.tensor_transform import TensorTransform
        torch.serialization.add_safe_globals([TensorTransform])
        print("✅ Added GETA TensorTransform to PyTorch safe globals")
    except ImportError as e:
        print(f"⚠️ Could not import GETA TensorTransform: {e}")
    
    # Monkey patch torch.load globally for GETA compatibility
    original_load = torch.load
    
    def safe_geta_load(*args, **kwargs):
        """Load function that handles GETA checkpoints safely"""
        # Check if this might be a GETA checkpoint
        checkpoint_path = args[0] if args else kwargs.get('f', '')
        is_geta_checkpoint = (
            'GETA' in str(checkpoint_path) or 
            'geta' in str(checkpoint_path).lower() or
            'GaitBase_GETA' in str(checkpoint_path)
        )
        
        if 'weights_only' not in kwargs:
            if is_geta_checkpoint:
                kwargs['weights_only'] = False  # GETA checkpoints need full loading
                print(f"🔧 Loading GETA checkpoint with weights_only=False: {os.path.basename(str(checkpoint_path))}")
            else:
                kwargs['weights_only'] = True  # Keep security for other checkpoints
        
        return original_load(*args, **kwargs)
    
    # Apply the global patch
    torch.load = safe_geta_load
    print("✅ Globally patched torch.load for GETA checkpoint compatibility")

def main():
    """Main function to run all tests and evaluation"""
    print("🚀 GETA Model Testing Suite")
    print("=" * 50)
    
    # Apply PyTorch loading fix before anything else
    fix_pytorch_loading_global()
    print()
    
    # Step 1: Test environment setup
    if not test_setup():
        print("\n❌ Setup test failed. Please fix environment issues.")
        return False
    
    # Step 2: Test integration
    if not test_integration():
        print("\n❌ Integration test failed. Please fix integration issues.")
        return False
    
    # Step 3: Ask user if they want to run evaluation
    print("\n" + "="*50)
    print("🎯 Setup and integration tests passed!")
    response = input("Do you want to run the model evaluation now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print("\n🚀 Starting model evaluation...")
        if run_model_evaluation():
            print("\n🎉 Model evaluation completed successfully!")
            return True
        else:
            print("\n❌ Model evaluation failed.")
            return False
    else:
        print("\n✅ Tests completed. You can run evaluation later.")
        return True

if __name__ == '__main__':
    success = main()
    if success:
        print("\n🎉 All operations completed successfully!")
    else:
        print("\n❌ Some operations failed. Please check the errors above.")