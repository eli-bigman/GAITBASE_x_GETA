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
        
        # Test working directory change
        original_cwd = os.getcwd()
        os.chdir(opengait_path)
        
        try:
            from opengait.utils import config_loader
            print("✅ OpenGait import successful")
            
            # Test config loading with default config
            default_config = os.path.join(opengait_path, 'configs', 'default.yaml')
            if os.path.exists(default_config):
                print(f"✅ Default config found: {default_config}")
                cfg = config_loader(default_config)
                print("✅ Config loading successful")
            else:
                print(f"❌ Default config not found: {default_config}")
                return False
                
        except ImportError as e:
            print(f"❌ OpenGait import failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Config loading failed: {e}")
            return False
        finally:
            os.chdir(original_cwd)
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
    else:
        print(f"❌ Custom config not found: {custom_config}")
        return False
    
    print("✅ All tests passed! Environment is ready.")
    return True

def test_integration():
    """Test the GETA-OpenGait integration"""
    print("\n=== Testing GETA-OpenGait Integration ===")
    
    try:
        from geta_opengait_integration import GETAOpenGaitTrainer
        print("✅ Integration module import successful")
        
        # Test config loading
        config_path = os.path.join(os.path.dirname(__file__), 'gaitbase_geta.yaml')
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

if __name__ == '__main__':
    success = test_setup()
    if success:
        success = test_integration()
    
    if success:
        print("\n🎉 All tests passed! You can proceed with training.")
    else:
        print("\n❌ Some tests failed. Please fix the issues before training.")
