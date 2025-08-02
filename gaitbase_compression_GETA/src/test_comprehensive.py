#!/usr/bin/env python3

"""
Comprehensive test script for GETA-OpenGait integration with distributed training fix
"""

import os
import sys

def test_distributed_initialization():
    """Test distributed training initialization"""
    print("🧪 Testing Distributed Training Initialization")
    print("=" * 60)
    
    try:
        import torch
        import torch.distributed as dist
        
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        
        # Check if already initialized
        try:
            is_init = dist.is_initialized()
            print(f"📊 Distributed already initialized: {is_init}")
        except Exception as e:
            print(f"⚠️ Cannot check if distributed is initialized: {e}")
            is_init = False
        
        if not is_init:
            # Set up environment variables
            os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
            os.environ.setdefault('MASTER_PORT', '29500')
            os.environ.setdefault('RANK', '0')
            os.environ.setdefault('WORLD_SIZE', '1')
            
            print("🔧 Environment variables:")
            print(f"   MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
            print(f"   MASTER_PORT: {os.environ.get('MASTER_PORT')}")
            print(f"   RANK: {os.environ.get('RANK')}")
            print(f"   WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
            
            # Initialize distributed training
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            print(f"🚀 Initializing with backend: {backend}")
            
            dist.init_process_group(
                backend=backend,
                init_method='env://',
                world_size=1,
                rank=0
            )
            
            print("✅ Distributed training initialized successfully!")
            print(f"   Rank: {dist.get_rank()}")
            print(f"   World Size: {dist.get_world_size()}")
            
            return True
        else:
            print("✅ Distributed training was already initialized")
            return True
        
    except Exception as e:
        print(f"❌ Distributed initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_opengait_with_distributed():
    """Test OpenGait components with distributed training"""
    print("\n🧪 Testing OpenGait with Distributed Training")
    print("=" * 60)
    
    try:
        # Add paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        opengait_path = os.path.join(current_dir, '../../OpenGait')
        
        if os.path.exists(opengait_path):
            sys.path.insert(0, opengait_path)
            print(f"✅ Added OpenGait path: {opengait_path}")
        else:
            print(f"❌ OpenGait not found: {opengait_path}")
            return False
        
        # Test OpenGait imports
        from opengait.utils import get_msg_mgr, config_loader
        print("✅ OpenGait imports successful")
        
        # Test message manager
        msg_mgr = get_msg_mgr()
        print("✅ Message manager created successfully")
        
        # Test basic functionality
        msg_mgr.log_info("Test message from comprehensive test script")
        print("✅ Message manager works correctly")
        
        # Test config loading
        default_config = os.path.join(opengait_path, 'configs', 'default.yaml')
        if os.path.exists(default_config):
            print(f"✅ Default config found: {default_config}")
            
            # Change to OpenGait directory temporarily
            original_cwd = os.getcwd()
            os.chdir(opengait_path)
            
            try:
                cfg = config_loader(default_config)
                print("✅ Config loading successful")
            finally:
                os.chdir(original_cwd)
        else:
            print(f"❌ Default config not found: {default_config}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ OpenGait test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_geta_opengait_integration():
    """Test the actual GETA-OpenGait integration"""
    print("\n🧪 Testing GETA-OpenGait Integration")
    print("=" * 60)
    
    try:
        # Test GETA import
        current_dir = os.path.dirname(os.path.abspath(__file__))
        geta_path = os.path.join(current_dir, '../../geta')
        
        if os.path.exists(geta_path):
            sys.path.insert(0, geta_path)
            print(f"✅ Added GETA path: {geta_path}")
        else:
            print(f"❌ GETA not found: {geta_path}")
            return False
        
        from only_train_once import OTO
        print("✅ GETA import successful")
        
        # Test integration import
        from geta_opengait_integration import GETAOpenGaitTrainer
        print("✅ Integration module import successful")
        
        # Test trainer initialization
        config_path = os.path.join(current_dir, 'gaitbase_geta.yaml')
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            return False
        
        print("🚀 Initializing GETA-OpenGait trainer...")
        trainer = GETAOpenGaitTrainer(config_path)
        print("✅ Trainer initialization successful")
        
        # Test model setup
        print("🏗️ Testing model setup...")
        trainer.setup_model()
        print("✅ Model setup successful")
        
        # Test compression compatibility
        print("🔍 Testing compression compatibility...")
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

def cleanup_distributed():
    """Clean up distributed training"""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
            print("✅ Distributed training cleaned up")
    except Exception as e:
        print(f"⚠️ Error during cleanup: {e}")

def main():
    """Run all tests"""
    print("🚀 GETA-OpenGait Comprehensive Test Suite")
    print("=" * 70)
    
    results = {
        'distributed_init': False,
        'opengait_test': False,
        'integration_test': False
    }
    
    # Test 1: Distributed Training Initialization
    results['distributed_init'] = test_distributed_initialization()
    
    # Test 2: OpenGait with Distributed Training
    if results['distributed_init']:
        results['opengait_test'] = test_opengait_with_distributed()
    else:
        print("\n⏭️ Skipping OpenGait test due to distributed initialization failure")
    
    # Test 3: GETA-OpenGait Integration
    if results['opengait_test']:
        results['integration_test'] = test_geta_opengait_integration()
    else:
        print("\n⏭️ Skipping integration test due to OpenGait test failure")
    
    # Cleanup
    cleanup_distributed()
    
    # Final Results
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title():<30}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your GETA-OpenGait integration is ready!")
        print("\nYou can now run:")
        print("  python src/quick_start.py --action train --config src/gaitbase_geta.yaml")
        return True
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please check the errors above.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
