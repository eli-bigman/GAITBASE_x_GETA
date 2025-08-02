#!/usr/bin/env python3

"""
Simple test script to verify distributed training initialization works
"""

import os
import sys

def test_distributed_init():
    """Test distributed training initialization"""
    print("=== Testing Distributed Training Initialization ===")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        
        import torch.distributed as dist
        print("✅ torch.distributed imported successfully")
        
        # Check if already initialized
        try:
            is_init = dist.is_initialized()
            print(f"📊 Distributed already initialized: {is_init}")
        except Exception as e:
            print(f"⚠️ Cannot check if distributed is initialized: {e}")
            is_init = False
        
        if not is_init:
            # Set up environment
            os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
            os.environ.setdefault('MASTER_PORT', '29500')
            os.environ.setdefault('RANK', '0')
            os.environ.setdefault('WORLD_SIZE', '1')
            
            print("🔧 Environment variables set:")
            print(f"   MASTER_ADDR: {os.environ['MASTER_ADDR']}")
            print(f"   MASTER_PORT: {os.environ['MASTER_PORT']}")
            print(f"   RANK: {os.environ['RANK']}")
            print(f"   WORLD_SIZE: {os.environ['WORLD_SIZE']}")
            
            # Try initialization
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            print(f"🚀 Attempting to initialize with backend: {backend}")
            
            dist.init_process_group(
                backend=backend,
                init_method='env://',
                world_size=1,
                rank=0
            )
            
            print("✅ Distributed training initialized successfully!")
            print(f"   Rank: {dist.get_rank()}")
            print(f"   World Size: {dist.get_world_size()}")
            
            # Clean up
            dist.destroy_process_group()
            print("✅ Distributed training cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Distributed initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_opengait_msgmgr():
    """Test OpenGait message manager after distributed init"""
    print("\n=== Testing OpenGait Message Manager ===")
    
    try:
        # Add OpenGait to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        opengait_path = os.path.join(current_dir, '../../OpenGait')
        
        if os.path.exists(opengait_path):
            sys.path.insert(0, opengait_path)
            print(f"✅ Added OpenGait path: {opengait_path}")
        else:
            print(f"❌ OpenGait not found: {opengait_path}")
            return False
        
        # Initialize distributed first
        if not test_distributed_init():
            print("❌ Cannot test message manager without distributed training")
            return False
        
        # Test message manager
        from opengait.utils import get_msg_mgr
        msg_mgr = get_msg_mgr()
        print("✅ Message manager created successfully")
        
        # Test basic functionality
        msg_mgr.log_info("Test message from distributed training test")
        print("✅ Message manager works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Message manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("🚀 Starting distributed training compatibility test")
    
    success1 = test_distributed_init()
    success2 = test_opengait_msgmgr()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Distributed training is ready.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        sys.exit(1)
