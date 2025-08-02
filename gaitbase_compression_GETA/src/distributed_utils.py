#!/usr/bin/env python3

import os
import torch
import torch.distributed as dist

def init_distributed_training(rank=0, world_size=1, backend=None):
    """
    Initialize distributed training for single or multi-GPU setups
    
    Args:
        rank (int): Process rank (0 for single GPU)
        world_size (int): Total number of processes (1 for single GPU)
        backend (str): Backend to use ('nccl' for GPU, 'gloo' for CPU)
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    
    try:
        # Check if already initialized
        if dist.is_initialized():
            print("✅ Distributed training already initialized")
            return True
        
        # Set environment variables
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29500')
        os.environ.setdefault('RANK', str(rank))
        os.environ.setdefault('WORLD_SIZE', str(world_size))
        
        # Auto-detect backend if not specified
        if backend is None:
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        
        # Initialize process group
        dist.init_process_group(
            backend=backend,
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        print(f"✅ Distributed training initialized:")
        print(f"   Backend: {backend}")
        print(f"   Rank: {rank}")
        print(f"   World Size: {world_size}")
        print(f"   Master: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize distributed training: {e}")
        return False

def cleanup_distributed():
    """Clean up distributed training"""
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
            print("✅ Distributed training cleaned up")
    except Exception as e:
        print(f"⚠️ Error during distributed cleanup: {e}")

def is_main_process():
    """Check if this is the main process (rank 0)"""
    try:
        return not dist.is_initialized() or dist.get_rank() == 0
    except:
        return True

def get_rank():
    """Get current process rank"""
    try:
        return dist.get_rank() if dist.is_initialized() else 0
    except:
        return 0

def get_world_size():
    """Get world size"""
    try:
        return dist.get_world_size() if dist.is_initialized() else 1
    except:
        return 1

if __name__ == '__main__':
    # Test the initialization
    print("Testing distributed training initialization...")
    
    success = init_distributed_training()
    
    if success:
        print(f"Main process: {is_main_process()}")
        print(f"Rank: {get_rank()}")
        print(f"World size: {get_world_size()}")
        cleanup_distributed()
    else:
        print("Initialization failed")
