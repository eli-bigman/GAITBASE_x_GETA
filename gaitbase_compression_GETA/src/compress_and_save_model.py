#!/usr/bin/env python3
"""
GETA Model Compression and Optimization Script
Properly compresses and saves GETA-trained models with maximum size reduction
"""

import os
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple

class ModelCompressor:
    """Advanced model compression utilities"""
    
    def __init__(self, compression_threshold: float = 1e-6):
        """
        Initialize model compressor
        
        Args:
            compression_threshold: Threshold below which weights are set to zero
        """
        self.compression_threshold = compression_threshold
        self.compression_stats = {}
    
    def analyze_model_format(self, model_path: str) -> Dict[str, Any]:
        """Analyze the format and content of a model"""
        
        if not os.path.exists(model_path):
            return {"error": f"Model not found: {model_path}"}
        
        file_size_mb = os.path.getsize(model_path) / (1024*1024)
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            analysis = {
                "file_size_mb": file_size_mb,
                "keys": list(checkpoint.keys()),
                "file_format": "PyTorch Checkpoint"
            }
            
            if 'model' in checkpoint:
                model_state = checkpoint['model']
                total_params = sum(p.numel() for p in model_state.values() if isinstance(p, torch.Tensor))
                zero_params = sum((p == 0).sum().item() for p in model_state.values() if isinstance(p, torch.Tensor))
                sparsity = zero_params / total_params if total_params > 0 else 0
                
                dtypes = set(str(p.dtype) for p in model_state.values() if isinstance(p, torch.Tensor))
                
                analysis.update({
                    "total_parameters": total_params,
                    "zero_parameters": zero_params,
                    "sparsity_ratio": sparsity,
                    "data_types": list(dtypes),
                    "layers": len([k for k in model_state.keys() if isinstance(model_state[k], torch.Tensor)])
                })
            
            return analysis
            
        except Exception as e:
            return {"error": f"Error loading model: {e}"}
    
    def apply_weight_pruning(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply additional weight pruning based on threshold"""
        
        pruned_state_dict = {}
        total_pruned = 0
        total_params = 0
        
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                original_zeros = (param == 0).sum().item()
                
                # Apply threshold pruning
                mask = torch.abs(param) > self.compression_threshold
                pruned_param = param * mask
                
                new_zeros = (pruned_param == 0).sum().item()
                total_pruned += new_zeros - original_zeros
                total_params += param.numel()
                
                pruned_state_dict[name] = pruned_param
                
                if new_zeros > original_zeros:
                    print(f"🔪 {name}: Pruned {new_zeros - original_zeros:,} additional weights")
            else:
                pruned_state_dict[name] = param
        
        print(f"✂️ Total additional weights pruned: {total_pruned:,}")
        return pruned_state_dict
    
    def convert_to_sparse_format(self, state_dict: Dict[str, torch.Tensor], 
                                sparsity_threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """Convert dense tensors to sparse format where beneficial"""
        
        sparse_state_dict = {}
        conversions = 0
        
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                sparsity = (param == 0).float().mean().item()
                
                if sparsity > sparsity_threshold and param.dim() >= 2:
                    # Convert to sparse COO format
                    sparse_param = param.to_sparse()
                    sparse_state_dict[name] = sparse_param
                    conversions += 1
                    print(f"🗜️ {name}: Converted to sparse format ({sparsity:.1%} sparse)")
                else:
                    sparse_state_dict[name] = param
            else:
                sparse_state_dict[name] = param
        
        print(f"🗜️ Converted {conversions} layers to sparse format")
        return sparse_state_dict
    
    def apply_precision_reduction(self, state_dict: Dict[str, torch.Tensor], 
                                 use_fp16: bool = True) -> Dict[str, torch.Tensor]:
        """Apply precision reduction (FP32 -> FP16) where appropriate"""
        
        reduced_state_dict = {}
        conversions = 0
        
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor) and use_fp16:
                # Convert weights to FP16, but keep biases and batch norm params in FP32
                if ('weight' in name and param.dim() > 1 and 
                    'bn' not in name.lower() and 'norm' not in name.lower()):
                    
                    if hasattr(param, 'to_sparse'):  # Already sparse
                        # Handle sparse tensors
                        values = param._values().half()
                        indices = param._indices()
                        reduced_param = torch.sparse_coo_tensor(indices, values, param.shape)
                    else:
                        reduced_param = param.half()
                    
                    reduced_state_dict[name] = reduced_param
                    conversions += 1
                    print(f"🔄 {name}: Converted to FP16")
                else:
                    reduced_state_dict[name] = param
            else:
                reduced_state_dict[name] = param
        
        print(f"🔄 Converted {conversions} layers to FP16")
        return reduced_state_dict
    
    def calculate_compression_stats(self, original_path: str, compressed_path: str) -> Dict[str, Any]:
        """Calculate compression statistics"""
        
        original_size = os.path.getsize(original_path) / (1024*1024)
        compressed_size = os.path.getsize(compressed_path) / (1024*1024)
        
        compression_ratio = compressed_size / original_size
        size_reduction = 1 - compression_ratio
        
        stats = {
            "original_size_mb": original_size,
            "compressed_size_mb": compressed_size,
            "compression_ratio": compression_ratio,
            "size_reduction_percent": size_reduction * 100,
            "space_saved_mb": original_size - compressed_size
        }
        
        return stats
    
    def compress_model(self, input_path: str, output_path: str, 
                      use_sparse: bool = True, use_fp16: bool = True,
                      additional_pruning: bool = True) -> str:
        """
        Comprehensive model compression
        
        Args:
            input_path: Path to input model
            output_path: Path to save compressed model
            use_sparse: Whether to use sparse tensor format
            use_fp16: Whether to use FP16 precision
            additional_pruning: Whether to apply additional weight pruning
        
        Returns:
            Path to compressed model
        """
        
        print(f"🚀 Starting model compression...")
        print(f"📂 Input: {input_path}")
        print(f"📂 Output: {output_path}")
        
        # Load original model
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input model not found: {input_path}")
        
        print(f"📥 Loading model...")
        checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
        
        # Extract state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        print(f"✅ Model loaded. Found {len(state_dict)} parameters.")
        
        # Apply compression steps
        if additional_pruning:
            print(f"\n🔪 Applying additional weight pruning...")
            state_dict = self.apply_weight_pruning(state_dict)
        
        if use_sparse:
            print(f"\n🗜️ Converting to sparse format...")
            state_dict = self.convert_to_sparse_format(state_dict)
        
        if use_fp16:
            print(f"\n🔄 Applying precision reduction...")
            state_dict = self.apply_precision_reduction(state_dict)
        
        # Calculate statistics
        total_params = 0
        zero_params = 0
        for param in state_dict.values():
            if isinstance(param, torch.Tensor):
                if hasattr(param, '_values'):  # Sparse tensor
                    total_params += param._values().numel()
                    zero_params += (param._values() == 0).sum().item()
                else:
                    total_params += param.numel()
                    zero_params += (param == 0).sum().item()
        
        final_sparsity = zero_params / total_params if total_params > 0 else 0
        
        # Prepare final checkpoint
        compressed_checkpoint = {
            'model': state_dict,
            'compression_info': {
                'method': 'GETA + Advanced Compression',
                'techniques': {
                    'sparse_format': use_sparse,
                    'fp16_precision': use_fp16,
                    'additional_pruning': additional_pruning
                },
                'final_sparsity': final_sparsity,
                'compression_threshold': self.compression_threshold,
                'total_parameters': total_params
            }
        }
        
        # Copy other important keys from original checkpoint
        for key in checkpoint.keys():
            if key not in ['model', 'state_dict']:
                if key not in compressed_checkpoint:
                    compressed_checkpoint[key] = checkpoint[key]
        
        # Save compressed model
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(compressed_checkpoint, output_path, _use_new_zipfile_serialization=True)
        
        # Calculate and display results
        stats = self.calculate_compression_stats(input_path, output_path)
        
        print(f"\n🎉 Compression completed!")
        print(f"📊 Original size: {stats['original_size_mb']:.2f} MB")
        print(f"📊 Compressed size: {stats['compressed_size_mb']:.2f} MB")
        print(f"🗜️ Size reduction: {stats['size_reduction_percent']:.1f}%")
        print(f"💾 Space saved: {stats['space_saved_mb']:.2f} MB")
        print(f"🎯 Final sparsity: {final_sparsity:.1%}")
        
        self.compression_stats = stats
        return output_path

def main():
    """Main compression script"""
    
    # Configuration
    INPUT_MODEL = '/kaggle/working/checkpoints/GaitBase_GETA_60K_Production-latest.pt'
    OUTPUT_DIR = '/kaggle/working/compressed_models_optimized'
    
    # Create compressor
    compressor = ModelCompressor(compression_threshold=1e-6)
    
    # Analyze original model
    print("🔍 Analyzing original model...")
    analysis = compressor.analyze_model_format(INPUT_MODEL)
    
    if "error" in analysis:
        print(f"❌ {analysis['error']}")
        return
    
    print(f"📊 Original Model Analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Compression variants
    compression_configs = [
        {
            'name': 'Maximum_Compression',
            'output': f'{OUTPUT_DIR}/GaitBase_GETA_MAX_COMPRESSED.pt',
            'sparse': True,
            'fp16': True,
            'pruning': True
        },
        {
            'name': 'Sparse_Only',
            'output': f'{OUTPUT_DIR}/GaitBase_GETA_SPARSE.pt',
            'sparse': True,
            'fp16': False,
            'pruning': False
        },
        {
            'name': 'FP16_Only',
            'output': f'{OUTPUT_DIR}/GaitBase_GETA_FP16.pt',
            'sparse': False,
            'fp16': True,
            'pruning': False
        }
    ]
    
    # Run compressions
    results = {}
    for config in compression_configs:
        print(f"\n{'='*60}")
        print(f"🗜️ Creating {config['name']} version...")
        
        try:
            output_path = compressor.compress_model(
                INPUT_MODEL,
                config['output'],
                use_sparse=config['sparse'],
                use_fp16=config['fp16'],
                additional_pruning=config['pruning']
            )
            results[config['name']] = compressor.compression_stats
            
        except Exception as e:
            print(f"❌ Failed to create {config['name']}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📋 COMPRESSION SUMMARY")
    print(f"{'='*60}")
    
    for name, stats in results.items():
        print(f"{name}:")
        print(f"  Size: {stats['compressed_size_mb']:.2f} MB")
        print(f"  Reduction: {stats['size_reduction_percent']:.1f}%")
        print()
    
    # Create download package
    print(f"📦 Creating download package...")
    
    import zipfile
    zip_path = '/kaggle/working/GETA_Compressed_Models.zip'
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(OUTPUT_DIR):
            if file.endswith(('.pt', '.pth')):
                file_path = os.path.join(OUTPUT_DIR, file)
                zipf.write(file_path, file)
                print(f"  ✅ Added: {file}")
    
    zip_size = os.path.getsize(zip_path) / (1024*1024)
    print(f"📦 Download package created: {zip_path} ({zip_size:.2f} MB)")
    print(f"📥 Ready for download from Kaggle's output panel!")

if __name__ == '__main__':
    main()
