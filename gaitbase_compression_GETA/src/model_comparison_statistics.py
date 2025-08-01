import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile, clever_format

# Load both models
full_model = torch.load(oto.full_group_sparse_model_path)
compressed_model = torch.load(oto.compressed_model_path)

def compare_model_architecture(model1, model2, input_size):
    print("=== ARCHITECTURE COMPARISON ===")
    
    # Model structure comparison
    print("\n1. Layer Count:")
    print(f"Full model layers: {len(list(model1.modules()))}")
    print(f"Compressed model layers: {len(list(model2.modules()))}")
    
    # Detailed layer analysis
    print("\n2. Layer-by-layer comparison:")
    for name, module in model1.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            try:
                compressed_module = dict(model2.named_modules())[name]
                original_shape = module.weight.shape
                compressed_shape = compressed_module.weight.shape
                print(f"{name}: {original_shape} → {compressed_shape}")
            except:
                print(f"{name}: REMOVED")

# Example usage
dummy_input = torch.rand(1, 3, 32, 32)  # Adjust based on your input
compare_model_architecture(full_model, compressed_model, (3, 32, 32))

#2. Computational Complexity (FLOPs & MACs)

from thop import profile, clever_format
import time

def analyze_computational_complexity(model1, model2, dummy_input):
    print("=== COMPUTATIONAL COMPLEXITY ===")
    
    # FLOPs analysis
    flops1, params1 = profile(model1, inputs=(dummy_input,), verbose=False)
    flops2, params2 = profile(model2, inputs=(dummy_input,), verbose=False)
    
    flops1, params1 = clever_format([flops1, params1], "%.3f")
    flops2, params2 = clever_format([flops2, params2], "%.3f")
    
    print(f"Full model - FLOPs: {flops1}, Params: {params1}")
    print(f"Compressed model - FLOPs: {flops2}, Params: {params2}")
    
    # Calculate reduction percentages
    flops_reduction = (1 - flops2_raw/flops1_raw) * 100
    params_reduction = (1 - params2_raw/params1_raw) * 100
    
    print(f"FLOPs reduction: {flops_reduction:.2f}%")
    print(f"Parameters reduction: {params_reduction:.2f}%")

# Get raw numbers for percentage calculation
flops1_raw, params1_raw = profile(full_model, inputs=(dummy_input,), verbose=False)
flops2_raw, params2_raw = profile(compressed_model, inputs=(dummy_input,), verbose=False)

analyze_computational_complexity(full_model, compressed_model, dummy_input)


# 3. Inference Speed & Memory Usage

import time
import torch
import psutil
import os

def benchmark_inference_speed(model1, model2, dummy_input, num_runs=100):
    print("=== INFERENCE SPEED COMPARISON ===")
    
    model1.eval()
    model2.eval()
    
    # Warm up
    for _ in range(10):
        _ = model1(dummy_input)
        _ = model2(dummy_input)
    
    # Benchmark full model
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model1(dummy_input)
    full_model_time = time.time() - start_time
    
    # Benchmark compressed model
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model2(dummy_input)
    compressed_model_time = time.time() - start_time
    
    speedup = full_model_time / compressed_model_time
    
    print(f"Full model avg time: {full_model_time/num_runs*1000:.3f} ms")
    print(f"Compressed model avg time: {compressed_model_time/num_runs*1000:.3f} ms")
    print(f"Speedup: {speedup:.2f}x")

def analyze_memory_usage(model1, model2):
    print("\n=== MEMORY USAGE ===")
    
    # Model size in memory
    def get_model_size_mb(model):
        param_size = 0
        buffer_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / 1024**2
    
    full_size = get_model_size_mb(model1)
    compressed_size = get_model_size_mb(model2)
    
    print(f"Full model memory: {full_size:.2f} MB")
    print(f"Compressed model memory: {compressed_size:.2f} MB")
    print(f"Memory reduction: {((full_size - compressed_size) / full_size * 100):.2f}%")

benchmark_inference_speed(full_model, compressed_model, dummy_input)
analyze_memory_usage(full_model, compressed_model)

#4. Detailed Parameter Analysis

def detailed_parameter_analysis(model1, model2):
    print("=== DETAILED PARAMETER ANALYSIS ===")
    
    # Count parameters by layer type
    def count_parameters_by_type(model):
        conv_params = 0
        bn_params = 0
        fc_params = 0
        other_params = 0
        
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                conv_params += sum(p.numel() for p in module.parameters())
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                bn_params += sum(p.numel() for p in module.parameters())
            elif isinstance(module, nn.Linear):
                fc_params += sum(p.numel() for p in module.parameters())
            else:
                other_params += sum(p.numel() for p in module.parameters())
        
        return conv_params, bn_params, fc_params, other_params
    
    conv1, bn1, fc1, other1 = count_parameters_by_type(model1)
    conv2, bn2, fc2, other2 = count_parameters_by_type(model2)
    
    print("Parameter breakdown:")
    print(f"Conv layers: {conv1:,} → {conv2:,} ({(1-conv2/conv1)*100:.1f}% reduction)")
    print(f"BatchNorm: {bn1:,} → {bn2:,} ({(1-bn2/bn1)*100:.1f}% reduction)")
    print(f"Linear layers: {fc1:,} → {fc2:,} ({(1-fc2/fc1)*100:.1f}% reduction)")
    print(f"Other: {other1:,} → {other2:,}")

detailed_parameter_analysis(full_model, compressed_model)

#5. Accuracy and Performance Metrics

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def comprehensive_accuracy_analysis(model1, model2, testloader):
    print("=== ACCURACY ANALYSIS ===")
    
    model1.eval()
    model2.eval()
    
    all_preds1, all_preds2, all_targets = [], [], []
    
    with torch.no_grad():
        for data, targets in testloader:
            data, targets = data.cuda(), targets.cuda()
            
            outputs1 = model1(data)
            outputs2 = model2(data)
            
            _, preds1 = torch.max(outputs1, 1)
            _, preds2 = torch.max(outputs2, 1)
            
            all_preds1.extend(preds1.cpu().numpy())
            all_preds2.extend(preds2.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate various metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    acc1 = accuracy_score(all_targets, all_preds1)
    acc2 = accuracy_score(all_targets, all_preds2)
    
    print(f"Full model accuracy: {acc1:.4f}")
    print(f"Compressed model accuracy: {acc2:.4f}")
    print(f"Accuracy drop: {(acc1-acc2):.4f} ({((acc1-acc2)/acc1)*100:.2f}%)")
    
    # Per-class performance
    print("\nPer-class F1 scores:")
    f1_full = f1_score(all_targets, all_preds1, average=None)
    f1_compressed = f1_score(all_targets, all_preds2, average=None)
    
    for i, (f1_f, f1_c) in enumerate(zip(f1_full, f1_compressed)):
        print(f"Class {i}: {f1_f:.3f} → {f1_c:.3f}")

comprehensive_accuracy_analysis(full_model, compressed_model, testloader)

# 7. For Your Gait Recognition Specific Analysis


def gait_specific_analysis(full_model, compressed_model, test_dataloader):
    print("=== GAIT-SPECIFIC ANALYSIS ===")
    
    # Analyze feature extraction quality
    def extract_features(model, dataloader, max_batches=10):
        features = []
        model.eval()
        
        with torch.no_grad():
            for i, (data, _) in enumerate(dataloader):
                if i >= max_batches:
                    break
                data = data.cuda()
                # Extract features before final classification
                # This depends on your model architecture
                feat = model.backbone(data)  # Adjust based on your model
                features.append(feat.cpu())
        
        return torch.cat(features, dim=0)
    
    full_features = extract_features(full_model, test_dataloader)
    compressed_features = extract_features(compressed_model, test_dataloader)
    
    # Feature similarity analysis
    cosine_sim = torch.nn.functional.cosine_similarity(
        full_features.flatten(1), 
        compressed_features.flatten(1), 
        dim=1
    ).mean()
    
    print(f"Feature similarity (cosine): {cosine_sim:.4f}")
    
    # Feature norm comparison
    full_norm = torch.norm(full_features.flatten(1), dim=1).mean()
    compressed_norm = torch.norm(compressed_features.flatten(1), dim=1).mean()
    
    print(f"Full model feature norm: {full_norm:.3f}")
    print(f"Compressed model feature norm: {compressed_norm:.3f}")

# Run gait-specific analysis
gait_specific_analysis(full_model, compressed_model, testloader)


