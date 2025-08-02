import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
import time
from collections import OrderedDict
import sys

# Add OpenGait to path 
sys.path.insert(0, '/kaggle/working/GAITBASE_x_GETA/OpenGait')

def analyze_gaitbase_model(model_path, config_path=None, test_data_path=None, device='cuda'):
    """
    Comprehensive analysis of GaitBase model with detailed metrics table.
    
    Args:
        model_path (str): Path to the saved model checkpoint
        config_path (str, optional): Path to the YAML config file
        test_data_path (str, optional): Path to test dataset for accuracy evaluation
        device (str): Device to load model on
    
    Returns:
        pd.DataFrame: Detailed analysis table
    """
    
    print("🔍 Starting comprehensive GaitBase model analysis...")
    
    # Initialize results dictionary
    analysis_results = OrderedDict()
    
    # =========================
    # 1. BASIC MODEL INFO
    # =========================
    print("📊 Analyzing basic model information...")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model information
        analysis_results['Model Path'] = model_path
        analysis_results['Checkpoint Keys'] = list(checkpoint.keys())
        
        if 'iteration' in checkpoint:
            analysis_results['Training Iteration'] = checkpoint['iteration']
        
        if 'model' in checkpoint:
            model_state = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        else:
            model_state = checkpoint
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return pd.DataFrame()
    
    # =========================
    # 2. PARAMETER ANALYSIS
    # =========================
    print("🔢 Analyzing model parameters...")
    
    def analyze_parameters(state_dict):
        total_params = 0
        trainable_params = 0
        layer_info = {}
        
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                param_count = param.numel()
                total_params += param_count
                trainable_params += param_count  # Assume all loaded params are trainable
                
                # Categorize layers
                if 'conv' in name.lower():
                    layer_info['Conv Layers'] = layer_info.get('Conv Layers', 0) + param_count
                elif 'bn' in name.lower() or 'batchnorm' in name.lower():
                    layer_info['BatchNorm Layers'] = layer_info.get('BatchNorm Layers', 0) + param_count
                elif 'fc' in name.lower() or 'linear' in name.lower():
                    layer_info['Linear Layers'] = layer_info.get('Linear Layers', 0) + param_count
                elif 'embed' in name.lower():
                    layer_info['Embedding Layers'] = layer_info.get('Embedding Layers', 0) + param_count
                else:
                    layer_info['Other Layers'] = layer_info.get('Other Layers', 0) + param_count
        
        return total_params, trainable_params, layer_info
    
    total_params, trainable_params, layer_breakdown = analyze_parameters(model_state)
    
    analysis_results['Total Parameters'] = f"{total_params:,}"
    analysis_results['Trainable Parameters'] = f"{trainable_params:,}"
    analysis_results['Model Size (MB)'] = f"{total_params * 4 / (1024**2):.2f}"  # Assuming float32
    
    # Add layer breakdown
    for layer_type, param_count in layer_breakdown.items():
        analysis_results[f'{layer_type} Parameters'] = f"{param_count:,}"
    
    # =========================
    # 3. MODEL ARCHITECTURE ANALYSIS
    # =========================
    print("🏗️ Analyzing model architecture...")
    
    try:
        # Try to load config if provided
        if config_path and os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            analysis_results['Model Type'] = config.get('model_cfg', {}).get('model', 'Unknown')
            analysis_results['Backbone'] = config.get('model_cfg', {}).get('backbone_cfg', {}).get('type', 'Unknown')
            analysis_results['Dataset'] = config.get('data_cfg', {}).get('dataset_name', 'Unknown')
            
            # Training configuration
            analysis_results['Learning Rate'] = config.get('optimizer_cfg', {}).get('lr', 'Unknown')
            analysis_results['Batch Size'] = config.get('trainer_cfg', {}).get('sampler', {}).get('batch_size', 'Unknown')
            analysis_results['Total Iterations'] = config.get('trainer_cfg', {}).get('total_iter', 'Unknown')
            
    except Exception as e:
        print(f"⚠️ Could not load config: {e}")
    
    # Analyze layer structure from state dict
    layer_names = list(model_state.keys())
    analysis_results['Total Layers'] = len(layer_names)
    
    # Count different layer types
    conv_layers = len([name for name in layer_names if 'conv' in name.lower() and 'weight' in name])
    bn_layers = len([name for name in layer_names if ('bn' in name.lower() or 'batchnorm' in name.lower()) and 'weight' in name])
    fc_layers = len([name for name in layer_names if ('fc' in name.lower() or 'linear' in name.lower()) and 'weight' in name])
    
    analysis_results['Conv Layers Count'] = conv_layers
    analysis_results['BatchNorm Layers Count'] = bn_layers
    analysis_results['Linear Layers Count'] = fc_layers
    
    # =========================
    # 4. MEMORY ANALYSIS
    # =========================
    print("💾 Analyzing memory requirements...")
    
    # Calculate memory footprint
    param_memory = sum(param.numel() * param.element_size() for param in model_state.values() if isinstance(param, torch.Tensor))
    analysis_results['Parameter Memory (MB)'] = f"{param_memory / (1024**2):.2f}"
    
    # Estimate activation memory (rough estimate for typical gait input)
    # Assuming input size: (batch=8, frames=30, channels=1, height=64, width=44)
    estimated_activation_memory = 8 * 30 * 1 * 64 * 44 * 4  # 4 bytes per float32
    analysis_results['Estimated Activation Memory (MB)'] = f"{estimated_activation_memory / (1024**2):.2f}"
    
    # =========================
    # 5. PERFORMANCE METRICS
    # =========================
    print("⚡ Analyzing performance characteristics...")
    
    # Try to estimate FLOPs (simplified)
    def estimate_flops(state_dict):
        total_flops = 0
        for name, param in state_dict.items():
            if 'conv' in name.lower() and 'weight' in name:
                # Rough FLOP estimation for conv layers
                if param.dim() == 4:  # Conv2D
                    out_channels, in_channels, kh, kw = param.shape
                    # Assuming typical gait input resolution
                    total_flops += out_channels * in_channels * kh * kw * 64 * 44  # Rough estimate
            elif ('fc' in name.lower() or 'linear' in name.lower()) and 'weight' in name:
                # FLOP estimation for linear layers
                if param.dim() == 2:
                    total_flops += param.shape[0] * param.shape[1]
        return total_flops
    
    estimated_flops = estimate_flops(model_state)
    analysis_results['Estimated FLOPs'] = f"{estimated_flops:,}"
    analysis_results['Estimated GFLOPs'] = f"{estimated_flops / (10**9):.2f}"
    
    # =========================
    # 6. FILE ANALYSIS
    # =========================
    print("📁 Analyzing checkpoint file...")
    
    file_size = os.path.getsize(model_path)
    analysis_results['Checkpoint Size (MB)'] = f"{file_size / (1024**2):.2f}"
    analysis_results['File Extension'] = Path(model_path).suffix
    
    # Check compression ratio
    try:
        import gzip
        with open(model_path, 'rb') as f:
            original_size = len(f.read())
        
        compressed_size = len(gzip.compress(open(model_path, 'rb').read()))
        compression_ratio = compressed_size / original_size
        analysis_results['Compression Ratio'] = f"{compression_ratio:.3f}"
    except:
        analysis_results['Compression Ratio'] = "N/A"
    
    # =========================
    # 7. LAYER DISTRIBUTION ANALYSIS
    # =========================
    print("📊 Analyzing layer distribution...")
    
    # Analyze parameter distribution across layers
    layer_sizes = {}
    for name, param in model_state.items():
        if isinstance(param, torch.Tensor) and 'weight' in name:
            layer_name = name.replace('.weight', '')
            layer_sizes[layer_name] = param.numel()
    
    if layer_sizes:
        # Find largest and smallest layers
        largest_layer = max(layer_sizes, key=layer_sizes.get)
        smallest_layer = min(layer_sizes, key=layer_sizes.get)
        
        analysis_results['Largest Layer'] = f"{largest_layer} ({layer_sizes[largest_layer]:,} params)"
        analysis_results['Smallest Layer'] = f"{smallest_layer} ({layer_sizes[smallest_layer]:,} params)"
        analysis_results['Avg Layer Size'] = f"{np.mean(list(layer_sizes.values())):.0f}"
    
    # =========================
    # 8. TRAINING INFORMATION
    # =========================
    print("🎯 Extracting training information...")
    
    if 'optimizer' in checkpoint:
        analysis_results['Optimizer State Available'] = "Yes"
    else:
        analysis_results['Optimizer State Available'] = "No"
    
    if 'scheduler' in checkpoint:
        analysis_results['Scheduler State Available'] = "Yes"
    else:
        analysis_results['Scheduler State Available'] = "No"
    
    # Extract any loss information
    if 'loss' in checkpoint:
        analysis_results['Final Loss'] = f"{checkpoint['loss']:.6f}"
    
    # =========================
    # 9. COMPATIBILITY ANALYSIS
    # =========================
    print("🔧 Analyzing compatibility...")
    
    analysis_results['PyTorch Version Compatible'] = f"PyTorch {torch.__version__}"
    analysis_results['CUDA Available'] = torch.cuda.is_available()
    
    if torch.cuda.is_available():
        analysis_results['CUDA Version'] = torch.version.cuda
        analysis_results['GPU Count'] = torch.cuda.device_count()
    
    # =========================
    # 10. CREATE SUMMARY TABLE
    # =========================
    print("📋 Creating summary table...")
    
    # Convert to DataFrame for nice display
    df = pd.DataFrame.from_dict(analysis_results, orient='index', columns=['Value'])
    df.index.name = 'Metric'
    
    # Add categories for better organization
    categories = []
    for metric in df.index:
        if metric in ['Model Path', 'Model Type', 'Backbone', 'Dataset']:
            categories.append('Basic Info')
        elif 'Parameters' in metric or 'Size' in metric:
            categories.append('Architecture')
        elif 'Memory' in metric or 'FLOP' in metric:
            categories.append('Performance')
        elif 'Layer' in metric:
            categories.append('Structure')
        elif any(word in metric for word in ['Iteration', 'Loss', 'Learning', 'Batch', 'Optimizer']):
            categories.append('Training')
        elif any(word in metric for word in ['File', 'Checkpoint', 'Compression']):
            categories.append('Storage')
        else:
            categories.append('Other')
    
    df['Category'] = categories
    
    # Reorder columns
    df = df[['Category', 'Value']]
    
    print("✅ Analysis complete!")
    
    return df

def save_analysis_table(df, output_path="model_analysis.csv", format_type="csv"):
    """Save the analysis table in various formats"""
    
    if format_type.lower() == "csv":
        df.to_csv(output_path)
    elif format_type.lower() == "excel":
        df.to_excel(output_path.replace('.csv', '.xlsx'))
    elif format_type.lower() == "html":
        df.to_html(output_path.replace('.csv', '.html'))
    
    print(f"💾 Analysis saved to {output_path}")

def display_analysis(df, group_by_category=True):
    """Display the analysis table in a formatted way"""
    
    print("=" * 80)
    print("🚀 GAITBASE MODEL ANALYSIS REPORT")
    print("=" * 80)
    
    if group_by_category:
        # Group by category for better readability
        for category in df['Category'].unique():
            print(f"\n📊 {category.upper()}")
            print("-" * 40)
            category_df = df[df['Category'] == category]
            for metric, row in category_df.iterrows():
                print(f"{metric:<30}: {row['Value']}")
    else:
        # Display all at once
        for metric, row in df.iterrows():
            print(f"{metric:<30}: {row['Value']}")
    
    print("=" * 80)



def analyze_my_gaitbase_model():
    ''' # Usage '''
    # Define paths 
    model_path = "models_output/CASIA-B/Baseline/GaitBase_DA/checkpoints/GaitBase_DA-09000.pt"
    config_path = "GAITBASE_x_GETA/OpenGait/configs/gaitbase/gaitbase_da_casiab.yaml"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("❌ Model checkpoint not found!")
        print("Available checkpoints:")
        checkpoint_dir = "models_output/CASIA-B/Baseline/GaitBase_DA/checkpoints/"
        if os.path.exists(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                if file.endswith('.pt'):
                    print(f"  📁 {file}")
        return None
    
    # Run analysis
    analysis_df = analyze_gaitbase_model(
        model_path=model_path,
        config_path=config_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    
    
    # Display results
    display_analysis(analysis_df)
    
    # Save results
    save_analysis_table(analysis_df, "gaitbase_analysis.csv")
    save_analysis_table(analysis_df, "gaitbase_analysis.xlsx", "excel")
    
    return analysis_df


# Run the analysis
if __name__ == "__main__":
    df = analyze_my_gaitbase_model()