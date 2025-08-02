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
from datetime import datetime

# Add OpenGait to path with better error handling
def setup_opengait_paths():
    """Setup OpenGait paths properly"""
    possible_paths = [
        '/kaggle/working/GAITBASE_x_GETA/OpenGait',
        '/kaggle/working/OpenGait',
        'GAITBASE_x_GETA/OpenGait',
        './OpenGait'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            if path not in sys.path:
                sys.path.insert(0, path)
            print(f"✅ Added OpenGait path: {path}")
            
            # Also add opengait subdirectory
            opengait_subdir = os.path.join(path, 'opengait')
            if os.path.exists(opengait_subdir) and opengait_subdir not in sys.path:
                sys.path.insert(0, opengait_subdir)
                print(f"✅ Added OpenGait subdir: {opengait_subdir}")
            return path  # Return the found path
    
    print("❌ OpenGait path not found")
    return None

def init_distributed_for_opengait():
    """Initialize distributed training for OpenGait compatibility"""
    try:
        import torch
        import torch.distributed as dist
        
        if dist.is_initialized():
            print("✅ Distributed training already initialized")
            return True
        
        # Set environment variables for single-process distributed training
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29500')
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('WORLD_SIZE', '1')
        
        # Initialize process group
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(
            backend=backend,
            init_method='env://',
            world_size=1,
            rank=0
        )
        
        print(f"✅ Initialized distributed training with {backend} backend")
        return True
        
    except Exception as e:
        print(f"⚠️ Could not initialize distributed training: {e}")
        return False

# Setup paths before importing
opengait_path = setup_opengait_paths()

# Initialize distributed training if OpenGait is found
if opengait_path:
    init_distributed_for_opengait()

def evaluate_model_accuracy_opengait(model_path, config_path, device='cuda'):
    """
    Evaluate model accuracy using OpenGait's actual evaluation pipeline
    Based on the evaluator.py structure shown in attachments
    """
    print("🎯 Starting OpenGait-based accuracy evaluation...")
    
    try:
        # Import OpenGait components
        import yaml
        from opengait.modeling import models
        from opengait.data import transform as base_transform
        from opengait.data.dataset import DataSet
        from opengait.utils import config_loader, get_msg_mgr
        from opengait.evaluation.evaluator import evaluate_indoor_dataset
        
        print("✅ Successfully imported OpenGait components")
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Loaded config from {config_path}")
        
        # Setup model
        Model = getattr(models, config['model_cfg']['model'])
        model = Model(config, training=False)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        model.eval()
        print("✅ Model loaded and set to evaluation mode")
        
        # Setup test dataset using OpenGait's pipeline
        test_transform = base_transform.get_transform(config['evaluator_cfg']['transform'])
        test_dataset = DataSet([config['data_cfg']], test_transform, training=False)
        
        print(f"✅ Test dataset loaded with {len(test_dataset)} samples")
        
        # Create test dataloader
        from torch.utils.data import DataLoader
        batch_size = config['evaluator_cfg']['sampler']['batch_size']
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config['data_cfg']['num_workers'],
            drop_last=False
        )
        
        print(f"✅ Test dataloader created with batch size {batch_size}")
        
        # Extract features using model
        print("🔄 Extracting features from test data...")
        all_embeddings = []
        all_labels = []
        all_types = []
        all_views = []
        
        with torch.no_grad():
            for i, batch_data in enumerate(test_loader):
                if i % 10 == 0:
                    print(f"  Processing batch {i+1}/{len(test_loader)}")
                
                # Handle OpenGait batch format
                inputs = batch_data[0]  # Silhouette data
                labels = batch_data[1] if len(batch_data) > 1 else None  # Identity labels
                types = batch_data[2] if len(batch_data) > 2 else None   # Sequence types
                views = batch_data[3] if len(batch_data) > 3 else None   # View angles
                
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                
                # Get model embeddings
                outputs = model(inputs)
                
                # Extract embeddings (typically the first output)
                if isinstance(outputs, (list, tuple)):
                    embeddings = outputs[0]  # Feature embeddings
                else:
                    embeddings = outputs
                
                # Store results
                all_embeddings.append(embeddings.cpu().numpy())
                if labels is not None:
                    all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
                if types is not None:
                    all_types.extend(types if isinstance(types, list) else types.numpy())
                if views is not None:
                    all_views.extend(views if isinstance(views, list) else views.numpy())
        
        # Concatenate all embeddings
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        print(f"✅ Feature extraction complete. Shape: {all_embeddings.shape}")
        
        # Prepare data for OpenGait evaluator
        evaluation_data = {
            'embeddings': all_embeddings,
            'labels': all_labels,
            'types': all_types,
            'views': all_views
        }
        
        # Use OpenGait's evaluation function
        dataset_name = config['data_cfg']['dataset_name']
        metric = config['evaluator_cfg'].get('metric', 'euc')
        
        print(f"🧮 Running OpenGait evaluation for {dataset_name} with {metric} metric...")
        
        # Call the appropriate evaluation function based on dataset
        if dataset_name in ('CASIA-B', 'OUMVLP', 'CASIA-E', 'SUSTech1K'):
            results = evaluate_indoor_dataset(evaluation_data, dataset_name, metric)
        else:
            # Fallback to manual calculation
            results = calculate_accuracy_manual(evaluation_data)
        
        print("✅ Accuracy evaluation completed!")
        return results
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return calculate_accuracy_fallback(model_path, config_path)
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return calculate_accuracy_fallback(model_path, config_path)

def calculate_accuracy_manual(data):
    """
    Manual accuracy calculation based on OpenGait's approach
    """
    print("🔄 Calculating accuracy manually...")
    
    embeddings = data['embeddings']
    labels = np.array(data['labels'])
    types = np.array(data['types']) if data['types'] else None
    views = np.array(data['views']) if data['views'] else None
    
    # Simple gallery-probe split based on CASIA-B protocol
    # Gallery: nm-01, nm-02, Probe: nm-03, nm-04, bg-01, bg-02, cl-01, cl-02
    gallery_types = ['nm-01', 'nm-02'] if types is not None else None
    probe_types = ['nm-03', 'nm-04', 'bg-01', 'bg-02', 'cl-01', 'cl-02'] if types is not None else None
    
    if types is not None and gallery_types and probe_types:
        # Use sequence type based splitting
        gallery_mask = np.isin(types, gallery_types)
        probe_mask = np.isin(types, probe_types)
    else:
        # Simple split: first half as gallery, second half as probe
        mid_point = len(embeddings) // 2
        gallery_mask = np.zeros(len(embeddings), dtype=bool)
        probe_mask = np.zeros(len(embeddings), dtype=bool)
        gallery_mask[:mid_point] = True
        probe_mask[mid_point:] = True
    
    gallery_embeddings = embeddings[gallery_mask]
    gallery_labels = labels[gallery_mask]
    probe_embeddings = embeddings[probe_mask]
    probe_labels = labels[probe_mask]
    
    print(f"Gallery samples: {len(gallery_embeddings)}, Probe samples: {len(probe_embeddings)}")
    
    if len(gallery_embeddings) == 0 or len(probe_embeddings) == 0:
        return {'Error': 'Insufficient data for gallery-probe evaluation'}
    
    # Calculate distances (Euclidean)
    from sklearn.metrics.pairwise import euclidean_distances
    distances = euclidean_distances(probe_embeddings, gallery_embeddings)
    
    # Calculate Rank-1 and Rank-5 accuracy
    rank1_correct = 0
    rank5_correct = 0
    rank10_correct = 0
    
    for i, probe_label in enumerate(probe_labels):
        # Get distances for this probe
        probe_distances = distances[i]
        
        # Sort gallery indices by distance
        sorted_indices = np.argsort(probe_distances)
        
        # Get corresponding gallery labels
        sorted_gallery_labels = gallery_labels[sorted_indices]
        
        # Check Rank-1
        if sorted_gallery_labels[0] == probe_label:
            rank1_correct += 1
        
        # Check Rank-5
        if probe_label in sorted_gallery_labels[:5]:
            rank5_correct += 1
        
        # Check Rank-10
        if probe_label in sorted_gallery_labels[:10]:
            rank10_correct += 1
    
    rank1_accuracy = rank1_correct / len(probe_labels)
    rank5_accuracy = rank5_correct / len(probe_labels)
    rank10_accuracy = rank10_correct / len(probe_labels)
    
    results = {
        'scalar/test_accuracy/Rank-1': rank1_accuracy * 100,
        'scalar/test_accuracy/Rank-5': rank5_accuracy * 100,
        'scalar/test_accuracy/Rank-10': rank10_accuracy * 100,
        'Gallery_Samples': len(gallery_embeddings),
        'Probe_Samples': len(probe_embeddings)
    }
    
    print(f"📊 Results: Rank-1: {rank1_accuracy*100:.2f}%, Rank-5: {rank5_accuracy*100:.2f}%, Rank-10: {rank10_accuracy*100:.2f}%")
    
    return results

def calculate_accuracy_fallback(model_path, config_path):
    """
    Fallback accuracy calculation when OpenGait imports fail
    """
    print("🔄 Using fallback accuracy calculation...")
    
    try:
        # Load checkpoint to check for any stored accuracy
        checkpoint = torch.load(model_path, map_location='cpu')
        
        accuracy_info = {}
        
        # Check for stored accuracy metrics
        accuracy_keys = [k for k in checkpoint.keys() if 'acc' in k.lower() or 'eval' in k.lower()]
        
        if accuracy_keys:
            for key in accuracy_keys:
                accuracy_info[f'Stored_{key}'] = str(checkpoint[key])
        
        # Extract training information
        if 'iteration' in checkpoint:
            accuracy_info['Training_Iteration'] = checkpoint['iteration']
        
        # If no accuracy found, provide informative message
        if not any('acc' in k.lower() for k in accuracy_info.keys()):
            accuracy_info = {
                'Status': 'No accuracy data available',
                'Note': 'Run test phase to generate accuracy metrics',
                'Suggestion': 'Use: python opengait/main.py --phase test --iter [iteration]'
            }
        
        return accuracy_info
        
    except Exception as e:
        return {'Error': f'Fallback calculation failed: {e}'}

def analyze_gaitbase_model(model_path, config_path=None, test_data_path=None, device='cuda', evaluate_accuracy=True):
    """
    Comprehensive analysis of GaitBase model with detailed metrics table including accuracy.
    
    Args:
        model_path (str): Path to the model checkpoint (.pt file)
        config_path (str, optional): Path to the YAML config file
        test_data_path (str, optional): Path to test dataset (currently unused)
        device (str): Device to run evaluation on ('cuda' or 'cpu')
        evaluate_accuracy (bool): Whether to evaluate model accuracy
    
    Returns:
        pd.DataFrame: Comprehensive analysis results
    """
    
    print("🔍 Starting comprehensive GaitBase model analysis...")
    print(f"📄 Model: {model_path}")
    print(f"⚙️ Config: {config_path}")
    
    # Validate inputs
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return pd.DataFrame()
    
    if config_path and not os.path.exists(config_path):
        print(f"⚠️ Config file not found: {config_path}")
        config_path = None
    
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
                trainable_params += param_count
                
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
    analysis_results['Model Size (MB)'] = f"{total_params * 4 / (1024**2):.2f}"
    
    # Add layer breakdown
    for layer_type, param_count in layer_breakdown.items():
        analysis_results[f'{layer_type} Parameters'] = f"{param_count:,}"
    
    # =========================
    # 3. MODEL ARCHITECTURE ANALYSIS
    # =========================
    print("🏗️ Analyzing model architecture...")
    
    config = None
    try:
        if config_path and os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            analysis_results['Model Type'] = config.get('model_cfg', {}).get('model', 'Unknown')
            analysis_results['Backbone'] = config.get('model_cfg', {}).get('backbone_cfg', {}).get('type', 'Unknown')
            analysis_results['Dataset'] = config.get('data_cfg', {}).get('dataset_name', 'Unknown')
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
    # 4. ACCURACY EVALUATION (ENHANCED!)
    # =========================
    if evaluate_accuracy and config_path:
        print("🎯 Evaluating model accuracy...")
        try:
            accuracy_results = evaluate_model_accuracy_opengait(model_path, config_path, device)
            
            # Add accuracy metrics to results
            for metric_name, value in accuracy_results.items():
                if isinstance(value, (int, float)):
                    analysis_results[f'Accuracy {metric_name}'] = f"{value:.4f}"
                else:
                    analysis_results[f'Accuracy {metric_name}'] = str(value)
                    
        except Exception as e:
            print(f"⚠️ Could not evaluate accuracy: {e}")
            analysis_results['Accuracy Evaluation'] = f"Failed: {str(e)}"
            analysis_results['Accuracy Note'] = "Run test phase for accuracy metrics"
    
    # =========================
    # 5. MEMORY & PERFORMANCE ANALYSIS
    # =========================
    print("💾 Analyzing memory and performance...")
    
    # Calculate memory footprint
    param_memory = sum(param.numel() * param.element_size() for param in model_state.values() if isinstance(param, torch.Tensor))
    analysis_results['Parameter Memory (MB)'] = f"{param_memory / (1024**2):.2f}"
    
    # Estimate activation memory
    estimated_activation_memory = 8 * 30 * 1 * 64 * 44 * 4  # 4 bytes per float32
    analysis_results['Estimated Activation Memory (MB)'] = f"{estimated_activation_memory / (1024**2):.2f}"
    
    # FLOP estimation
    def estimate_flops(state_dict):
        total_flops = 0
        for name, param in state_dict.items():
            if 'conv' in name.lower() and 'weight' in name:
                if param.dim() == 4:  # Conv2D
                    out_channels, in_channels, kh, kw = param.shape
                    total_flops += out_channels * in_channels * kh * kw * 64 * 44
            elif ('fc' in name.lower() or 'linear' in name.lower()) and 'weight' in name:
                if param.dim() == 2:
                    total_flops += param.shape[0] * param.shape[1]
        return total_flops
    
    estimated_flops = estimate_flops(model_state)
    analysis_results['Estimated FLOPs'] = f"{estimated_flops:,}"
    analysis_results['Estimated GFLOPs'] = f"{estimated_flops / (10**9):.2f}"
    
    # =========================
    # 6. FILE ANALYSIS
    # =========================
    print("📁 Analyzing file properties...")
    
    file_size = os.path.getsize(model_path)
    analysis_results['Checkpoint Size (MB)'] = f"{file_size / (1024**2):.2f}"
    analysis_results['File Extension'] = Path(model_path).suffix
    
    # =========================
    # 7. TRAINING INFORMATION
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
    
    if 'loss' in checkpoint:
        analysis_results['Final Loss'] = f"{checkpoint['loss']:.6f}"
    
    # =========================
    # 8. COMPATIBILITY ANALYSIS
    # =========================
    print("🔧 Analyzing compatibility...")
    
    analysis_results['PyTorch Version Compatible'] = f"PyTorch {torch.__version__}"
    analysis_results['CUDA Available'] = torch.cuda.is_available()
    
    if torch.cuda.is_available():
        analysis_results['CUDA Version'] = torch.version.cuda
        analysis_results['GPU Count'] = torch.cuda.device_count()
    
    # =========================
    # 9. CREATE SUMMARY TABLE
    # =========================
    print("📋 Creating summary table...")
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(analysis_results, orient='index', columns=['Value'])
    df.index.name = 'Metric'
    
    # Add categories
    categories = []
    for metric in df.index:
        if metric in ['Model Path', 'Model Type', 'Backbone', 'Dataset']:
            categories.append('Basic Info')
        elif 'Parameters' in metric or 'Size' in metric:
            categories.append('Architecture')
        elif 'Memory' in metric or 'FLOP' in metric:
            categories.append('Performance')
        elif 'Accuracy' in metric:
            categories.append('Accuracy')
        elif 'Layer' in metric:
            categories.append('Structure')
        elif any(word in metric for word in ['Iteration', 'Loss', 'Learning', 'Batch', 'Optimizer']):
            categories.append('Training')
        elif any(word in metric for word in ['File', 'Checkpoint', 'Compression']):
            categories.append('Storage')
        else:
            categories.append('Other')
    
    df['Category'] = categories
    df = df[['Category', 'Value']]
    
    print("✅ Analysis complete!")
    return df

def generate_unique_filename(model_path, base_name="gaitbase_analysis"):
    """Generate unique filename based on model characteristics"""
    path_parts = Path(model_path).parts
    
    model_id = "unknown"
    iteration = "unknown"
    
    for part in path_parts:
        if "GaitBase" in part:
            model_id = part
        elif part.endswith('.pt'):
            if '-' in part:
                iteration = part.split('-')[1].replace('.pt', '')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = f"{model_id}_iter{iteration}_{timestamp}"
    
    return f"{base_name}_{unique_id}"

def save_analysis_table(df, model_path, base_output_name="gaitbase_analysis", format_type="csv"):
    """Save the analysis table with unique model-specific filename"""
    unique_name = generate_unique_filename(model_path, base_output_name)
    
    if format_type.lower() == "csv":
        output_path = f"{unique_name}.csv"
        df.to_csv(output_path)
    elif format_type.lower() == "excel":
        output_path = f"{unique_name}.xlsx"
        df.to_excel(output_path)
    elif format_type.lower() == "html":
        output_path = f"{unique_name}.html"
        df.to_html(output_path)
    
    print(f"💾 Analysis saved to {output_path}")
    return output_path

def display_analysis(df, group_by_category=True):
    """Display the analysis table in a formatted way"""
    print("=" * 80)
    print("🚀 GAITBASE MODEL ANALYSIS REPORT")
    print("=" * 80)
    
    if group_by_category:
        for category in df['Category'].unique():
            print(f"\n📊 {category.upper()}")
            print("-" * 40)
            category_df = df[df['Category'] == category]
            for metric, row in category_df.iterrows():
                print(f"{metric:<30}: {row['Value']}")
    else:
        for metric, row in df.iterrows():
            print(f"{metric:<30}: {row['Value']}")
    
    print("=" * 80)

# ✅ NEW: Dynamic analysis function with flexible parameters
def analyze_model_dynamic(model_path, config_path=None, output_dir="./", 
                         evaluate_accuracy=True, save_formats=['csv'], 
                         device='auto', display_results=True):
    """
    Dynamic model analysis function with flexible parameters
    
    Args:
        model_path (str): Path to model checkpoint
        config_path (str, optional): Path to config file
        output_dir (str): Directory to save results
        evaluate_accuracy (bool): Whether to evaluate accuracy
        save_formats (list): List of formats to save ['csv', 'excel', 'html']
        device (str): Device to use ('auto', 'cuda', 'cpu')
        display_results (bool): Whether to display results
    
    Returns:
        pd.DataFrame: Analysis results
    """
    
    # Auto-detect device if needed
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🚀 Dynamic Model Analysis")
    print(f"📄 Model: {model_path}")
    print(f"⚙️ Config: {config_path}")
    print(f"💾 Output: {output_dir}")
    print(f"🔧 Device: {device}")
    print(f"🎯 Accuracy: {evaluate_accuracy}")
    
    # Validate model path
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run analysis
    analysis_df = analyze_gaitbase_model(
        model_path=model_path,
        config_path=config_path,
        device=device,
        evaluate_accuracy=evaluate_accuracy
    )
    
    if analysis_df.empty:
        print("❌ Analysis failed")
        return None
    
    # Display results if requested
    if display_results:
        display_analysis(analysis_df)
    
    # Save in requested formats
    saved_files = []
    for format_type in save_formats:
        try:
            output_path = save_analysis_table(
                analysis_df, 
                model_path, 
                os.path.join(output_dir, "gaitbase_analysis"), 
                format_type
            )
            saved_files.append(output_path)
        except Exception as e:
            print(f"❌ Failed to save {format_type}: {e}")
    
    print(f"\n📁 Files saved: {len(saved_files)}")
    for file in saved_files:
        print(f"  📄 {file}")
    
    return analysis_df

# ✅ NEW: Convenience functions for different use cases
def quick_analyze(model_path, config_path=None):
    """Quick analysis without saving files"""
    return analyze_model_dynamic(
        model_path=model_path,
        config_path=config_path,
        evaluate_accuracy=False,
        save_formats=[],
        display_results=True
    )

def full_analyze(model_path, config_path=None, output_dir="./results/"):
    """Full analysis with accuracy and all formats"""
    return analyze_model_dynamic(
        model_path=model_path,
        config_path=config_path,
        output_dir=output_dir,
        evaluate_accuracy=True,
        save_formats=['csv', 'excel', 'html'],
        display_results=True
    )

def batch_analyze(model_paths, config_paths=None, output_dir="./batch_results/"):
    """Analyze multiple models"""
    results = {}
    
    if config_paths is None:
        config_paths = [None] * len(model_paths)
    elif len(config_paths) != len(model_paths):
        print("⚠️ Config paths length doesn't match model paths. Using None for missing configs.")
        config_paths.extend([None] * (len(model_paths) - len(config_paths)))
    
    for i, (model_path, config_path) in enumerate(zip(model_paths, config_paths)):
        print(f"\n🔄 Analyzing model {i+1}/{len(model_paths)}: {model_path}")
        
        try:
            result = analyze_model_dynamic(
                model_path=model_path,
                config_path=config_path,
                output_dir=os.path.join(output_dir, f"model_{i+1}"),
                evaluate_accuracy=True,
                save_formats=['csv', 'excel'],
                display_results=False
            )
            results[model_path] = result
        except Exception as e:
            print(f"❌ Failed to analyze {model_path}: {e}")
            results[model_path] = None
    
    print(f"\n✅ Batch analysis complete! Analyzed {len([r for r in results.values() if r is not None])}/{len(model_paths)} models")
    return results

# ✅ UPDATED: Example usage function - now uses dynamic paths
def analyze_my_gaitbase_model(model_path=None, config_path=None):
    """
    Enhanced analysis with dynamic path support
    
    Args:
        model_path (str, optional): Path to model checkpoint
        config_path (str, optional): Path to config file
    """
    
    # Default paths if none provided
    if model_path is None:
        model_path = "models_output/CASIA-B/Baseline/GaitBase_DA/checkpoints/GaitBase_DA-09000.pt"
    
    if config_path is None:
        config_path = "GAITBASE_x_GETA/OpenGait/configs/gaitbase/gaitbase_da_casiab.yaml"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("❌ Model checkpoint not found!")
        print(f"Searched: {model_path}")
        
        # Try to find available checkpoints
        checkpoint_dir = os.path.dirname(model_path)
        if os.path.exists(checkpoint_dir):
            print("Available checkpoints:")
            for file in os.listdir(checkpoint_dir):
                if file.endswith('.pt'):
                    full_path = os.path.join(checkpoint_dir, file)
                    print(f"  📁 {full_path}")
        return None
    
    # Run full analysis
    return full_analyze(model_path=model_path, config_path=config_path)

# Run the analysis
import argparse

def main():
    parser = argparse.ArgumentParser(description="Analyze a GaitBase model checkpoint.")
    
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--config_path", type=str, default=None,
        help="Path to the config YAML file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results/",
        help="Directory where the analysis results will be saved"
    )
    parser.add_argument(
        "--device", type=str, choices=["auto", "cpu", "cuda"], default="auto",
        help="Device to use for evaluation"
    )
    parser.add_argument(
        "--formats", nargs="+", default=["csv"],
        help="File formats to save: any combination of 'csv', 'excel', 'html'"
    )
    parser.add_argument(
        "--no_accuracy", action="store_true",
        help="Disable accuracy evaluation"
    )
    parser.add_argument(
        "--no_display", action="store_true",
        help="Disable display of results in console"
    )

    args = parser.parse_args()

    # Run analysis
    analyze_model_dynamic(
        model_path=args.model_path,
        config_path=args.config_path,
        output_dir=args.output_dir,
        evaluate_accuracy=not args.no_accuracy,
        save_formats=args.formats,
        device=args.device,
        display_results=not args.no_display
    )

if __name__ == "__main__":
    main()

