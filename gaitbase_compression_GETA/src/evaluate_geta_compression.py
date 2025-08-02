#!/usr/bin/env python3

import argparse
import os
import sys
import torch
import json
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Evaluate GETA compressed models')
    parser.add_argument('--config', '-c', 
                       default='gaitbase_geta.yaml',
                       help='Path to config file')
    parser.add_argument('--original_model', 
                       help='Path to original model checkpoint')
    parser.add_argument('--compressed_model', 
                       help='Path to compressed model')
    parser.add_argument('--full_sparse_model',
                       help='Path to full sparse model')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device')
    parser.add_argument('--output_dir', default='./evaluation_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Add paths dynamically
    current_dir = os.path.dirname(os.path.abspath(__file__))
    geta_path = os.path.join(current_dir, '../../geta')
    opengait_path = os.path.join(current_dir, '../../OpenGait')
    
    if os.path.exists(geta_path):
        sys.path.insert(0, geta_path)
        print(f"✅ Added GETA path: {geta_path}")
    else:
        print(f"❌ GETA path not found: {geta_path}")
        return
    
    if os.path.exists(opengait_path):
        sys.path.insert(0, opengait_path)
        print(f"✅ Added OpenGait path: {opengait_path}")
    else:
        print(f"❌ OpenGait path not found: {opengait_path}")
        return

    # Import after path setup
    from geta_opengait_integration import GETAOpenGaitTrainer

    # Initialize trainer
    trainer = GETAOpenGaitTrainer(args.config)
    
    # Setup model and data
    print("Setting up model...")
    trainer.setup_model()
    
    print("Setting up data...")
    trainer.setup_data()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluation results
    evaluation_results = {
        'timestamp': datetime.now().isoformat(),
        'config_path': args.config,
        'gpu': args.gpu
    }
    
    # Evaluate original model if provided
    if args.original_model and os.path.exists(args.original_model):
        print("Evaluating original model...")
        try:
            original_results = trainer.evaluate_model(args.original_model)
            evaluation_results['original_model'] = {
                'path': args.original_model,
                'results': original_results,
                'file_size_mb': os.path.getsize(args.original_model) / (1024**2)
            }
        except Exception as e:
            print(f"❌ Failed to evaluate original model: {e}")
            evaluation_results['original_model'] = {'error': str(e)}
    
    # Evaluate compressed model if provided
    if args.compressed_model and os.path.exists(args.compressed_model):
        print("Evaluating compressed model...")
        try:
            compressed_results = trainer.evaluate_model(args.compressed_model)
            evaluation_results['compressed_model'] = {
                'path': args.compressed_model,
                'results': compressed_results,
                'file_size_mb': os.path.getsize(args.compressed_model) / (1024**2)
            }
        except Exception as e:
            print(f"❌ Failed to evaluate compressed model: {e}")
            evaluation_results['compressed_model'] = {'error': str(e)}
    
    # Evaluate full sparse model if provided
    if args.full_sparse_model and os.path.exists(args.full_sparse_model):
        print("Evaluating full sparse model...")
        try:
            sparse_results = trainer.evaluate_model(args.full_sparse_model)
            evaluation_results['full_sparse_model'] = {
                'path': args.full_sparse_model,
                'results': sparse_results,
                'file_size_mb': os.path.getsize(args.full_sparse_model) / (1024**2)
            }
        except Exception as e:
            print(f"❌ Failed to evaluate full sparse model: {e}")
            evaluation_results['full_sparse_model'] = {'error': str(e)}
    
    # Calculate compression metrics
    if 'original_model' in evaluation_results and 'compressed_model' in evaluation_results:
        if 'error' not in evaluation_results['original_model'] and 'error' not in evaluation_results['compressed_model']:
            original_size = evaluation_results['original_model']['file_size_mb']
            compressed_size = evaluation_results['compressed_model']['file_size_mb']
            
            evaluation_results['compression_metrics'] = {
                'size_reduction_mb': original_size - compressed_size,
                'size_reduction_percentage': ((original_size - compressed_size) / original_size) * 100,
                'compression_ratio': compressed_size / original_size
            }
            
            # Performance comparison if both evaluations succeeded
            original_perf = evaluation_results['original_model']['results']
            compressed_perf = evaluation_results['compressed_model']['results']
            
            if isinstance(original_perf, dict) and isinstance(compressed_perf, dict):
                performance_retention = {}
                for metric in original_perf:
                    if metric in compressed_perf and isinstance(original_perf[metric], (int, float)):
                        retention = (compressed_perf[metric] / original_perf[metric]) * 100
                        performance_retention[metric] = retention
                
                evaluation_results['performance_retention'] = performance_retention
    
    # Save results
    results_file = os.path.join(args.output_dir, f'evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Results saved to: {results_file}")
    
    if 'compression_metrics' in evaluation_results:
        metrics = evaluation_results['compression_metrics']
        print(f"Size reduction: {metrics['size_reduction_percentage']:.2f}%")
        print(f"Compression ratio: {metrics['compression_ratio']:.3f}")
    
    if 'performance_retention' in evaluation_results:
        print("Performance retention:")
        for metric, retention in evaluation_results['performance_retention'].items():
            print(f"  {metric}: {retention:.2f}%")
    
    print("Evaluation completed!")

if __name__ == '__main__':
    main()
