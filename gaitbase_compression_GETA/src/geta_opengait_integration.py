import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import os

# Dynamic path detection - remove hardcoded paths
def setup_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try different path configurations
    possible_geta_paths = [
        '/kaggle/working/geta',
        '/kaggle/working/GAITBASE_x_GETA/geta',
        os.path.join(current_dir, '../../geta')
    ]
    
    possible_opengait_paths = [
        '/kaggle/working/OpenGait',
        '/kaggle/working/GAITBASE_x_GETA/OpenGait',
        os.path.join(current_dir, '../../OpenGait')
    ]
    
    # Add GETA path
    geta_path = None
    for path in possible_geta_paths:
        if os.path.exists(path):
            sys.path.insert(0, path)
            geta_path = path
            print(f"✅ Found GETA at: {path}")
            break
    else:
        raise ImportError("❌ Could not find GETA directory")
    
    # Add OpenGait path and store for working directory
    opengait_path = None
    for path in possible_opengait_paths:
        if os.path.exists(path):
            sys.path.insert(0, path)
            opengait_path = path
            print(f"✅ Found OpenGait at: {path}")
            break
    else:
        raise ImportError("❌ Could not find OpenGait directory")
    
    return geta_path, opengait_path

# Setup paths before imports
geta_path, opengait_path = setup_paths()

# Also check environment variables set by the main script
if 'OPENGAIT_PATH' in os.environ:
    opengait_path = os.environ['OPENGAIT_PATH']
if 'GETA_PATH' in os.environ:
    geta_path = os.environ['GETA_PATH']

# GETA imports
from only_train_once import OTO

# OpenGait imports
from opengait.data import transform as base_transform
from opengait.data.dataset import DataSet
from opengait.modeling import models
from opengait.utils import config_loader, get_msg_mgr
from opengait.modeling.losses import TripletLoss

class GETAOpenGaitTrainer:
    def __init__(self, config_path):
        self.config_path = config_path
        
        # Store original working directory
        self.original_cwd = os.getcwd()
        
        # CRITICAL FIX: Initialize distributed training FIRST
        self._init_simple_distributed()
        
        # Change to OpenGait directory for config loading
        os.chdir(opengait_path)
        print(f"📁 Changed working directory to: {os.getcwd()}")
        
        try:
            self.cfg = config_loader(config_path)
            
            # Fix dataset partition path to be relative to OpenGait directory
            if 'data_cfg' in self.cfg and 'dataset_partition' in self.cfg['data_cfg']:
                partition_path = self.cfg['data_cfg']['dataset_partition']
                if partition_path.startswith('./'):
                    # Convert to absolute path relative to OpenGait directory
                    abs_partition_path = os.path.join(opengait_path, partition_path[2:])
                    self.cfg['data_cfg']['dataset_partition'] = abs_partition_path
                    print(f"🔧 Fixed dataset partition path: {abs_partition_path}")
            
            # Now it's safe to get the message manager
            try:
                self.msg_mgr = get_msg_mgr()
                # Check if logger is properly initialized
                if not hasattr(self.msg_mgr, 'logger') or self.msg_mgr.logger is None:
                    print("⚠️ Message manager logger not initialized, initializing it...")
                    # Initialize the message manager properly
                    import tempfile
                    import os.path as osp
                    temp_save_path = tempfile.mkdtemp()
                    self.msg_mgr.init_manager(
                        save_path=temp_save_path,
                        log_to_file=False,  # Don't log to file for now
                        log_iter=100,
                        iteration=0
                    )
                    print("✅ Message manager properly initialized")
                else:
                    print("✅ Message manager already properly initialized")
            except Exception as e:
                print(f"⚠️ Failed to get/initialize message manager: {e}")
                # Create simple fallback
                self.msg_mgr = self._create_simple_msg_mgr()
            
        finally:
            # Return to original directory
            os.chdir(self.original_cwd)
            print(f"📁 Restored working directory to: {os.getcwd()}")
    
    def _init_simple_distributed(self):
        """Simple distributed training initialization"""
        try:
            import torch.distributed as dist
            
            if dist.is_initialized():
                print("✅ Distributed training already initialized")
                return
            
            # Set environment for single GPU
            os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
            os.environ.setdefault('MASTER_PORT', '29500')
            os.environ.setdefault('RANK', '0')
            os.environ.setdefault('WORLD_SIZE', '1')
            
            # Initialize
            import torch
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(
                backend=backend,
                init_method='env://',
                world_size=1,
                rank=0
            )
            print(f"✅ Initialized distributed training with {backend}")
            
        except Exception as e:
            print(f"⚠️ Distributed training init failed: {e}")
    
    def _create_simple_msg_mgr(self):
        """Create simple message manager fallback that matches OpenGait's interface"""
        class SimpleMsgMgr:
            def __init__(self):
                # Create a simple logger-like object
                import logging
                self.logger = logging.getLogger("SimpleMsgMgr")
                self.logger.setLevel(logging.INFO)
                
                # Add console handler if not exists
                if not self.logger.handlers:
                    handler = logging.StreamHandler()
                    formatter = logging.Formatter('%(levelname)s: %(message)s')
                    handler.setFormatter(formatter)
                    self.logger.addHandler(handler)
            
            def log_info(self, *args, **kwargs):
                if hasattr(self, 'logger') and self.logger:
                    # Convert args to string for logging
                    message = ' '.join(str(arg) for arg in args)
                    self.logger.info(message, **kwargs)
                else:
                    print("INFO:", *args, **kwargs)
                    
            def log_warning(self, *args, **kwargs):
                if hasattr(self, 'logger') and self.logger:
                    message = ' '.join(str(arg) for arg in args)
                    self.logger.warning(message, **kwargs)
                else:
                    print("WARNING:", *args, **kwargs)
                    
            def log_error(self, *args, **kwargs):
                if hasattr(self, 'logger') and self.logger:
                    message = ' '.join(str(arg) for arg in args)
                    self.logger.error(message, **kwargs)
                else:
                    print("ERROR:", *args, **kwargs)
        
        return SimpleMsgMgr()
        
    def setup_model(self):
        """Setup the GaitBase model from OpenGait"""
        # Change to OpenGait directory for model setup (important for dataset loading)
        os.chdir(opengait_path)
        print(f"📁 Changed to OpenGait directory for model setup: {os.getcwd()}")
        
        try:
            Model = getattr(models, self.cfg['model_cfg']['model'])
            self.model = Model(self.cfg, training=True)
            
            # Move to GPU
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                
            print("✅ Model setup successful")
            return self.model
            
        finally:
            # Return to original directory
            os.chdir(self.original_cwd)
            print(f"📁 Restored working directory after model setup: {os.getcwd()}")
    
    def setup_data(self):
        """Setup CASIA-B dataset with OpenGait's data pipeline"""
        # Training data
        train_transform = base_transform.get_transform(self.cfg['trainer_cfg']['transform'])
        self.train_dataset = DataSet(
            [self.cfg['data_cfg']], 
            train_transform, 
            training=True
        )
        
        # Test data  
        test_transform = base_transform.get_transform(self.cfg['evaluator_cfg']['transform'])
        self.test_dataset = DataSet(
            [self.cfg['data_cfg']], 
            test_transform, 
            training=False
        )
        
        # ✅ FIX: Use OpenGait's batch sampler approach like main.py
        from opengait.data.sampler import TripletSampler
        
        sampler_cfg = self.cfg['trainer_cfg']['sampler']
        self.train_sampler = TripletSampler(
            self.train_dataset, 
            batch_size=sampler_cfg['batch_size']
        )
        
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_sampler=self.train_sampler,
            num_workers=self.cfg['data_cfg']['num_workers']
        )
        
        # Test loader (simpler)
        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.cfg['evaluator_cfg']['sampler']['batch_size'],
            shuffle=False,
            num_workers=self.cfg['data_cfg']['num_workers']
        )
        
    def create_dummy_input(self):
        """Create appropriate dummy input for GaitBase model"""
        # Get configuration
        frames_num = self.cfg['trainer_cfg']['sampler'].get('frames_num_fixed', 30)
        
        # Create dummy sequence lengths - let's try with proper batch structure
        # If the issue is that seqL[0] becomes an int instead of list, we need multiple sequences
        batch_size = 2  # Use batch size 2 to avoid the single-item issue
        
        # Create dummy sequence data - OpenGait expects [batch_size, frames, height, width]
        # Note: OpenGait models typically work with silhouettes (grayscale)
        dummy_seq = torch.rand(batch_size, frames_num, 64, 44)
        
        # Create dummy labels (batch_size,)
        dummy_labs = torch.randint(0, 74, (batch_size,))  # CASIA-B has 74 training subjects
        
        # Create dummy types (batch_size,) - typically walking types
        dummy_typs = ['nm'] * batch_size  # normal walking type
        
        # Create dummy views (batch_size,) - camera angles 
        dummy_vies = ['000'] * batch_size  # camera view angle
        
        # Create dummy sequence lengths - should be a Python list/numpy array as real data
        # The real OpenGait data loader returns sequence lengths as lists, not tensors
        import numpy as np
        dummy_seqL = [frames_num] * batch_size  # Python list: [30, 30]
        
        # Move tensors to GPU if available
        if torch.cuda.is_available():
            dummy_seq = dummy_seq.cuda()
            dummy_labs = dummy_labs.cuda()
            # Note: seqL stays as Python list - will be converted by OpenGait's np2var()
        
        # OpenGait expects inputs as a tuple of (seqs, labs, typs, vies, seqL)
        # where seqs is a list of sequences (for different input types)
        dummy_input = ([dummy_seq], dummy_labs, dummy_typs, dummy_vies, dummy_seqL)
        
        return dummy_input
        
    def setup_geta_oto(self):
        """Setup GETA OTO for compression"""
        # Create appropriate dummy input
        dummy_input = self.create_dummy_input()
        print(f"Using dummy input with sequence shape: {dummy_input[0][0].shape}")
            
        # Initialize OTO
        self.oto = OTO(model=self.model, dummy_input=dummy_input)
        
        # Setup HESSO optimizer with conservative settings for gait
        total_steps = self.cfg['trainer_cfg']['total_iter']
        
        self.optimizer = self.oto.hesso(
            variant='sgd',
            lr=self.cfg['optimizer_cfg']['lr'],
            weight_decay=self.cfg['optimizer_cfg']['weight_decay'],
            momentum=self.cfg['optimizer_cfg']['momentum'],
            target_group_sparsity=0.6,  # Start conservative for gait recognition
            start_pruning_step=total_steps // 5,  # Wait longer before pruning
            pruning_periods=15,  # More gradual pruning
            pruning_steps=total_steps // 3,  # Longer pruning period
        )
        
    def setup_losses(self):
        """Setup loss functions from config - match OpenGait structure"""
        self.losses = []
        for loss_cfg in self.cfg['loss_cfg']:
            if loss_cfg['type'] == 'TripletLoss':
                loss_fn = TripletLoss(margin=loss_cfg['margin'])
            elif loss_cfg['type'] == 'CrossEntropyLoss':
                loss_fn = nn.CrossEntropyLoss()
            else:
                raise ValueError(f"Unknown loss type: {loss_cfg['type']}")
                
            self.losses.append({
                'loss_fn': loss_fn,
                'weight': loss_cfg['loss_term_weight'],
                'type': loss_cfg['type']
            })
    
    def validate_compression_compatibility(self):
        """Check if model is compatible with GETA compression"""
        print("=== COMPRESSION COMPATIBILITY CHECK ===")
        
        # Test forward pass with dummy input
        dummy_input = self.create_dummy_input()
        print(f"Created dummy input with sequence shape: {dummy_input[0][0].shape}")
        
        try:
            with torch.no_grad():
                output = self.model(dummy_input)
            print("✅ Forward pass successful")
            print(f"Output type: {type(output)}")
            if hasattr(output, 'shape'):
                print(f"Output shape: {output.shape}")
            elif isinstance(output, (list, tuple)):
                print(f"Output is {type(output)} with {len(output)} elements")
                for i, item in enumerate(output):
                    if hasattr(item, 'shape'):
                        print(f"  Output[{i}] shape: {item.shape}")
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Check for problematic layers
        problematic_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
                problematic_layers.append(f"{name}: {type(module).__name__}")
        
        if problematic_layers:
            print("⚠️ Found potentially problematic layers for structured pruning:")
            for layer in problematic_layers:
                print(f"  - {layer}")
            print("Consider using lower compression ratios")
        
        return True
    
    def train(self):
        """Main training loop with GETA compression"""
        self.model.train()
        total_iter = self.cfg['trainer_cfg']['total_iter']
        log_iter = self.cfg['trainer_cfg']['log_iter']
        save_iter = self.cfg['trainer_cfg']['save_iter']
        
        # Learning rate scheduler
        milestones = self.cfg['scheduler_cfg']['milestones']
        gamma = self.cfg['scheduler_cfg']['gamma']
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=milestones, gamma=gamma
        )
        
        data_iter = iter(self.train_loader)
        
        for iteration in range(total_iter):
            try:
                batch_data = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch_data = next(data_iter)
            
            # ✅ FIX: Handle OpenGait's data format properly
            if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                inputs = batch_data[0]
                labels = batch_data[1]
            else:
                print(f"❌ Unexpected batch data format: {type(batch_data)}")
                continue
            
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # ✅ FIX: Handle multiple outputs properly (like main.py)
            total_loss = 0
            loss_info = {}
            
            for i, loss_config in enumerate(self.losses):
                if loss_config['type'] == 'TripletLoss':
                    # TripletLoss expects feature embeddings
                    if isinstance(outputs, (list, tuple)):
                        loss_val = loss_config['loss_fn'](outputs[0], labels)
                    else:
                        loss_val = loss_config['loss_fn'](outputs, labels)
                elif loss_config['type'] == 'CrossEntropyLoss':
                    # CrossEntropyLoss expects logits
                    if isinstance(outputs, (list, tuple)):
                        loss_val = loss_config['loss_fn'](outputs[1], labels)
                    else:
                        loss_val = loss_config['loss_fn'](outputs, labels)
                
                weighted_loss = loss_val * loss_config['weight']
                total_loss += weighted_loss
                loss_info[f'loss_{i}'] = loss_val.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            scheduler.step()
            
            # Logging
            if iteration % log_iter == 0:
                opt_metrics = self.optimizer.compute_metrics()
                self.msg_mgr.log_info(
                    f"Iter: {iteration}/{total_iter}, "
                    f"Loss: {total_loss.item():.4f}, "
                    f"Group Sparsity: {opt_metrics.group_sparsity:.3f}, "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}"
                )
            
            # Save checkpoint
            if iteration % save_iter == 0 and iteration > 0:
                self.save_checkpoint(iteration)
        
        # Final compression
        self.compress_model()
    
    def save_checkpoint(self, iteration):
        """Save training checkpoint"""
        save_name = self.cfg['trainer_cfg']['save_name']
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iteration': iteration,
            'config': self.cfg
        }
        
        os.makedirs('./checkpoints', exist_ok=True)
        torch.save(checkpoint, f'./checkpoints/{save_name}_iter_{iteration}.pth')
        
    def compress_model(self):
        """Generate compressed model using GETA"""
        print("Compressing model with GETA...")
        
        # Create output directory
        os.makedirs('./compressed_models', exist_ok=True)
        
        # Construct compressed subnet
        self.oto.construct_subnet(out_dir='./compressed_models')
        
        print(f"Compressed model saved to: {self.oto.compressed_model_path}")
        print(f"Full sparse model saved to: {self.oto.full_group_sparse_model_path}")
        
        return self.oto.compressed_model_path, self.oto.full_group_sparse_model_path
    
    def evaluate_compression(self):
        """Evaluate the compression results"""
        # Load both models
        full_model = torch.load(self.oto.full_group_sparse_model_path)
        compressed_model = torch.load(self.oto.compressed_model_path)
        
        # Compare model sizes
        full_size = os.path.getsize(self.oto.full_group_sparse_model_path)
        compressed_size = os.path.getsize(self.oto.compressed_model_path)
        
        print(f"\n=== COMPRESSION RESULTS ===")
        print(f"Full model size: {full_size / (1024**2):.2f} MB")
        print(f"Compressed model size: {compressed_size / (1024**2):.2f} MB")
        print(f"Size reduction: {((full_size - compressed_size) / full_size) * 100:.2f}%")
        
        return {
            'full_size_mb': full_size / (1024**2),
            'compressed_size_mb': compressed_size / (1024**2),
            'compression_ratio': compressed_size / full_size
        }
    
    def evaluate_model(self, model_path=None, model=None):
        """Evaluate model performance using OpenGait's evaluation framework"""
        print("=== MODEL EVALUATION ===")
        
        if model_path:
            # Load model from path
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        elif model:
            self.model = model
        
        # Set to evaluation mode
        self.model.eval()
        
        # Change to OpenGait directory for evaluation
        os.chdir(opengait_path)
        
        try:
            # Import evaluation functions
            from opengait.evaluation import evaluator as eval_functions
            
            # Get evaluation function
            eval_func_name = self.cfg['evaluator_cfg']['eval_func']
            eval_func = getattr(eval_functions, eval_func_name)
            
            # Setup evaluation dataset if not already done
            if not hasattr(self, 'test_loader'):
                self.setup_data()
            
            # Run evaluation
            with torch.no_grad():
                results = eval_func(self.model, self.test_loader, self.cfg)
            
            print(f"Evaluation Results: {results}")
            return results
            
        finally:
            # Return to original directory
            os.chdir(self.original_cwd)
    
    def compare_models(self, original_model_path, compressed_model_path):
        """Compare performance between original and compressed models"""
        print("=== MODEL COMPARISON ===")
        
        # Evaluate original model
        print("Evaluating original model...")
        original_results = self.evaluate_model(original_model_path)
        
        # Evaluate compressed model
        print("Evaluating compressed model...")
        compressed_results = self.evaluate_model(compressed_model_path)
        
        # Calculate performance retention
        comparison = {
            'original_performance': original_results,
            'compressed_performance': compressed_results,
            'performance_retention': {}
        }
        
        if isinstance(original_results, dict) and isinstance(compressed_results, dict):
            for metric in original_results:
                if metric in compressed_results:
                    retention = (compressed_results[metric] / original_results[metric]) * 100
                    comparison['performance_retention'][metric] = retention
        
        print(f"Performance Comparison: {comparison}")
        return comparison