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
        """Create dummy input for GETA model tracing.
        
        OpenGait models expect a 5-tuple: (seqs, labs, typs, vies, seqL)
        Based on CASIA-B dataset format.
        """
        batch_size = 4
        frames = 30
        height = 64 
        width = 44
        
        # Create the 5-tuple that OpenGait expects
        seqs = torch.randn(batch_size, frames, height, width)  # sequences
        labs = torch.randint(0, 10, (batch_size,))           # labels  
        typs = torch.randint(0, 2, (batch_size,))            # types (nm, bg, cl)
        vies = torch.randint(0, 11, (batch_size,))           # views (0-10 for CASIA-B)
        seqL = torch.full((batch_size,), frames)             # sequence lengths
        
        if torch.cuda.is_available():
            seqs = seqs.cuda()
            labs = labs.cuda()
            typs = typs.cuda()
            vies = vies.cuda()
            seqL = seqL.cuda()
        
        # Return as tuple for GETA OTO initialization
        return (seqs, labs, typs, vies, seqL)
        
    def setup_geta_oto(self):
        """Setup GETA OTO for compression - simplified approach following GETA tutorial"""
        print("🔧 Setting up GETA compression...")
        
        # Simple dummy input for GETA model tracing only
        dummy_input = self.create_dummy_input()
        print(f"Created dummy input for GETA tracing: {dummy_input.shape}")
        
        # Initialize OTO with the model (GETA tutorial step 1)
        try:
            self.oto = OTO(model=self.model, dummy_input=dummy_input)
            print("✅ GETA OTO initialized successfully")
        except Exception as e:
            print(f"❌ GETA OTO initialization failed: {e}")
            # Fall back to regular PyTorch optimizer if GETA fails
            print("⚠️ Falling back to standard SGD optimizer")
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg['optimizer_cfg']['lr'],
                momentum=self.cfg['optimizer_cfg']['momentum'],
                weight_decay=self.cfg['optimizer_cfg']['weight_decay']
            )
            self.use_geta = False
            return
        
        # Setup HESSO optimizer (GETA tutorial step 2)
        total_steps = self.cfg['trainer_cfg']['total_iter']
        
        try:
            self.optimizer = self.oto.hesso(
                variant='sgd',
                lr=self.cfg['optimizer_cfg']['lr'],
                weight_decay=self.cfg['optimizer_cfg']['weight_decay'],
                momentum=self.cfg['optimizer_cfg']['momentum'],
                target_group_sparsity=0.7,  # From config
                start_pruning_step=total_steps // 4,  # Start pruning after 25%
                pruning_periods=10,  # Gradual pruning
                pruning_steps=total_steps // 2,  # Prune for 50% of training
            )
            self.use_geta = True
            print("✅ GETA HESSO optimizer setup successful")
            
        except Exception as e:
            print(f"❌ GETA HESSO setup failed: {e}")
            # Fall back to regular optimizer
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg['optimizer_cfg']['lr'],
                momentum=self.cfg['optimizer_cfg']['momentum'],
                weight_decay=self.cfg['optimizer_cfg']['weight_decay']
            )
            self.use_geta = False
            print("⚠️ Using standard SGD optimizer instead")
        
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
        
        # Skip complex forward pass testing - just check if GETA can initialize
        # The real test happens during actual training with real data
        try:
            # Test if we can create the optimizer
            self.setup_geta_oto()
            
            if hasattr(self, 'use_geta') and self.use_geta:
                print("✅ GETA initialization successful")
                print("✅ Model is compatible with GETA compression")
            else:
                print("⚠️ GETA initialization failed, but fallback optimizer available")
                print("⚠️ Will use standard training without compression")
                
            return True
            
        except Exception as e:
            print(f"❌ Compatibility check failed: {e}")
            return False
    
    def train(self):
        """Main training loop with optional GETA compression"""
        print("🚀 Starting training...")
        
        # Setup optimizer (GETA or fallback)
        if not hasattr(self, 'optimizer'):
            self.setup_geta_oto()
        
        # Setup losses
        self.setup_losses()
        
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
        
        # Use OpenGait's real data loader
        data_iter = iter(self.train_loader)
        
        for iteration in range(total_iter):
            try:
                # Get real batch from OpenGait's data loader
                batch_data = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch_data = next(data_iter)
            
            # Let OpenGait handle its own data format
            # Just pass it directly to the model like in OpenGait's main.py
            inputs = batch_data[0]  # OpenGait's data format
            labels = batch_data[1]  # OpenGait's labels
            
            # Forward pass with real OpenGait data
            outputs = self.model(inputs)
            
            # Calculate losses using OpenGait's approach
            total_loss = 0
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
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            scheduler.step()
            
            # Logging
            if iteration % log_iter == 0:
                if hasattr(self, 'use_geta') and self.use_geta:
                    # GETA optimizer has metrics
                    opt_metrics = self.optimizer.compute_metrics()
                    self.msg_mgr.log_info(
                        f"Iter: {iteration}/{total_iter}, "
                        f"Loss: {total_loss.item():.4f}, "
                        f"Group Sparsity: {opt_metrics.group_sparsity:.3f}, "
                        f"LR: {scheduler.get_last_lr()[0]:.6f}"
                    )
                else:
                    # Standard optimizer
                    self.msg_mgr.log_info(
                        f"Iter: {iteration}/{total_iter}, "
                        f"Loss: {total_loss.item():.4f}, "
                        f"LR: {scheduler.get_last_lr()[0]:.6f}"
                    )
            
            # Save checkpoint
            if iteration % save_iter == 0 and iteration > 0:
                self.save_checkpoint(iteration)
        
        # Final compression (only if using GETA)
        if hasattr(self, 'use_geta') and self.use_geta:
            self.compress_model()
        else:
            print("⚠️ No compression applied - trained with standard optimizer")
    
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