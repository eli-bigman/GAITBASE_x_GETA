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
        '/kaggle/working/GAITBASE_x_GETA/geta',
        '/kaggle/working/GAITBASE_x_GETA/geta'
    ]
    
    possible_opengait_paths = [
        '/kaggle/working/OpenGait',
        '/kaggle/working/GAITBASE_x_GETA/OpenGait'
    ]
    
    # Add GETA path
    for path in possible_geta_paths:
        if os.path.exists(path):
            sys.path.insert(0, path)
            print(f"✅ Found GETA at: {path}")
            break
    else:
        raise ImportError("❌ Could not find GETA directory")
    
    # Add OpenGait path  
    for path in possible_opengait_paths:
        if os.path.exists(path):
            sys.path.insert(0, path)
            print(f"✅ Found OpenGait at: {path}")
            break
    else:
        raise ImportError("❌ Could not find OpenGait directory")

# Setup paths before imports
setup_paths()

# GETA imports
from only_train_once import OTO

# OpenGait imports
from opengait.data import transform as base_transform
from opengait.data.dataset import DataSet
from opengait.modeling import models
from opengait.utils import config_loader, get_ddp_module, get_msg_mgr, is_main_process
from opengait.modeling.losses import TripletLoss

class GETAOpenGaitTrainer:
    def __init__(self, config_path):
        self.config_path = config_path
        self.cfg = config_loader(config_path)
        self.msg_mgr = get_msg_mgr()
        
    def setup_model(self):
        """Setup the GaitBase model from OpenGait"""
        Model = getattr(models, self.cfg['model_cfg']['model'])
        self.model = Model(self.cfg, training=True)
        
        # Move to GPU
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            
        return self.model
    
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
        """Create appropriate dummy input for GaitBase"""
        # Get frame configuration from config
        frames_num = self.cfg['trainer_cfg']['sampler'].get('frames_num_fixed', 30)
        
        # For CASIA-B: silhouettes are grayscale (1 channel)
        # Typical size after preprocessing: 64x44
        batch_size = 1
        channels = 1
        height = 64  # Adjust based on your preprocessing
        width = 44   # Adjust based on your preprocessing
        
        dummy_input = torch.rand(batch_size, frames_num, channels, height, width)
        
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
        
        return dummy_input
        
    def setup_geta_oto(self):
        """Setup GETA OTO for compression"""
        # Create appropriate dummy input
        dummy_input = self.create_dummy_input()
        print(f"Using dummy input shape: {dummy_input.shape}")
            
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
        
        try:
            with torch.no_grad():
                output = self.model(dummy_input)
            print("✅ Forward pass successful")
            print(f"Output shape: {output.shape if hasattr(output, 'shape') else type(output)}")
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
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