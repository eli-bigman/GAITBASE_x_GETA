# GETA-OpenGait Integration: GaitBase Compression with GETA

This project integrates GETA (Gradient-based and Efficient Training Algorithm) with OpenGait's GaitBase model for neural network compression while maintaining gait recognition performance.

## 🚨 CRITICAL: PyTorch Version Requirement

**⚠️ MUST READ FIRST ⚠️**

GETA requires **EXACTLY** PyTorch 2.0.1! Install the correct version before proceeding:

```bash
# REQUIRED: GETA Dependencies - EXACT versions
pip install "torch==2.0.1+cu117" \
            "torchvision==0.15.2+cu117" \
            "torchaudio==2.0.2" \
            --extra-index-url https://download.pytorch.org/whl/cu117
```

❌ **PyTorch 2.1+ will fail** with ONNX export errors  
❌ **PyTorch 1.x will fail** with missing features  
✅ **PyTorch 2.0.1+cu117 works perfectly**

See `SETUP_GUIDE.md` for complete setup instructions.

## 📁 Project Structure

```
gaitbase_compression_GETA/
├── src/
│   ├── geta_opengait_integration.py     # Main integration class
│   ├── train_gaitbase_with_geta.py      # Training script
│   ├── evaluate_geta_compression.py     # Evaluation script
│   ├── model_comparison_statistics.py   # Model comparison utilities
│   ├── test_setup.py                    # Environment setup validation
│   └── gaitbase_geta.yaml               # Configuration file
├── notebooks/
│   ├── GETA_OpenGait_Integration_Tutorial.ipynb
│   └── geta-testing.ipynb
└── README.md
```

## 🚀 Quick Start

### 🎯 **RECOMMENDED: Use the Quick Start Script**

For the easiest setup, use the all-in-one quick start script:

```bash
# Test environment setup
python src/quick_start.py --action test

# Run training with GETA compression
python src/quick_start.py --action train --config src/gaitbase_geta.yaml --gpu 0

# Evaluate a trained model
python src/quick_start.py --action evaluate --config src/gaitbase_geta.yaml --model checkpoints/model.pth
```

This script automatically handles:
- ✅ Path detection and setup
- ✅ Distributed training initialization  
- ✅ Environment validation
- ✅ Error handling and fallbacks

### 🔧 **Alternative: Manual Setup**

If you prefer manual control:

#### 1. Environment Setup

First, test if your environment is properly configured:

```bash
python src/test_setup.py
```

This will verify:
- ✅ GETA and OpenGait paths are correctly set
- ✅ All required imports work
- ✅ Configuration files are found
- ✅ GPU availability
- ✅ Distributed training compatibility

#### 2. Training with GETA Compression

Use the fixed training script that handles distributed training:

```bash
python src/train_gaitbase_with_geta_fixed.py \
    --config src/gaitbase_geta.yaml \
    --gpu 0 \
    --validate
```

Or use the original script if distributed training is already set up:

```bash
python src/train_gaitbase_with_geta.py \
    --config src/gaitbase_geta.yaml \
    --gpu 0 \
    --validate
```

Parameters:
- `--config`: Path to configuration file
- `--gpu`: GPU device ID
- `--validate`: Run compatibility check before training

#### 3. Evaluation

After training, evaluate the compressed models:

```bash
python src/evaluate_geta_compression.py \
    --config src/gaitbase_geta.yaml \
    --original_model checkpoints/original_model.pth \
    --compressed_model compressed_models/compressed_model.pth \
    --full_sparse_model compressed_models/full_sparse_model.pth \
    --gpu 0
```

## 🔧 Key Features

### GETA Integration Features:
- **Structured Pruning**: Group-wise sparsity for efficient compression
- **HESSO Optimizer**: Gradient-based optimization with sparsity constraints
- **Progressive Compression**: Gradual pruning during training
- **Compatibility Validation**: Automatic checks for problematic layers

### OpenGait Integration Features:
- **Native Data Pipeline**: Uses OpenGait's TripletSampler and DataSet
- **Loss Functions**: Supports TripletLoss and CrossEntropyLoss
- **Evaluation Framework**: Built-in accuracy assessment
- **Model Architecture**: Full GaitBase model support

## ⚙️ Configuration

Key configuration parameters in `gaitbase_geta.yaml`:

```yaml
# GETA-specific settings (handled in code)
target_group_sparsity: 0.6        # Target compression ratio
start_pruning_step: 16000          # When to start pruning
pruning_periods: 15                # Gradual pruning steps
pruning_steps: 26667               # Total pruning duration

# OpenGait settings
model_cfg:
  model: Baseline                  # GaitBase model
  backbone_cfg:
    type: ResNet9                  # Backbone architecture

trainer_cfg:
  total_iter: 80000               # Training iterations
  save_iter: 2000                 # Checkpoint frequency
  log_iter: 100                   # Logging frequency
```

## 🧪 Troubleshooting

### ⚡ **Critical Issue: Distributed Training Error**

**Error**: `ValueError: Default process group has not been initialized, please make sure to call init_process_group.`

**Why**: OpenGait requires PyTorch distributed training to be initialized, even for single-GPU training.

**Solutions**:
1. **✅ RECOMMENDED**: Use `quick_start.py` which handles this automatically
2. **🔧 Manual Fix**: Use `train_gaitbase_with_geta_fixed.py` instead of the original training script
3. **🛠️ Test First**: Run `python src/test_distributed.py` to verify distributed setup

### Common Issues:

1. **FileNotFoundError: './configs/default.yaml'**
   - **Cause**: OpenGait expects to run from its root directory
   - **Solution**: The integration automatically handles working directory changes

2. **Import Errors**
   - **Cause**: Missing paths to GETA or OpenGait
   - **Solution**: Run `test_setup.py` to verify all paths are correct

3. **CUDA Errors**
   - **Cause**: GPU configuration issues
   - **Solution**: Check `nvidia-smi` and ensure PyTorch CUDA version matches

4. **Memory Issues**
   - **Cause**: Large batch sizes or model size
   - **Solution**: Reduce batch_size in config or use gradient accumulation

5. **Distributed Training Issues**
   - **Cause**: OpenGait's message manager requires distributed training
   - **Solution**: The integration now automatically initializes single-GPU distributed training

### 🔧 Error Fixes Applied:

✅ **Fixed distributed training initialization**: Automatic setup for single-GPU training  
✅ **Fixed working directory issue**: Integration automatically changes to OpenGait directory for config loading  
✅ **Added path detection**: Dynamic path resolution for both Kaggle and local environments  
✅ **Enhanced error handling**: Better error messages and validation  
✅ **Added evaluation framework**: Comprehensive model testing and comparison  
✅ **Created fallback message manager**: Works when distributed training fails  

## 📊 Output Files

### Training Outputs:
- `checkpoints/`: Model checkpoints during training
- `compressed_models/`: Final compressed models
- `logs/`: Training logs and metrics

### Evaluation Outputs:
- `evaluation_results/`: JSON files with detailed metrics
- Model size comparisons
- Performance retention analysis
- Compression ratio statistics

## 🔍 Model Compatibility

The integration checks for:
- ✅ Forward pass compatibility
- ✅ Gradient flow validation
- ⚠️ Problematic layers (LSTM, GRU, RNN)
- ✅ Memory requirements

## 📈 Expected Results

Typical compression results:
- **Size Reduction**: 40-70% model size reduction
- **Performance Retention**: 85-95% accuracy retention
- **Inference Speed**: 2-3x faster inference

## 🤝 Contributing

When modifying the integration:

1. Test with `test_setup.py` first
2. Validate compression compatibility
3. Run full evaluation pipeline
4. Update configuration examples

## 📚 References

- **GETA**: Gradient-based Efficient Training Algorithm
- **OpenGait**: Open source gait recognition framework
- **GaitBase**: Baseline model for gait recognition
- **CASIA-B**: Gait recognition dataset

## 🆘 Support

If you encounter issues:

1. Run `test_setup.py` for environment validation
2. Check the error logs in the output directories
3. Verify GPU memory availability
4. Ensure dataset paths are correct in the config

The integration handles most common issues automatically, but manual intervention may be needed for dataset-specific configurations.
