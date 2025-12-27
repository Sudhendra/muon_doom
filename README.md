# muon_doom

Training reinforcement learning agents on DOOM using PufferLib PPO with comparative experiments between Adam and Muon optimizers.

## Project Overview

This project implements a complete RL training pipeline for ViZDoom's Basic scenario using:
- **Environment**: ViZDoom Basic (Gymnasium v1)
- **Preprocessing**: Grayscale + 84x84 resize + 4-frame stacking
- **Trainer**: PufferLib PPO (high-performance RL at millions of SPS)
- **Experiments**: PPO with Adam vs PPO with Muon optimizer

## Setup

### 1. Prerequisites

- Python 3.9+
- macOS/Linux (Windows support via WSL)
- Boost libraries (for ViZDoom)
  - macOS: `brew install boost cmake`
  - Linux: `sudo apt-get install libboost-all-dev cmake`

### 2. Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # or `venv/bin/activate` on some systems

# Install dependencies
pip install --upgrade pip
pip install -e .

# Or install manually
pip install gymnasium numpy opencv-python vizdoom pufferlib torch heavyball
```

## Project Structure

```
muon_doom/
├── muon_doom/
│   ├── envs/
│   │   ├── vizdoom_basic.py      # ViZDoom wrapper with preprocessing
│   │   └── puffer_wrapper.py     # PufferLib integration
│   └── training/
│       └── train.py               # Main training script
├── Plan.md                        # Detailed experiment plan
├── README.md                      # This file
└── requirements.txt               # Dependencies
```

## Usage

### Quick Start - Train with Adam

```bash
# Activate venv
source venv/bin/activate

# Train with Adam optimizer
PYTHONPATH=. python muon_doom/training/train.py \
    --optimizer adam \
    --learning-rate 3e-4 \
    --total-timesteps 1000000 \
    --num-envs 8 \
    --batch-size 2048 \
    --bptt-horizon 16
```

### Train with Muon Optimizer

```bash
# Train with Muon optimizer (uses larger LR)
PYTHONPATH=. python muon_doom/training/train.py \
    --optimizer muon \
    --learning-rate 0.02 \
    --total-timesteps 1000000 \
    --num-envs 8 \
    --batch-size 2048 \
    --bptt-horizon 16
```

### Training Arguments

```
Environment:
  --num-envs N              Number of parallel environments (default: 8)
  --num-workers N           Number of worker processes (default: 2)
  --backend {Serial,Multiprocessing}  Vectorization backend

Optimizer:
  --optimizer {adam,muon}   Optimizer choice
  --learning-rate LR        Learning rate (adam: ~3e-4, muon: ~0.02)
  --anneal-lr               Enable LR annealing

Training:
  --total-timesteps N       Total environment steps (default: 1M)
  --batch-size N            Batch size (default: 2048)
  --bptt-horizon N          Sequence length (default: 16)
  --minibatch-size N        Minibatch size (default: 512)
  --update-epochs N         PPO update epochs (default: 4)

PPO Hyperparameters:
  --gamma G                 Discount factor (default: 0.99)
  --gae-lambda L            GAE lambda (default: 0.95)
  --clip-coef C             PPO clip coefficient (default: 0.2)
  --vf-coef C               Value function coefficient (default: 0.5)
  --ent-coef C              Entropy coefficient (default: 0.01)
  --max-grad-norm N         Max gradient norm (default: 0.5)

Misc:
  --seed S                  Random seed (default: 42)
  --device {cpu,cuda,mps}   Device (default: auto-detect)
  --data-dir DIR            Checkpoint directory (default: experiments/)
```

## Implementation Details

### Environment Pipeline

1. **Base**: ViZDoom Gymnasium Basic scenario (v1)
   - RGB 320×240 observations
   - 4 discrete actions (attack, left, right, forward)
   
2. **Preprocessing**:
   - Convert RGB → Grayscale
   - Resize to 84×84
   - Stack 4 consecutive frames
   - Output: `Box(0, 255, (4, 84, 84), uint8)`

3. **PufferLib Integration**:
   - Wrapped with `GymnasiumPufferEnv` for compatibility
   - Vectorized via `pufferlib.vector.make`
   - Supports Serial and Multiprocessing backends

### Policy Network

- **Architecture**: NatureCNN (Convolutional)
  - 3 conv layers: 32→64→64 filters
  - Flattened to 512-dim hidden layer
  - Separate action and value heads
  - ~1.7M parameters

### Experiment Protocol

See `Plan.md` for the full experimental protocol including:
- Per-optimizer LR tuning procedure
- Metrics tracking (episodic return, SPS, wall-clock time)
- Multi-seed evaluation strategy

## Current Status

✅ Environment wrapper implemented and tested  
✅ PufferLib vectorization working  
✅ Training script functional with both Adam and Muon  
⏳ LR tuning sweeps (planned)  
⏳ Full multi-seed evaluation runs (planned)  

## Known Issues

1. Minor PufferLib compatibility issue with `explained_variance` calculation when NaN (does not affect training)
2. macOS multiprocessing requires proper `__main__` guards in scripts

## References

- [PufferLib](https://github.com/PufferAI/PufferLib) - High-performance RL training
- [ViZDoom](https://github.com/Farama-Foundation/ViZDoom) - DOOM-based RL environments
- [Muon Optimizer](https://github.com/KellerJordan/Muon) - Geometric optimizer for neural networks

## License

MIT
