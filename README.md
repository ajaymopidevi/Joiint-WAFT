# Joint-WAFT: Joint Stereo Disparity & Optical Flow Transformer

Joint-WAFT extends the **Warped Attention Flow Transformer** paradigm into a multi-task framework for stereo video sequences ($I_L^t, I_R^t, I_L^{t+1}, I_R^{t+1}$).

## Architecture Overview

```
Input Video: [I_L^t, I_R^t, I_L^{t+1}] ──► Shared Backbone Encoder ──► Task Adapters
                                                                     ├──► Stereo Head (1D Epipolar Warping) ──► Disparity Map
                                                                     └──► Flow Head   (2D Motion Warping)   ──► Optical Flow
```

- **Shared Backbone**: Reuses feature extraction across all frames to minimize compute and memory overhead.
- **Task Adapters**: Decouple 1D epipolar disparity representations from 2D temporal motion representations to avoid negative transfer.
- **Coupled Recurrent Unit**: Joint iterative refinement exchanging hidden state cues between motion and depth.
- **Loss / Supervision**: Multi-task iterative Laplace & L1 losses + 4-view cyclic consistency.

## Usage

### Run Quick Test
```bash
cd Joint-WAFT
python test_joint.py
```

### Train on Multi-View Dataset with Monitoring
```bash
python main.py --config-file configs/joint_waft.yaml --num-gpus 4
```

### Experiment Monitoring

#### 1. TensorBoard (Default)
TensorBoard event logs are automatically saved under the checkpoint directory (e.g. `ckpts/joint_waft/joint_waft/42/tb_logs`) or custom directory via `--tb-dir`:

```bash
tensorboard --logdir ckpts/joint_waft/
```

Logged metrics include:
- `train/lr`: Learning rate
- `train/total_loss`: Combined multi-task objective
- `train/loss_disp`: Disparity sequence + initialization loss
- `train/loss_flow`: Optical flow sequence loss
- `train/Disp_EPE`, `train/Disp_D1`: Stereo error metrics
- `train/Flow_EPE`, `train/Flow_1px`: Flow error metrics

#### 2. Weights & Biases (Optional)
To log to WandB in addition to TensorBoard:
```bash
python main.py --config-file configs/joint_waft.yaml --num-gpus 4 --use-wandb
```
