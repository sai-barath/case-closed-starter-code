# Hybrid CNN+BFS DQN Implementation

## Overview
This is a **hybrid architecture** that combines the best of both worlds:
- **CNN path**: Spatial awareness of trails, walls, and player positions
- **BFS path**: Strategic lookahead of reachable spaces in each direction

## Architecture Details

### Input Representation
1. **Spatial Input** (5 channels, 18×20 grid):
   - Channel 0: My trail positions
   - Channel 1: Opponent trail positions
   - Channel 2: Walls/obstacles
   - Channel 3: My current position (hot encoded)
   - Channel 4: Opponent current position (hot encoded)

2. **BFS Features** (8 values):
   - `space_up`: Reachable cells if moving UP
   - `space_down`: Reachable cells if moving DOWN
   - `space_left`: Reachable cells if moving LEFT
   - `space_right`: Reachable cells if moving RIGHT
   - `dist_to_opponent`: Manhattan distance (normalized)
   - `my_boosts`: Number of boosts I have (normalized)
   - `opp_boosts`: Number of boosts opponent has (normalized)
   - `turn_ratio`: Current turn / max_turns (normalized)

### Network Architecture

```
CNN Path:
  Spatial Input (5×18×20)
    ↓
  Conv2d(5→32, 3×3) + ReLU + MaxPool(2×2)  [9×10]
    ↓
  Conv2d(32→64, 3×3) + ReLU + MaxPool(2×2)  [4×5]
    ↓
  Conv2d(64→64, 3×3) + ReLU
    ↓
  Flatten → 1280 features
    ↓
  Linear(1280→256) + ReLU
    ↓
  256 features

BFS Path:
  BFS Input (8)
    ↓
  Linear(8→64) + ReLU
    ↓
  Linear(64→128) + ReLU
    ↓
  128 features

Combined:
  Concat(CNN 256 + BFS 128) = 384
    ↓
  Linear(384→256) + LayerNorm + ReLU + Dropout(0.2)
    ↓
  Linear(256→128) + ReLU
    ↓
  Linear(128→4)  [Q-values for UP, DOWN, LEFT, RIGHT]
```

## Training Strategy

### Opponent Rotation
Trains against **5 diverse heuristic agents** (rotating each episode):
1. **GreedySpaceMaximizer**: Maximizes reachable space via flood-fill
2. **AggressiveChaser**: Chases opponent using Manhattan distance
3. **SmartAvoider**: 60% distance + 40% space hybrid
4. **TerritorialDefender**: 50% space + 30% center control + 20% distance
5. **AdaptiveHybrid**: Dynamic strategy switching based on game state

Opponent selection: `opponents[episode % 5]`

### Position Alternation
- **50% Player 1**: Episodes where `episode % 2 == 0`
- **50% Player 2**: Episodes where `episode % 2 == 1`

This ensures balanced learning from both positions.

### Training Parameters
- **Learning Rate**: 0.001
- **Gamma**: 0.99
- **Epsilon**: 1.0 → 0.01 (exponential decay)
- **Replay Buffer**: 30,000 experiences
- **Batch Size**: 256
- **Target Update**: Every 500 steps

### Checkpointing
- **Every 100 episodes**: Saves checkpoint with current stats
- **Every 200 episodes**: Full evaluation against all 5 opponents (100 games each)
- **Final episode**: Saves `dqn_fast_final.pt` with complete training stats

## Files

### Core Implementation
- **`dqn_fast.py`**: Complete training system
  - `CompactDQNetwork`: Hybrid CNN+BFS architecture
  - `FastDQNAgent`: Agent with dual-tensor state encoding
  - `train_fast_dqn()`: Training loop with opponent rotation

- **`agent.py`**: Flask API server for competition
  - Synced architecture with `dqn_fast.py`
  - Loads trained model for inference
  - Safety checks and boost logic

- **`heuristic_agents.py`**: 5 sophisticated opponents
  - Each implements `get_move(state) → action`
  - Wrapper functions for training compatibility

- **`training_env.py`**: Fast training environment (189 games/sec capability)

## Usage

### 1. Train the Model
```bash
# Train for 1000 episodes (recommended for full convergence)
python dqn_fast.py 1000

# Or start with shorter run for testing
python dqn_fast.py 200
```

**Expected Output:**
```
============================================================
Models will be saved to: models_fast_20251108_XXXXXX
============================================================

Training against 5 opponents with position alternation...

Episode 100/1000 | Avg Reward: 0.45 | Win Rate: 62.0% | Epsilon: 0.89
  Opponent Stats: [Greedy: 70%, Chaser: 58%, Avoider: 65%, Defender: 55%, Hybrid: 60%]
  Position Stats: [P1: 64%, P2: 60%]
...
```

### 2. Update Flask Server
After training completes, the model is automatically saved. The `agent.py` file is already configured to use the latest model directory.

### 3. Start the Server
```bash
python agent.py
```

The server will load the trained hybrid CNN+BFS model and serve on port 5008.

## Performance Metrics

### Training Speed
- **Target**: 20+ episodes/min (despite CNN overhead)
- **Previous BFS-only**: 23.1 episodes/min
- **Previous CNN-only**: 6.6 episodes/min

The hybrid should be slightly slower than pure BFS but much faster than pure CNN.

### Win Rates
Track both:
- **Per-opponent win rates**: Performance against each heuristic type
- **Position-based win rates**: P1 vs P2 balance

### Model Size
Expected size: ~50-100 MB (less than original 138MB CNN, more than pure BFS)

## Advantages of Hybrid Approach

### CNN Path Benefits
- **Spatial pattern recognition**: Sees trails, walls, and positions as 2D patterns
- **Local context**: Understands nearby obstacles and opponent proximity visually
- **Position awareness**: Hot-encoded positions provide clear location information

### BFS Path Benefits
- **Strategic lookahead**: Knows exactly how much space is available in each direction
- **Future planning**: BFS provides concrete reachability metrics
- **Computational efficiency**: Pre-computed features are fast to use

### Combined Power
- **Tactical + Strategic**: Short-term spatial awareness + long-term planning
- **Redundancy**: If CNN misses a pattern, BFS might catch it (and vice versa)
- **Balanced learning**: Both paths trained simultaneously through backpropagation

## Debugging

### Check Architecture
```python
from dqn_fast import CompactDQNetwork
import torch

net = CompactDQNetwork()
spatial = torch.randn(1, 5, 18, 20)
bfs = torch.randn(1, 8)
output = net(spatial, bfs)
print(output.shape)  # Should be: torch.Size([1, 4])
```

### Verify State Encoding
```python
from dqn_fast import FastDQNAgent

agent = FastDQNAgent()
# Mock state
state = {
    'board': [[0]*20 for _ in range(18)],
    'my_position': (5, 5),
    'opponent_position': (10, 10),
    'my_direction': (1, 0),
    'player_number': 1,
    'turn_count': 50,
    'my_boosts': 2,
    'agent2_boosts': 1
}

spatial, bfs = agent.state_to_tensor(state)
print(f"Spatial: {spatial.shape}")  # (1, 5, 18, 20)
print(f"BFS: {bfs.shape}")  # (1, 8)
```

## Next Steps After Training

1. **Evaluate Performance**:
   ```bash
   # Check final win rates from training output
   # Compare per-opponent performance
   ```

2. **Test Flask Server**:
   ```bash
   # Start server
   python agent.py
   
   # Test endpoint (in another terminal)
   curl -X POST http://localhost:5008/get-move \
     -H "Content-Type: application/json" \
     -d '{"board": [...], "my_position": [5,5], ...}'
   ```

3. **Compare to Previous Models**:
   - Original CNN (138MB): Spatial awareness, slow training
   - BFS-only (smaller): Fast training, strategic planning
   - **Hybrid (this)**: Best of both worlds

4. **Fine-tune if Needed**:
   - Adjust training episodes (500-2000 range)
   - Modify architecture (layer sizes, dropout rate)
   - Change training parameters (learning rate, epsilon decay)

## Troubleshooting

### "RuntimeError: size mismatch"
- Architecture mismatch between training and inference
- Solution: Ensure `agent.py` and `dqn_fast.py` have identical network definitions

### "Invalid move" or crashes
- Agent trained to crash naturally when trapped (by design)
- This is expected behavior - no invalid moves are attempted

### Slow training
- Check CUDA availability: `torch.cuda.is_available()`
- Reduce batch size or network complexity
- Ensure no unnecessary logging in training loop

### Poor win rates
- Train longer (1000+ episodes)
- Check opponent rotation is working
- Verify position alternation (50/50 split)
- Consider adjusting reward shaping

## Model Files

After training `N` episodes, you'll have:
```
models_fast_20251108_XXXXXX/
├── dqn_fast_ep100.pt          # Checkpoint at episode 100
├── dqn_fast_ep200.pt          # Checkpoint at episode 200
├── ...
├── dqn_fast_final.pt          # Final trained model
└── training_stats.txt         # Training statistics
```

Each `.pt` file contains:
- `model_state_dict`: Network weights
- `optimizer_state_dict`: Optimizer state
- `episode`: Episode number
- `epsilon`: Current exploration rate
- `win_rate`: Overall win rate
- `per_opponent_stats`: Win rate vs each heuristic
- `position_stats`: P1 and P2 win rates

---

**Ready to train!** Run: `python dqn_fast.py 1000`
