# Deep Q-Network (DQN) Training Guide for Case Closed

## 🎯 Overview

This guide explains how to train a DQN agent for the Case Closed challenge using PyTorch. The system trains on GPU (if available) and saves models that can be loaded for CPU inference in `agent.py`.

## 🏗️ Architecture

### Network Design

The DQN uses a convolutional neural network to process spatial information:

```
Input: State representation (7 channels × 18 × 20 grid + 8 extra features)
  ↓
Conv2D (32 filters, 3×3)
  ↓
Conv2D (64 filters, 3×3)
  ↓
Conv2D (64 filters, 3×3)
  ↓
Flatten → Concatenate with extra features
  ↓
FC (512 units)
  ↓
FC (256 units)
  ↓
Output: Q-values for 4 actions (UP, DOWN, LEFT, RIGHT)
```

### State Representation

**Spatial Channels (7 total):**
1. **My Trail**: Binary mask of my trail positions
2. **Opponent Trail**: Binary mask of opponent's trail
3. **Walls**: Binary mask of occupied cells
4. **My Position**: One-hot encoding of my current position
5. **Opponent Position**: One-hot encoding of opponent's position
6. **Valid Moves**: Binary mask of cells I can safely move to
7. **Danger Zones**: Cells opponent might move to next

**Extra Features (8 total):**
1. My boosts remaining (normalized 0-1)
2. Opponent boosts remaining (normalized 0-1)
3. Turn count (normalized 0-1)
4. My trail length (normalized 0-1)
5. Opponent trail length (normalized 0-1)
6. Distance to opponent (normalized 0-1)
7. Current direction X component
8. Current direction Y component

### Reward Structure

```python
Survival: +1.0 per step (encourages staying alive)
Win: +100.0 (winning the game)
Loss: -100.0 (losing the game)
Draw: -50.0 (discourages draws)
```

You can experiment with different rewards:
- Reward for territorial control (trail length)
- Penalty for risky moves
- Bonus for aggressive play near opponent
- Distance-based shaping rewards

## 🚀 Quick Start

### 1. Training from Scratch

```bash
# Train for 10,000 episodes (default)
python agent2.py train

# Train for custom number of episodes
python agent2.py train 20000
```

### 2. Resume Training

```bash
# Resume from saved checkpoint
python agent2.py train 10000 models/dqn_agent_ep5000_20250108_143022.pt
```

### 3. Evaluate Trained Model

```bash
# Evaluate against baseline agents
python agent2.py eval models/dqn_agent_final.pt
```

### 4. Demo Mode (Quick Test)

```bash
# Just run the script - trains for 100 episodes as demo
python agent2.py
```

## 📊 Training Output

During training, you'll see output like:

```
Starting DQN Training on cuda
==================================================================

Episode    10 | Reward:   45.20 | Length:  38.5 | Win Rate: 55.00% | Loss:  0.0234 | Epsilon: 0.9850 | Buffer: 770
Episode    20 | Reward:   52.10 | Length:  42.3 | Win Rate: 60.00% | Loss:  0.0198 | Epsilon: 0.9702 | Buffer: 1685
...

==================================================================
Evaluation at Episode 100
==================================================================

DQN vs Wall Avoider (50 games)...
  DQN Win Rate: 68.0%

DQN vs Greedy Space Agent (50 games)...
  DQN Win Rate: 45.0%
==================================================================

Model saved to models/dqn_agent_ep100_20250108_143522.pt
```

### Metrics Explained

- **Reward**: Average cumulative reward over last 10 episodes
- **Length**: Average game length (steps) over last 10 episodes
- **Win Rate**: Percentage of wins in last 100 self-play games
- **Loss**: Average TD loss over last 100 training steps
- **Epsilon**: Current exploration rate (starts at 1.0, decays to 0.01)
- **Buffer**: Number of experiences in replay buffer

## 🎮 Hyperparameters

You can tune these in the `DQNAgent` class:

```python
learning_rate = 0.0001      # Adam optimizer learning rate
gamma = 0.99                 # Discount factor for future rewards
epsilon_start = 1.0          # Initial exploration rate
epsilon_end = 0.01           # Minimum exploration rate
epsilon_decay = 0.995        # Decay multiplier per episode
buffer_capacity = 100000     # Max experiences in replay buffer
batch_size = 64              # Batch size for training
target_update_freq = 1000    # Steps between target network updates
```

### Hyperparameter Tuning Tips

**Learning Rate:**
- Too high → unstable learning, oscillating performance
- Too low → very slow learning
- Sweet spot: 0.0001 - 0.001

**Gamma (Discount Factor):**
- Higher (0.99) → values long-term rewards (strategic play)
- Lower (0.9) → values immediate rewards (aggressive play)

**Epsilon Decay:**
- Faster decay → exploits learned policy earlier
- Slower decay → explores longer (better for complex environments)

**Batch Size:**
- Larger → more stable gradients, slower training
- Smaller → noisier gradients, faster updates

## 💾 Model Files

Models are saved to `models/` directory with timestamped names:

```
models/
├── dqn_agent_ep500_20250108_143022.pt   # Checkpoint at episode 500
├── dqn_agent_ep1000_20250108_144530.pt  # Checkpoint at episode 1000
└── dqn_agent_final.pt                    # Final trained model
```

Each model file contains:
- Policy network weights
- Optimizer state
- Training statistics
- Metadata (episode number, win rate, etc.)

## 🔬 Advanced Usage

### Custom Training Loop

```python
from agent2 import DQNAgent, TrainingEnvironment
from training_env import wall_avoider_agent

# Initialize agent
agent = DQNAgent(
    learning_rate=0.0005,
    gamma=0.95,
    epsilon_decay=0.99
)

# Custom training loop
for episode in range(1000):
    env = TrainingEnvironment()
    state, _ = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state, training=True)
        next_state, _, reward, _, done, _ = env.step(action, "UP")
        
        agent.replay_buffer.push(state, action, reward, next_state, done)
        agent.train_step()
        
        state = next_state
    
    if episode % 100 == 0:
        agent.save_model(f"models/custom_ep{episode}.pt")
```

### Loading Model for Inference

```python
from agent2 import create_inference_agent
from training_env import run_episode, wall_avoider_agent

# Load trained model
dqn_policy = create_inference_agent("models/dqn_agent_final.pt")

# Use in games
result, history = run_episode(dqn_policy, wall_avoider_agent, render=True)
print(f"Result: {result}, Length: {len(history)}")
```

### Integrating with agent.py

To use the trained model in your competition agent:

```python
# In agent.py

import torch
import torch.nn as nn
from agent2 import DQNetwork
import numpy as np

# Load model (CPU mode)
model = DQNetwork()
checkpoint = torch.load("models/dqn_agent_final.pt", map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def get_move(state):
    """
    Use trained DQN to select move.
    
    Args:
        state: Game state from judge engine
        
    Returns:
        Move string (e.g., "UP", "DOWN:BOOST")
    """
    # Convert state to tensor (implement state_to_tensor logic)
    spatial_tensor, extra_features = state_to_tensor(state)
    
    with torch.no_grad():
        q_values = model(spatial_tensor, extra_features)
        q_values = q_values.cpu().numpy()[0]
    
    # Select best valid action
    valid_actions = get_valid_actions(state)
    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    masked_q = q_values.copy()
    for idx, action in enumerate(actions):
        if action not in valid_actions:
            masked_q[idx] = -1e9
    
    best_action = actions[np.argmax(masked_q)]
    
    # Optional: Use boost if advantageous
    # (add boost logic based on Q-values or heuristics)
    
    return best_action
```

## 🧪 Experimentation Ideas

### 1. Reward Shaping

Try different reward structures:

```python
def _calculate_rewards(self, result):
    if result is None:
        # Reward for controlling more territory
        territory_bonus = (self.game.agent1.length - self.game.agent2.length) * 0.1
        return 1.0 + territory_bonus, 1.0 - territory_bonus
    
    # ... rest of rewards
```

### 2. Curriculum Learning

Start with easier opponents, gradually increase difficulty:

```python
# Episode 0-1000: Train against random agent
# Episode 1000-3000: Train against wall avoider
# Episode 3000+: Train against self
```

### 3. Prioritized Experience Replay

Sample important experiences more frequently:

```python
# Store experiences with priority based on TD error
# Sample proportional to priority
```

### 4. Dueling DQN

Separate value and advantage streams:

```python
self.value_stream = nn.Linear(256, 1)
self.advantage_stream = nn.Linear(256, 4)

q_values = value + (advantage - advantage.mean())
```

### 5. Double DQN

Use policy network for action selection, target network for evaluation:

```python
# In train_step():
next_actions = self.policy_net(next_states).argmax(1)
next_q_values = self.target_net(next_states).gather(1, next_actions)
```

## 📈 Expected Performance

After different training durations:

| Episodes | Win vs Random | Win vs Wall Avoider | Win vs Greedy Space |
|----------|---------------|---------------------|---------------------|
| 100      | ~70%          | ~40%                | ~30%                |
| 1,000    | ~95%          | ~65%                | ~45%                |
| 5,000    | ~99%          | ~80%                | ~60%                |
| 10,000   | ~100%         | ~90%                | ~75%                |

*Note: Results vary based on random seed and hyperparameters*

## 🐛 Troubleshooting

### Issue: Training is very slow
**Solution:** Ensure CUDA is available and being used:
```python
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show your GPU
```

### Issue: Win rate stuck around 50%
**Solution:** 
- Increase training episodes
- Adjust learning rate
- Try different reward structure
- Check if agent is exploring enough (epsilon)

### Issue: Agent makes invalid moves
**Solution:** The `select_action` function already masks invalid moves, but double-check:
- Board wrapping logic (torus topology)
- Reversal prevention
- Wall checking

### Issue: High loss values not decreasing
**Solution:**
- Reduce learning rate
- Increase batch size
- Check reward scale (normalize if too large)
- Ensure target network is updating

### Issue: Agent learns then "forgets"
**Solution:**
- Increase replay buffer size
- Decrease epsilon decay rate
- Add dropout for regularization
- Save more frequent checkpoints

## 🎓 Learning Resources

### Reinforcement Learning Fundamentals
- [Sutton & Barto: RL Book](http://incompleteideas.net/book/the-book-2nd.html)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [DeepMind x UCL RL Course](https://www.deepmind.com/learning-resources/reinforcement-learning-lecture-series-2021)

### DQN Papers
- [Playing Atari with Deep RL (Mnih et al., 2013)](https://arxiv.org/abs/1312.5602)
- [Human-level control through deep RL (Nature, 2015)](https://www.nature.com/articles/nature14236)
- [Rainbow: Combining Improvements in DRL (2017)](https://arxiv.org/abs/1710.02298)

### PyTorch Resources
- [PyTorch DQN Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## 📝 Next Steps

1. **Train baseline model**: Run for 5000-10000 episodes
2. **Analyze performance**: Plot rewards, win rates over time
3. **Experiment with rewards**: Try territorial control bonuses
4. **Test against humans**: Deploy and play against other competitors
5. **Advanced algorithms**: Try PPO, A3C, or Rainbow DQN

## 🏆 Competition Tips

1. **Ensemble multiple models**: Train several models with different seeds, vote on actions
2. **Add heuristics**: Combine RL policy with rule-based safety checks
3. **Opponent modeling**: Track opponent behavior patterns
4. **Adaptive strategy**: Switch between aggressive/defensive based on game state
5. **Boost management**: Save boosts for critical moments (escape, attack)

Good luck with your training! 🚀
