"""
Deep Q-Network (DQN) Training System for Case Closed Challenge

This module implements a complete DQN training pipeline with:
- PyTorch-based neural network architecture
- Experience replay buffer
- Target network for stable learning
- Self-play training against copies of itself
- GPU training with CPU-compatible model saving
- Epsilon-greedy exploration strategy

The trained model can be loaded into agent.py for CPU inference.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional
import copy
import os
import json
from datetime import datetime
import time

from training_env import TrainingEnvironment, run_episode, evaluate_agents
from case_closed_game import GameResult

# Check for CUDA availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNetwork(nn.Module):
    """
    Deep Q-Network architecture for Case Closed.
    
    Input: State tensor (channels × height × width)
    Output: Q-values for each action (4 actions: UP, DOWN, LEFT, RIGHT)
    
    Architecture:
    - 3 convolutional layers to process spatial information
    - 2 fully connected layers for decision making
    - Outputs Q-value for each of the 4 possible moves
    """
    
    def __init__(self, input_channels: int = 7, height: int = 18, width: int = 20):
        """
        Initialize the DQN architecture.
        
        Args:
            input_channels: Number of input channels (default 7):
                - Channel 0: My trail
                - Channel 1: Opponent trail
                - Channel 2: Walls/boundaries
                - Channel 3: My position
                - Channel 4: Opponent position
                - Channel 5: My valid moves mask
                - Channel 6: Danger zones (cells that might kill me)
            height: Board height (18)
            width: Board width (20)
        """
        super(DQNetwork, self).__init__()
        
        self.input_channels = input_channels
        self.height = height
        self.width = width
        
        # Convolutional layers to process spatial information
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Calculate size after convolutions (no pooling, so same size)
        conv_output_size = height * width * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size + 8, 512)  # +8 for extra features
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)  # 4 actions: UP, DOWN, LEFT, RIGHT
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, state_tensor: torch.Tensor, extra_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state_tensor: Spatial state (batch × channels × height × width)
            extra_features: Non-spatial features (batch × 8):
                - My boosts remaining
                - Opponent boosts remaining
                - Turn count (normalized)
                - My trail length (normalized)
                - Opponent trail length (normalized)
                - Distance to opponent (normalized)
                - My direction (one-hot encoded as 2 values: dx, dy)
                
        Returns:
            Q-values for each action (batch × 4)
        """
        # Convolutional layers with ReLU activation
        x = F.relu(self.conv1(state_tensor))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten spatial features
        x = x.view(x.size(0), -1)
        
        # Concatenate with extra features
        x = torch.cat([x, extra_features], dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    
    Stores past experiences and samples random batches for training.
    This breaks correlation between consecutive samples and stabilizes learning.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add an experience to the buffer."""
        self.buffer.append(Experience(state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a random batch of experiences."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent with experience replay and target network.
    
    This agent learns to play Case Closed through self-play using DQN algorithm.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_capacity: int = 50000,  # Reduced from 100k for faster training
                 batch_size: int = 128,  # Increased from 64 for more stable updates
                 target_update_freq: int = 1000,
                 save_dir: Optional[str] = None):
        """
        Initialize DQN agent.
        
        Args:
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            buffer_capacity: Size of replay buffer (50k for speed)
            batch_size: Batch size for training (128 for stability)
            target_update_freq: Steps between target network updates
            save_dir: Directory to save models (timestamped if None)
        """
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Create timestamped model directory
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"models_{timestamp}"
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Models will be saved to: {self.save_dir}")
        print(f"{'='*60}\n")
        
        # Action mapping
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.action_to_idx = {action: idx for idx, action in enumerate(self.actions)}
        
        # Direction deltas for action validation
        self.direction_deltas = {
            'UP': (0, -1),
            'DOWN': (0, 1),
            'LEFT': (-1, 0),
            'RIGHT': (1, 0),
        }
        
        # Networks
        self.policy_net = DQNetwork().to(DEVICE)
        self.target_net = DQNetwork().to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is never trained directly
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Training statistics
        self.steps = 0
        self.episodes = 0
        self.losses = []
        self.rewards = []
        
    def state_to_tensor(self, state: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert game state dictionary to neural network input tensors.
        
        Args:
            state: Game state dictionary from TrainingEnvironment
            
        Returns:
            Tuple of (spatial_tensor, extra_features)
            - spatial_tensor: (1 × 7 × 18 × 20) tensor with spatial information
            - extra_features: (1 × 8) tensor with non-spatial features
        """
        board = state['board']
        height = state['board_height']
        width = state['board_width']
        my_pos = state['my_position']
        opp_pos = state['opponent_position']
        my_trail = set(state['my_trail'])
        opp_trail = set(state['opponent_trail'])
        my_direction = state['my_direction']
        
        # Create spatial channels
        channels = []
        
        # Channel 0: My trail
        my_trail_channel = np.zeros((height, width), dtype=np.float32)
        for x, y in my_trail:
            my_trail_channel[y][x] = 1.0
        channels.append(my_trail_channel)
        
        # Channel 1: Opponent trail
        opp_trail_channel = np.zeros((height, width), dtype=np.float32)
        for x, y in opp_trail:
            opp_trail_channel[y][x] = 1.0
        channels.append(opp_trail_channel)
        
        # Channel 2: Walls (any occupied cell)
        wall_channel = np.array(board, dtype=np.float32)
        channels.append(wall_channel)
        
        # Channel 3: My position
        my_pos_channel = np.zeros((height, width), dtype=np.float32)
        my_pos_channel[my_pos[1]][my_pos[0]] = 1.0
        channels.append(my_pos_channel)
        
        # Channel 4: Opponent position
        opp_pos_channel = np.zeros((height, width), dtype=np.float32)
        opp_pos_channel[opp_pos[1]][opp_pos[0]] = 1.0
        channels.append(opp_pos_channel)
        
        # Channel 5: Valid moves mask
        valid_moves_channel = np.zeros((height, width), dtype=np.float32)
        current_dir = my_direction.value if my_direction else (1, 0)
        for action in self.actions:
            dx, dy = self.direction_deltas[action]
            if (dx, dy) == (-current_dir[0], -current_dir[1]):
                continue  # Skip reversal
            new_x = (my_pos[0] + dx) % width
            new_y = (my_pos[1] + dy) % height
            if board[new_y][new_x] == 0:
                valid_moves_channel[new_y][new_x] = 1.0
        channels.append(valid_moves_channel)
        
        # Channel 6: Danger zones (opponent's possible next moves)
        danger_channel = np.zeros((height, width), dtype=np.float32)
        for action in self.actions:
            dx, dy = self.direction_deltas[action]
            new_x = (opp_pos[0] + dx) % width
            new_y = (opp_pos[1] + dy) % height
            danger_channel[new_y][new_x] = 0.5
        channels.append(danger_channel)
        
        # Stack channels and convert to tensor
        spatial_tensor = np.stack(channels, axis=0)  # (7 × 18 × 20)
        spatial_tensor = torch.FloatTensor(spatial_tensor).unsqueeze(0).to(DEVICE)  # (1 × 7 × 18 × 20)
        
        # Extra features
        my_boosts = state['my_boosts'] / 3.0  # Normalize (max 3 boosts)
        opp_boosts = state.get('agent2_boosts', 0) / 3.0 if state['player_number'] == 1 else state.get('agent1_boosts', 0) / 3.0
        turn_count = state['turn_count'] / 500.0  # Normalize (max ~500 turns)
        my_length = len(my_trail) / 360.0  # Normalize (max board size)
        opp_length = len(opp_trail) / 360.0
        
        # Distance to opponent (Manhattan distance with torus wrapping)
        dx = abs(my_pos[0] - opp_pos[0])
        dy = abs(my_pos[1] - opp_pos[1])
        dx = min(dx, width - dx)  # Torus wrapping
        dy = min(dy, height - dy)
        distance = (dx + dy) / (width + height)  # Normalize
        
        # Current direction
        dir_x = current_dir[0]
        dir_y = current_dir[1]
        
        extra_features = torch.FloatTensor([
            my_boosts, opp_boosts, turn_count, my_length, opp_length, distance, dir_x, dir_y
        ]).unsqueeze(0).to(DEVICE)  # (1 × 8)
        
        return spatial_tensor, extra_features
    
    def select_action(self, state: Dict, training: bool = True) -> str:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Game state dictionary
            training: If True, use epsilon-greedy; if False, use greedy
            
        Returns:
            Action string (e.g., "UP", "DOWN", "LEFT", "RIGHT")
        """
        # Get valid actions (not reversal, not into walls)
        valid_actions = self._get_valid_actions(state)
        
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions) if valid_actions else random.choice(self.actions)
        
        # Greedy action selection
        with torch.no_grad():
            spatial_tensor, extra_features = self.state_to_tensor(state)
            q_values = self.policy_net(spatial_tensor, extra_features)
            q_values = q_values.cpu().numpy()[0]
        
        # Mask invalid actions with very negative Q-values
        masked_q_values = q_values.copy()
        for idx, action in enumerate(self.actions):
            if action not in valid_actions:
                masked_q_values[idx] = -1e9
        
        # Select action with highest Q-value
        action_idx = np.argmax(masked_q_values)
        return self.actions[action_idx]
    
    def _get_valid_actions(self, state: Dict) -> List[str]:
        """Get list of valid actions (not reversal, not into walls)."""
        board = state['board']
        my_pos = state['my_position']
        my_direction = state['my_direction']
        height = state['board_height']
        width = state['board_width']
        
        current_dir = my_direction.value if my_direction else (1, 0)
        valid_actions = []
        
        for action in self.actions:
            dx, dy = self.direction_deltas[action]
            
            # Check reversal
            if (dx, dy) == (-current_dir[0], -current_dir[1]):
                continue
            
            # Check if cell is empty
            new_x = (my_pos[0] + dx) % width
            new_y = (my_pos[1] + dy) % height
            
            if board[new_y][new_x] == 0:
                valid_actions.append(action)
        
        return valid_actions
    
    def train_step(self):
        """
        Perform one training step (sample batch and update network).
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Prepare batch tensors
        states_spatial = []
        states_extra = []
        actions = []
        rewards = []
        next_states_spatial = []
        next_states_extra = []
        dones = []
        
        for exp in batch:
            # Current state
            spatial, extra = self.state_to_tensor(exp.state)
            states_spatial.append(spatial)
            states_extra.append(extra)
            
            # Action
            action_idx = self.action_to_idx[exp.action]
            actions.append(action_idx)
            
            # Reward
            rewards.append(exp.reward)
            
            # Next state
            next_spatial, next_extra = self.state_to_tensor(exp.next_state)
            next_states_spatial.append(next_spatial)
            next_states_extra.append(next_extra)
            
            # Done flag
            dones.append(exp.done)
        
        # Convert to tensors
        states_spatial = torch.cat(states_spatial, dim=0)
        states_extra = torch.cat(states_extra, dim=0)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states_spatial = torch.cat(next_states_spatial, dim=0)
        next_states_extra = torch.cat(next_states_extra, dim=0)
        dones = torch.FloatTensor(dones).to(DEVICE)
        
        # Current Q-values
        current_q_values = self.policy_net(states_spatial, states_extra)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states_spatial, next_states_extra)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)  # Gradient clipping
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save_model(self, filepath: str, metadata: dict = None):
        """
        Save model to file (CPU-compatible).
        
        Args:
            filepath: Path to save model
            metadata: Optional metadata dictionary
        """
        # Move model to CPU for saving
        self.policy_net.cpu()
        
        save_dict = {
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'metadata': metadata or {}
        }
        
        torch.save(save_dict, filepath)
        
        # Move model back to training device
        self.policy_net.to(DEVICE)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
        """
        checkpoint = torch.load(filepath, map_location=DEVICE)
        
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        
        # Update target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        print(f"Model loaded from {filepath}")
        print(f"  Episodes: {self.episodes}, Steps: {self.steps}, Epsilon: {self.epsilon:.4f}")


def visualize_board(state: Dict, last_action: str = None, result: str = None):
    """
    Print a visual representation of the game board.
    
    Args:
        state: Game state dictionary
        last_action: Last action taken (optional)
        result: Game result string (optional)
    """
    board = state['board']
    height = state['board_height']
    width = state['board_width']
    my_pos = state['my_position']
    opp_pos = state['opponent_position']
    
    print(f"\n{'='*60}")
    if result:
        print(f"GAME RESULT: {result}")
    if last_action:
        print(f"Last Action: {last_action}")
    print(f"Turn: {state['turn_count']} | My Boosts: {state['my_boosts']} | Player: {state['player_number']}")
    print(f"{'='*60}")
    
    # Create visual board
    visual = []
    for y in range(height):
        row = []
        for x in range(width):
            if (x, y) == my_pos:
                row.append('🔵')  # My position
            elif (x, y) == opp_pos:
                row.append('🔴')  # Opponent position
            elif board[y][x] == 1:
                row.append('██')  # Wall/trail
            else:
                row.append('  ')  # Empty
        visual.append(''.join(row))
    
    # Print with border
    print('┌' + '─' * (width * 2) + '┐')
    for row in visual:
        print('│' + row + '│')
    print('└' + '─' * (width * 2) + '┘')
    print()


def train_dqn_agent(num_episodes: int = 10000,
                    save_freq: int = 50,  # Save every 50 episodes (was 500)
                    eval_freq: int = 200,  # Evaluate every 200 episodes (was 100)
                    save_dir: Optional[str] = None,
                    resume_from: Optional[str] = None):
    """
    Train DQN agent through self-play.
    
    Args:
        num_episodes: Number of training episodes
        save_freq: Episodes between model saves (default 50)
        eval_freq: Episodes between evaluations (default 200)
        save_dir: Directory to save models (timestamped if None)
        resume_from: Optional path to resume training from
    """
    # Initialize agent (will create timestamped directory)
    agent = DQNAgent(save_dir=save_dir)
    
    # Save initial model immediately
    initial_path = os.path.join(agent.save_dir, "dqn_agent_ep0_initial.pt")
    agent.save_model(initial_path, metadata={'episode': 0, 'note': 'Initial model'})
    
    # Resume training if specified
    if resume_from and os.path.exists(resume_from):
        agent.load_model(resume_from)
        print(f"Resuming training from episode {agent.episodes}")
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    win_history = deque(maxlen=100)  # Last 100 games
    
    # Training loop
    print(f"\n{'='*70}")
    print(f"Starting DQN Training on {DEVICE}")
    print(f"Save Frequency: Every {save_freq} episodes")
    print(f"Eval Frequency: Every {eval_freq} episodes")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    for episode in range(agent.episodes, agent.episodes + num_episodes):
        env = TrainingEnvironment()
        state1, state2 = env.reset()
        
        # Self-play: agent plays against slightly older version of itself
        # Agent 1 is the learning agent, Agent 2 uses target network (more stable opponent)
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Agent 1 (learning) selects action
            action1 = agent.select_action(state1, training=True)
            
            # Agent 2 (target network opponent) selects action
            # Use epsilon-greedy with lower epsilon for more stable opponent
            old_epsilon = agent.epsilon
            agent.epsilon = 0.1  # Fixed low exploration for opponent
            action2 = agent.select_action(state2, training=True)
            agent.epsilon = old_epsilon
            
            # Execute step
            next_state1, next_state2, reward1, reward2, done, result = env.step(action1, action2)
            
            # Store experience for Agent 1
            agent.replay_buffer.push(state1, action1, reward1, next_state1, done)
            
            # Also store experience for Agent 2 (learning from both perspectives)
            agent.replay_buffer.push(state2, action2, reward2, next_state2, done)
            
            # Train (only if buffer is large enough)
            if len(agent.replay_buffer) >= agent.batch_size:
                loss = agent.train_step()
                if loss is not None:
                    agent.losses.append(loss)
            
            # Update state
            state1 = next_state1
            state2 = next_state2
            episode_reward += reward1
            episode_length += 1
        
        # Record statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if result == GameResult.AGENT1_WIN:
            win_history.append(1)
        elif result == GameResult.AGENT2_WIN:
            win_history.append(0)
        else:
            win_history.append(0.5)  # Draw
        
        agent.episodes += 1
        
        # Visualize board every 50 episodes (was 25 - reduce overhead)
        if (episode + 1) % 50 == 0 and (episode + 1) <= 200:  # Only first 200 episodes
            result_str = str(result).replace('GameResult.', '')
            visualize_board(state1, action1, result_str)
        
        # Print progress every 10 episodes with timing info
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            win_rate = np.mean(win_history) if win_history else 0.5
            avg_loss = np.mean(agent.losses[-100:]) if agent.losses else 0
            
            # Calculate episodes per minute
            elapsed = time.time() - start_time
            eps_per_min = (episode + 1 - agent.episodes + num_episodes) / elapsed * 60 if elapsed > 0 else 0
            
            print(f"Ep {episode + 1:4d} | "
                  f"Reward: {avg_reward:6.1f} | "
                  f"Length: {avg_length:4.1f} | "
                  f"WinRate: {win_rate:.2%} | "
                  f"Loss: {avg_loss:6.4f} | "
                  f"ε: {agent.epsilon:.4f} | "
                  f"Buf: {len(agent.replay_buffer):5d} | "
                  f"{eps_per_min:.0f} ep/min")
        
        # Evaluation against baseline agents
        if (episode + 1) % eval_freq == 0:
            print(f"\n{'='*70}")
            print(f"Evaluation at Episode {episode + 1}")
            print(f"{'='*70}")
            
            # Create evaluation policy (no exploration)
            def dqn_policy(state):
                return agent.select_action(state, training=False)
            
            # Evaluate against wall avoider
            from training_env import wall_avoider_agent, greedy_space_agent
            
            print("\nDQN vs Wall Avoider (50 games)...")
            stats1 = evaluate_agents(dqn_policy, wall_avoider_agent, num_episodes=50, verbose=False)
            print(f"  DQN Win Rate: {stats1['agent1_win_rate']:.1%}")
            
            print("\nDQN vs Greedy Space Agent (50 games)...")
            stats2 = evaluate_agents(dqn_policy, greedy_space_agent, num_episodes=50, verbose=False)
            print(f"  DQN Win Rate: {stats2['agent1_win_rate']:.1%}")
            
            print(f"{'='*70}\n")
        
        # Save model
        if (episode + 1) % save_freq == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(agent.save_dir, f"dqn_agent_ep{episode + 1}_{timestamp}.pt")
            
            metadata = {
                'episode': episode + 1,
                'avg_reward_last_100': np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards),
                'avg_length_last_100': np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths),
                'win_rate_last_100': np.mean(win_history),
                'epsilon': agent.epsilon,
                'timestamp': timestamp
            }
            
            agent.save_model(save_path, metadata)
            print(f"✓ Checkpoint saved: {save_path}")
    
    # Final save
    final_path = os.path.join(agent.save_dir, "dqn_agent_final.pt")
    final_metadata = {
        'total_episodes': agent.episodes,
        'final_epsilon': agent.epsilon,
        'training_complete': True,
        'final_win_rate': np.mean(win_history) if win_history else 0.5,
        'total_training_time': time.time() - start_time
    }
    agent.save_model(final_path, metadata=final_metadata)
    
    # Training summary
    total_time = time.time() - start_time
    eps_per_min = num_episodes / (total_time / 60)
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Total Episodes:     {agent.episodes}")
    print(f"Total Time:         {total_time/60:.2f} minutes")
    print(f"Episodes/Minute:    {eps_per_min:.1f}")
    print(f"Final Win Rate:     {np.mean(win_history):.2%}" if win_history else "N/A")
    print(f"Final Epsilon:      {agent.epsilon:.4f}")
    print(f"Replay Buffer Size: {len(agent.replay_buffer)}")
    print(f"Final Model:        {final_path}")
    print(f"Model Directory:    {agent.save_dir}")
    print(f"{'='*70}\n")


def create_inference_agent(model_path: str):
    """
    Create a policy function that can be used in agent.py.
    
    This loads the trained model and returns a function that takes
    game state and returns an action string.
    
    Args:
        model_path: Path to trained model
        
    Returns:
        Policy function for inference
    """
    # Load model
    agent = DQNAgent()
    agent.load_model(model_path)
    agent.policy_net.eval()
    
    # Create policy function
    def policy(state: Dict) -> str:
        return agent.select_action(state, training=False)
    
    return policy


if __name__ == "__main__":
    import sys
    
    print(f"\n{'='*70}")
    print(f"Case Closed - DQN Training System")
    print(f"{'='*70}\n")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            # Training mode
            num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
            resume_from = sys.argv[3] if len(sys.argv) > 3 else None
            
            train_dqn_agent(num_episodes=num_episodes, resume_from=resume_from)
        
        elif sys.argv[1] == "eval":
            # Evaluation mode
            if len(sys.argv) < 3:
                print("Usage: python agent2.py eval <model_path>")
                sys.exit(1)
            
            model_path = sys.argv[2]
            dqn_policy = create_inference_agent(model_path)
            
            from training_env import wall_avoider_agent, greedy_space_agent
            
            print(f"Evaluating model: {model_path}\n")
            
            print("DQN vs Wall Avoider (100 games)...")
            stats1 = evaluate_agents(dqn_policy, wall_avoider_agent, num_episodes=100, verbose=True)
            
            print("\nDQN vs Greedy Space Agent (100 games)...")
            stats2 = evaluate_agents(dqn_policy, greedy_space_agent, num_episodes=100, verbose=True)
            
            print("\nDQN vs DQN (100 games)...")
            stats3 = evaluate_agents(dqn_policy, dqn_policy, num_episodes=100, verbose=True)
    
    else:
        # Demo mode - quick test
        print("Demo Mode: Training for 100 episodes\n")
        print("For full training, run: python agent2.py train 10000")
        print("For evaluation, run: python agent2.py eval models/dqn_agent_final.pt\n")
        
        train_dqn_agent(num_episodes=100, save_freq=50, eval_freq=50)
