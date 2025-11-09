"""
Fast DQN Training System with BFS-based State Encoding

Key improvements:
1. State = reachable space in each direction (4 values) + extra features
2. Much simpler network architecture (faster training)
3. NO INVALID MOVES - pre-filters valid actions
4. Optimized for speed (target: 60+ episodes/min)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional
import os
import json
from datetime import datetime
import time

from training_env import TrainingEnvironment, run_episode, evaluate_agents
from case_closed_game import GameResult

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bfs_reachable_space(start_pos: Tuple[int, int], board: List[List[int]], 
                        width: int, height: int, max_depth: int = 30) -> int:
    """
    BFS to count reachable empty cells from start position.
    Limited to max_depth for speed. Reduced from 50 to 30 for faster training.
    """
    visited = set()
    queue = deque([(start_pos, 0)])
    visited.add(start_pos)
    count = 0
    
    while queue:
        (x, y), depth = queue.popleft()
        if depth > max_depth:
            continue
        count += 1
        
        # Early exit if we've found enough space
        if count > 100:  # If we have 100+ reachable cells, that's enough info
            return count
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x = (x + dx) % width
            new_y = (y + dy) % height
            
            if (new_x, new_y) not in visited and board[new_y][new_x] == 0:
                visited.add((new_x, new_y))
                queue.append(((new_x, new_y), depth + 1))
    
    return count


class CompactDQNetwork(nn.Module):
    """
    Strategic DQN using BFS-based state encoding with heavy architecture.
    
    Focus: Learn aggressive space control and risk-taking behavior.
    
    Input: 12 strategic features (BFS-computed space control)
    Output: 4 Q-values (UP, DOWN, LEFT, RIGHT)
    """
    
    def __init__(self, input_size: int = 12):
        super(CompactDQNetwork, self).__init__()
        
        # HEAVY architecture for complex strategic learning
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, 4)
        
        # Layer normalization for stability
        self.ln1 = nn.LayerNorm(1024)
        self.ln2 = nn.LayerNorm(1024)
        self.ln3 = nn.LayerNorm(512)
        self.ln4 = nn.LayerNorm(512)
        self.ln5 = nn.LayerNorm(256)
        
        self.dropout = nn.Dropout(0.2)  # Lower dropout for more aggressive learning
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Very deep network for strategic planning
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.dropout(x)
        
        x = F.relu(self.ln4(self.fc4(x)))
        x = self.dropout(x)
        
        x = F.relu(self.ln5(self.fc5(x)))
        x = self.dropout(x)
        
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        
        return x


def print_board_state(state: Dict, action: str = None):
    """Print the board state in a clear format for debugging."""
    board = state['board']
    my_pos = state['my_position']
    opp_pos = state['opponent_position']
    height = state['board_height']
    width = state['board_width']
    
    print(f"\n{'='*60}")
    print(f"Turn: {state.get('turn_count', 0)} | Player: {state.get('player_number', '?')}")
    print(f"My Pos: {my_pos} | Opp Pos: {opp_pos}")
    if action:
        print(f"Selected Action: {action}")
    print(f"{'='*60}")
    
    # Print board
    for y in range(height):
        row = ""
        for x in range(width):
            if (x, y) == my_pos:
                row += "M "  # Me
            elif (x, y) == opp_pos:
                row += "O "  # Opponent
            elif board[y][x] == 1:
                row += "# "  # Wall
            else:
                row += ". "  # Empty
        print(row)
    print(f"{'='*60}\n")


def validate_move(state: Dict, action: str) -> Tuple[bool, str]:
    """
    Validate if a move is legal.
    Returns (is_valid, reason)
    """
    board = state['board']
    my_pos = state['my_position']
    my_direction = state.get('my_direction', (1, 0))
    
    # Handle Direction enum
    if hasattr(my_direction, 'value'):
        my_direction = my_direction.value
    
    height = state['board_height']
    width = state['board_width']
    
    direction_deltas = {
        'UP': (0, -1),
        'DOWN': (0, 1),
        'LEFT': (-1, 0),
        'RIGHT': (1, 0),
    }
    
    if action not in direction_deltas:
        return False, f"Invalid action '{action}' not in {list(direction_deltas.keys())}"
    
    dx, dy = direction_deltas[action]
    
    # Check for reversal
    if (dx, dy) == (-my_direction[0], -my_direction[1]):
        return False, f"Reversal: {action} reverses direction {my_direction}"
    
    # Calculate new position
    new_x = (my_pos[0] + dx) % width
    new_y = (my_pos[1] + dy) % height
    
    # Check for collision (board[y][x] != 0 means occupied)
    if board[new_y][new_x] != 0:
        return False, f"Collision: position ({new_x}, {new_y}) is occupied (value={board[new_y][new_x]})"
    
    return True, "Valid move"


class FastDQNAgent:
    """Fast DQN agent with BFS-based state encoding."""
    
    def __init__(self, 
                 learning_rate: float = 0.0003,  # Lower LR for stability
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,  # Higher min exploration
                 epsilon_decay: float = 0.997,  # Slower decay
                 buffer_capacity: int = 50000,
                 batch_size: int = 256,
                 target_update_freq: int = 1000,
                 save_dir: Optional[str] = None):
        
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Create timestamped model directory
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"models_fast_{timestamp}"
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Models will be saved to: {self.save_dir}")
        print(f"{'='*60}\n")
        
        # Action mapping
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.action_to_idx = {action: idx for idx, action in enumerate(self.actions)}
        self.direction_deltas = {
            'UP': (0, -1),
            'DOWN': (0, 1),
            'LEFT': (-1, 0),
            'RIGHT': (1, 0),
        }
        
        # Networks
        self.policy_net = CompactDQNetwork().to(DEVICE)
        self.target_net = CompactDQNetwork().to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_capacity)
        
        # Training statistics
        self.steps = 0
        self.episodes = 0
        self.losses = []
    
    def get_valid_actions(self, state: Dict) -> List[str]:
        """Get list of valid actions (no reversals, no collisions)."""
        board = state['board']
        my_pos = state['my_position']
        my_direction = state.get('my_direction', (1, 0))
        
        # Handle Direction enum
        if hasattr(my_direction, 'value'):
            my_direction = my_direction.value
        
        height = state['board_height']
        width = state['board_width']
        
        valid = []
        for action in self.actions:
            dx, dy = self.direction_deltas[action]
            
            # Skip reversals
            if (dx, dy) == (-my_direction[0], -my_direction[1]):
                continue
            
            # Check if move is valid
            new_x = (my_pos[0] + dx) % width
            new_y = (my_pos[1] + dy) % height
            
            if board[new_y][new_x] == 0:
                valid.append(action)
        
        # If no valid moves, return empty list (agent will crash)
        # DO NOT return a fallback move - that might be invalid!
        return valid
    
    def state_to_tensor(self, state: Dict) -> torch.Tensor:
        """
        Convert game state to strategic feature tensor.
        
        Returns 12 strategic features focusing on space control and positioning:
        1-4: BFS reachable space in each direction (normalized)
        5: My space ratio (my_reachable / total_empty)
        6: Distance to opponent (normalized)
        7-8: Boosts (mine, opponent's)
        9: Turn ratio
        10-11: My current direction (dx, dy)
        12: Territory advantage (my_space - opp_space)
        """
        board = state['board']
        my_pos = state['my_position']
        opp_pos = state['opponent_position']
        height = state['board_height']
        width = state['board_width']
        turn_count = state.get('turn_count', 0)
        my_boosts = state.get('my_boosts', 0)
        my_direction = state.get('my_direction', (1, 0))
        
        # Handle Direction enum
        if hasattr(my_direction, 'value'):
            my_direction = my_direction.value
        
        # Get opponent boosts
        player_num = state.get('player_number', 1)
        if player_num == 1:
            opp_boosts = state.get('agent2_boosts', 0)
        else:
            opp_boosts = state.get('agent1_boosts', 0)
        
        # Feature 1-4: BFS reachable space in each direction
        spaces = []
        for action in self.actions:
            dx, dy = self.direction_deltas[action]
            new_x = (my_pos[0] + dx) % width
            new_y = (my_pos[1] + dy) % height
            
            if board[new_y][new_x] == 0:
                # Make temporary board copy
                board_copy = [row[:] for row in board]
                board_copy[new_y][new_x] = 1
                space = bfs_reachable_space((new_x, new_y), board_copy, width, height, max_depth=50)
                spaces.append(space / 360.0)  # Normalize by max board size
            else:
                spaces.append(0.0)  # Blocked direction
        
        # Feature 5: My total reachable space ratio
        board_copy = [row[:] for row in board]
        my_total_space = bfs_reachable_space(my_pos, board_copy, width, height, max_depth=50)
        total_empty = sum(row.count(0) for row in board)
        my_space_ratio = my_total_space / max(total_empty, 1)
        
        # Feature 6: Distance to opponent (Manhattan with wrapping)
        dx = abs(my_pos[0] - opp_pos[0])
        dy = abs(my_pos[1] - opp_pos[1])
        dx = min(dx, width - dx)
        dy = min(dy, height - dy)
        distance = (dx + dy) / (width + height)
        
        # Feature 7-8: Boosts
        my_boost_norm = my_boosts / 3.0
        opp_boost_norm = opp_boosts / 3.0
        
        # Feature 9: Turn ratio
        turn_ratio = turn_count / 500.0
        
        # Feature 10-11: Current direction
        dir_x = my_direction[0]
        dir_y = my_direction[1]
        
        # Feature 12: Territory advantage (estimate opponent's space too)
        board_copy_opp = [row[:] for row in board]
        opp_total_space = bfs_reachable_space(opp_pos, board_copy_opp, width, height, max_depth=50)
        territory_advantage = (my_total_space - opp_total_space) / 360.0
        
        # Combine all features
        features = spaces + [
            my_space_ratio,
            distance,
            my_boost_norm,
            opp_boost_norm,
            turn_ratio,
            dir_x,
            dir_y,
            territory_advantage
        ]
        
        return torch.FloatTensor(features).unsqueeze(0).to(DEVICE)
    
    def select_action(self, state: Dict, training: bool = True, debug: bool = False) -> str:
        """
        Select action using epsilon-greedy with VALID actions only.
        CRITICAL: NEVER returns a reversal move (opposite of current direction).
        """
        # Get current direction for reversal check
        my_direction = state.get('my_direction', (1, 0))
        if hasattr(my_direction, 'value'):
            my_direction = my_direction.value
        
        # Get valid actions (already filters reversals and collisions)
        valid_actions = self.get_valid_actions(state)
        
        if debug:
            print(f"\nCurrent direction: {my_direction}")
            print(f"Valid actions (no reversals, no walls): {valid_actions}")
        
        # If no valid moves, agent is trapped - pick any non-reversal move
        # Game will crash agent, but at least we didn't try to reverse!
        if not valid_actions:
            # Find ANY non-reversal move
            for action in self.actions:
                dx, dy = self.direction_deltas[action]
                if (dx, dy) != (-my_direction[0], -my_direction[1]):
                    if debug:
                        print(f"⚠ Trapped! Picking non-reversal: {action}")
                    return action
            # Absolute last resort (shouldn't happen)
            return self.actions[0]
        
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            action = random.choice(valid_actions)
            if debug:
                print(f"🎲 Random action: {action}")
            return action
        
        # Greedy selection - get Q-values and rank valid actions
        with torch.no_grad():
            state_tensor = self.state_to_tensor(state)
            q_values = self.policy_net(state_tensor)
            q_values_np = q_values.cpu().numpy()[0]
            
            if debug:
                print(f"Q-values: {dict(zip(self.actions, q_values_np))}")
            
            # Rank valid actions by Q-value (highest first)
            action_q_pairs = [(action, q_values_np[self.action_to_idx[action]]) 
                             for action in valid_actions]
            action_q_pairs.sort(key=lambda x: x[1], reverse=True)
            
            if debug:
                print(f"Ranked valid actions: {action_q_pairs}")
            
            # Pick highest Q-value action from valid actions
            best_action = action_q_pairs[0][0]
            
            # CRITICAL SAFETY CHECK: Ensure action is not a reversal
            dx, dy = self.direction_deltas[best_action]
            if (dx, dy) == (-my_direction[0], -my_direction[1]):
                # This should NEVER happen (valid_actions already filters)
                # But if it does, pick second choice
                print(f"🚨 CRITICAL: Best action {best_action} is reversal! Picking 2nd choice")
                if len(action_q_pairs) > 1:
                    best_action = action_q_pairs[1][0]
                else:
                    # Find first non-reversal
                    for action in self.actions:
                        dx2, dy2 = self.direction_deltas[action]
                        if (dx2, dy2) != (-my_direction[0], -my_direction[1]):
                            best_action = action
                            break
            
            if debug:
                print(f"✓ Final action: {best_action}")
            
            return best_action
    
    def train_step(self):
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        # Prepare tensors
        spatial_states = []
        bfs_states = []
        actions = []
        rewards = []
        spatial_next_states = []
        bfs_next_states = []
        dones = []
        
        for exp in batch:
            state_tensor = self.state_to_tensor(exp.state)
            spatial_states.append(state_tensor)
            actions.append(self.action_to_idx[exp.action])
            rewards.append(exp.reward)
            next_state_tensor = self.state_to_tensor(exp.next_state)
            spatial_next_states.append(next_state_tensor)
            dones.append(exp.done)
        
        spatial_states = torch.cat(spatial_states, dim=0)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        spatial_next_states = torch.cat(spatial_next_states, dim=0)
        dones = torch.FloatTensor(dones).to(DEVICE)
        
        # Current Q-values
        current_q = self.policy_net(spatial_states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q = self.target_net(spatial_next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save_model(self, filepath: str, metadata: dict = None):
        """Save model to file."""
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
        self.policy_net.to(DEVICE)
        print(f"✓ Model saved: {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=DEVICE)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        print(f"✓ Model loaded: {filepath}")


def calculate_space_reward(state: Dict, action: str) -> float:
    """
    Calculate reward bonus for moving into open spaces.
    Uses BFS to measure reachable space after taking action.
    
    Returns reward in range [0, 1] based on how much space the move opens up.
    """
    board = state['board']
    my_pos = state['my_position']
    height = state['board_height']
    width = state['board_width']
    
    # Map action to delta
    direction_deltas = {
        'UP': (0, -1),
        'DOWN': (0, 1),
        'LEFT': (-1, 0),
        'RIGHT': (1, 0),
    }
    
    dx, dy = direction_deltas[action]
    new_x = (my_pos[0] + dx) % width
    new_y = (my_pos[1] + dy) % height
    
    # If moving into wall, no space reward
    if board[new_y][new_x] != 0:
        return 0.0
    
    # Calculate reachable space from new position
    board_copy = [row[:] for row in board]
    board_copy[new_y][new_x] = 1
    reachable = bfs_reachable_space((new_x, new_y), board_copy, width, height, max_depth=50)
    
    # Normalize to [0, 1] range
    max_possible = width * height
    space_ratio = reachable / max_possible
    
    return space_ratio


def train_fast_dqn(num_episodes: int = 1000,
                   save_freq: int = 100,
                   eval_freq: int = 200):
    """Train fast DQN agent against diverse heuristic opponents."""
    
    agent = FastDQNAgent()
    
    # Import all heuristic agents
    from heuristic_agents import (
        greedy_space_agent, 
        aggressive_chaser_agent, 
        smart_avoider_agent,
        territorial_defender_agent,
        adaptive_hybrid_agent
    )
    
    heuristic_agents = [
        ('GreedySpace', greedy_space_agent),
        ('AggressiveChaser', aggressive_chaser_agent),
        ('SmartAvoider', smart_avoider_agent),
        ('TerritorialDefender', territorial_defender_agent),
        ('AdaptiveHybrid', adaptive_hybrid_agent),
    ]
    
    # Save initial model
    initial_path = os.path.join(agent.save_dir, "dqn_fast_ep0_initial.pt")
    agent.save_model(initial_path, metadata={'episode': 0})
    
    # Training stats
    episode_rewards = []
    episode_lengths = []
    win_history = deque(maxlen=100)
    opponent_wins = {name: [] for name, _ in heuristic_agents}
    player_position_wins = {'player1': [], 'player2': []}
    
    print(f"\n{'='*70}")
    print(f"Starting Fast DQN Training on {DEVICE}")
    print(f"Training against 5 diverse heuristic opponents")
    print(f"50% as Player 1, 50% as Player 2")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        env = TrainingEnvironment()
        
        # Rotate through opponents
        opp_name, opp_policy = heuristic_agents[episode % len(heuristic_agents)]
        
        # 50% as player 1, 50% as player 2
        as_player_1 = (episode % 2 == 0)
        
        state1, state2 = env.reset()
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        move_count = 0
        while not done:
            move_count += 1
            
            if as_player_1:
                # DQN is player 1
                action1 = agent.select_action(state1, training=True, debug=False)
                action2 = opp_policy(state2)
                dqn_reward_key = 'reward1'
            else:
                # DQN is player 2
                action1 = opp_policy(state1)
                action2 = agent.select_action(state2, training=True, debug=False)
                dqn_reward_key = 'reward2'
            
            # Step
            next_state1, next_state2, reward1, reward2, done, result = env.step(action1, action2)
            
            # Enhanced reward shaping: Add space control bonus
            if as_player_1:
                space_bonus = calculate_space_reward(state1, action1) * 0.5  # Scale to [0, 0.5]
                shaped_reward1 = reward1 + space_bonus
                agent.replay_buffer.append(Experience(state1, action1, shaped_reward1, next_state1, done))
                episode_reward += shaped_reward1
            else:
                space_bonus = calculate_space_reward(state2, action2) * 0.5
                shaped_reward2 = reward2 + space_bonus
                agent.replay_buffer.append(Experience(state2, action2, shaped_reward2, next_state2, done))
                episode_reward += shaped_reward2
            
            # Train
            if len(agent.replay_buffer) >= agent.batch_size:
                loss = agent.train_step()
                if loss is not None:
                    agent.losses.append(loss)
            
            state1 = next_state1
            state2 = next_state2
            episode_length += 1
        
        # Record stats
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Determine if DQN won
        if as_player_1:
            dqn_won = (result == GameResult.AGENT1_WIN)
            player_pos = 'player1'
        else:
            dqn_won = (result == GameResult.AGENT2_WIN)
            player_pos = 'player2'
        
        win_history.append(1 if dqn_won else 0)
        opponent_wins[opp_name].append(1 if dqn_won else 0)
        player_position_wins[player_pos].append(1 if dqn_won else 0)
        
        agent.episodes += 1
        
        # Print progress
        if (episode + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eps_per_min = (episode + 1) / elapsed * 60
            
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            win_rate = np.mean(win_history) if win_history else 0.5
            avg_loss = np.mean(agent.losses[-100:]) if agent.losses else 0
            
            # Calculate position-specific win rates
            p1_wr = np.mean(player_position_wins['player1'][-50:]) if player_position_wins['player1'] else 0.5
            p2_wr = np.mean(player_position_wins['player2'][-50:]) if player_position_wins['player2'] else 0.5
            
            print(f"Ep {episode + 1:4d} | "
                  f"Opp: {opp_name:18s} | "
                  f"Pos: {'P1' if as_player_1 else 'P2'} | "
                  f"Win: {win_rate:.2%} | "
                  f"P1_WR: {p1_wr:.2%} | "
                  f"P2_WR: {p2_wr:.2%} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"{eps_per_min:.0f} ep/min")
        
        # Detailed evaluation
        if (episode + 1) % eval_freq == 0:
            print(f"\n{'='*70}")
            print(f"Evaluation at Episode {episode + 1}")
            print(f"{'='*70}")
            
            def dqn_policy(state):
                return agent.select_action(state, training=False)
            
            # Test against each opponent
            for opp_name, opp_policy in heuristic_agents:
                stats = evaluate_agents(dqn_policy, opp_policy, num_episodes=40, verbose=False)
                print(f"vs {opp_name:20s}: {stats['agent1_win_rate']:5.1%} win rate")
            
            # Show per-opponent training stats
            print(f"\nTraining Win Rates (last 100 games per opponent):")
            for opp_name in opponent_wins:
                if opponent_wins[opp_name]:
                    wr = np.mean(opponent_wins[opp_name][-100:])
                    print(f"  vs {opp_name:20s}: {wr:.1%}")
            
            print(f"\nPosition-based Win Rates:")
            print(f"  As Player 1: {np.mean(player_position_wins['player1'][-100:]):.1%}")
            print(f"  As Player 2: {np.mean(player_position_wins['player2'][-100:]):.1%}")
            print(f"{'='*70}\n")
        
        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(agent.save_dir, f"dqn_fast_ep{episode + 1}_{timestamp}.pt")
            
            agent.save_model(save_path, metadata={
                'episode': episode + 1,
                'avg_reward': np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards),
                'win_rate': np.mean(win_history),
                'epsilon': agent.epsilon,
                'p1_win_rate': np.mean(player_position_wins['player1'][-100:]) if player_position_wins['player1'] else 0.5,
                'p2_win_rate': np.mean(player_position_wins['player2'][-100:]) if player_position_wins['player2'] else 0.5,
            })
    
    # Final save
    total_time = time.time() - start_time
    eps_per_min = num_episodes / (total_time / 60)
    
    final_path = os.path.join(agent.save_dir, "dqn_fast_final.pt")
    agent.save_model(final_path, metadata={
        'total_episodes': agent.episodes,
        'final_win_rate': np.mean(win_history),
        'training_time': total_time,
        'eps_per_min': eps_per_min,
        'p1_win_rate': np.mean(player_position_wins['player1']),
        'p2_win_rate': np.mean(player_position_wins['player2']),
    })
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Total Episodes: {agent.episodes}")
    print(f"Total Time: {total_time/60:.2f} minutes")
    print(f"Speed: {eps_per_min:.1f} episodes/min")
    print(f"Final Win Rate: {np.mean(win_history):.2%}")
    print(f"As Player 1: {np.mean(player_position_wins['player1']):.2%}")
    print(f"As Player 2: {np.mean(player_position_wins['player2']):.2%}")
    print(f"\nPer-Opponent Win Rates:")
    for opp_name in opponent_wins:
        if opponent_wins[opp_name]:
            print(f"  vs {opp_name:20s}: {np.mean(opponent_wins[opp_name]):.1%}")
    print(f"Final Model: {final_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        num_eps = int(sys.argv[1])
    else:
        num_eps = 500
    
    print(f"\n{'='*70}")
    print(f"Fast DQN Training - BFS State Encoding")
    print(f"Training for {num_eps} episodes")
    print(f"{'='*70}\n")
    
    train_fast_dqn(num_episodes=num_eps)
