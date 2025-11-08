"""
DQN Inference Module for agent.py Integration

This module provides utility functions to load and use trained DQN models
in the competition agent.py file. It's designed for CPU inference.

Usage in agent.py:
    from dqn_inference import DQNInference
    
    # Initialize once
    dqn = DQNInference("models/dqn_agent_final.pt")
    
    # Use in get_move function
    move = dqn.get_move(game_state)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import os


class DQNetwork(nn.Module):
    """DQN architecture - must match agent2.py exactly."""
    
    def __init__(self, input_channels: int = 7, height: int = 18, width: int = 20):
        super(DQNetwork, self).__init__()
        
        self.input_channels = input_channels
        self.height = height
        self.width = width
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        conv_output_size = height * width * 64
        self.fc1 = nn.Linear(conv_output_size + 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, state_tensor: torch.Tensor, extra_features: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(state_tensor))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, extra_features], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class DQNInference:
    """
    DQN inference wrapper for easy integration into agent.py.
    
    This class handles:
    - Loading trained model for CPU inference
    - Converting game state to neural network input
    - Selecting best valid action
    - Optional boost usage
    """
    
    def __init__(self, model_path: str):
        """
        Initialize DQN inference.
        
        Args:
            model_path: Path to trained model checkpoint
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model on CPU
        self.device = torch.device("cpu")
        self.model = DQNetwork().to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Action mapping
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.direction_deltas = {
            'UP': (0, -1),
            'DOWN': (0, 1),
            'LEFT': (-1, 0),
            'RIGHT': (1, 0),
        }
        
        print(f"✓ DQN model loaded from {model_path}")
        print(f"✓ Running on {self.device}")
        if 'metadata' in checkpoint and checkpoint['metadata']:
            metadata = checkpoint['metadata']
            if 'episode' in metadata:
                print(f"✓ Trained for {metadata['episode']} episodes")
    
    def get_move(self, state: Dict, use_boost_threshold: float = 0.2) -> str:
        """
        Get best move from trained DQN.
        
        Args:
            state: Game state dictionary (from judge engine or training env)
            use_boost_threshold: Q-value advantage threshold for using boost
            
        Returns:
            Move string (e.g., "UP", "DOWN:BOOST")
        """
        # Get valid actions
        valid_actions = self._get_valid_actions(state)
        
        if not valid_actions:
            # No valid moves - shouldn't happen but fallback
            return 'UP'
        
        # Convert state to tensors
        spatial_tensor, extra_features = self._state_to_tensor(state)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.model(spatial_tensor, extra_features)
            q_values = q_values.cpu().numpy()[0]
        
        # Mask invalid actions
        masked_q = q_values.copy()
        for idx, action in enumerate(self.actions):
            if action not in valid_actions:
                masked_q[idx] = -1e9
        
        # Select best action
        best_action_idx = np.argmax(masked_q)
        best_action = self.actions[best_action_idx]
        best_q_value = q_values[best_action_idx]
        
        # Decide if we should use boost
        # Use boost if:
        # 1. We have boosts available
        # 2. The Q-value advantage is significant (suggests good move)
        use_boost = False
        if state.get('my_boosts', 0) > 0 or state.get('agent1_boosts', 0) > 0:
            avg_q_value = np.mean([q_values[i] for i, a in enumerate(self.actions) if a in valid_actions])
            q_advantage = best_q_value - avg_q_value
            
            if q_advantage > use_boost_threshold:
                use_boost = True
        
        # Format move string
        move = best_action
        if use_boost:
            move += ":BOOST"
        
        return move
    
    def _get_valid_actions(self, state: Dict) -> List[str]:
        """Get list of valid actions (not reversal, not into walls)."""
        board = state.get('board', [])
        height = len(board)
        width = len(board[0]) if board else 20
        
        # Get my position
        my_pos = state.get('my_position')
        if not my_pos:
            # Fallback: get from trail
            my_trail = state.get('my_trail', [])
            if my_trail:
                my_pos = my_trail[-1]
            else:
                # Last resort: use agent1 trail if we're player 1
                agent1_trail = state.get('agent1_trail', [])
                if agent1_trail:
                    my_pos = agent1_trail[-1]
                else:
                    return self.actions  # Can't determine position, return all
        
        # Get current direction
        my_direction = state.get('my_direction')
        if my_direction:
            if hasattr(my_direction, 'value'):
                current_dir = my_direction.value
            else:
                current_dir = my_direction
        else:
            current_dir = (1, 0)  # Default to RIGHT
        
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
        
        # If no safe moves, return non-reversal moves
        if not valid_actions:
            for action in self.actions:
                dx, dy = self.direction_deltas[action]
                if (dx, dy) != (-current_dir[0], -current_dir[1]):
                    valid_actions.append(action)
        
        return valid_actions
    
    def _state_to_tensor(self, state: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert game state to neural network input tensors."""
        board = state.get('board', [])
        height = len(board)
        width = len(board[0]) if board else 20
        
        # Get positions and trails
        my_pos = state.get('my_position')
        opp_pos = state.get('opponent_position')
        
        if not my_pos:
            my_trail = state.get('my_trail', [])
            my_pos = my_trail[-1] if my_trail else (0, 0)
        
        if not opp_pos:
            opp_trail = state.get('opponent_trail', [])
            opp_pos = opp_trail[-1] if opp_trail else (0, 0)
        
        my_trail = set(state.get('my_trail', []))
        opp_trail = set(state.get('opponent_trail', []))
        
        # Get direction
        my_direction = state.get('my_direction')
        if my_direction:
            if hasattr(my_direction, 'value'):
                current_dir = my_direction.value
            else:
                current_dir = my_direction
        else:
            current_dir = (1, 0)
        
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
        
        # Channel 2: Walls
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
        for action in self.actions:
            dx, dy = self.direction_deltas[action]
            if (dx, dy) == (-current_dir[0], -current_dir[1]):
                continue
            new_x = (my_pos[0] + dx) % width
            new_y = (my_pos[1] + dy) % height
            if board[new_y][new_x] == 0:
                valid_moves_channel[new_y][new_x] = 1.0
        channels.append(valid_moves_channel)
        
        # Channel 6: Danger zones
        danger_channel = np.zeros((height, width), dtype=np.float32)
        for action in self.actions:
            dx, dy = self.direction_deltas[action]
            new_x = (opp_pos[0] + dx) % width
            new_y = (opp_pos[1] + dy) % height
            danger_channel[new_y][new_x] = 0.5
        channels.append(danger_channel)
        
        # Stack and convert to tensor
        spatial_tensor = np.stack(channels, axis=0)
        spatial_tensor = torch.FloatTensor(spatial_tensor).unsqueeze(0)
        
        # Extra features
        my_boosts = state.get('my_boosts', state.get('agent1_boosts', 0)) / 3.0
        opp_boosts = state.get('opponent_boosts', state.get('agent2_boosts', 0)) / 3.0
        turn_count = state.get('turn_count', 0) / 500.0
        my_length = len(my_trail) / 360.0
        opp_length = len(opp_trail) / 360.0
        
        dx = abs(my_pos[0] - opp_pos[0])
        dy = abs(my_pos[1] - opp_pos[1])
        dx = min(dx, width - dx)
        dy = min(dy, height - dy)
        distance = (dx + dy) / (width + height)
        
        dir_x = current_dir[0]
        dir_y = current_dir[1]
        
        extra_features = torch.FloatTensor([
            my_boosts, opp_boosts, turn_count, my_length, opp_length, distance, dir_x, dir_y
        ]).unsqueeze(0)
        
        return spatial_tensor, extra_features


# Example integration code for agent.py
def example_integration():
    """
    Example of how to integrate DQN into agent.py.
    
    Copy this pattern into your agent.py file.
    """
    print("""
# Add to top of agent.py:
from dqn_inference import DQNInference

# Initialize DQN (do this once, outside the request handler)
dqn_model = DQNInference("models/dqn_agent_final.pt")

# In your get_move() function or route handler:
@app.route('/move', methods=['POST'])
def move():
    data = request.json
    
    # Convert judge engine data to state dict
    state = {
        'board': data['board'],
        'my_position': tuple(data['you']['position']),
        'opponent_position': tuple(data['opponent']['position']),
        'my_trail': [tuple(pos) for pos in data['you']['trail']],
        'opponent_trail': [tuple(pos) for pos in data['opponent']['trail']],
        'my_boosts': data['you'].get('boosts_remaining', 0),
        'opponent_boosts': data['opponent'].get('boosts_remaining', 0),
        'turn_count': data.get('turn', 0),
        'my_direction': tuple(data['you'].get('direction', [1, 0])),
        'board_height': len(data['board']),
        'board_width': len(data['board'][0]),
    }
    
    # Get move from DQN
    move = dqn_model.get_move(state)
    
    return jsonify({'move': move})
    """)


if __name__ == "__main__":
    print("="*70)
    print("DQN Inference Module")
    print("="*70)
    print("\nThis module provides utilities for loading trained DQN models")
    print("and using them in agent.py for CPU inference.\n")
    
    example_integration()
