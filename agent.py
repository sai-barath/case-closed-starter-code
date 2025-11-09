import os
import uuid
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

from case_closed_game import Game, Direction, GameResult

# ============================================================================
# DQN NEURAL NETWORK ARCHITECTURE (FAST BFS-BASED)
# ============================================================================

def bfs_reachable_space(start_pos: Tuple[int, int], board: list, 
                        width: int, height: int, max_depth: int = 50) -> int:
    """BFS to count reachable empty cells from start position."""
    visited = set()
    queue = deque([(start_pos, 0)])
    visited.add(start_pos)
    count = 0
    
    while queue:
        (x, y), depth = queue.popleft()
        if depth > max_depth:
            continue
        count += 1
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x = (x + dx) % width
            new_y = (y + dy) % height
            
            if (new_x, new_y) not in visited and board[new_y][new_x] == 0:
                visited.add((new_x, new_y))
                queue.append(((new_x, new_y), depth + 1))
    
    return count


class DQNetwork(nn.Module):
    """
    Hybrid CNN+BFS DQN architecture.
    Combines spatial CNN processing with strategic BFS features.
    """
    
    def __init__(self):
        super(DQNetwork, self).__init__()
        
        # CNN path for spatial processing (5 channels: my_trail, opp_trail, walls, my_pos, opp_pos)
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # After 2 pooling layers: 18×20 → 9×10 → 4×5 = 20 cells
        # With 64 channels: 64 * 4 * 5 = 1280 features
        self.fc_cnn = nn.Linear(64 * 4 * 5, 256)
        
        # BFS path for strategic features (8 features)
        self.fc_bfs1 = nn.Linear(8, 64)
        self.fc_bfs2 = nn.Linear(64, 128)
        
        # Combined path (256 from CNN + 128 from BFS = 384)
        self.fc_combined1 = nn.Linear(384, 256)
        self.fc_combined2 = nn.Linear(256, 128)
        self.fc_combined3 = nn.Linear(128, 4)  # 4 actions
        
        self.dropout = nn.Dropout(0.2)
        self.layer_norm_combined = nn.LayerNorm(256)
        
    def forward(self, spatial_input: torch.Tensor, bfs_input: torch.Tensor) -> torch.Tensor:
        # CNN path
        x_cnn = F.relu(self.conv1(spatial_input))
        x_cnn = self.pool(x_cnn)  # 18×20 → 9×10
        x_cnn = F.relu(self.conv2(x_cnn))
        x_cnn = self.pool(x_cnn)  # 9×10 → 4×5
        x_cnn = F.relu(self.conv3(x_cnn))
        x_cnn = x_cnn.view(x_cnn.size(0), -1)  # Flatten
        x_cnn = F.relu(self.fc_cnn(x_cnn))
        
        # BFS path
        x_bfs = F.relu(self.fc_bfs1(bfs_input))
        x_bfs = F.relu(self.fc_bfs2(x_bfs))
        
        # Combine
        x = torch.cat([x_cnn, x_bfs], dim=1)
        x = F.relu(self.layer_norm_combined(self.fc_combined1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc_combined2(x))
        x = self.fc_combined3(x)
        
        return x


class DQNAgent:
    """Hybrid CNN+BFS DQN Agent."""
    
    def __init__(self, model_path: str):
        self.device = torch.device("cpu")
        self.model = DQNetwork().to(self.device)
        
        # Load trained model
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"✓ Hybrid CNN+BFS DQN model loaded from {model_path}")
        else:
            print(f"⚠ Model file not found: {model_path}")
            print(f"⚠ Using untrained model (will play randomly)")
        
        # Action mapping
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.direction_deltas = {
            'UP': (0, -1),
            'DOWN': (0, 1),
            'LEFT': (-1, 0),
            'RIGHT': (1, 0),
        }
    
    def state_to_tensor(self, state_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert game state to dual tensors: spatial (5×18×20) and BFS features (8).
        
        Spatial channels:
        0: My trail
        1: Opponent trail
        2: Walls/obstacles
        3: My position (hot encoded)
        4: Opponent position (hot encoded)
        
        BFS features: [space_up, space_down, space_left, space_right, 
                       dist_to_opp, my_boosts, opp_boosts, turn_ratio]
        """
        board = state_dict.get('board', [])
        my_pos = state_dict.get('my_position', (0, 0))
        opp_pos = state_dict.get('opponent_position', (0, 0))
        height = len(board)
        width = len(board[0]) if board else 20
        turn_count = state_dict.get('turn_count', 0)
        my_boosts = state_dict.get('my_boosts', 0)
        
        # Get opponent boosts
        player_num = state_dict.get('player_number', 1)
        if player_num == 1:
            opp_boosts = state_dict.get('agent2_boosts', 0)
        else:
            opp_boosts = state_dict.get('agent1_boosts', 0)
        
        # Create spatial channels (5 channels, 18×20)
        spatial = np.zeros((5, height, width), dtype=np.float32)
        
        # Channel 0: My trail (1 where I've been)
        # Channel 1: Opponent trail (1 where opponent has been)
        # Channel 2: Walls/obstacles (1 where blocked, excluding player positions)
        for y in range(height):
            for x in range(width):
                if board[y][x] == player_num:
                    spatial[0, y, x] = 1.0  # My trail
                elif board[y][x] == (3 - player_num):  # Other player
                    spatial[1, y, x] = 1.0  # Opponent trail
                elif board[y][x] > 0:
                    spatial[2, y, x] = 1.0  # Wall/obstacle
        
        # Channel 3: My position (hot encoded)
        spatial[3, my_pos[1], my_pos[0]] = 1.0
        
        # Channel 4: Opponent position (hot encoded)
        spatial[4, opp_pos[1], opp_pos[0]] = 1.0
        
        # BFS features - Calculate reachable space in each direction
        bfs_features = []
        for action in self.actions:
            dx, dy = self.direction_deltas[action]
            new_x = (my_pos[0] + dx) % width
            new_y = (my_pos[1] + dy) % height
            
            if board[new_y][new_x] == 0:
                # Make temporary board copy
                board_copy = [row[:] for row in board]
                board_copy[new_y][new_x] = 1
                space = bfs_reachable_space((new_x, new_y), board_copy, width, height)
                bfs_features.append(space / 360.0)  # Normalize
            else:
                bfs_features.append(0.0)  # Blocked
        
        # Distance to opponent (Manhattan with wrapping)
        dx = abs(my_pos[0] - opp_pos[0])
        dy = abs(my_pos[1] - opp_pos[1])
        dx = min(dx, width - dx)
        dy = min(dy, height - dy)
        distance = (dx + dy) / (width + height)
        
        # Combine all BFS features
        bfs_features.extend([
            distance,
            my_boosts / 3.0,
            opp_boosts / 3.0,
            turn_count / 500.0
        ])
        
        # Convert to tensors
        spatial_tensor = torch.FloatTensor(spatial).unsqueeze(0).to(self.device)
        bfs_tensor = torch.FloatTensor(bfs_features).unsqueeze(0).to(self.device)
        
        return spatial_tensor, bfs_tensor
    
    def count_reachable_spaces(self, board: list, start_pos: Tuple[int, int], max_depth: int = 15) -> int:
        """BFS to count reachable empty spaces from a position."""
        height = len(board)
        width = len(board[0]) if board else 20
        
        visited = set()
        queue = deque([start_pos])
        visited.add(start_pos)
        count = 0
        depth = 0
        
        while queue and depth < max_depth:
            level_size = len(queue)
            for _ in range(level_size):
                x, y = queue.popleft()
                count += 1
                
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    new_x = (x + dx) % width
                    new_y = (y + dy) % height
                    
                    if (new_x, new_y) not in visited and board[new_y][new_x] == 0:
                        visited.add((new_x, new_y))
                        queue.append((new_x, new_y))
            depth += 1
        
        return count
    
    def is_safe_move(self, board: list, pos: Tuple[int, int], action: str, my_direction: Tuple[int, int]) -> bool:
        """Check if a move is safe (not into immediate trap)."""
        height = len(board)
        width = len(board[0]) if board else 20
        
        dx, dy = self.direction_deltas[action]
        
        # Check reversal
        if (dx, dy) == (-my_direction[0], -my_direction[1]):
            return False
        
        # Check if cell is empty
        new_x = (pos[0] + dx) % width
        new_y = (pos[1] + dy) % height
        
        if board[new_y][new_x] != 0:
            return False
        
        # Check if move leads to a space with at least 2 exits
        exits = 0
        for next_dx, next_dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            # Skip reversal from new position
            if (next_dx, next_dy) == (-dx, -dy):
                continue
            
            check_x = (new_x + next_dx) % width
            check_y = (new_y + next_dy) % height
            
            if board[check_y][check_x] == 0:
                exits += 1
        
        return exits >= 1  # At least 1 exit (other than where we came from)
    
    def get_valid_actions(self, state_dict: Dict) -> list:
        """Get list of valid actions with safety checks."""
        board = state_dict.get('board', [])
        my_pos = state_dict.get('my_position', (0, 0))
        my_direction = state_dict.get('my_direction', (1, 0))
        height = len(board)
        width = len(board[0]) if board else 20
        
        safe_actions = []
        basic_valid = []
        
        for action in self.actions:
            dx, dy = self.direction_deltas[action]
            
            # Check reversal
            if (dx, dy) == (-my_direction[0], -my_direction[1]):
                continue
            
            # Check if cell is empty
            new_x = (my_pos[0] + dx) % width
            new_y = (my_pos[1] + dy) % height
            
            if board[new_y][new_x] == 0:
                basic_valid.append(action)
                
                # Additional safety check: count reachable spaces
                reachable = self.count_reachable_spaces(board, (new_x, new_y), max_depth=10)
                
                # Consider safe if leads to reasonable territory
                if reachable >= 8 and self.is_safe_move(board, my_pos, action, my_direction):
                    safe_actions.append(action)
        
        # Prefer safe actions, fallback to basic valid, then any non-reversal
        if safe_actions:
            return safe_actions
        elif basic_valid:
            return basic_valid
        else:
            # Last resort: any non-reversal move
            fallback = []
            for action in self.actions:
                dx, dy = self.direction_deltas[action]
                if (dx, dy) != (-my_direction[0], -my_direction[1]):
                    fallback.append(action)
            return fallback if fallback else ['RIGHT']
    
    def is_boost_safe(self, board: list, pos: Tuple[int, int], action: str, my_direction: Tuple[int, int]) -> bool:
        """Check if using boost is safe (both first and second move are valid)."""
        height = len(board)
        width = len(board[0]) if board else 20
        
        dx, dy = self.direction_deltas[action]
        
        # First move position
        first_x = (pos[0] + dx) % width
        first_y = (pos[1] + dy) % height
        
        if board[first_y][first_x] != 0:
            return False
        
        # Second move position (same direction)
        second_x = (first_x + dx) % width
        second_y = (first_y + dy) % height
        
        if board[second_y][second_x] != 0:
            return False
        
        # Check that second position has at least 1 exit
        exits = 0
        for next_dx, next_dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            if (next_dx, next_dy) == (-dx, -dy):
                continue
            
            check_x = (second_x + next_dx) % width
            check_y = (second_y + next_dy) % height
            
            if board[check_y][check_x] == 0:
                exits += 1
        
        # Need at least 1 exit after boost
        if exits < 1:
            return False
        
        # Check reachable space after boost
        reachable = self.count_reachable_spaces(board, (second_x, second_y), max_depth=8)
        return reachable >= 6
    
    def get_move(self, state_dict: Dict) -> str:
        """Get best move from DQN with safety checks."""
        valid_actions = self.get_valid_actions(state_dict)
        
        if not valid_actions:
            return 'RIGHT'
        
        board = state_dict.get('board', [])
        my_pos = state_dict.get('my_position', (0, 0))
        my_direction = state_dict.get('my_direction', (1, 0))
        my_boosts = state_dict.get('my_boosts', 0)
        
        # Convert state to dual tensors (spatial + BFS)
        spatial_tensor, bfs_tensor = self.state_to_tensor(state_dict)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.model(spatial_tensor, bfs_tensor)
            q_values = q_values.cpu().numpy()[0]
        
        # Mask invalid actions
        masked_q = q_values.copy()
        for idx, action in enumerate(self.actions):
            if action not in valid_actions:
                masked_q[idx] = -1e9
        
        # Rank actions by Q-value
        action_q_pairs = [(self.actions[i], q_values[i]) for i in range(len(self.actions)) if self.actions[i] in valid_actions]
        action_q_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Select best SAFE action with secondary validation
        best_action = None
        best_q_value = -1e9
        
        for action, q_val in action_q_pairs:
            # Double-check safety
            if self.is_safe_move(board, my_pos, action, my_direction):
                best_action = action
                best_q_value = q_val
                break
        
        # If no safe action found, take the highest Q-value action
        if best_action is None:
            best_action = action_q_pairs[0][0] if action_q_pairs else 'RIGHT'
            best_q_value = action_q_pairs[0][1] if action_q_pairs else 0
        
        # Smart boost decision
        use_boost = False
        if my_boosts > 0 and best_action:
            # Check if boost is safe
            if self.is_boost_safe(board, my_pos, best_action, my_direction):
                # Calculate advantage
                other_q_values = [q for a, q in action_q_pairs if a != best_action]
                if other_q_values:
                    avg_q = np.mean(other_q_values)
                    q_advantage = best_q_value - avg_q
                    
                    # Use boost if significant advantage and lots of space ahead
                    reachable_after = self.count_reachable_spaces(board, my_pos, max_depth=12)
                    
                    # More conservative boost usage
                    if q_advantage > 0.3 and reachable_after > 20:
                        use_boost = True
                    # Or if we're in open space and need to claim territory
                    elif reachable_after > 40 and my_boosts > 1:
                        use_boost = True
        
        move = best_action
        if use_boost:
            move += ":BOOST"
        
        return move


# ============================================================================
# FLASK API SERVER SETUP
# ============================================================================

# Initialize DQN agent (loads model once at startup)
# Using trained Hybrid CNN+BFS DQN model from timestamped directory
# IMPORTANT: Train the model first with: python dqn_fast.py 1000
# Then update MODEL_PATH below to point to the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models_fast_20251108_172653", "dqn_fast_final.pt")
dqn_agent = DQNAgent(MODEL_PATH)

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "ParticipantX"
AGENT_NAME = "AgentX"


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.

    Returns participant and agent_name (so Judge.check_latency can create Agent objects).
    """
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.

    The judge posts a dictionary with keys matching the Judge.send_state payload
    (board, agent1_trail, agent2_trail, agent1_length, agent2_length, agent1_alive,
    agent2_alive, agent1_boosts, agent2_boosts, turn_count).
    """
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"]) 
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"]) 
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server.

    The agent should update its local representation and return 200.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.

    Query params the judge sends (optional): player_number, attempt_number,
    random_moves_left, turn_count. Agents can use this to decide.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    where DIRECTION is UP, DOWN, LEFT, or RIGHT
    and :BOOST is optional to use a speed boost (move twice)
    """
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        state = dict(LAST_POSTED_STATE)   
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        opp_agent = GLOBAL_GAME.agent2 if player_number == 1 else GLOBAL_GAME.agent1
        boosts_remaining = my_agent.boosts_remaining
   
    # ============================================================================
    # DQN AGENT MOVE SELECTION
    # ============================================================================
    
    # Prepare state dictionary for DQN
    board = state.get('board', GLOBAL_GAME.board.grid)
    
    # Get trails
    my_trail = list(my_agent.trail) if hasattr(my_agent, 'trail') else []
    opp_trail = list(opp_agent.trail) if hasattr(opp_agent, 'trail') else []
    
    # Get positions (last element of trail)
    my_position = my_trail[-1] if my_trail else (0, 0)
    opp_position = opp_trail[-1] if opp_trail else (0, 0)
    
    # Get direction
    my_direction = my_agent.direction.value if hasattr(my_agent.direction, 'value') else (1, 0)
    
    # Create state dict for DQN
    dqn_state = {
        'board': board,
        'my_position': my_position,
        'opponent_position': opp_position,
        'my_trail': my_trail,
        'opponent_trail': opp_trail,
        'my_boosts': boosts_remaining,
        'opponent_boosts': opp_agent.boosts_remaining if hasattr(opp_agent, 'boosts_remaining') else 0,
        'turn_count': state.get('turn_count', GLOBAL_GAME.turns),
        'my_direction': my_direction,
        'board_height': len(board),
        'board_width': len(board[0]) if board else 20,
    }
    
    # Get move from DQN agent
    try:
        move = dqn_agent.get_move(dqn_state)
    except Exception as e:
        print(f"Error in DQN agent: {e}")
        # Fallback to a safe default
        move = "RIGHT"

    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We update local state for record-keeping and return OK.
    """
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=True)
