"""
Advanced Heuristic Agents for Case Closed Game

These agents implement sophisticated strategies for training DQN opponents.
All agents operate locally without network calls.
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Set, Optional
import random


class BaseHeuristicAgent:
    """Base class for all heuristic agents."""
    
    def __init__(self):
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.direction_deltas = {
            'UP': (0, -1),
            'DOWN': (0, 1),
            'LEFT': (-1, 0),
            'RIGHT': (1, 0),
        }
    
    def get_valid_moves(self, state: Dict) -> List[str]:
        """Get list of valid moves (non-reversing, non-collision)."""
        board = state['board']
        my_pos = state['my_position']
        my_direction = state.get('my_direction', (1, 0))
        
        # Handle Direction enum OR tuple
        if hasattr(my_direction, 'value'):
            my_direction = my_direction.value
        
        height = state['board_height']
        width = state['board_width']
        
        valid = []
        for action in self.actions:
            dx, dy = self.direction_deltas[action]
            
            # CRITICAL: Skip if it's a reversal (opposite of current direction)
            if (dx, dy) == (-my_direction[0], -my_direction[1]):
                continue
            
            # Check if move is valid (not into wall)
            new_x = (my_pos[0] + dx) % width
            new_y = (my_pos[1] + dy) % height
            
            if board[new_y][new_x] == 0:  # Empty cell
                valid.append(action)
        
        # IMPORTANT: If no valid moves, return non-reversal moves (agent will crash but not cause invalid move)
        if not valid:
            for action in self.actions:
                dx, dy = self.direction_deltas[action]
                if (dx, dy) != (-my_direction[0], -my_direction[1]):
                    valid.append(action)
        
        return valid if valid else self.actions  # Last resort fallback
    
    def flood_fill(self, start: Tuple[int, int], board: List[List[int]], 
                   width: int, height: int) -> int:
        """
        Flood fill to count reachable empty cells from a starting position.
        Returns the number of reachable cells.
        """
        visited = set()
        queue = deque([start])
        visited.add(start)
        count = 0
        
        while queue:
            x, y = queue.popleft()
            count += 1
            
            # Check all 4 directions
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x = (x + dx) % width
                new_y = (y + dy) % height
                
                if (new_x, new_y) not in visited and board[new_y][new_x] == 0:
                    visited.add((new_x, new_y))
                    queue.append((new_x, new_y))
        
        return count
    
    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int],
                          width: int, height: int) -> int:
        """Calculate Manhattan distance with torus wrapping."""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        
        # Account for wrapping
        dx = min(dx, width - dx)
        dy = min(dy, height - dy)
        
        return dx + dy
    
    def get_move(self, state: Dict) -> str:
        """Must be implemented by subclasses."""
        raise NotImplementedError


class GreedySpaceMaximizer(BaseHeuristicAgent):
    """
    🧩 The Greedy Space Maximizer
    
    Strategy: Always move toward the direction with the most open space.
    Uses flood-fill to calculate reachable cells from each possible move.
    Excellent at avoiding self-trapping situations.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Greedy Space Maximizer"
    
    def get_move(self, state: Dict) -> str:
        """Select move that maximizes reachable space."""
        valid_moves = self.get_valid_moves(state)
        
        if not valid_moves:
            return random.choice(self.actions)
        
        board = state['board']
        my_pos = state['my_position']
        height = state['board_height']
        width = state['board_width']
        
        best_move = valid_moves[0]
        best_space = -1
        
        # Evaluate each valid move
        for action in valid_moves:
            dx, dy = self.direction_deltas[action]
            new_x = (my_pos[0] + dx) % width
            new_y = (my_pos[1] + dy) % height
            
            # Simulate the move - create a copy of the board with this position marked
            board_copy = [row[:] for row in board]
            board_copy[new_y][new_x] = 1  # Mark as occupied
            
            # Flood fill from new position to count reachable space
            reachable = self.flood_fill((new_x, new_y), board_copy, width, height)
            
            if reachable > best_space:
                best_space = reachable
                best_move = action
        
        return best_move


class AggressiveChaser(BaseHeuristicAgent):
    """
    🧠 The Aggressive Chaser
    
    Strategy: Always move toward the opponent's position.
    Uses Manhattan distance to find the closest path.
    Forces the DQN to learn defensive tactics and evasion.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Aggressive Chaser"
    
    def get_move(self, state: Dict) -> str:
        """Select move that brings us closest to opponent."""
        valid_moves = self.get_valid_moves(state)
        
        if not valid_moves:
            return random.choice(self.actions)
        
        my_pos = state['my_position']
        opp_pos = state['opponent_position']
        height = state['board_height']
        width = state['board_width']
        
        best_move = valid_moves[0]
        best_distance = float('inf')
        
        # Find move that minimizes distance to opponent
        for action in valid_moves:
            dx, dy = self.direction_deltas[action]
            new_x = (my_pos[0] + dx) % width
            new_y = (my_pos[1] + dy) % height
            
            distance = self.manhattan_distance((new_x, new_y), opp_pos, width, height)
            
            if distance < best_distance:
                best_distance = distance
                best_move = action
        
        return best_move


class SmartAvoider(BaseHeuristicAgent):
    """
    ⚔️ The Smart Avoider
    
    Strategy: Stay as far away from opponent as possible while maintaining space.
    Combines distance maximization with space awareness.
    Teaches the DQN to balance aggression with survival.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Smart Avoider"
    
    def get_move(self, state: Dict) -> str:
        """Select move that maximizes distance from opponent while avoiding traps."""
        valid_moves = self.get_valid_moves(state)
        
        if not valid_moves:
            return random.choice(self.actions)
        
        board = state['board']
        my_pos = state['my_position']
        opp_pos = state['opponent_position']
        height = state['board_height']
        width = state['board_width']
        
        best_move = valid_moves[0]
        best_score = -float('inf')
        
        # Evaluate each move based on distance and space
        for action in valid_moves:
            dx, dy = self.direction_deltas[action]
            new_x = (my_pos[0] + dx) % width
            new_y = (my_pos[1] + dy) % height
            
            # Calculate distance from opponent (want to maximize this)
            distance = self.manhattan_distance((new_x, new_y), opp_pos, width, height)
            
            # Calculate reachable space (also want to maximize this)
            board_copy = [row[:] for row in board]
            board_copy[new_y][new_x] = 1
            reachable = self.flood_fill((new_x, new_y), board_copy, width, height)
            
            # Score combines distance (60%) and space (40%)
            score = distance * 0.6 + reachable * 0.4
            
            if score > best_score:
                best_score = score
                best_move = action
        
        return best_move


class TerritorialDefender(BaseHeuristicAgent):
    """
    🛡️ The Territorial Defender
    
    Strategy: Control the center and maximize controlled territory.
    Moves to positions that give the most strategic advantage.
    Combines space control with positional awareness.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Territorial Defender"
    
    def get_move(self, state: Dict) -> str:
        """Select move that maximizes territorial control."""
        valid_moves = self.get_valid_moves(state)
        
        if not valid_moves:
            return random.choice(self.actions)
        
        board = state['board']
        my_pos = state['my_position']
        opp_pos = state['opponent_position']
        height = state['board_height']
        width = state['board_width']
        
        center_x = width // 2
        center_y = height // 2
        
        best_move = valid_moves[0]
        best_score = -float('inf')
        
        for action in valid_moves:
            dx, dy = self.direction_deltas[action]
            new_x = (my_pos[0] + dx) % width
            new_y = (my_pos[1] + dy) % height
            
            # Calculate multiple factors
            # 1. Distance to center (closer is better early game)
            dist_to_center = self.manhattan_distance((new_x, new_y), (center_x, center_y), width, height)
            center_score = max(0, (width + height) - dist_to_center)
            
            # 2. Reachable space
            board_copy = [row[:] for row in board]
            board_copy[new_y][new_x] = 1
            reachable = self.flood_fill((new_x, new_y), board_copy, width, height)
            
            # 3. Distance from opponent (maintain some distance)
            opp_distance = self.manhattan_distance((new_x, new_y), opp_pos, width, height)
            
            # Combined score: space (50%), center control (30%), opponent distance (20%)
            score = reachable * 0.5 + center_score * 0.3 + opp_distance * 0.2
            
            if score > best_score:
                best_score = score
                best_move = action
        
        return best_move


class AdaptiveHybrid(BaseHeuristicAgent):
    """
    🎯 The Adaptive Hybrid
    
    Strategy: Dynamically switches between aggressive, defensive, and space-maximizing
    based on game state (turns remaining, space available, opponent proximity).
    
    The ultimate heuristic opponent that adapts to the situation.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Adaptive Hybrid"
        self.space_maximizer = GreedySpaceMaximizer()
        self.chaser = AggressiveChaser()
        self.avoider = SmartAvoider()
        self.defender = TerritorialDefender()
    
    def analyze_game_state(self, state: Dict) -> str:
        """Determine which strategy to use based on game state."""
        board = state['board']
        my_pos = state['my_position']
        opp_pos = state['opponent_position']
        height = state['board_height']
        width = state['board_width']
        turn_count = state.get('turn_count', 0)
        
        # Calculate key metrics
        total_cells = height * width
        occupied_cells = sum(sum(row) for row in board)
        free_space_ratio = (total_cells - occupied_cells) / total_cells
        
        distance_to_opp = self.manhattan_distance(my_pos, opp_pos, width, height)
        
        # Calculate our reachable space
        board_copy = [row[:] for row in board]
        my_reachable = self.flood_fill(my_pos, board_copy, width, height)
        
        # Decision logic
        # Early game (< 30 turns): Control territory
        if turn_count < 30:
            return "defender"
        
        # If very close to opponent (< 5 tiles): Avoid collision
        if distance_to_opp < 5:
            return "avoider"
        
        # If low on space (< 20% free): Focus on space
        if free_space_ratio < 0.2:
            return "space_maximizer"
        
        # If we have significantly less reachable space: Get more space
        if my_reachable < total_cells * 0.15:
            return "space_maximizer"
        
        # Mid game with good position: Be aggressive
        if free_space_ratio > 0.3 and my_reachable > total_cells * 0.2:
            return "chaser"
        
        # Default: Maximize space (safest)
        return "space_maximizer"
    
    def get_move(self, state: Dict) -> str:
        """Adaptively select strategy and make move."""
        strategy = self.analyze_game_state(state)
        
        if strategy == "space_maximizer":
            return self.space_maximizer.get_move(state)
        elif strategy == "chaser":
            return self.chaser.get_move(state)
        elif strategy == "avoider":
            return self.avoider.get_move(state)
        elif strategy == "defender":
            return self.defender.get_move(state)
        else:
            return self.space_maximizer.get_move(state)


# ============================================================================
# Convenience functions for easy integration
# ============================================================================

def create_heuristic_agent(agent_type: str = "hybrid"):
    """
    Factory function to create heuristic agents.
    
    Args:
        agent_type: One of ["space", "chaser", "avoider", "defender", "hybrid"]
    
    Returns:
        Heuristic agent instance
    """
    agents = {
        "space": GreedySpaceMaximizer,
        "chaser": AggressiveChaser,
        "avoider": SmartAvoider,
        "defender": TerritorialDefender,
        "hybrid": AdaptiveHybrid,
    }
    
    agent_class = agents.get(agent_type.lower(), AdaptiveHybrid)
    return agent_class()


def get_all_agents() -> List[BaseHeuristicAgent]:
    """Return instances of all heuristic agents for testing."""
    return [
        GreedySpaceMaximizer(),
        AggressiveChaser(),
        SmartAvoider(),
        TerritorialDefender(),
        AdaptiveHybrid(),
    ]


# ============================================================================
# Wrapper functions for use with training_env.py
# ============================================================================

def greedy_space_agent(state: Dict) -> str:
    """Wrapper for GreedySpaceMaximizer - compatible with training_env."""
    agent = GreedySpaceMaximizer()
    return agent.get_move(state)


def aggressive_chaser_agent(state: Dict) -> str:
    """Wrapper for AggressiveChaser - compatible with training_env."""
    agent = AggressiveChaser()
    return agent.get_move(state)


def smart_avoider_agent(state: Dict) -> str:
    """Wrapper for SmartAvoider - compatible with training_env."""
    agent = SmartAvoider()
    return agent.get_move(state)


def territorial_defender_agent(state: Dict) -> str:
    """Wrapper for TerritorialDefender - compatible with training_env."""
    agent = TerritorialDefender()
    return agent.get_move(state)


def adaptive_hybrid_agent(state: Dict) -> str:
    """Wrapper for AdaptiveHybrid - compatible with training_env."""
    agent = AdaptiveHybrid()
    return agent.get_move(state)


# ============================================================================
# Testing and Demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Advanced Heuristic Agents for Case Closed")
    print("=" * 70)
    
    # Create sample state for testing
    sample_state = {
        'board': [[0] * 20 for _ in range(18)],
        'board_height': 18,
        'board_width': 20,
        'my_position': (10, 9),
        'opponent_position': (15, 9),
        'my_direction': (1, 0),
        'turn_count': 10,
        'my_trail': [(9, 9), (10, 9)],
        'opponent_trail': [(14, 9), (15, 9)],
    }
    
    # Mark some occupied cells
    sample_state['board'][9][9] = 1
    sample_state['board'][9][10] = 1
    sample_state['board'][9][14] = 1
    sample_state['board'][9][15] = 1
    
    print("\nTesting all agents with sample state:")
    print(f"My Position: {sample_state['my_position']}")
    print(f"Opponent Position: {sample_state['opponent_position']}")
    print()
    
    agents = get_all_agents()
    for agent in agents:
        move = agent.get_move(sample_state)
        print(f"{agent.name:30} -> {move}")
    
    print("\n" + "=" * 70)
    print("All agents ready for training!")
    print("=" * 70)
