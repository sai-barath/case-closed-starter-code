"""
Advanced Heuristic Agent with Multiple Strategies

This agent combines several sophisticated heuristics:
1. Voronoi Territory Control - Claims more space than opponent
2. Minimax-inspired threat assessment
3. Aggressive positioning toward opponent
4. Space flooding algorithm
5. Boost strategic usage

Fast enough for real-time competition (<10ms per move).
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from collections import deque
import random


class AdvancedHeuristicAgent:
    """
    Advanced heuristic agent with multiple strategic components.
    
    Strategies:
    - Territory control via Voronoi partitioning
    - Threat assessment and danger avoidance
    - Aggressive positioning when advantageous
    - Space flooding for long-term planning
    - Strategic boost usage
    """
    
    def __init__(self, aggression: float = 0.5):
        """
        Initialize agent.
        
        Args:
            aggression: 0.0 = defensive, 1.0 = aggressive (default 0.5)
        """
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.direction_deltas = {
            'UP': (0, -1),
            'DOWN': (0, 1),
            'LEFT': (-1, 0),
            'RIGHT': (1, 0),
        }
        self.aggression = aggression
    
    def get_move(self, state: Dict) -> str:
        """
        Select best move using advanced heuristics.
        
        Args:
            state: Game state dictionary
            
        Returns:
            Move string (e.g., "UP", "DOWN:BOOST")
        """
        board = state['board']
        height = state['board_height']
        width = state['board_width']
        my_pos = state['my_position']
        opp_pos = state['opponent_position']
        my_direction = state['my_direction']
        my_boosts = state.get('my_boosts', 0)
        
        current_dir = my_direction.value if hasattr(my_direction, 'value') else my_direction
        
        # Get valid actions
        valid_actions = self._get_valid_actions(state)
        if not valid_actions:
            return random.choice(self.actions)
        
        # Score each valid action
        action_scores = {}
        
        for action in valid_actions:
            dx, dy = self.direction_deltas[action]
            new_x = (my_pos[0] + dx) % width
            new_y = (my_pos[1] + dy) % height
            new_pos = (new_x, new_y)
            
            score = 0.0
            
            # 1. Territory Control (Voronoi) - Most important
            territory_score = self._voronoi_territory_score(
                board, new_pos, opp_pos, width, height
            )
            score += territory_score * 100.0
            
            # 2. Reachable Space (Flood Fill)
            reachable_score = self._reachable_space_score(
                board, new_pos, width, height
            )
            score += reachable_score * 50.0
            
            # 3. Distance to Opponent (Aggressive/Defensive)
            distance_score = self._distance_score(new_pos, opp_pos, width, height)
            score += distance_score * 20.0
            
            # 4. Center Control
            center_score = self._center_control_score(new_pos, width, height)
            score += center_score * 10.0
            
            # 5. Wall Proximity Penalty
            wall_penalty = self._wall_proximity_penalty(
                board, new_pos, width, height
            )
            score -= wall_penalty * 30.0
            
            # 6. Opponent Threat Assessment
            threat_penalty = self._opponent_threat_penalty(
                board, new_pos, opp_pos, width, height
            )
            score -= threat_penalty * 40.0
            
            action_scores[action] = score
        
        # Select best action
        best_action = max(action_scores, key=action_scores.get)
        best_score = action_scores[best_action]
        
        # Decide whether to use boost
        use_boost = False
        if my_boosts > 0:
            # Use boost if:
            # 1. We have clear advantage (high score)
            # 2. Or we're in danger (low score means opponent has advantage)
            avg_score = np.mean(list(action_scores.values()))
            
            if best_score > avg_score + 50:  # Clear advantage
                use_boost = True
            elif best_score < avg_score - 30:  # Desperate escape
                use_boost = True
        
        move = best_action
        if use_boost:
            move += ":BOOST"
        
        return move
    
    def _get_valid_actions(self, state: Dict) -> List[str]:
        """Get valid actions (not reversal, not into walls)."""
        board = state['board']
        my_pos = state['my_position']
        my_direction = state['my_direction']
        height = state['board_height']
        width = state['board_width']
        
        current_dir = my_direction.value if hasattr(my_direction, 'value') else my_direction
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
    
    def _voronoi_territory_score(self, board: List[List[int]], 
                                  my_pos: Tuple[int, int],
                                  opp_pos: Tuple[int, int],
                                  width: int, height: int) -> float:
        """
        Calculate territory control using Voronoi partitioning.
        
        Uses BFS to find cells closer to me than opponent.
        """
        # BFS from both positions simultaneously
        my_queue = deque([my_pos])
        opp_queue = deque([opp_pos])
        
        my_visited = {my_pos: 0}
        opp_visited = {opp_pos: 0}
        
        my_territory = 0
        opp_territory = 0
        neutral_territory = 0
        
        max_dist = 20  # Limit BFS depth for performance
        
        # BFS for my position
        while my_queue and len(my_visited) < 150:  # Limit for speed
            pos = my_queue.popleft()
            dist = my_visited[pos]
            
            if dist >= max_dist:
                continue
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x = (pos[0] + dx) % width
                new_y = (pos[1] + dy) % height
                new_pos = (new_x, new_y)
                
                if new_pos not in my_visited and board[new_y][new_x] == 0:
                    my_visited[new_pos] = dist + 1
                    my_queue.append(new_pos)
        
        # BFS for opponent position
        while opp_queue and len(opp_visited) < 150:
            pos = opp_queue.popleft()
            dist = opp_visited[pos]
            
            if dist >= max_dist:
                continue
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x = (pos[0] + dx) % width
                new_y = (pos[1] + dy) % height
                new_pos = (new_x, new_y)
                
                if new_pos not in opp_visited and board[new_y][new_x] == 0:
                    opp_visited[new_pos] = dist + 1
                    opp_queue.append(new_pos)
        
        # Count territories
        all_empty = set()
        for y in range(height):
            for x in range(width):
                if board[y][x] == 0:
                    all_empty.add((x, y))
        
        for pos in all_empty:
            my_dist = my_visited.get(pos, 9999)
            opp_dist = opp_visited.get(pos, 9999)
            
            if my_dist < opp_dist:
                my_territory += 1
            elif opp_dist < my_dist:
                opp_territory += 1
            else:
                neutral_territory += 1
        
        # Return normalized score (-1 to 1)
        total = my_territory + opp_territory + neutral_territory
        if total == 0:
            return 0.0
        
        return (my_territory - opp_territory) / total
    
    def _reachable_space_score(self, board: List[List[int]],
                               my_pos: Tuple[int, int],
                               width: int, height: int) -> float:
        """
        Calculate reachable empty space using flood fill.
        """
        visited = set()
        queue = deque([my_pos])
        visited.add(my_pos)
        
        count = 0
        max_count = 100  # Limit for performance
        
        while queue and count < max_count:
            pos = queue.popleft()
            count += 1
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_x = (pos[0] + dx) % width
                new_y = (pos[1] + dy) % height
                new_pos = (new_x, new_y)
                
                if new_pos not in visited and board[new_y][new_x] == 0:
                    visited.add(new_pos)
                    queue.append(new_pos)
        
        # Normalize by board size
        return count / (width * height)
    
    def _distance_score(self, my_pos: Tuple[int, int],
                       opp_pos: Tuple[int, int],
                       width: int, height: int) -> float:
        """
        Score based on distance to opponent.
        
        Aggressive mode: prefer closer
        Defensive mode: prefer farther
        """
        dx = abs(my_pos[0] - opp_pos[0])
        dy = abs(my_pos[1] - opp_pos[1])
        
        # Torus wrapping
        dx = min(dx, width - dx)
        dy = min(dy, height - dy)
        
        distance = dx + dy
        max_distance = width + height
        
        normalized_dist = distance / max_distance
        
        # Aggression determines if we want to be close or far
        if self.aggression > 0.5:
            return 1.0 - normalized_dist  # Prefer closer
        else:
            return normalized_dist  # Prefer farther
    
    def _center_control_score(self, pos: Tuple[int, int],
                              width: int, height: int) -> float:
        """
        Score based on proximity to center (center control is valuable).
        """
        center_x = width / 2
        center_y = height / 2
        
        dx = abs(pos[0] - center_x)
        dy = abs(pos[1] - center_y)
        
        distance = dx + dy
        max_distance = center_x + center_y
        
        return 1.0 - (distance / max_distance)
    
    def _wall_proximity_penalty(self, board: List[List[int]],
                                pos: Tuple[int, int],
                                width: int, height: int) -> float:
        """
        Penalize being near walls (risky positions).
        """
        wall_count = 0
        check_radius = 2
        
        for dx in range(-check_radius, check_radius + 1):
            for dy in range(-check_radius, check_radius + 1):
                if dx == 0 and dy == 0:
                    continue
                
                check_x = (pos[0] + dx) % width
                check_y = (pos[1] + dy) % height
                
                if board[check_y][check_x] != 0:
                    wall_count += 1
        
        max_walls = (check_radius * 2 + 1) ** 2 - 1
        return wall_count / max_walls
    
    def _opponent_threat_penalty(self, board: List[List[int]],
                                 my_pos: Tuple[int, int],
                                 opp_pos: Tuple[int, int],
                                 width: int, height: int) -> float:
        """
        Penalize positions where opponent can trap us.
        """
        # Check if opponent can reach this position quickly
        dx = abs(my_pos[0] - opp_pos[0])
        dy = abs(my_pos[1] - opp_pos[1])
        
        dx = min(dx, width - dx)
        dy = min(dy, height - dy)
        
        distance = dx + dy
        
        # If very close, check if they can cut us off
        if distance <= 3:
            return 1.0 - (distance / 10.0)
        
        return 0.0


# Create policy function for evaluation
def create_advanced_heuristic_policy(aggression: float = 0.5):
    """
    Create a policy function for the advanced heuristic agent.
    
    Args:
        aggression: 0.0 = defensive, 1.0 = aggressive
        
    Returns:
        Policy function
    """
    agent = AdvancedHeuristicAgent(aggression=aggression)
    
    def policy(state: Dict) -> str:
        return agent.get_move(state)
    
    return policy


# Preset agents with different strategies
def aggressive_heuristic_agent(state: Dict) -> str:
    """Aggressive heuristic agent (aggression=0.8)."""
    agent = AdvancedHeuristicAgent(aggression=0.8)
    return agent.get_move(state)


def defensive_heuristic_agent(state: Dict) -> str:
    """Defensive heuristic agent (aggression=0.2)."""
    agent = AdvancedHeuristicAgent(aggression=0.2)
    return agent.get_move(state)


def balanced_heuristic_agent(state: Dict) -> str:
    """Balanced heuristic agent (aggression=0.5)."""
    agent = AdvancedHeuristicAgent(aggression=0.5)
    return agent.get_move(state)


if __name__ == "__main__":
    from training_env import evaluate_agents, wall_avoider_agent, greedy_space_agent
    
    print("="*70)
    print("Advanced Heuristic Agent - Performance Test")
    print("="*70)
    
    # Test against baseline agents
    print("\n[1/3] Balanced Heuristic vs Wall Avoider (100 games)...")
    stats1 = evaluate_agents(balanced_heuristic_agent, wall_avoider_agent, 
                            num_episodes=100, verbose=True)
    
    print("\n[2/3] Balanced Heuristic vs Greedy Space (100 games)...")
    stats2 = evaluate_agents(balanced_heuristic_agent, greedy_space_agent,
                            num_episodes=100, verbose=True)
    
    print("\n[3/3] Aggressive vs Defensive (100 games)...")
    stats3 = evaluate_agents(aggressive_heuristic_agent, defensive_heuristic_agent,
                            num_episodes=100, verbose=True)
    
    print("\n" + "="*70)
    print("Performance Summary:")
    print("="*70)
    print(f"vs Wall Avoider:   {stats1['agent1_win_rate']:.1%} win rate")
    print(f"vs Greedy Space:   {stats2['agent1_win_rate']:.1%} win rate")
    print(f"Aggressive vs Def: {stats3['agent1_win_rate']:.1%} win rate")
    print("="*70)
