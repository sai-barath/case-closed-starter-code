#!/usr/bin/env python3
"""
Heuristic Data Collector - Runs games and collects training data for weight optimization

Collects heuristic values at each turn before agents separate into disconnected components.
Labels each position with the final outcome (component size difference when separated).
"""

import requests
import sys
import time
import os
import csv
from collections import deque
from case_closed_game import Game, Direction, GameResult

TIMEOUT = 4
BOARD_HEIGHT = 18
BOARD_WIDTH = 20

class PlayerAgent:
    def __init__(self, participant, agent_name):
        self.participant = participant
        self.agent_name = agent_name
        self.latency = 0.0

def torus_check(pos):
    """Normalize position for torus wraparound"""
    x, y = pos
    return (x % BOARD_WIDTH, y % BOARD_HEIGHT)

def torus_distance(p1, p2):
    """Calculate Manhattan distance on torus"""
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    
    if dx > BOARD_WIDTH // 2:
        dx = BOARD_WIDTH - dx
    if dy > BOARD_HEIGHT // 2:
        dy = BOARD_HEIGHT - dy
    
    return dx + dy

def calculate_voronoi_control(agent1_trail, agent2_trail, board_grid):
    """Calculate Voronoi territory control for both agents"""
    if not agent1_trail or not agent2_trail:
        return 0, 0, {}
    
    a1_head = agent1_trail[-1]
    a2_head = agent2_trail[-1]
    
    occupied = set(agent1_trail) | set(agent2_trail)
    
    visited = {}
    queue = deque()
    
    queue.append((a1_head, 1))
    queue.append((a2_head, 2))
    visited[a1_head] = 1
    visited[a2_head] = 2
    
    while queue:
        pos, owner = queue.popleft()
        x, y = pos
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx = (x + dx) % BOARD_WIDTH
            ny = (y + dy) % BOARD_HEIGHT
            npos = (nx, ny)
            
            if npos in occupied or npos in visited:
                continue
            
            visited[npos] = owner
            queue.append((npos, owner))
    
    p1_territory = sum(1 for v in visited.values() if v == 1)
    p2_territory = sum(1 for v in visited.values() if v == 2)
    
    return p1_territory, p2_territory, visited

def count_reachable_space(agent_trail, board_grid, max_depth=15):
    """BFS to count reachable empty space from agent head"""
    if not agent_trail:
        return 0
    
    head = agent_trail[-1]
    occupied = set(agent_trail)
    
    visited = set()
    queue = deque([head])
    visited.add(head)
    count = 0
    
    while queue and count < max_depth:
        x, y = queue.popleft()
        count += 1
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx = (x + dx) % BOARD_WIDTH
            ny = (y + dy) % BOARD_HEIGHT
            npos = (nx, ny)
            
            if npos not in visited and board_grid[ny][nx] == 0:  # EMPTY
                visited.add(npos)
                queue.append(npos)
    
    return count

def calculate_edge_bonus(control, owner):
    """Count edge cells (cells with empty neighbors) in controlled territory"""
    bonus = 0
    
    for (x, y), ctrl_owner in control.items():
        if ctrl_owner != owner:
            continue
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx = (x + dx) % BOARD_WIDTH
            ny = (y + dy) % BOARD_HEIGHT
            npos = (nx, ny)
            
            # Check if neighbor is empty (not in control means not occupied)
            if npos not in control:
                bonus += 1
    
    return bonus

def are_separated(agent1_trail, agent2_trail, board_grid):
    """Check if agents are in separate components. Returns (separated, p1_size, p2_size)"""
    if not agent1_trail or not agent2_trail:
        return False, 0, 0
    
    a1_head = agent1_trail[-1]
    a2_head = agent2_trail[-1]
    
    occupied = set(agent1_trail) | set(agent2_trail)
    
    # BFS from agent 1 head
    visited = set()
    queue = deque([a1_head])
    visited.add(a1_head)
    
    while queue:
        x, y = queue.popleft()
        
        # If we reach agent 2 head, they're connected
        if (x, y) == a2_head:
            return False, 0, 0
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx = (x + dx) % BOARD_WIDTH
            ny = (y + dy) % BOARD_HEIGHT
            npos = (nx, ny)
            
            if npos in occupied or npos in visited:
                continue
            
            visited.add(npos)
            queue.append(npos)
    
    # Separated! Count component sizes
    p1_component = len(visited)
    total_empty = BOARD_WIDTH * BOARD_HEIGHT - len(occupied)
    p2_component = total_empty - p1_component
    
    return True, p1_component, p2_component

class HeuristicCollector:
    """Collects heuristic data during a game"""
    
    def __init__(self):
        self.snapshots = []
        self.separation_turn = None
        self.final_p1_space = None
        self.final_p2_space = None
        self.game_result = None
    
    def record_turn(self, game):
        """Record heuristic values for current turn"""
        if not game.agent1.alive or not game.agent2.alive:
            return
        
        a1_trail = game.agent1.get_trail_positions()
        a2_trail = game.agent2.get_trail_positions()
        
        # Check if separated
        separated, p1_comp, p2_comp = are_separated(a1_trail, a2_trail, game.board.grid)
        
        if separated and self.separation_turn is None:
            self.separation_turn = game.turns
            self.final_p1_space = p1_comp
            self.final_p2_space = p2_comp
            print(f"  [Separation detected at turn {game.turns}: P1={p1_comp}, P2={p2_comp}]")
        
        # Only collect data before separation
        if not separated:
            # Calculate all heuristics
            p1_territory, p2_territory, control = calculate_voronoi_control(
                a1_trail, a2_trail, game.board.grid
            )
            
            # Count valid moves (freedom)
            p1_head = a1_trail[-1]
            p2_head = a2_trail[-1]
            occupied = set(a1_trail) | set(a2_trail)
            
            p1_valid_moves = 0
            p2_valid_moves = 0
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                p1_next = ((p1_head[0] + dx) % BOARD_WIDTH, (p1_head[1] + dy) % BOARD_HEIGHT)
                if p1_next not in occupied:
                    p1_valid_moves += 1
                
                p2_next = ((p2_head[0] + dx) % BOARD_WIDTH, (p2_head[1] + dy) % BOARD_HEIGHT)
                if p2_next not in occupied:
                    p2_valid_moves += 1
            
            # Reachable space
            p1_reachable = count_reachable_space(a1_trail, game.board.grid, 15)
            p2_reachable = count_reachable_space(a2_trail, game.board.grid, 15)
            
            # Edge bonus
            p1_edge = calculate_edge_bonus(control, 1)
            p2_edge = calculate_edge_bonus(control, 2)
            
            # Head distance
            head_dist = torus_distance(p1_head, p2_head)
            
            snapshot = {
                'turn': game.turns,
                'territory_diff': p1_territory - p2_territory,
                'freedom_diff': p1_valid_moves - p2_valid_moves,
                'reachable_diff': p1_reachable - p2_reachable,
                'boost_diff': game.agent1.boosts_remaining - game.agent2.boosts_remaining,
                'edge_diff': p1_edge - p2_edge,
                'head_distance': head_dist,
                'p1_valid_moves': p1_valid_moves,
                'p2_valid_moves': p2_valid_moves,
                'p1_territory': p1_territory,
                'p2_territory': p2_territory,
            }
            
            self.snapshots.append(snapshot)
    
    def finalize(self, game_result):
        """Set final game result and calculate outcome scores"""
        self.game_result = game_result
        
        if game_result == GameResult.AGENT1_WIN:
            outcome_score = 100
        elif game_result == GameResult.AGENT2_WIN:
            outcome_score = -100
        else:
            outcome_score = 0
        
        for snap in self.snapshots:
            snap['outcome_score'] = outcome_score
            snap['p1_won'] = 1 if game_result == GameResult.AGENT1_WIN else 0
    
    def has_valid_data(self):
        """Check if we collected useful data"""
        return len(self.snapshots) > 5

class Judge:
    def __init__(self, p1_url, p2_url):
        self.p1_url = p1_url
        self.p2_url = p2_url
        self.game = Game()
        self.p1_agent = None
        self.p2_agent = None
        self.collector = HeuristicCollector()

    def check_latency(self):
        """Check latency for both players"""
        try:
            start = time.time()
            response = requests.get(self.p1_url, timeout=TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                self.p1_agent = PlayerAgent(
                    data.get("participant", "P1"),
                    data.get("agent_name", "Agent1")
                )
                self.p1_agent.latency = time.time() - start
            else:
                return False
        except:
            return False

        try:
            start = time.time()
            response = requests.get(self.p2_url, timeout=TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                self.p2_agent = PlayerAgent(
                    data.get("participant", "P2"),
                    data.get("agent_name", "Agent2")
                )
                self.p2_agent.latency = time.time() - start
            else:
                return False
        except:
            return False

        return True

    def send_state(self, player_num):
        """Send game state to player"""
        url = self.p1_url if player_num == 1 else self.p2_url
        
        state_data = {
            "board": self.game.board.grid,
            "agent1_trail": self.game.agent1.get_trail_positions(),
            "agent2_trail": self.game.agent2.get_trail_positions(),
            "agent1_length": self.game.agent1.length,
            "agent2_length": self.game.agent2.length,
            "agent1_alive": self.game.agent1.alive,
            "agent2_alive": self.game.agent2.alive,
            "agent1_boosts": self.game.agent1.boosts_remaining,
            "agent2_boosts": self.game.agent2.boosts_remaining,
            "turn_count": self.game.turns,
            "player_number": player_num,
        }
        
        try:
            response = requests.post(f"{url}/send-state", json=state_data, timeout=TIMEOUT)
            return response.status_code == 200
        except:
            return False

    def get_move(self, player_num):
        """Get move from player"""
        url = self.p1_url if player_num == 1 else self.p2_url
        
        params = {
            "player_number": player_num,
            "attempt_number": 1,
            "random_moves_left": 0,
            "turn_count": self.game.turns,
        }
        
        try:
            response = requests.get(f"{url}/send-move", params=params, timeout=TIMEOUT)
            if response.status_code == 200:
                return response.json().get('move')
        except:
            pass
        
        return None

    def end_game(self, result):
        """Notify players of game end"""
        end_data = {
            "board": self.game.board.grid,
            "agent1_trail": self.game.agent1.get_trail_positions(),
            "agent2_trail": self.game.agent2.get_trail_positions(),
            "agent1_length": self.game.agent1.length,
            "agent2_length": self.game.agent2.length,
            "agent1_alive": self.game.agent1.alive,
            "agent2_alive": self.game.agent2.alive,
            "agent1_boosts": self.game.agent1.boosts_remaining,
            "agent2_boosts": self.game.agent2.boosts_remaining,
            "turn_count": self.game.turns,
            "result": result.name if isinstance(result, GameResult) else str(result),
        }
        
        try:
            requests.post(f"{self.p1_url}/end", json=end_data, timeout=TIMEOUT)
            requests.post(f"{self.p2_url}/end", json=end_data, timeout=TIMEOUT)
        except:
            pass

    def run_game(self):
        """Run a complete game and collect data. Returns True if data collected."""
        # Send initial state
        self.send_state(1)
        self.send_state(2)
        
        direction_map = {
            'UP': Direction.UP,
            'DOWN': Direction.DOWN,
            'LEFT': Direction.LEFT,
            'RIGHT': Direction.RIGHT,
        }
        
        # Game loop
        while self.game.turns < 500:
            # Record heuristics BEFORE moves
            self.collector.record_turn(self.game)
            
            # Get moves
            p1_move_str = self.get_move(1)
            p2_move_str = self.get_move(2)
            
            if not p1_move_str or not p2_move_str:
                # Connection failed, abort
                return False
            
            # Parse moves
            p1_parts = p1_move_str.upper().split(':')
            p2_parts = p2_move_str.upper().split(':')
            
            p1_dir = direction_map.get(p1_parts[0], Direction.RIGHT)
            p2_dir = direction_map.get(p2_parts[0], Direction.LEFT)
            
            p1_boost = len(p1_parts) > 1 and p1_parts[1] == 'BOOST'
            p2_boost = len(p2_parts) > 1 and p2_parts[1] == 'BOOST'
            
            # Execute move
            result = self.game.step(p1_dir, p2_dir, p1_boost, p2_boost)
            
            # Send updated state
            self.send_state(1)
            self.send_state(2)
            
            if result:
                self.end_game(result)
                self.collector.finalize(result)
                return self.collector.has_valid_data()
        
        # Max turns reached
        self.end_game(GameResult.DRAW)
        self.collector.finalize(GameResult.DRAW)
        return self.collector.has_valid_data()

def write_snapshots_to_csv(snapshots, filename, mode='a'):
    """Append snapshots to CSV file"""
    if not snapshots:
        return
    
    file_exists = os.path.exists(filename)
    
    with open(filename, mode, newline='') as f:
        fieldnames = [
            'turn', 'territory_diff', 'freedom_diff', 'reachable_diff',
            'boost_diff', 'edge_diff', 'head_distance', 
            'p1_valid_moves', 'p2_valid_moves',
            'p1_territory', 'p2_territory', 'outcome_score'
        ]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists or mode == 'w':
            writer.writeheader()
        
        for snap in snapshots:
            writer.writerow(snap)

def main():
    if len(sys.argv) < 3:
        print("Usage: python heuristic_data_collector.py <url1> <url2> [num_games]")
        print("Example: python heuristic_data_collector.py http://localhost:5008 http://localhost:5009 100")
        sys.exit(1)
    
    url1 = sys.argv[1]
    url2 = sys.argv[2]
    num_games = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    output_file = "heuristic_training_data.csv"
    
    # Clear file
    with open(output_file, 'w') as f:
        f.write("")
    
    print(f"Collecting data from {num_games} games...")
    print(f"Player 1: {url1}")
    print(f"Player 2: {url2}")
    print(f"Output: {output_file}\n")
    
    games_with_data = 0
    total_snapshots = 0
    
    for game_num in range(num_games):
        print(f"Game {game_num + 1}/{num_games}...", end=" ")
        
        # Alternate who goes first
        if game_num % 2 == 0:
            judge = Judge(url1, url2)
        else:
            judge = Judge(url2, url1)
        
        # Wait for agents
        time.sleep(0.5)
        
        if not judge.check_latency():
            print("Failed to connect")
            continue
        
        # Run game
        success = judge.run_game()
        
        if success:
            # Write data
            write_snapshots_to_csv(judge.collector.snapshots, output_file, 'a')
            games_with_data += 1
            total_snapshots += len(judge.collector.snapshots)
            print(f"✓ ({len(judge.collector.snapshots)} snapshots, sep turn {judge.collector.separation_turn})")
        else:
            print("✗ (no separation or error)")
        
        # Small delay between games
        time.sleep(0.2)
    
    print(f"\n{'='*60}")
    print(f"Data collection complete!")
    print(f"Games with valid data: {games_with_data}/{num_games}")
    print(f"Total snapshots: {total_snapshots}")
    print(f"Output file: {output_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
