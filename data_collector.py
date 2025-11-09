#!/usr/bin/env python3
import subprocess
import threading
import time
import sys
import os
import traceback
from collections import deque
from case_closed_game import Game, Direction, GameResult

BOARD_HEIGHT = 18
BOARD_WIDTH = 20

class BotProcess:
    def __init__(self, binary_path, port):
        self.binary_path = binary_path
        self.port = port
        self.process = None
        
    def start(self):
        env = os.environ.copy()
        env['PORT'] = str(self.port)
        self.process = subprocess.Popen(
            [self.binary_path],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(0.5)
        
    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=2)

def calculate_voronoi_metrics(agent1_trail, agent2_trail):
    visited = {}
    queue = deque()
    
    if not agent1_trail or not agent2_trail:
        return 0, 0, 0, 0
    
    a1_head = agent1_trail[-1]
    a2_head = agent2_trail[-1]
    
    occupied = set(agent1_trail) | set(agent2_trail)
    
    queue.append((a1_head, 1))
    queue.append((a2_head, 2))
    visited[a1_head] = 1
    visited[a2_head] = 2
    
    while queue:
        (x, y), owner = queue.popleft()
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx = (x + dx) % BOARD_WIDTH
            ny = (y + dy) % BOARD_HEIGHT
            npos = (nx, ny)
            
            if npos in occupied or npos in visited:
                continue
                
            visited[npos] = owner
            queue.append((npos, owner))
    
    p1_nodes = sum(1 for v in visited.values() if v == 1)
    p2_nodes = sum(1 for v in visited.values() if v == 2)
    
    p1_edges = 0
    p2_edges = 0
    
    for (x, y), owner in visited.items():
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx = (x + dx) % BOARD_WIDTH
            ny = (y + dy) % BOARD_HEIGHT
            npos = (nx, ny)
            
            if npos not in occupied and npos not in visited:
                if owner == 1:
                    p1_edges += 1
                elif owner == 2:
                    p2_edges += 1
    
    return p1_nodes, p2_nodes, p1_edges, p2_edges

def are_separated(agent1_trail, agent2_trail):
    if not agent1_trail or not agent2_trail:
        return False, None, None
    
    a1_head = agent1_trail[-1]
    a2_head = agent2_trail[-1]
    
    occupied = set(agent1_trail) | set(agent2_trail)
    
    visited = set()
    queue = deque([a1_head])
    visited.add(a1_head)
    
    while queue:
        x, y = queue.popleft()
        
        if (x, y) == a2_head:
            return False, None, None
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx = (x + dx) % BOARD_WIDTH
            ny = (y + dy) % BOARD_HEIGHT
            npos = (nx, ny)
            
            if npos in occupied or npos in visited:
                continue
            
            visited.add(npos)
            queue.append(npos)
    
    p1_component = len(visited)
    p2_component = (BOARD_WIDTH * BOARD_HEIGHT) - len(occupied) - p1_component
    
    return True, p1_component, p2_component

def run_game_silent(bot1_port, bot2_port, game_id, output_file):
    import requests
    
    try:
        bot1_url = f"http://localhost:{bot1_port}"
        bot2_url = f"http://localhost:{bot2_port}"
        
        game = Game()
        
        try:
            requests.get(bot1_url, timeout=2)
            requests.get(bot2_url, timeout=2)
        except Exception as e:
            print(f"Failed to connect to bots: {e}")
            traceback.print_exc()
            return
        
        game_states = []
        separation_turn = None
        final_p1_space = None
        final_p2_space = None
        
        for turn in range(500):
            a1_trail = game.agent1.get_trail_positions()
            a2_trail = game.agent2.get_trail_positions()
            
            state_data = {
                "board": game.board.grid,
                "agent1_trail": a1_trail,
                "agent2_trail": a2_trail,
                "agent1_length": game.agent1.length,
                "agent2_length": game.agent2.length,
                "agent1_alive": game.agent1.alive,
                "agent2_alive": game.agent2.alive,
                "agent1_boosts": game.agent1.boosts_remaining,
                "agent2_boosts": game.agent2.boosts_remaining,
                "turn_count": turn,
                "player_number": 1,
            }
            
            try:
                requests.post(f"{bot1_url}/send-state", json=state_data, timeout=2)
                state_data["player_number"] = 2
                requests.post(f"{bot2_url}/send-state", json=state_data, timeout=2)
            except Exception as e:
                print(f"Failed to send state: {e}")
                pass
            
            p1_nodes, p2_nodes, p1_edges, p2_edges = calculate_voronoi_metrics(a1_trail, a2_trail)
            
            separated, p1_comp, p2_comp = are_separated(a1_trail, a2_trail)
            
            if separated and separation_turn is None:
                separation_turn = turn
                final_p1_space = p1_comp
                final_p2_space = p2_comp
            
            if separation_turn is None:
                game_states.append({
                    'turn': turn,
                    'p1_nodes': p1_nodes,
                    'p2_nodes': p2_nodes,
                    'p1_edges': p1_edges,
                    'p2_edges': p2_edges,
                })
            
            try:
                params = {"player_number": 1, "attempt_number": 1, "random_moves_left": 5, "turn_count": turn}
                r1 = requests.get(f"{bot1_url}/send-move", params=params, timeout=2)
                move1 = r1.json().get('move', 'RIGHT')
                
                params["player_number"] = 2
                r2 = requests.get(f"{bot2_url}/send-move", params=params, timeout=2)
                move2 = r2.json().get('move', 'LEFT')
            except Exception as e:
                print(f"Failed to get moves: {e}")
                break
            
            move1_parts = move1.upper().split(':')
            dir1_str = move1_parts[0]
            boost1 = len(move1_parts) > 1 and move1_parts[1] == 'BOOST'
            
            move2_parts = move2.upper().split(':')
            dir2_str = move2_parts[0]
            boost2 = len(move2_parts) > 1 and move2_parts[1] == 'BOOST'
            
            dir_map = {'UP': Direction.UP, 'DOWN': Direction.DOWN, 'LEFT': Direction.LEFT, 'RIGHT': Direction.RIGHT}
            dir1 = dir_map.get(dir1_str, Direction.RIGHT)
            dir2 = dir_map.get(dir2_str, Direction.LEFT)
            
            result = game.step(dir1, dir2, boost1, boost2)
            
            if result:
                break
        
        if separation_turn is not None and final_p1_space is not None and final_p2_space is not None and len(game_states) > 0:
            print(f"Game {game_id}: Separation at turn {separation_turn}, writing {len(game_states)} states")
            with open(output_file, 'a') as f:
                for state in game_states:
                    node_diff = state['p1_nodes'] - state['p2_nodes']
                    edge_diff = state['p1_edges'] - state['p2_edges']
                    space_diff = final_p1_space - final_p2_space
                    
                    f.write(f"{node_diff},{edge_diff},{space_diff}\n")
        else:
            print(f"Game {game_id}: No separation occurred or no data collected (sep_turn={separation_turn}, states={len(game_states)})")
    
    except Exception as e:
        print(f"Error in run_game_silent: {e}")
        traceback.print_exc()

def worker(binary1, binary2, port1, port2, output_file, worker_id):
    try:
        bot1 = BotProcess(binary1, port1)
        bot2 = BotProcess(binary2, port2)
        
        bot1.start()
        bot2.start()
        
        time.sleep(2)
        
        game_count = 0
        try:
            while True:
                game_count += 1
                print(f"Worker {worker_id}: Starting game {game_count}")
                run_game_silent(port1, port2, game_count, output_file)
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            bot1.stop()
            bot2.stop()
    except Exception as e:
        print(f"Worker {worker_id} error: {e}")
        traceback.print_exc()

def main():
    if len(sys.argv) < 3:
        print("Usage: python data_collector.py <binary1_path> <binary2_path> [num_workers]")
        sys.exit(1)
    
    binary1 = sys.argv[1]
    binary2 = sys.argv[2]
    num_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    
    output_file = "training_data.csv"
    
    with open(output_file, 'w') as f:
        f.write("node_diff,edge_diff,final_space_diff\n")
    
    print(f"Starting {num_workers} workers. Press Ctrl+C to stop.")
    print(f"Data will be saved to {output_file}")
    
    base_port = 10000
    
    threads = []
    for i in range(num_workers):
        port1 = base_port + (i * 2)
        port2 = base_port + (i * 2) + 1
        
        t = threading.Thread(
            target=worker,
            args=(binary1, binary2, port1, port2, output_file, i)
        )
        t.daemon = True
        t.start()
        threads.append(t)
        time.sleep(0.5)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping workers...")
        print(f"Data saved to {output_file}")

if __name__ == "__main__":
    main()
