#!/usr/bin/env python3

import subprocess
import time
import sys
import os
import glob

def check_port_available(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def wait_for_agent(port, timeout=5):
    import requests
    url = f"http://localhost:{port}"
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.1)
    return False

def run_game(agent1_exe, agent2_exe, port1, port2):
    agent1_process = None
    agent2_process = None
    
    try:
        if not check_port_available(port1) or not check_port_available(port2):
            print(f"Ports {port1} or {port2} already in use")
            return None
        
        env1 = os.environ.copy()
        env1["PORT"] = str(port1)
        env1["AGENT_NAME"] = f"{os.path.basename(agent1_exe)}-P1"
        env1["PARTICIPANT"] = os.path.basename(agent1_exe)
        
        agent1_process = subprocess.Popen(
            [agent1_exe],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env1
        )
        
        env2 = os.environ.copy()
        env2["PORT"] = str(port2)
        env2["AGENT_NAME"] = f"{os.path.basename(agent2_exe)}-P2"
        env2["PARTICIPANT"] = os.path.basename(agent2_exe)
        
        agent2_process = subprocess.Popen(
            [agent2_exe],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env2
        )
        
        if not wait_for_agent(port1) or not wait_for_agent(port2):
            print(f"Agents failed to start")
            return None
        
        env_judge = os.environ.copy()
        env_judge["PLAYER1_URL"] = f"http://localhost:{port1}"
        env_judge["PLAYER2_URL"] = f"http://localhost:{port2}"
        
        result = subprocess.run(
            ["uv", "run", "judge_engine.py"],
            capture_output=True,
            text=True,
            env=env_judge
        )
        
        return result.stdout
        
    except Exception as e:
        print(f"Error: {e}")
        return None
    finally:
        if agent1_process:
            agent1_process.terminate()
            try:
                agent1_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                agent1_process.kill()
                agent1_process.wait()
        if agent2_process:
            agent2_process.terminate()
            try:
                agent2_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                agent2_process.kill()
                agent2_process.wait()
        time.sleep(0.5)

def parse_wins(output):
    lines = output.strip().split('\n')
    url1_wins = 0
    url2_wins = 0
    ties = 0
    for line in lines[-10:]:
        if "URL1" in line and "Wins:" in line:
            url1_wins = int(line.split("Wins:")[1].strip())
        elif "URL2" in line and "Wins:" in line:
            url2_wins = int(line.split("Wins:")[1].strip())
        elif "Ties:" in line:
            ties = int(line.split("Ties:")[1].strip())
    return url1_wins, url2_wins, ties

if __name__ == "__main__":
    current_exe = "./steamroller0/steamroller"
    prev_dir = "./prev"
    
    if not os.path.exists(current_exe):
        print(f"Current agent not found: {current_exe}")
        print("Run: cd steamroller0 && go build -o steamroller")
        sys.exit(1)
    
    prev_exes = sorted(glob.glob(f"{prev_dir}/*"))
    if not prev_exes:
        print(f"No previous agents found in {prev_dir}")
        sys.exit(1)
    
    print(f"Testing {current_exe} against {len(prev_exes)} previous versions\n")
    print("="*60)
    
    total_wins = 0
    total_losses = 0
    total_ties = 0
    
    for prev_exe in prev_exes:
        prev_name = os.path.basename(prev_exe)
        print(f"\n{prev_name}:")
        
        output = run_game(current_exe, prev_exe, 10001, 10002)
        
        if output is None:
            print(f"  Failed to run game")
            continue
        
        wins, losses, ties = parse_wins(output)
        total_wins += wins
        total_losses += losses
        total_ties += ties
        
        print(f"  Current: {wins} wins | {prev_name}: {losses} wins | Ties: {ties}")
    
    print("\n" + "="*60)
    print(f"OVERALL STATS")
    print("="*60)
    print(f"Total Wins: {total_wins}")
    print(f"Total Losses: {total_losses}")
    print(f"Total Ties: {total_ties}")
    print(f"Win Rate: {total_wins}/{total_wins + total_losses + total_ties}")
    print("="*60)
    
    sys.exit(0)
