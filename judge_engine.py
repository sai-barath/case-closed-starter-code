import requests
import sys
import time
import os
from case_closed_game import Game, Direction, GameResult
import random

class Colors:
    RED = '\033[91m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

TIMEOUT = 4

class BotInfo:
    def __init__(self, participant, agent_name):
        self.participant = participant
        self.agent_name = agent_name
        self.latency = 0.0

class Judge:
    def __init__(self, red_url, blue_url):
        self.red_url = red_url
        self.blue_url = blue_url
        self.red_info = None
        self.blue_info = None
        self.game = Game()
        self.game_str = ""
    
    def connect_to_bots(self):
        try:
            start = time.time()
            resp = requests.get(self.red_url, timeout=TIMEOUT)
            if resp.status_code != 200:
                return False
            data = resp.json()
            self.red_info = BotInfo(data.get("participant", "RedPlayer"), data.get("agent_name", "RedAgent"))
            self.red_info.latency = time.time() - start
        except (requests.RequestException, requests.Timeout):
            return False
        
        try:
            start = time.time()
            resp = requests.get(self.blue_url, timeout=TIMEOUT)
            if resp.status_code != 200:
                return False
            data = resp.json()
            self.blue_info = BotInfo(data.get("participant", "BluePlayer"), data.get("agent_name", "BlueAgent"))
            self.blue_info.latency = time.time() - start
        except (requests.RequestException, requests.Timeout):
            return False
        
        return True
    
    def send_state_to_bot(self, color):
        url = self.red_url if color == "RED" else self.blue_url
        player_num = 1 if color == "RED" else 2
        
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
            resp = requests.post(f"{url}/send-state", json=state_data, timeout=TIMEOUT)
            return resp.status_code == 200
        except (requests.RequestException, requests.Timeout):
            return False
    
    def request_move_from_bot(self, color, attempt_num, random_left):
        url = self.red_url if color == "RED" else self.blue_url
        player_num = 1 if color == "RED" else 2
        
        params = {
            "player_number": player_num,
            "attempt_number": attempt_num,
            "random_moves_left": random_left,
            "turn_count": self.game.turns,
        }
        
        try:
            start = time.time()
            resp = requests.get(f"{url}/send-move", params=params, timeout=TIMEOUT)
            elapsed = time.time() - start
            
            if color == "RED":
                self.red_info.latency = elapsed
            else:
                self.blue_info.latency = elapsed
            
            if resp.status_code == 200:
                return resp.json().get('move')
            return None
        except (requests.RequestException, requests.Timeout):
            return None
    
    def parse_and_validate_move(self, move_str, color):
        info = self.red_info if color == "RED" else self.blue_info
        agent = self.game.agent1 if color == "RED" else self.game.agent2
        
        if not isinstance(move_str, str):
            print(f"Invalid move format from {color} ({info.agent_name}): not a string")
            return None
        
        parts = move_str.upper().split(':')
        dir_str = parts[0]
        use_boost = len(parts) > 1 and parts[1] == 'BOOST'
        
        dir_map = {'UP': Direction.UP, 'DOWN': Direction.DOWN, 'LEFT': Direction.LEFT, 'RIGHT': Direction.RIGHT}
        
        if dir_str not in dir_map:
            print(f"Invalid direction from {color} ({info.agent_name}): {dir_str}")
            return None
        
        direction = dir_map[dir_str]
        
        cur_dx, cur_dy = agent.direction.value
        req_dx, req_dy = direction.value
        if (req_dx, req_dy) == (-cur_dx, -cur_dy):
            print(f"{color} ({info.agent_name}) sent opposite direction, using current direction instead")
            direction = agent.direction
            dir_str = {Direction.UP: 'UP', Direction.DOWN: 'DOWN', Direction.LEFT: 'LEFT', Direction.RIGHT: 'RIGHT'}[direction]
        
        return (direction, dir_str, use_boost)
    
    def display_move(self, color, dir_str, use_boost, is_random):
        info = self.red_info if color == "RED" else self.blue_info
        col = Colors.RED if color == "RED" else Colors.BLUE
        boost_txt = f" {Colors.YELLOW}⚡BOOST{Colors.RESET}" if use_boost else ""
        rand_txt = f" {Colors.DIM}(random){Colors.RESET}" if is_random else ""
        print(f"{col}{color} ({info.agent_name}){Colors.RESET}: {Colors.BOLD}{dir_str}{Colors.RESET}{boost_txt}{rand_txt}")
    
    def display_board(self):
        agent1_trail = set(self.game.agent1.get_trail_positions())
        agent2_trail = set(self.game.agent2.get_trail_positions())
        agent1_head = self.game.agent1.trail[-1] if self.game.agent1.alive else None
        agent2_head = self.game.agent2.trail[-1] if self.game.agent2.alive else None
        
        print(f"\n{Colors.CYAN}╔{'═' * (self.game.board.width * 2)}╗{Colors.RESET}")
        
        for y in range(self.game.board.height):
            print(f"{Colors.CYAN}║{Colors.RESET}", end='')
            for x in range(self.game.board.width):
                pos = (x, y)
                if pos == agent1_head:
                    print(f"{Colors.RED}{Colors.BOLD}●{Colors.RESET} ", end='')
                elif pos == agent2_head:
                    print(f"{Colors.BLUE}{Colors.BOLD}●{Colors.RESET} ", end='')
                elif pos in agent1_trail:
                    print(f"{Colors.RED}◼{Colors.RESET} ", end='')
                elif pos in agent2_trail:
                    print(f"{Colors.BLUE}◼{Colors.RESET} ", end='')
                else:
                    print(f"{Colors.DIM}·{Colors.RESET} ", end='')
            print(f"{Colors.CYAN}║{Colors.RESET}")
        
        print(f"{Colors.CYAN}╚{'═' * (self.game.board.width * 2)}╝{Colors.RESET}\n")
        
        print(f"{Colors.RED}RED ({self.red_info.agent_name}){Colors.RESET}: Trail={self.game.agent1.length} | Alive={self.game.agent1.alive} | Boosts={self.game.agent1.boosts_remaining}")
        print(f"{Colors.BLUE}BLUE ({self.blue_info.agent_name}){Colors.RESET}: Trail={self.game.agent2.length} | Alive={self.game.agent2.alive} | Boosts={self.game.agent2.boosts_remaining}")
    
    def notify_end(self, result):
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
            requests.post(f"{self.red_url}/end", json=end_data, timeout=TIMEOUT)
            requests.post(f"{self.blue_url}/end", json=end_data, timeout=TIMEOUT)
        except:
            pass
        
        if isinstance(result, GameResult):
            if result == GameResult.AGENT1_WIN:
                print(f"\n{Colors.GREEN}{Colors.BOLD}🏆 WINNER: RED ({self.red_info.agent_name}) 🏆{Colors.RESET}\n")
            elif result == GameResult.AGENT2_WIN:
                print(f"\n{Colors.GREEN}{Colors.BOLD}🏆 WINNER: BLUE ({self.blue_info.agent_name}) 🏆{Colors.RESET}\n")
            else:
                print(f"\n{Colors.YELLOW}{Colors.BOLD}⚖️  DRAW ⚖️{Colors.RESET}\n")
        
        return result

def get_random_move():
    return random.choice([Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT])

def run_game(judge):
    judge.game.reset()
    judge.game_str = ""
    
    print(f"{Colors.RED}RED ({judge.red_info.agent_name}){Colors.RESET} = game.agent1 at (1,2)")
    print(f"{Colors.BLUE}BLUE ({judge.blue_info.agent_name}){Colors.RESET} = game.agent2 at (17,15)\n")
    
    if not judge.send_state_to_bot("RED") or not judge.send_state_to_bot("BLUE"):
        print(f"{Colors.RED}Failed to send initial state{Colors.RESET}")
        return None
    
    red_random_left = 5
    blue_random_left = 5
    
    while True:
        print(f"\n{Colors.BOLD}{'─' * 60}")
        print(f"Turn {judge.game.turns + 1}")
        print(f"{'─' * 60}{Colors.RESET}")
        
        red_dir = None
        red_dir_str = None
        red_boost = False
        
        for attempt in range(1, 3):
            move_str = judge.request_move_from_bot("RED", attempt, red_random_left)
            if move_str:
                parsed = judge.parse_and_validate_move(move_str, "RED")
                if parsed:
                    red_dir, red_dir_str, red_boost = parsed
                    judge.display_move("RED", red_dir_str, red_boost, False)
                    break
                else:
                    print(f"{Colors.RED}RED forfeited (invalid move){Colors.RESET}")
                    return judge.notify_end(GameResult.AGENT2_WIN)
            if attempt < 2:
                print(f"{Colors.DIM}  RED attempt {attempt} failed, retrying...{Colors.RESET}")
        
        if not red_dir:
            if red_random_left > 0:
                print(f"{Colors.YELLOW}Using random move for RED ({red_random_left} left){Colors.RESET}")
                red_dir = get_random_move()
                red_dir_str = {Direction.UP: 'UP', Direction.DOWN: 'DOWN', Direction.LEFT: 'LEFT', Direction.RIGHT: 'RIGHT'}[red_dir]
                red_boost = False
                red_random_left -= 1
                judge.display_move("RED", red_dir_str, red_boost, True)
            else:
                print(f"{Colors.RED}RED out of random moves, forfeiting{Colors.RESET}")
                return judge.notify_end(GameResult.AGENT2_WIN)
        
        blue_dir = None
        blue_dir_str = None
        blue_boost = False
        
        for attempt in range(1, 3):
            move_str = judge.request_move_from_bot("BLUE", attempt, blue_random_left)
            if move_str:
                parsed = judge.parse_and_validate_move(move_str, "BLUE")
                if parsed:
                    blue_dir, blue_dir_str, blue_boost = parsed
                    judge.display_move("BLUE", blue_dir_str, blue_boost, False)
                    break
                else:
                    print(f"{Colors.RED}BLUE forfeited (invalid move){Colors.RESET}")
                    return judge.notify_end(GameResult.AGENT1_WIN)
            if attempt < 2:
                print(f"{Colors.DIM}  BLUE attempt {attempt} failed, retrying...{Colors.RESET}")
        
        if not blue_dir:
            if blue_random_left > 0:
                print(f"{Colors.YELLOW}Using random move for BLUE ({blue_random_left} left){Colors.RESET}")
                blue_dir = get_random_move()
                blue_dir_str = {Direction.UP: 'UP', Direction.DOWN: 'DOWN', Direction.LEFT: 'LEFT', Direction.RIGHT: 'RIGHT'}[blue_dir]
                blue_boost = False
                blue_random_left -= 1
                judge.display_move("BLUE", blue_dir_str, blue_boost, True)
            else:
                print(f"{Colors.RED}BLUE out of random moves, forfeiting{Colors.RESET}")
                return judge.notify_end(GameResult.AGENT1_WIN)
        
        result = judge.game.step(red_dir, blue_dir, red_boost, blue_boost)
        
        judge.send_state_to_bot("RED")
        judge.send_state_to_bot("BLUE")
        
        judge.display_board()
        
        if result is not None:
            judge.notify_end(result)
            print(f"{Colors.DIM}Game String: {judge.game_str}{Colors.RESET}")
            return result
        
        if judge.game.turns >= 500:
            print(f"{Colors.YELLOW}Max turns reached{Colors.RESET}")
            result = judge.notify_end(GameResult.DRAW)
            print(f"{Colors.DIM}Game String: {judge.game_str}{Colors.RESET}")
            return result
        
        time.sleep(0.1)

def main():
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║          CASE CLOSED - JUDGE ENGINE v3.0                  ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}\n")
    
    PLAYER1_URL = os.getenv("PLAYER1_URL", "http://localhost:5008")
    PLAYER2_URL = os.getenv("PLAYER2_URL", "http://localhost:5009")
    NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "1"))
    
    print(f"Connecting to Player1={PLAYER1_URL} and Player2={PLAYER2_URL}...")
    
    temp_judge = Judge(PLAYER1_URL, PLAYER2_URL)
    if not temp_judge.connect_to_bots():
        print(f"{Colors.RED}Failed to connect{Colors.RESET}")
        return
    
    player1_name = temp_judge.red_info.agent_name
    player1_participant = temp_judge.red_info.participant
    player2_name = temp_judge.blue_info.agent_name
    player2_participant = temp_judge.blue_info.participant
    
    print(f"\n{Colors.GREEN}✓ Connected!{Colors.RESET}")
    print(f"Player 1: {player1_name} ({player1_participant})")
    print(f"Player 2: {player2_name} ({player2_participant})\n")
    
    player1_wins = 0
    player2_wins = 0
    draws = 0
    
    total_games = NUM_ROUNDS * 2
    for round_num in range(NUM_ROUNDS):
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 60}")
        print(f"          GAME {round_num * 2 + 1} of {total_games}")
        print(f"{'=' * 60}{Colors.RESET}\n")
        
        judge1 = Judge(PLAYER1_URL, PLAYER2_URL)
        judge1.connect_to_bots()
        print(f"Player 1 ({player1_name}) = RED, Player 2 ({player2_name}) = BLUE")
        
        result1 = run_game(judge1)
        
        if result1 == GameResult.AGENT1_WIN:
            player1_wins += 1
        elif result1 == GameResult.AGENT2_WIN:
            player2_wins += 1
        else:
            draws += 1
        
        print(f"\n{Colors.CYAN}{Colors.BOLD}Score:{Colors.RESET}")
        print(f"Player 1 ({player1_name}): {player1_wins} | Player 2 ({player2_name}): {player2_wins} | Draws: {draws}\n")
        
        time.sleep(2)
        
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 60}")
        print(f"          GAME {round_num * 2 + 2} of {total_games}")
        print(f"{'=' * 60}{Colors.RESET}\n")
        
        judge2 = Judge(PLAYER2_URL, PLAYER1_URL)
        judge2.connect_to_bots()
        print(f"Player 1 ({player1_name}) = BLUE, Player 2 ({player2_name}) = RED")
        
        result2 = run_game(judge2)
        
        if result2 == GameResult.AGENT2_WIN:
            player1_wins += 1
        elif result2 == GameResult.AGENT1_WIN:
            player2_wins += 1
        else:
            draws += 1
        
        print(f"\n{Colors.CYAN}{Colors.BOLD}Score:{Colors.RESET}")
        print(f"Player 1 ({player1_name}): {player1_wins} | Player 2 ({player2_name}): {player2_wins} | Draws: {draws}\n")
        
        if round_num < NUM_ROUNDS - 1:
            time.sleep(2)
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                    FINAL RESULTS                           ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}\n")
    
    print(f"{Colors.BOLD}Total Games: {total_games}{Colors.RESET}")
    print(f"Player 1 ({player1_name}): {player1_wins}")
    print(f"Player 2 ({player2_name}): {player2_wins}")
    print(f"Draws: {draws}\n")
    
    if player1_wins > player2_wins:
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 OVERALL WINNER: Player 1 ({player1_name}) 🎉{Colors.RESET}\n")
    elif player2_wins > player1_wins:
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 OVERALL WINNER: Player 2 ({player2_name}) 🎉{Colors.RESET}\n")
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚖️  TIE ⚖️{Colors.RESET}\n")

if __name__ == "__main__":
    main()
    sys.exit(0)
