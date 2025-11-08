import requests
import sys
import time
import os
from case_closed_game import Game, Direction, GameResult
import random

# ANSI color codes for pretty output
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

class RandomPlayer:
    def __init__(self, player_id=1):
        self.player_id = player_id
    
    def get_possible_moves(self):
        """Returns list of all possible directions for agent."""
        return [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
        
    def get_best_move(self):
        """Returns a random valid direction."""
        possible_moves = self.get_possible_moves()
        return random.choice(possible_moves)

TIMEOUT = 4  # time for each move

class PlayerAgent:
    def __init__(self, participant, agent_name):
        self.participant = participant
        self.agent_name = agent_name
        self.latency = None

class Judge:
    def __init__(self, p1_url, p2_url, swap_positions=False):
        self.p1_url = p1_url
        self.p2_url = p2_url
        self.game = Game()
        self.p1_agent = None
        self.p2_agent = None
        self.game_str = ""  # Track game moves as string
        self.swap_positions = swap_positions  # If True, swap starting positions

    def check_latency(self):
        """Check latency for both players and create their agents"""
        # Check P1
        try:
            start_time = time.time()
            response = requests.get(self.p1_url, timeout=TIMEOUT)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                self.p1_agent = PlayerAgent(data.get("participant", "Participant1"), 
                                     data.get("agent_name", "Agent1"))
                self.p1_agent.latency = (end_time - start_time)
            else:
                return False
                
        except (requests.RequestException, requests.Timeout):
            return False

        # Check P2
        try:
            start_time = time.time()
            response = requests.get(self.p2_url, timeout=TIMEOUT)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                self.p2_agent = PlayerAgent(data.get("participant", "Participant2"), 
                                     data.get("agent_name", "Agent2"))
                self.p2_agent.latency = (end_time - start_time)
            else:
                return False
                
        except (requests.RequestException, requests.Timeout):
            return False

        return True

    def send_state(self, player_num):
        """Send current game state to a player via POST"""
        url = self.p1_url if player_num == 1 else self.p2_url
        
        # If positions are swapped, adjust which agent data we send
        if self.swap_positions:
            state_data = {
                "board": self.game.board.grid,
                "agent1_trail": self.game.agent2.get_trail_positions(),
                "agent2_trail": self.game.agent1.get_trail_positions(),
                "agent1_length": self.game.agent2.length,
                "agent2_length": self.game.agent1.length,
                "agent1_alive": self.game.agent2.alive,
                "agent2_alive": self.game.agent1.alive,
                "agent1_boosts": self.game.agent2.boosts_remaining,
                "agent2_boosts": self.game.agent1.boosts_remaining,
                "turn_count": self.game.turns,
                "player_number": player_num,
            }
        else:
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
        except (requests.RequestException, requests.Timeout):
            return False

    def get_move(self, player_num, attempt_number, random_moves_left):
        """Request a move from a player via GET with query parameters"""
        url = self.p1_url if player_num == 1 else self.p2_url
        
        # Build query parameters for GET request
        params = {
            "player_number": player_num,
            "attempt_number": attempt_number,
            "random_moves_left": random_moves_left,
            "turn_count": self.game.turns,
        }
        
        try:
            start_time = time.time()
            response = requests.get(f"{url}/send-move", params=params, timeout=TIMEOUT)
            end_time = time.time()
            
            if player_num == 1:
                self.p1_agent.latency = (end_time - start_time)
            else:
                self.p2_agent.latency = (end_time - start_time)
            
            if response.status_code == 200:
                move = response.json()
                return move.get('move')
            else:
                return None
                
        except (requests.RequestException, requests.Timeout):
            return None

    def display_board_pretty(self):
        """Display the board with pretty colors and blocks"""
        agent1_trail = set(self.game.agent1.get_trail_positions())
        agent2_trail = set(self.game.agent2.get_trail_positions())
        agent1_head = self.game.agent1.trail[-1] if self.game.agent1.alive else None
        agent2_head = self.game.agent2.trail[-1] if self.game.agent2.alive else None
        
        # Top border
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
        
        # Bottom border
        print(f"{Colors.CYAN}╚{'═' * (self.game.board.width * 2)}╝{Colors.RESET}\n")
    
    def end_game(self, result):
        """End the game and notify both players"""
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
            
            if isinstance(result, GameResult):
                if result == GameResult.AGENT1_WIN:
                    print(f"\n{Colors.GREEN}{Colors.BOLD}🏆 WINNER: Agent 1 ({self.p1_agent.agent_name}) 🏆{Colors.RESET}\n")
                elif result == GameResult.AGENT2_WIN:
                    print(f"\n{Colors.GREEN}{Colors.BOLD}🏆 WINNER: Agent 2 ({self.p2_agent.agent_name}) 🏆{Colors.RESET}\n")
                else:
                    print(f"\n{Colors.YELLOW}{Colors.BOLD}⚖️  DRAW ⚖️{Colors.RESET}\n")
            else:
                print(f"\n{Colors.MAGENTA}Game ended: {result}{Colors.RESET}\n")
        except (requests.RequestException, requests.Timeout):
            return False
        
        return result

    def handle_move(self, move, player_num, is_random=False):
        """Validate and execute a move. Returns 'forfeit' or tuple (valid, boost_flag, direction)"""
        
        # Validate move format
        if not isinstance(move, str):
            print(f"Invalid move format by Player {player_num}: move must be a string")
            return "forfeit"
        
        # Parse move - can be "DIRECTION" or "DIRECTION:BOOST"
        move_parts = move.upper().split(':')
        direction_str = move_parts[0]
        use_boost = len(move_parts) > 1 and move_parts[1] == 'BOOST'
        
        # Convert move string to Direction
        direction_map = {
            'UP': Direction.UP,
            'DOWN': Direction.DOWN,
            'LEFT': Direction.LEFT,
            'RIGHT': Direction.RIGHT,
        }
        
        if direction_str not in direction_map:
            print(f"Invalid direction by Player {player_num}: {direction_str}")
            return "forfeit"
        
        direction = direction_map[direction_str]
        
        # Check if move is opposite to current direction (invalid move)
        agent = self.game.agent1 if player_num == 1 else self.game.agent2
        current_dir = agent.direction
        
        # Check if requested direction is opposite to current
        cur_dx, cur_dy = current_dir.value
        req_dx, req_dy = direction.value
        if (req_dx, req_dy) == (-cur_dx, -cur_dy):
            print(f"Player {player_num} attempted invalid move (opposite direction). Using current direction instead.")
            direction = current_dir
            direction_str = {Direction.UP: 'UP', Direction.DOWN: 'DOWN', 
                           Direction.LEFT: 'LEFT', Direction.RIGHT: 'RIGHT'}[direction]
        
        # Pretty print the move with colors
        color = Colors.RED if player_num == 1 else Colors.BLUE
        boost_text = f" {Colors.YELLOW}⚡BOOST{Colors.RESET}" if use_boost else ""
        random_text = f" {Colors.DIM}(random){Colors.RESET}" if is_random else ""
        print(f"{color}Player {player_num}{Colors.RESET}'s move: {Colors.BOLD}{direction_str}{Colors.RESET}{boost_text}{random_text}")
        
        # Record move in game string with improved format
        move_abbrev = {'UP': 'U', 'DOWN': 'D', 'LEFT': 'L', 'RIGHT': 'R'}
        boost_marker = 'B' if use_boost else ''
        random_marker = 'R' if is_random else ''
        self.game_str += f"{player_num}{move_abbrev[direction_str]}{boost_marker}{random_marker}-"
        
        return (True, use_boost, direction)  # Return tuple: (valid, boost_flag, direction)
            

def run_single_game(judge, game_number, total_games):
    """Run a single game and return the result"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 60}")
    print(f"          GAME {game_number} of {total_games}")
    print(f"{'=' * 60}{Colors.RESET}\n")
    
    if judge.swap_positions:
        print(f"{Colors.MAGENTA}Positions SWAPPED this round!{Colors.RESET}")
        print(f"{Colors.BLUE}Player 1 ({judge.p1_agent.agent_name}){Colors.RESET} starts at position (17,15)")
        print(f"{Colors.RED}Player 2 ({judge.p2_agent.agent_name}){Colors.RESET} starts at position (1,2)\n")
    else:
        print(f"{Colors.RED}Player 1 ({judge.p1_agent.agent_name}){Colors.RESET} starts at position (1,2)")
        print(f"{Colors.BLUE}Player 2 ({judge.p2_agent.agent_name}){Colors.RESET} starts at position (17,15)\n")
    
    # Reset game
    judge.game.reset()
    judge.game_str = ""
    
    # Send initial state to both players
    print("Sending initial game state...")
    if not judge.send_state(1) or not judge.send_state(2):
        print(f"{Colors.RED}Failed to send initial state{Colors.RESET}")
        return None

    # Random moves left for p1 and p2
    p1_random = 5
    p2_random = 5

    # Game loop
    while True:
        print(f"\n{Colors.BOLD}{'─' * 60}")
        print(f"Turn {judge.game.turns + 1}")
        print(f"{'─' * 60}{Colors.RESET}")
        
        # Get moves from both players
        p1_move = None
        p2_move = None
        p1_boost = False
        p2_boost = False
        
        # Player 1 move
        for attempt in range(1, 3):  # 2 attempts
            p1_move = judge.get_move(1, attempt, p1_random)
            if p1_move:
                validation = judge.handle_move(p1_move, 1, is_random=False)
                if validation == "forfeit":
                    print(f"{Colors.RED}Player 1 forfeited{Colors.RESET}")
                    return judge.end_game(GameResult.AGENT2_WIN)
                elif validation:
                    p1_boost = validation[1]
                    p1_direction = validation[2]
                    break
            if attempt < 2:
                print(f"{Colors.DIM}  Attempt {attempt} failed, retrying...{Colors.RESET}")
        
        # If both attempts failed, use random move or forfeit
        if not p1_move or not validation:
            if p1_random > 0:
                print(f"{Colors.YELLOW}Using random move for Player 1 ({p1_random} random moves left){Colors.RESET}")
                random_agent = RandomPlayer(1)
                p1_direction = random_agent.get_best_move()
                p1_random -= 1
                dir_to_str = {Direction.UP: 'UP', Direction.DOWN: 'DOWN', Direction.LEFT: 'LEFT', Direction.RIGHT: 'RIGHT'}
                validation = judge.handle_move(dir_to_str[p1_direction], 1, is_random=True)
                p1_boost = False
            else:
                print(f"{Colors.RED}Player 1 has no random moves left. Forfeiting.{Colors.RESET}")
                return judge.end_game(GameResult.AGENT2_WIN)
        
        # Player 2 move
        for attempt in range(1, 3):  # 2 attempts
            p2_move = judge.get_move(2, attempt, p2_random)
            if p2_move:
                validation = judge.handle_move(p2_move, 2, is_random=False)
                if validation == "forfeit":
                    print(f"{Colors.RED}Player 2 forfeited{Colors.RESET}")
                    return judge.end_game(GameResult.AGENT1_WIN)
                elif validation:
                    p2_boost = validation[1]
                    p2_direction = validation[2]
                    break
            if attempt < 2:
                print(f"{Colors.DIM}  Attempt {attempt} failed, retrying...{Colors.RESET}")
        
        # If both attempts failed, use random move or forfeit
        if not p2_move or not validation:
            if p2_random > 0:
                print(f"{Colors.YELLOW}Using random move for Player 2 ({p2_random} random moves left){Colors.RESET}")
                random_agent = RandomPlayer(2)
                p2_direction = random_agent.get_best_move()
                p2_random -= 1
                dir_to_str = {Direction.UP: 'UP', Direction.DOWN: 'DOWN', Direction.LEFT: 'LEFT', Direction.RIGHT: 'RIGHT'}
                validation = judge.handle_move(dir_to_str[p2_direction], 2, is_random=True)
                p2_boost = False
            else:
                print(f"{Colors.RED}Player 2 has no random moves left. Forfeiting.{Colors.RESET}")
                return judge.end_game(GameResult.AGENT1_WIN)
        
        # Execute both moves simultaneously
        result = judge.game.step(p1_direction, p2_direction, p1_boost, p2_boost)
        
        # Send updated state to both players
        judge.send_state(1)
        judge.send_state(2)
        
        # Display current board state with pretty colors
        judge.display_board_pretty()
        
        # Display stats
        print(f"{Colors.RED}Agent 1{Colors.RESET}: Trail={judge.game.agent1.length} | Alive={judge.game.agent1.alive} | Boosts={judge.game.agent1.boosts_remaining}")
        print(f"{Colors.BLUE}Agent 2{Colors.RESET}: Trail={judge.game.agent2.length} | Alive={judge.game.agent2.alive} | Boosts={judge.game.agent2.boosts_remaining}")
        
        # Check for game end
        if result is not None:
            final_result = judge.end_game(result)
            print(f"{Colors.DIM}Game String: {judge.game_str}{Colors.RESET}")
            return final_result
        
        # Check for max turns (safety)
        if judge.game.turns >= 500:
            print(f"{Colors.YELLOW}Maximum turns reached{Colors.RESET}")
            final_result = judge.end_game(GameResult.DRAW)
            print(f"{Colors.DIM}Game String: {judge.game_str}{Colors.RESET}")
            return final_result
        
        time.sleep(0.1)  # Small delay for readability

def main():
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║          CASE CLOSED - JUDGE ENGINE v2.0                  ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}\n")
    print("Judge engine starting up, waiting for agents...")
    time.sleep(3)

    # Get agent URLs from environment variables
    PLAYER1_URL = os.getenv("PLAYER1_URL", "http://localhost:5008")
    PLAYER2_URL = os.getenv("PLAYER2_URL", "http://localhost:5009")
    NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "1"))  # Number of rounds (each round plays both ways)

    # Creating judges for both configurations
    print(f"Connecting to agents at {PLAYER1_URL} and {PLAYER2_URL}...")
    judge_normal = Judge(PLAYER1_URL, PLAYER2_URL, swap_positions=False)
    
    # Check connectivity and latency
    if not judge_normal.check_latency():
        print(f"{Colors.RED}Failed to connect to one or both players{Colors.RESET}")
        return
    
    print(f"\n{Colors.GREEN}✓ Connected successfully!{Colors.RESET}")    
    print(f"{Colors.RED}Player 1{Colors.RESET}: {judge_normal.p1_agent.agent_name} ({judge_normal.p1_agent.participant})")
    print(f"{Colors.BLUE}Player 2{Colors.RESET}: {judge_normal.p2_agent.agent_name} ({judge_normal.p2_agent.participant})")
    print(f"{Colors.DIM}Initial latencies - P1: {judge_normal.p1_agent.latency:.3f}s, P2: {judge_normal.p2_agent.latency:.3f}s{Colors.RESET}\n")
    
    # Score tracking
    p1_wins = 0
    p2_wins = 0
    draws = 0
    
    # Run multiple rounds
    total_games = NUM_ROUNDS * 2
    for round_num in range(NUM_ROUNDS):
        # Game 1: Normal positions
        judge1 = Judge(PLAYER1_URL, PLAYER2_URL, swap_positions=False)
        judge1.p1_agent = judge_normal.p1_agent
        judge1.p2_agent = judge_normal.p2_agent
        
        result1 = run_single_game(judge1, round_num * 2 + 1, total_games)
        
        if result1 == GameResult.AGENT1_WIN:
            p1_wins += 1
        elif result1 == GameResult.AGENT2_WIN:
            p2_wins += 1
        else:
            draws += 1
        
        # Show current score
        print(f"\n{Colors.CYAN}{Colors.BOLD}Current Score:{Colors.RESET}")
        print(f"{Colors.RED}Player 1{Colors.RESET}: {p1_wins} wins | {Colors.BLUE}Player 2{Colors.RESET}: {p2_wins} wins | {Colors.YELLOW}Draws{Colors.RESET}: {draws}\n")
        
        time.sleep(2)
        
        # Game 2: Swapped positions
        judge2 = Judge(PLAYER1_URL, PLAYER2_URL, swap_positions=True)
        judge2.p1_agent = judge_normal.p1_agent
        judge2.p2_agent = judge_normal.p2_agent
        
        result2 = run_single_game(judge2, round_num * 2 + 2, total_games)
        
        if result2 == GameResult.AGENT1_WIN:
            p1_wins += 1
        elif result2 == GameResult.AGENT2_WIN:
            p2_wins += 1
        else:
            draws += 1
        
        # Show current score
        print(f"\n{Colors.CYAN}{Colors.BOLD}Current Score:{Colors.RESET}")
        print(f"{Colors.RED}Player 1{Colors.RESET}: {p1_wins} wins | {Colors.BLUE}Player 2{Colors.RESET}: {p2_wins} wins | {Colors.YELLOW}Draws{Colors.RESET}: {draws}\n")
        
        if round_num < NUM_ROUNDS - 1:
            time.sleep(2)
    
    # Final results
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                    FINAL RESULTS                           ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}")
    
    print(f"\n{Colors.BOLD}Total Games Played: {total_games}{Colors.RESET}")
    print(f"{Colors.RED}Player 1 ({judge_normal.p1_agent.agent_name}){Colors.RESET}: {p1_wins} wins")
    print(f"{Colors.BLUE}Player 2 ({judge_normal.p2_agent.agent_name}){Colors.RESET}: {p2_wins} wins")
    print(f"{Colors.YELLOW}Draws{Colors.RESET}: {draws}\n")
    
    if p1_wins > p2_wins:
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 OVERALL WINNER: Player 1 ({judge_normal.p1_agent.agent_name}) 🎉{Colors.RESET}\n")
    elif p2_wins > p1_wins:
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 OVERALL WINNER: Player 2 ({judge_normal.p2_agent.agent_name}) 🎉{Colors.RESET}\n")
    else:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚖️  OVERALL RESULT: TIE ⚖️{Colors.RESET}\n")


if __name__ == "__main__":
    main()
    sys.exit(0)
