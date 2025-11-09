#!/usr/bin/env python3
"""Test edge case where P1 moves up to attack, but P2 moves down (escaping)"""

from case_closed_game import Game, Direction, GameResult, Agent, GameBoard

def test_p1_attacks_up_p2_escapes_right():
    """
    Test scenario:
    - P1 is below P2 and moves UP (trying to hit P2's current position)
    - P2 moves RIGHT (escaping from that position)
    
    Expected: P2 should survive because they moved away before P1 arrived
    """
    print("=== Test: P1 attacks UP, P2 escapes RIGHT ===\n")
    
    game = Game()
    board = game.board
    
    # Clear the board and create a custom scenario
    board.grid = [[0 for _ in range(20)] for _ in range(18)]
    
    # Create P1 at (10, 11) facing UP
    game.agent1 = Agent(agent_id="1", start_pos=(10, 12), start_dir=Direction.UP, board=board)
    
    # Create P2 at (10, 10) facing RIGHT (directly above P1)
    game.agent2 = Agent(agent_id="2", start_pos=(9, 10), start_dir=Direction.RIGHT, board=board)
    
    print("Initial state:")
    print(f"P1 head: {game.agent1.trail[-1]}, direction: {game.agent1.direction}")
    print(f"P2 head: {game.agent2.trail[-1]}, direction: {game.agent2.direction}")
    print(game)
    
    # P1 moves UP (trying to reach (10, 10))
    # P2 moves RIGHT (moving away from (10, 10) to (11, 10))
    print("\nTurn 1: P1 moves UP, P2 moves RIGHT")
    result = game.step(Direction.UP, Direction.RIGHT)
    
    print(f"P1 head: {game.agent1.trail[-1]}, alive: {game.agent1.alive}")
    print(f"P2 head: {game.agent2.trail[-1]}, alive: {game.agent2.alive}")
    print(game)
    
    if result:
        print(f"\nGame Result: {result}")
    else:
        print("\nNo winner yet")
    
    print("\n" + "="*50 + "\n")


def test_p2_attacks_up_p1_escapes_right_FLIPPED():
    """
    Same test but FLIPPED: P2 attacks UP, P1 escapes RIGHT
    """
    print("=== Test: P2 attacks UP, P1 escapes RIGHT (FLIPPED) ===\n")
    
    game = Game()
    board = game.board
    
    # Clear the board
    board.grid = [[0 for _ in range(20)] for _ in range(18)]
    
    # Create P2 at (10, 11) facing UP (attacker position - swapped from P1)
    game.agent2 = Agent(agent_id="2", start_pos=(10, 12), start_dir=Direction.UP, board=board)
    
    # Create P1 at (10, 10) facing RIGHT (escape position - swapped from P2)
    game.agent1 = Agent(agent_id="1", start_pos=(9, 10), start_dir=Direction.RIGHT, board=board)
    
    print("Initial state:")
    print(f"P1 head: {game.agent1.trail[-1]}, direction: {game.agent1.direction}")
    print(f"P2 head: {game.agent2.trail[-1]}, direction: {game.agent2.direction}")
    print(game)
    
    # P2 moves UP (trying to reach (10, 10))
    # P1 moves RIGHT (moving away from (10, 10) to (11, 10))
    print("\nTurn 1: P1 moves RIGHT, P2 moves UP")
    result = game.step(Direction.RIGHT, Direction.UP)
    
    print(f"P1 head: {game.agent1.trail[-1]}, alive: {game.agent1.alive}")
    print(f"P2 head: {game.agent2.trail[-1]}, alive: {game.agent2.alive}")
    print(game)
    
    if result:
        print(f"\nGame Result: {result}")
    else:
        print("\nNo winner yet")
    
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    test_p1_attacks_up_p2_escapes_right()
    test_p2_attacks_up_p1_escapes_right_FLIPPED()
