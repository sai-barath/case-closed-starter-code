"""Test to reproduce the Turn 8 collision scenario from the summary."""

from case_closed_game import Agent, Direction, GameBoard, AGENT

def test_sequential_move_order_scenario():
    """
    Scenario from summary:
    - RED (player 1) at (10, 2) moves RIGHT → lands at (11, 2)
    - BLUE (player 2) at (11, 2) moves DOWN:BOOST
    - Result: Both agents died (draw), but BLUE should have won?
    """
    board = GameBoard()
    
    # Set up RED with head at (10, 2) 
    # If we want head at (10,2) after initial setup, start at (9,2) facing RIGHT
    # This gives trail [(9,2), (10,2)], head is (10,2)
    red = Agent(agent_id="RED", start_pos=(9, 2), start_dir=Direction.RIGHT, board=board)
    
    # Set up BLUE with head at (11, 2)
    # start_pos=(11,1), start_dir=DOWN gives trail [(11,1), (11,2)], head is (11,2)
    blue = Agent(agent_id="BLUE", start_pos=(11, 1), start_dir=Direction.DOWN, board=board)
    
    print(f"Initial state:")
    print(f"RED at {red.trail[-1]}, alive={red.alive}")
    print(f"BLUE at {blue.trail[-1]}, alive={blue.alive}")
    
    # Execute moves in sequence (like the game engine does)
    print(f"\nRED moves RIGHT...")
    red_alive = red.move(Direction.RIGHT, other_agent=blue, use_boost=False)
    print(f"RED now at {red.trail[-1]}, alive={red_alive}")
    print(f"BLUE still at {blue.trail[-1]}, alive={blue.alive}")
    
    print(f"\nBLUE moves DOWN with BOOST...")
    blue_alive = blue.move(Direction.DOWN, other_agent=red, use_boost=True)
    print(f"BLUE now at {blue.trail[-1]}, alive={blue_alive}")
    
    print(f"\nFinal result:")
    print(f"RED alive: {red_alive}")
    print(f"BLUE alive: {blue_alive}")
    
    if not red_alive and not blue_alive:
        print("Result: DRAW (both died)")
    elif red_alive and not blue_alive:
        print("Result: RED wins")
    elif not red_alive and blue_alive:
        print("Result: BLUE wins")
    else:
        print("Result: Both alive (game continues)")

if __name__ == "__main__":
    test_sequential_move_order_scenario()
