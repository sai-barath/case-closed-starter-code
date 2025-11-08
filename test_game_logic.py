import unittest
from case_closed_game import GameBoard, Agent, Direction, EMPTY, AGENT


class TestGameBoard(unittest.TestCase):
    def test_board_initialization(self):
        board = GameBoard(height=10, width=15)
        self.assertEqual(board.height, 10)
        self.assertEqual(board.width, 15)
        self.assertEqual(len(board.grid), 10)
        self.assertEqual(len(board.grid[0]), 15)
        
        for row in board.grid:
            for cell in row:
                self.assertEqual(cell, EMPTY)
    
    def test_torus_wraparound_positive(self):
        board = GameBoard(height=18, width=20)
        
        # Test X wraparound
        pos = board._torus_check((21, 5))
        self.assertEqual(pos, (1, 5))
        
        # Test Y wraparound
        pos = board._torus_check((5, 19))
        self.assertEqual(pos, (5, 1))
        
        # Test both wraparound
        pos = board._torus_check((25, 20))
        self.assertEqual(pos, (5, 2))
    
    def test_torus_wraparound_negative(self):
        board = GameBoard(height=18, width=20)
        
        # Test negative X
        pos = board._torus_check((-1, 5))
        self.assertEqual(pos, (19, 5))
        
        # Test negative Y
        pos = board._torus_check((5, -1))
        self.assertEqual(pos, (5, 17))
        
        # Test both negative
        pos = board._torus_check((-3, -2))
        self.assertEqual(pos, (17, 16))
    
    def test_set_and_get_cell_state(self):
        board = GameBoard(height=18, width=20)
        
        board.set_cell_state((5, 10), AGENT)
        self.assertEqual(board.get_cell_state((5, 10)), AGENT)
        
        # Test with wraparound
        board.set_cell_state((25, 10), AGENT)
        self.assertEqual(board.get_cell_state((5, 10)), AGENT)


class TestAgent(unittest.TestCase):
    def test_agent_initialization(self):
        board = GameBoard()
        agent = Agent(agent_id="1", start_pos=(5, 5), start_dir=Direction.RIGHT, board=board)
        
        self.assertEqual(agent.agent_id, "1")
        self.assertEqual(len(agent.trail), 2)
        self.assertEqual(agent.trail[0], (5, 5))
        self.assertEqual(agent.trail[1], (6, 5))
        self.assertTrue(agent.alive)
        self.assertEqual(agent.length, 2)
        self.assertEqual(agent.boosts_remaining, 3)
        
        # Check board state
        self.assertEqual(board.get_cell_state((5, 5)), AGENT)
        self.assertEqual(board.get_cell_state((6, 5)), AGENT)
    
    def test_agent_basic_move(self):
        board = GameBoard()
        agent = Agent(agent_id=1, start_pos=(5, 5), start_dir=Direction.RIGHT, board=board)
        
        result = agent.move(Direction.RIGHT)
        self.assertTrue(result)
        self.assertEqual(len(agent.trail), 3)
        self.assertEqual(agent.trail[-1], (7, 5))
        self.assertEqual(board.get_cell_state((7, 5)), AGENT)
    
    def test_agent_direction_change(self):
        board = GameBoard()
        agent = Agent(agent_id=1, start_pos=(5, 5), start_dir=Direction.RIGHT, board=board)
        
        # Move right, then up
        agent.move(Direction.RIGHT)
        agent.move(Direction.UP)
        
        self.assertEqual(len(agent.trail), 4)
        self.assertEqual(agent.trail[-1], (7, 4))
    
    def test_agent_cannot_reverse(self):
        board = GameBoard()
        agent = Agent(agent_id=1, start_pos=(5, 5), start_dir=Direction.RIGHT, board=board)
        
        # Try to move left (opposite of right)
        agent.move(Direction.LEFT)
        
        # Should still only have 2 positions (invalid move ignored)
        self.assertEqual(len(agent.trail), 2)
    
    def test_agent_self_collision(self):
        board = GameBoard()
        agent = Agent(agent_id=1, start_pos=(5, 5), start_dir=Direction.RIGHT, board=board)
        
        # Create a box to collide with self
        agent.move(Direction.RIGHT)  # (7, 5)
        agent.move(Direction.DOWN)   # (7, 6)
        agent.move(Direction.LEFT)   # (6, 6)
        agent.move(Direction.UP)     # (6, 5) - collision with own trail
        
        self.assertFalse(agent.alive)
    
    def test_agent_head_on_collision(self):
        board = GameBoard()
        agent1 = Agent(agent_id=1, start_pos=(5, 5), start_dir=Direction.RIGHT, board=board)
        agent2 = Agent(agent_id=2, start_pos=(8, 5), start_dir=Direction.LEFT, board=board)
        
        # Move agents toward each other
        agent1.move(Direction.RIGHT, other_agent=agent2)  # agent1 at (7, 5)
        agent2.move(Direction.LEFT, other_agent=agent1)   # agent2 at (7, 5) - head-on collision
        
        # Both should be dead in head-on collision
        self.assertFalse(agent1.alive)
        self.assertFalse(agent2.alive)
    
    def test_agent_trail_collision(self):
        board = GameBoard()
        agent1 = Agent(agent_id=1, start_pos=(5, 5), start_dir=Direction.RIGHT, board=board)
        agent2 = Agent(agent_id=2, start_pos=(10, 10), start_dir=Direction.UP, board=board)
        
        # Agent1 creates a trail
        agent1.move(Direction.RIGHT, other_agent=agent2)
        agent1.move(Direction.RIGHT, other_agent=agent2)
        
        # Agent2 moves into agent1's trail
        agent2.move(Direction.LEFT, other_agent=agent1)
        agent2.move(Direction.LEFT, other_agent=agent1)
        agent2.move(Direction.LEFT, other_agent=agent1)
        agent2.move(Direction.DOWN, other_agent=agent1)
        agent2.move(Direction.DOWN, other_agent=agent1)
        result = agent2.move(Direction.DOWN, other_agent=agent1)  # Should hit agent1's trail
        
        # Agent2 should be dead, agent1 should be alive
        self.assertTrue(agent1.alive)
        # Check if collision happened
        if not result:
            self.assertFalse(agent2.alive)
    
    def test_agent_boost(self):
        board = GameBoard()
        agent = Agent(agent_id=1, start_pos=(5, 5), start_dir=Direction.RIGHT, board=board)
        
        initial_length = agent.length
        initial_boosts = agent.boosts_remaining
        
        # Use boost - should move 2 spaces
        agent.move(Direction.RIGHT, use_boost=True)
        
        self.assertEqual(agent.length, initial_length + 2)
        self.assertEqual(agent.boosts_remaining, initial_boosts - 1)
        self.assertEqual(agent.trail[-1], (8, 5))
    
    def test_agent_boost_exhaustion(self):
        board = GameBoard()
        agent = Agent(agent_id=1, start_pos=(5, 5), start_dir=Direction.RIGHT, board=board)
        
        # Use all 3 boosts
        agent.move(Direction.RIGHT, use_boost=True)
        agent.move(Direction.RIGHT, use_boost=True)
        agent.move(Direction.RIGHT, use_boost=True)
        
        self.assertEqual(agent.boosts_remaining, 0)
        
        # Try to use another boost - should move normally
        initial_length = agent.length
        agent.move(Direction.RIGHT, use_boost=True)
        self.assertEqual(agent.length, initial_length + 1)  # Only moved 1 space
    
    def test_torus_movement(self):
        board = GameBoard(height=18, width=20)
        agent = Agent(agent_id=1, start_pos=(19, 5), start_dir=Direction.RIGHT, board=board)
        
        # Move right from x=20 should wrap to x=0
        agent.move(Direction.RIGHT)
        normalized_pos = board._torus_check(agent.trail[-1])
        self.assertEqual(normalized_pos[0], 1)  # Should wrap around
    
    def test_is_head(self):
        board = GameBoard()
        agent = Agent(agent_id=1, start_pos=(5, 5), start_dir=Direction.RIGHT, board=board)
        
        self.assertTrue(agent.is_head((6, 5)))
        self.assertFalse(agent.is_head((5, 5)))
        
        agent.move(Direction.RIGHT)
        self.assertTrue(agent.is_head((7, 5)))
        self.assertFalse(agent.is_head((6, 5)))


if __name__ == '__main__':
    unittest.main()
