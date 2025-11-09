package main

import "testing"

func TestBoardInitialization(t *testing.T) {
	board := NewGameBoard(10, 15)
	if board.Height != 10 {
		t.Errorf("Expected height 10, got %d", board.Height)
	}
	if board.Width != 15 {
		t.Errorf("Expected width 15, got %d", board.Width)
	}
	if len(board.Grid) != 10 {
		t.Errorf("Expected 10 rows, got %d", len(board.Grid))
	}
	if len(board.Grid[0]) != 15 {
		t.Errorf("Expected 15 columns, got %d", len(board.Grid[0]))
	}

	for y := 0; y < board.Height; y++ {
		for x := 0; x < board.Width; x++ {
			if board.Grid[y][x] != EMPTY {
				t.Errorf("Expected EMPTY at (%d, %d), got %d", x, y, board.Grid[y][x])
			}
		}
	}
}

func TestTorusWraparoundPositive(t *testing.T) {
	board := NewGameBoard(18, 20)

	// Test X wraparound
	pos := board.TorusCheck(Position{X: 21, Y: 5})
	if pos.X != 1 || pos.Y != 5 {
		t.Errorf("Expected (1, 5), got (%d, %d)", pos.X, pos.Y)
	}

	// Test Y wraparound
	pos = board.TorusCheck(Position{X: 5, Y: 19})
	if pos.X != 5 || pos.Y != 1 {
		t.Errorf("Expected (5, 1), got (%d, %d)", pos.X, pos.Y)
	}

	// Test both wraparound
	pos = board.TorusCheck(Position{X: 25, Y: 20})
	if pos.X != 5 || pos.Y != 2 {
		t.Errorf("Expected (5, 2), got (%d, %d)", pos.X, pos.Y)
	}
}

func TestTorusWraparoundNegative(t *testing.T) {
	board := NewGameBoard(18, 20)

	// Test negative X
	pos := board.TorusCheck(Position{X: -1, Y: 5})
	if pos.X != 19 || pos.Y != 5 {
		t.Errorf("Expected (19, 5), got (%d, %d)", pos.X, pos.Y)
	}

	// Test negative Y
	pos = board.TorusCheck(Position{X: 5, Y: -1})
	if pos.X != 5 || pos.Y != 17 {
		t.Errorf("Expected (5, 17), got (%d, %d)", pos.X, pos.Y)
	}

	// Test both negative
	pos = board.TorusCheck(Position{X: -3, Y: -2})
	if pos.X != 17 || pos.Y != 16 {
		t.Errorf("Expected (17, 16), got (%d, %d)", pos.X, pos.Y)
	}
}

func TestSetAndGetCellState(t *testing.T) {
	board := NewGameBoard(18, 20)

	board.SetCellState(Position{X: 5, Y: 10}, AGENT)
	if board.GetCellState(Position{X: 5, Y: 10}) != AGENT {
		t.Errorf("Expected AGENT at (5, 10)")
	}

	// Test with wraparound
	board.SetCellState(Position{X: 25, Y: 10}, AGENT)
	if board.GetCellState(Position{X: 5, Y: 10}) != AGENT {
		t.Errorf("Expected AGENT at (5, 10) after wraparound")
	}
}

func TestAgentInitialization(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)

	if agent.AgentID != 1 {
		t.Errorf("Expected agent ID 1, got %d", agent.AgentID)
	}
	if len(agent.Trail) != 2 {
		t.Errorf("Expected trail length 2, got %d", len(agent.Trail))
	}
	if agent.Trail[0].X != 5 || agent.Trail[0].Y != 5 {
		t.Errorf("Expected first position (5, 5), got (%d, %d)", agent.Trail[0].X, agent.Trail[0].Y)
	}
	if agent.Trail[1].X != 6 || agent.Trail[1].Y != 5 {
		t.Errorf("Expected second position (6, 5), got (%d, %d)", agent.Trail[1].X, agent.Trail[1].Y)
	}
	if !agent.Alive {
		t.Errorf("Expected agent to be alive")
	}
	if agent.Length != 2 {
		t.Errorf("Expected length 2, got %d", agent.Length)
	}
	if agent.BoostsRemaining != 3 {
		t.Errorf("Expected 3 boosts, got %d", agent.BoostsRemaining)
	}

	// Check board state
	if board.GetCellState(Position{X: 5, Y: 5}) != AGENT {
		t.Errorf("Expected AGENT at (5, 5)")
	}
	if board.GetCellState(Position{X: 6, Y: 5}) != AGENT {
		t.Errorf("Expected AGENT at (6, 5)")
	}
}

func TestAgentBasicMove(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)

	result := agent.Move(RIGHT, nil, false)
	if !result {
		t.Errorf("Expected move to succeed")
	}
	if len(agent.Trail) != 3 {
		t.Errorf("Expected trail length 3, got %d", len(agent.Trail))
	}
	if agent.Trail[len(agent.Trail)-1].X != 7 || agent.Trail[len(agent.Trail)-1].Y != 5 {
		t.Errorf("Expected head at (7, 5), got (%d, %d)", agent.Trail[len(agent.Trail)-1].X, agent.Trail[len(agent.Trail)-1].Y)
	}
	if board.GetCellState(Position{X: 7, Y: 5}) != AGENT {
		t.Errorf("Expected AGENT at (7, 5)")
	}
}

func TestAgentDirectionChange(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)

	agent.Move(RIGHT, nil, false)
	agent.Move(UP, nil, false)

	if len(agent.Trail) != 4 {
		t.Errorf("Expected trail length 4, got %d", len(agent.Trail))
	}
	if agent.Trail[len(agent.Trail)-1].X != 7 || agent.Trail[len(agent.Trail)-1].Y != 4 {
		t.Errorf("Expected head at (7, 4), got (%d, %d)", agent.Trail[len(agent.Trail)-1].X, agent.Trail[len(agent.Trail)-1].Y)
	}
}

func TestAgentCannotReverse(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)

	agent.Move(LEFT, nil, false)

	if len(agent.Trail) != 2 {
		t.Errorf("Expected trail length 2 (invalid move ignored), got %d", len(agent.Trail))
	}
}

func TestAgentSelfCollision(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)

	agent.Move(RIGHT, nil, false) // (7, 5)
	agent.Move(DOWN, nil, false)  // (7, 6)
	agent.Move(LEFT, nil, false)  // (6, 6)
	agent.Move(UP, nil, false)    // (6, 5) - collision with own trail

	if agent.Alive {
		t.Errorf("Expected agent to be dead after self-collision")
	}
}

func TestAgentHeadOnCollision(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent1 := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)
	agent2 := NewAgent(2, Position{X: 8, Y: 5}, LEFT, board)

	agent1.Move(RIGHT, agent2, false) // agent1 at (7, 5)
	agent2.Move(LEFT, agent1, false)  // agent2 at (7, 5) - head-on collision

	if agent1.Alive || agent2.Alive {
		t.Errorf("Expected both agents to be dead after head-on collision")
	}
}

func TestAgentBoost(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)

	initialLength := agent.Length
	initialBoosts := agent.BoostsRemaining

	agent.Move(RIGHT, nil, true)

	if agent.Length != initialLength+2 {
		t.Errorf("Expected length %d after boost, got %d", initialLength+2, agent.Length)
	}
	if agent.BoostsRemaining != initialBoosts-1 {
		t.Errorf("Expected %d boosts remaining, got %d", initialBoosts-1, agent.BoostsRemaining)
	}
	if agent.Trail[len(agent.Trail)-1].X != 8 || agent.Trail[len(agent.Trail)-1].Y != 5 {
		t.Errorf("Expected head at (8, 5) after boost, got (%d, %d)", agent.Trail[len(agent.Trail)-1].X, agent.Trail[len(agent.Trail)-1].Y)
	}
}

func TestAgentBoostExhaustion(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)

	// Use all 3 boosts
	agent.Move(RIGHT, nil, true)
	agent.Move(RIGHT, nil, true)
	agent.Move(RIGHT, nil, true)

	if agent.BoostsRemaining != 0 {
		t.Errorf("Expected 0 boosts remaining, got %d", agent.BoostsRemaining)
	}

	// Try to use another boost - should move normally
	initialLength := agent.Length
	agent.Move(RIGHT, nil, true)
	if agent.Length != initialLength+1 {
		t.Errorf("Expected length %d (only 1 space), got %d", initialLength+1, agent.Length)
	}
}

func TestTorusMovement(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 19, Y: 5}, RIGHT, board)

	agent.Move(RIGHT, nil, false)
	normalizedPos := board.TorusCheck(agent.Trail[len(agent.Trail)-1])
	if normalizedPos.X != 1 {
		t.Errorf("Expected X to wrap to 1, got %d", normalizedPos.X)
	}
}

func TestIsHead(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)

	if !agent.IsHead(Position{X: 6, Y: 5}) {
		t.Errorf("Expected (6, 5) to be head")
	}
	if agent.IsHead(Position{X: 5, Y: 5}) {
		t.Errorf("Expected (5, 5) to not be head")
	}

	agent.Move(RIGHT, nil, false)
	if !agent.IsHead(Position{X: 7, Y: 5}) {
		t.Errorf("Expected (7, 5) to be head after move")
	}
	if agent.IsHead(Position{X: 6, Y: 5}) {
		t.Errorf("Expected (6, 5) to not be head after move")
	}
}

func TestContainsPosition(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)

	if !agent.ContainsPosition(Position{X: 5, Y: 5}) {
		t.Errorf("Expected trail to contain (5, 5)")
	}
	if !agent.ContainsPosition(Position{X: 6, Y: 5}) {
		t.Errorf("Expected trail to contain (6, 5)")
	}
	if agent.ContainsPosition(Position{X: 7, Y: 5}) {
		t.Errorf("Expected trail to not contain (7, 5)")
	}

	agent.Move(RIGHT, nil, false)
	if !agent.ContainsPosition(Position{X: 7, Y: 5}) {
		t.Errorf("Expected trail to contain (7, 5) after move")
	}
}

func TestMultipleDirectionChanges(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 10, Y: 10}, RIGHT, board)

	moves := []struct {
		dir       Direction
		expectedX int
		expectedY int
	}{
		{RIGHT, 12, 10},
		{DOWN, 12, 11},
		{LEFT, 11, 11},
		{DOWN, 11, 12},
	}

	for i, move := range moves {
		agent.Move(move.dir, nil, false)
		head := agent.Trail[len(agent.Trail)-1]
		if head.X != move.expectedX || head.Y != move.expectedY {
			t.Errorf("Move %d: expected (%d, %d), got (%d, %d)", i, move.expectedX, move.expectedY, head.X, head.Y)
		}
	}

	if agent.Length != 6 {
		t.Errorf("Expected length 6 after 4 moves, got %d", agent.Length)
	}
}

func TestBoostWithDirectionChange(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)

	agent.Move(RIGHT, nil, true)

	if agent.Length != 4 {
		t.Errorf("Expected length 4 after boost, got %d", agent.Length)
	}
	if agent.Trail[len(agent.Trail)-1].X != 8 {
		t.Errorf("Expected final X at 8, got %d", agent.Trail[len(agent.Trail)-1].X)
	}

	agent.Move(DOWN, nil, true)
	if agent.Length != 6 {
		t.Errorf("Expected length 6 after second boost, got %d", agent.Length)
	}
	if agent.Trail[len(agent.Trail)-1].Y != 7 {
		t.Errorf("Expected final Y at 7, got %d", agent.Trail[len(agent.Trail)-1].Y)
	}
}

func TestAgentTrailCollisionWithOther(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent1 := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)
	agent2 := NewAgent(2, Position{X: 6, Y: 8}, UP, board)

	agent1.Move(RIGHT, agent2, false)

	agent2.Move(UP, agent1, false)
	agent2.Move(UP, agent1, false)

	if agent2.Alive {
		t.Errorf("Expected agent2 to die after colliding with agent1's trail")
		t.Logf("Agent1 trail: %v", agent1.Trail)
		t.Logf("Agent2 trail: %v", agent2.Trail)
		t.Logf("Agent2 head: %v", agent2.Trail[len(agent2.Trail)-1])
	}
	if !agent1.Alive {
		t.Errorf("Expected agent1 to remain alive")
	}
}

func TestTorusEdgeCasesAllDirections(t *testing.T) {
	board := NewGameBoard(18, 20)

	tests := []struct {
		pos       Position
		expectedX int
		expectedY int
	}{
		{Position{X: 20, Y: 0}, 0, 0},
		{Position{X: 0, Y: 18}, 0, 0},
		{Position{X: -1, Y: 0}, 19, 0},
		{Position{X: 0, Y: -1}, 0, 17},
		{Position{X: 40, Y: 36}, 0, 0},
		{Position{X: -20, Y: -18}, 0, 0},
		{Position{X: 23, Y: 19}, 3, 1},
	}

	for _, test := range tests {
		result := board.TorusCheck(test.pos)
		if result.X != test.expectedX || result.Y != test.expectedY {
			t.Errorf("TorusCheck(%d, %d): expected (%d, %d), got (%d, %d)",
				test.pos.X, test.pos.Y, test.expectedX, test.expectedY, result.X, result.Y)
		}
	}
}

func TestBoostAtBoardEdge(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 18, Y: 5}, RIGHT, board)

	agent.Move(RIGHT, nil, true)

	if !agent.Alive {
		t.Errorf("Expected agent to remain alive after boost at edge")
	}
	if agent.Trail[len(agent.Trail)-1].X != 1 {
		t.Errorf("Expected X to wrap to 1 after boost, got %d", agent.Trail[len(agent.Trail)-1].X)
	}
}

func TestBoostSelfCollision(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)

	agent.Move(RIGHT, nil, false)
	agent.Move(DOWN, nil, false)
	agent.Move(DOWN, nil, false)
	agent.Move(LEFT, nil, false)
	agent.Move(UP, nil, true)

	if agent.Alive {
		t.Errorf("Expected agent to die from boost self-collision")
	}
}

func TestDeadAgentCannotMove(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)

	agent.Move(RIGHT, nil, false)
	agent.Move(DOWN, nil, false)
	agent.Move(LEFT, nil, false)
	agent.Move(UP, nil, false)

	if agent.Alive {
		t.Errorf("Agent should be dead from self-collision")
	}

	initialLength := agent.Length
	result := agent.Move(RIGHT, nil, false)

	if result {
		t.Errorf("Dead agent should not be able to move")
	}
	if agent.Length != initialLength {
		t.Errorf("Dead agent length should not change")
	}
}

func TestAgentInvalidMoveDoesNotChangeDirection(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)

	initialDir := agent.Direction
	agent.Move(LEFT, nil, false)

	if agent.Direction != initialDir {
		t.Errorf("Direction should not change after invalid reverse move")
	}
}

func TestLongTrailIntegrity(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 10, Y: 10}, RIGHT, board)

	for i := 0; i < 50; i++ {
		if i%10 == 9 {
			agent.Move(DOWN, nil, false)
		} else {
			agent.Move(RIGHT, nil, false)
		}
	}

	if agent.Length != 52 {
		t.Errorf("Expected length 52, got %d", agent.Length)
	}

	if len(agent.Trail) != 52 {
		t.Errorf("Expected trail slice length 52, got %d", len(agent.Trail))
	}

	for i, pos := range agent.Trail {
		if board.GetCellState(pos) != AGENT {
			t.Errorf("Trail position %d (%d, %d) not marked on board", i, pos.X, pos.Y)
		}
	}
}

func TestBoostCollisionWithOtherAgent(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent1 := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)
	agent2 := NewAgent(2, Position{X: 9, Y: 5}, LEFT, board)

	agent1.Move(RIGHT, agent2, true)

	if agent1.Alive {
		t.Errorf("Expected agent1 to die from boost collision with agent2's trail")
	}
}

func TestBothAgentsMoveToSameSpot(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent1 := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)
	agent2 := NewAgent(2, Position{X: 8, Y: 5}, LEFT, board)

	agent1.Move(RIGHT, agent2, false)
	agent2.Move(LEFT, agent1, false)

	if agent1.Alive || agent2.Alive {
		t.Errorf("Both agents should be dead from head-on collision")
	}
}

func TestCircularPathNoCollision(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 10, Y: 10}, RIGHT, board)

	moves := []Direction{RIGHT, RIGHT, RIGHT, DOWN, DOWN, DOWN, LEFT, LEFT, LEFT, UP, UP}
	for _, dir := range moves {
		agent.Move(dir, nil, false)
	}

	if !agent.Alive {
		t.Errorf("Agent should survive circular path without closing the loop")
	}
}

func TestCircularPathWithCollision(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 10, Y: 10}, RIGHT, board)

	moves := []Direction{RIGHT, RIGHT, RIGHT, DOWN, DOWN, DOWN, LEFT, LEFT, LEFT, UP, UP, UP}
	for _, dir := range moves {
		agent.Move(dir, nil, false)
	}

	if agent.Alive {
		t.Errorf("Agent should die from closing the circular loop")
	}
}

func TestBoostExhaustionThenNormalMove(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)

	agent.Move(RIGHT, nil, true)
	agent.Move(RIGHT, nil, true)
	agent.Move(RIGHT, nil, true)

	lengthAfterBoosts := agent.Length

	agent.Move(RIGHT, nil, false)
	if agent.Length != lengthAfterBoosts+1 {
		t.Errorf("Expected normal move after boost exhaustion")
	}

	agent.Move(RIGHT, nil, true)
	if agent.Length != lengthAfterBoosts+2 {
		t.Errorf("Expected boost to fail gracefully and move normally")
	}
}

func TestMultipleAgentsIndependentMovement(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent1 := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)
	agent2 := NewAgent(2, Position{X: 15, Y: 15}, LEFT, board)

	for i := 0; i < 5; i++ {
		agent1.Move(RIGHT, agent2, false)
		agent2.Move(LEFT, agent1, false)
	}

	if !agent1.Alive || !agent2.Alive {
		t.Errorf("Both agents should be alive with independent paths")
	}
	if agent1.Length != 7 {
		t.Errorf("Expected agent1 length 7, got %d", agent1.Length)
	}
	if agent2.Length != 7 {
		t.Errorf("Expected agent2 length 7, got %d", agent2.Length)
	}
}

func TestAgentStartingAtBoundary(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 0, Y: 0}, DOWN, board)

	if !agent.Alive {
		t.Errorf("Agent should be alive when starting at boundary")
	}

	agent.Move(LEFT, nil, false)
	if agent.Trail[len(agent.Trail)-1].X != 19 {
		t.Errorf("Expected X to wrap to 19, got %d", agent.Trail[len(agent.Trail)-1].X)
	}
}

func TestComplexBoostScenario(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 10, Y: 10}, RIGHT, board)

	agent.Move(RIGHT, nil, true)
	agent.Move(DOWN, nil, false)
	agent.Move(DOWN, nil, true)

	if agent.BoostsRemaining != 1 {
		t.Errorf("Expected 1 boost remaining, got %d", agent.BoostsRemaining)
	}
	if !agent.Alive {
		t.Errorf("Agent should be alive after boost pattern")
	}
	if agent.Length != 7 {
		t.Errorf("Expected length 7 (2 initial + 2 boost + 1 normal + 2 boost), got %d", agent.Length)
	}
}

func TestHeadOnCollisionExactTiming(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent1 := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)
	agent2 := NewAgent(2, Position{X: 10, Y: 5}, LEFT, board)

	for i := 0; i < 2; i++ {
		agent1.Move(RIGHT, agent2, false)
		agent2.Move(LEFT, agent1, false)
	}

	if agent1.Alive || agent2.Alive {
		t.Errorf("Both agents should die from head-on collision")
	}
}

func TestAgentMovementWithEmptyOtherAgent(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)

	for i := 0; i < 10; i++ {
		result := agent.Move(RIGHT, nil, false)
		if !result {
			t.Errorf("Move %d failed with nil other agent", i)
		}
	}

	if !agent.Alive {
		t.Errorf("Agent should be alive when moving with nil other agent")
	}
}

func TestBoardStatePersistence(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)

	agent.Move(RIGHT, nil, false)
	agent.Move(DOWN, nil, false)
	agent.Move(LEFT, nil, false)

	for _, pos := range agent.Trail {
		if board.GetCellState(pos) != AGENT {
			t.Errorf("Position (%d, %d) should be marked as AGENT", pos.X, pos.Y)
		}
	}
}

func TestMassiveTorusWrap(t *testing.T) {
	board := NewGameBoard(18, 20)
	tests := []struct {
		x, y       int
		expX, expY int
	}{
		{1000, 1000, 1000 % 20, 1000 % 18},
		{-1000, -1000, ((-1000 % 20) + 20) % 20, ((-1000 % 18) + 18) % 18},
		{500, -500, 500 % 20, ((-500 % 18) + 18) % 18},
	}

	for _, test := range tests {
		result := board.TorusCheck(Position{X: test.x, Y: test.y})
		if result.X != test.expX || result.Y != test.expY {
			t.Errorf("TorusCheck(%d, %d): expected (%d, %d), got (%d, %d)",
				test.x, test.y, test.expX, test.expY, result.X, result.Y)
		}
	}
}

func TestTrailGrowthConsistency(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 10, Y: 10}, RIGHT, board)

	for i := 2; i < 20; i++ {
		agent.Move(RIGHT, nil, false)
		if agent.Length != i+1 {
			t.Errorf("After move %d, expected length %d, got %d", i-1, i+1, agent.Length)
		}
		if len(agent.Trail) != i+1 {
			t.Errorf("After move %d, expected trail size %d, got %d", i-1, i+1, len(agent.Trail))
		}
	}
}

func TestReverseMoveDuringBoost(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 10, Y: 10}, RIGHT, board)

	initialLength := agent.Length
	agent.Move(LEFT, nil, true)

	if agent.Length != initialLength {
		t.Errorf("Expected no length change on reverse boost, got length %d", agent.Length)
	}
	if agent.BoostsRemaining != 2 {
		t.Errorf("Expected boost to be consumed even on invalid move, got %d", agent.BoostsRemaining)
	}
}

func TestAlternatingDirections(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 10, Y: 10}, RIGHT, board)

	for i := 0; i < 10; i++ {
		if i%2 == 0 {
			agent.Move(RIGHT, nil, false)
		} else {
			agent.Move(DOWN, nil, false)
		}
	}

	if !agent.Alive {
		t.Errorf("Agent should survive alternating pattern")
	}
	if agent.Length != 12 {
		t.Errorf("Expected length 12, got %d", agent.Length)
	}
}

func TestBoostIntoSelfTrail(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 10, Y: 10}, RIGHT, board)

	agent.Move(RIGHT, nil, false)
	agent.Move(RIGHT, nil, false)
	agent.Move(DOWN, nil, false)
	agent.Move(DOWN, nil, false)
	agent.Move(LEFT, nil, false)
	agent.Move(UP, nil, true)

	if agent.Alive {
		t.Errorf("Expected agent to die from boost into own trail")
	}
}

func TestSimultaneousBoostCollision(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent1 := NewAgent(1, Position{X: 5, Y: 10}, RIGHT, board)
	agent2 := NewAgent(2, Position{X: 11, Y: 10}, LEFT, board)

	agent1.Move(RIGHT, agent2, true)
	agent2.Move(LEFT, agent1, true)

	if agent1.Alive || agent2.Alive {
		t.Errorf("Both agents should die from simultaneous boost collision")
	}
}

func TestAgentLengthTracking(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 10, Y: 10}, RIGHT, board)

	if agent.Length != len(agent.Trail) {
		t.Errorf("Initial length mismatch: Length=%d, Trail=%d", agent.Length, len(agent.Trail))
	}

	for i := 0; i < 20; i++ {
		agent.Move(RIGHT, nil, false)
		if agent.Length != len(agent.Trail) {
			t.Errorf("After move %d, Length=%d but Trail=%d", i, agent.Length, len(agent.Trail))
		}
	}
}

func TestZigZagPattern(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)

	pattern := []Direction{RIGHT, DOWN, RIGHT, DOWN, RIGHT, DOWN, LEFT, DOWN, LEFT, DOWN, LEFT}
	for _, dir := range pattern {
		if !agent.Move(dir, nil, false) {
			t.Errorf("Move failed unexpectedly in zigzag pattern")
		}
	}

	if !agent.Alive {
		t.Errorf("Agent should survive zigzag pattern")
	}
}

func TestBoostAtTorusEdge(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 19, Y: 10}, RIGHT, board)

	agent.Move(RIGHT, nil, true)

	if !agent.Alive {
		t.Errorf("Agent should survive boost across torus edge")
	}

	head := agent.Trail[len(agent.Trail)-1]
	if head.X > 2 {
		t.Errorf("Expected head to wrap around, got X=%d", head.X)
	}
}

func TestMultipleAgentsNoInteraction(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent1 := NewAgent(1, Position{X: 2, Y: 2}, RIGHT, board)
	agent2 := NewAgent(2, Position{X: 15, Y: 15}, LEFT, board)

	for i := 0; i < 8; i++ {
		agent1.Move(RIGHT, agent2, false)
		agent2.Move(LEFT, agent1, false)
	}

	if !agent1.Alive {
		t.Errorf("Agent1 should be alive")
	}
	if !agent2.Alive {
		t.Errorf("Agent2 should be alive")
	}

	boardHasAgent1 := false
	boardHasAgent2 := false
	for _, pos := range agent1.Trail {
		if board.GetCellState(pos) == AGENT {
			boardHasAgent1 = true
			break
		}
	}
	for _, pos := range agent2.Trail {
		if board.GetCellState(pos) == AGENT {
			boardHasAgent2 = true
			break
		}
	}

	if !boardHasAgent1 || !boardHasAgent2 {
		t.Errorf("Both agents should have trails on board")
	}
}

func TestAgentDiesReturnsFalse(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 10, Y: 10}, RIGHT, board)

	agent.Move(RIGHT, nil, false)
	agent.Move(DOWN, nil, false)
	agent.Move(LEFT, nil, false)
	result := agent.Move(UP, nil, false)

	if result {
		t.Errorf("Expected Move to return false when agent dies")
	}
	if agent.Alive {
		t.Errorf("Agent should be dead")
	}
}

func TestBoostDecrementsTiming(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 10, Y: 10}, RIGHT, board)

	if agent.BoostsRemaining != 3 {
		t.Errorf("Expected 3 initial boosts, got %d", agent.BoostsRemaining)
	}

	agent.Move(RIGHT, nil, true)
	if agent.BoostsRemaining != 2 {
		t.Errorf("Expected 2 boosts after first boost, got %d", agent.BoostsRemaining)
	}

	agent.Move(DOWN, nil, false)
	if agent.BoostsRemaining != 2 {
		t.Errorf("Boosts should not decrement on normal move, got %d", agent.BoostsRemaining)
	}

	agent.Move(DOWN, nil, true)
	agent.Move(LEFT, nil, true)
	if agent.BoostsRemaining != 0 {
		t.Errorf("Expected 0 boosts after using all 3, got %d", agent.BoostsRemaining)
	}
}

func TestHeadPositionAfterMultipleMoves(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 10, Y: 10}, RIGHT, board)

	agent.Move(RIGHT, nil, false)
	if !agent.IsHead(Position{X: 12, Y: 10}) {
		t.Errorf("Head should be at (12, 10)")
	}

	agent.Move(DOWN, nil, false)
	if !agent.IsHead(Position{X: 12, Y: 11}) {
		t.Errorf("Head should be at (12, 11)")
	}
	if agent.IsHead(Position{X: 12, Y: 10}) {
		t.Errorf("(12, 10) should no longer be head")
	}
}

func TestCellStateAfterAgentPasses(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)

	pos1 := Position{X: 5, Y: 5}
	pos2 := Position{X: 6, Y: 5}

	if board.GetCellState(pos1) != AGENT {
		t.Errorf("Initial position should be AGENT")
	}

	agent.Move(RIGHT, nil, false)
	if board.GetCellState(pos1) != AGENT {
		t.Errorf("Trail should remain marked as AGENT")
	}
	if board.GetCellState(pos2) != AGENT {
		t.Errorf("New position should be marked as AGENT")
	}
}

func TestSpiralPattern(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 10, Y: 10}, RIGHT, board)

	moves := []Direction{
		RIGHT, RIGHT, RIGHT,
		DOWN, DOWN, DOWN,
		LEFT, LEFT, LEFT,
		UP, UP,
	}

	for i, dir := range moves {
		if !agent.Move(dir, nil, false) {
			t.Errorf("Move %d failed in spiral pattern", i)
		}
	}

	if !agent.Alive {
		t.Errorf("Agent should survive incomplete spiral")
	}
}

func TestAllDirectionsFromCenter(t *testing.T) {
	directions := []Direction{UP, DOWN, LEFT, RIGHT}
	for i, dir := range directions {
		board := NewGameBoard(18, 20)
		agent := NewAgent(1, Position{X: 10, Y: 10}, dir, board)

		for j := 0; j < 5; j++ {
			if !agent.Move(dir, nil, false) {
				t.Errorf("Direction %d, move %d failed", i, j)
			}
		}

		if !agent.Alive {
			t.Errorf("Agent should survive moving in direction %d", i)
		}
		if agent.Length != 7 {
			t.Errorf("Direction %d: expected length 7, got %d", i, agent.Length)
		}
	}
}

func TestUndoableMoveBasic(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)

	initialLength := agent.Length
	initialHead := agent.GetHead()

	success, state := agent.UndoableMove(RIGHT, nil, false)
	if !success {
		t.Errorf("Expected move to succeed")
	}

	newHead := agent.GetHead()
	if newHead.X != 7 || newHead.Y != 5 {
		t.Errorf("Expected head at (7, 5), got (%d, %d)", newHead.X, newHead.Y)
	}

	agent.UndoMove(state, nil)

	if agent.Length != initialLength {
		t.Errorf("Expected length %d after undo, got %d", initialLength, agent.Length)
	}
	if agent.GetHead().X != initialHead.X || agent.GetHead().Y != initialHead.Y {
		t.Errorf("Expected head back at (%d, %d), got (%d, %d)",
			initialHead.X, initialHead.Y, agent.GetHead().X, agent.GetHead().Y)
	}
	if board.GetCellState(Position{X: 7, Y: 5}) != EMPTY {
		t.Errorf("Expected (7, 5) to be EMPTY after undo")
	}
}

func TestUndoableMoveWithBoost(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)

	initialLength := agent.Length
	initialBoosts := agent.BoostsRemaining

	success, state := agent.UndoableMove(RIGHT, nil, true)
	if !success {
		t.Errorf("Expected boost move to succeed")
	}

	if agent.Length != initialLength+2 {
		t.Errorf("Expected length %d after boost, got %d", initialLength+2, agent.Length)
	}
	if agent.BoostsRemaining != initialBoosts-1 {
		t.Errorf("Expected %d boosts remaining, got %d", initialBoosts-1, agent.BoostsRemaining)
	}

	agent.UndoMove(state, nil)

	if agent.Length != initialLength {
		t.Errorf("Expected length %d after undo, got %d", initialLength, agent.Length)
	}
	if agent.BoostsRemaining != initialBoosts {
		t.Errorf("Expected %d boosts after undo, got %d", initialBoosts, agent.BoostsRemaining)
	}
}

func TestUndoableMoveCollision(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)

	agent.Move(RIGHT, nil, false)
	agent.Move(DOWN, nil, false)
	agent.Move(LEFT, nil, false)

	success, state := agent.UndoableMove(UP, nil, false)
	if success {
		t.Errorf("Expected collision move to fail")
	}
	if agent.Alive {
		t.Errorf("Expected agent to be dead after collision")
	}

	agent.UndoMove(state, nil)

	if !agent.Alive {
		t.Errorf("Expected agent to be alive after undo")
	}
}

func TestUndoableMoveHeadOnCollision(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent1 := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)
	agent2 := NewAgent(2, Position{X: 8, Y: 5}, LEFT, board)

	success, state := agent1.UndoableMove(RIGHT, agent2, false)
	if success {
		t.Errorf("Expected head-on collision to fail")
	}
	if agent1.Alive || agent2.Alive {
		t.Errorf("Expected both agents to be dead after head-on collision")
	}

	agent1.UndoMove(state, agent2)

	if !agent1.Alive || !agent2.Alive {
		t.Errorf("Expected both agents to be alive after undo")
	}
}

func TestUndoableMultipleMoves(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 10, Y: 10}, RIGHT, board)

	var states []MoveState
	moves := []Direction{RIGHT, RIGHT, DOWN, DOWN}

	for _, dir := range moves {
		_, state := agent.UndoableMove(dir, nil, false)
		states = append(states, state)
	}

	if agent.Length != 6 {
		t.Errorf("Expected length 6 after 4 moves, got %d", agent.Length)
	}

	for i := len(states) - 1; i >= 0; i-- {
		agent.UndoMove(states[i], nil)
	}

	if agent.Length != 2 {
		t.Errorf("Expected length 2 after undoing all moves, got %d", agent.Length)
	}
	if agent.GetHead().X != 11 || agent.GetHead().Y != 10 {
		t.Errorf("Expected head at (11, 10), got (%d, %d)", agent.GetHead().X, agent.GetHead().Y)
	}
}

func TestUndoableDirectionRestore(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)

	initialDir := agent.Direction

	_, state := agent.UndoableMove(DOWN, nil, false)
	if agent.Direction.DX != DOWN.DX || agent.Direction.DY != DOWN.DY {
		t.Errorf("Expected direction to change to DOWN")
	}

	agent.UndoMove(state, nil)
	if agent.Direction.DX != initialDir.DX || agent.Direction.DY != initialDir.DY {
		t.Errorf("Expected direction to restore to original")
	}
}

func TestAggressiveBoostDrawScenario(t *testing.T) {
	// Test the scenario from the game log where P1 boosts RIGHT aggressively
	// P2 moves DOWN, both end up at same position and should draw
	board := NewGameBoard(18, 20)

	// Set up P1 at position similar to game log - heading RIGHT
	agent1 := NewAgent(1, Position{X: 10, Y: 11}, RIGHT, board)
	// Set up P2 directly above and to the right - heading DOWN
	agent2 := NewAgent(2, Position{X: 13, Y: 10}, DOWN, board)

	t.Logf("Initial - Agent1 head: %v, Agent2 head: %v", agent1.GetHead(), agent2.GetHead())

	// P1 moves RIGHT with BOOST (moves to 12,11 then 13,11)
	// But P2 moves DOWN (moves to 13,11)
	// They collide at 13,11 - but is it head-on or trail collision?

	// Sequential execution: P1 moves first
	agent1Alive := agent1.Move(RIGHT, agent2, true)
	t.Logf("After P1 boost RIGHT - Agent1 head: %v, alive: %v", agent1.GetHead(), agent1.Alive)

	// Then P2 moves
	agent2Alive := agent2.Move(DOWN, agent1, false)
	t.Logf("After P2 DOWN - Agent2 head: %v, alive: %v", agent2.GetHead(), agent2.Alive)

	// Both should be dead (head-on collision at 13,11)
	if agent1Alive {
		t.Error("Agent1 should have died in collision")
	}
	if agent2Alive {
		t.Error("Agent2 should have died in collision")
	}

	if agent1.Alive {
		t.Error("Agent1 should be marked as dead")
	}
	if agent2.Alive {
		t.Error("Agent2 should be marked as dead")
	}
}

func TestGameStepDrawResult(t *testing.T) {
	// Test that Game.Step correctly returns Draw when both agents die
	board := NewGameBoard(18, 20)
	agent1 := NewAgent(1, Position{X: 10, Y: 11}, RIGHT, board)
	agent2 := NewAgent(2, Position{X: 13, Y: 10}, DOWN, board)

	game := &Game{
		Board:  board,
		Agent1: agent1,
		Agent2: agent2,
		Turns:  0,
	}

	result := game.Step(RIGHT, DOWN, true, false)

	if result == nil {
		t.Fatal("Expected game to end with a result, got nil")
	}

	if *result != Draw {
		t.Errorf("Expected Draw result when both agents die, got %v", *result)
	}

	if agent1.Alive || agent2.Alive {
		t.Error("Both agents should be dead after collision")
	}
}

func TestSequentialMoveExecution(t *testing.T) {
	// Test that moves are executed sequentially: P1 first, then P2
	// P2 sees P1's updated position and collides with it
	board := NewGameBoard(18, 20)

	// P1 at (10, 12) with trail [(10,12), (10,11)], will move UP to (10,10)
	agent1 := NewAgent(1, Position{X: 10, Y: 12}, UP, board)
	// P2 at (10, 8) with trail [(10,8), (10,9)], will move UP to (10,10)
	agent2 := NewAgent(2, Position{X: 10, Y: 8}, DOWN, board)

	t.Logf("Initial - Agent1 trail: %v, Agent2 trail: %v",
		agent1.Trail, agent2.Trail)

	// P1 moves UP first to (10, 10)
	agent1Alive := agent1.Move(UP, agent2, false)
	t.Logf("After P1 UP - Agent1 head: %v, alive: %v", agent1.GetHead(), agent1Alive)

	// P2 moves DOWN to (10, 10) - should hit P1's new head/trail
	agent2Alive := agent2.Move(DOWN, agent1, false)
	t.Logf("After P2 DOWN - Agent2 head: %v, alive: %v", agent2.GetHead(), agent2Alive)

	// P1 survives its move
	// P2 hits P1's trail and dies
	if !agent1Alive {
		t.Error("Agent1 should survive its move")
	}
	if agent2Alive {
		t.Error("Agent2 should die hitting Agent1's trail")
	}
}
