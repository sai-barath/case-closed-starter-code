package main

import (
	"testing"
)

// ============================================================================
// Endgame Tests
// ============================================================================

func TestHilbertIndex(t *testing.T) {
	testCases := []struct {
		name  string
		x, y  int
		order int
	}{
		{"Origin", 0, 0, 3},
		{"Middle", 4, 4, 3},
		{"Edge", 7, 0, 3},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			idx := hilbertIndex(tc.x, tc.y, tc.order)
			if idx < 0 {
				t.Errorf("Hilbert index should be non-negative, got %d", idx)
			}
		})
	}
}

func TestGenerateHilbertPath(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	agent := &Agent{
		AgentID:  1,
		Trail:    []Position{{X: 10, Y: 10}},
		TrailSet: make(map[Position]bool),
		Board:    board,
		Alive:    true,
	}

	path := generateHilbertPath(agent)

	if len(path) == 0 {
		t.Error("Hilbert path should not be empty on open board")
	}

	// Path should contain reachable cells
	if len(path) < BOARD_HEIGHT*BOARD_WIDTH/2 {
		t.Logf("Warning: Hilbert path seems short: %d cells", len(path))
	}
}

func TestGreedyFurthestPoint(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	agent := &Agent{
		AgentID:  1,
		Trail:    []Position{{X: 10, Y: 10}},
		TrailSet: make(map[Position]bool),
		Board:    board,
		Alive:    true,
	}

	path := greedyFurthestPoint(agent)

	if len(path) == 0 {
		t.Error("Greedy path should not be empty on open board")
	}

	// First position should be the agent's head
	if path[0] != agent.Trail[0] {
		t.Errorf("Path should start at agent head %v, got %v", agent.Trail[0], path[0])
	}
}

func TestGetReachableCells(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	agent := &Agent{
		AgentID:  1,
		Trail:    []Position{{X: 10, Y: 10}},
		TrailSet: make(map[Position]bool),
		Board:    board,
		Alive:    true,
	}

	reachable := getReachableCells(agent)
	totalCells := BOARD_HEIGHT * BOARD_WIDTH

	if len(reachable) < totalCells-10 {
		t.Errorf("Should reach most cells on empty board, got %d", len(reachable))
	}

	// Add walls to create an enclosed area (box) - since board is torus,
	// need to create a box to truly limit reachable cells
	for x := 5; x < 15; x++ {
		board.SetCellState(Position{X: x, Y: 5}, AGENT)
		board.SetCellState(Position{X: x, Y: 15}, AGENT)
	}
	for y := 5; y < 15; y++ {
		board.SetCellState(Position{X: 5, Y: y}, AGENT)
		board.SetCellState(Position{X: 15, Y: y}, AGENT)
	}

	reachable = getReachableCells(agent)
	// Agent at (10,10) is inside the box, should only reach cells in ~10x10 area minus walls
	if len(reachable) >= totalCells/2 {
		t.Errorf("Walls should limit reachable cells, got %d", len(reachable))
	}
}

func TestLongestPathScore(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	agent := &Agent{
		AgentID:  1,
		Trail:    []Position{{X: 10, Y: 10}},
		TrailSet: make(map[Position]bool),
		Board:    board,
		Alive:    true,
	}

	score := longestPathScore(agent)
	if score <= 0 {
		t.Errorf("Longest path score should be positive, got %d", score)
	}
}

func TestEvaluateEndgame(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	myAgent := &Agent{
		AgentID:  1,
		Trail:    []Position{{X: 10, Y: 10}},
		TrailSet: make(map[Position]bool),
		Board:    board,
		Alive:    true,
	}

	oppAgent := &Agent{
		AgentID:  2,
		Trail:    []Position{{X: 5, Y: 5}},
		TrailSet: make(map[Position]bool),
		Board:    board,
		Alive:    true,
	}

	// Equal space
	score := evaluateEndgame(myAgent, oppAgent, 100, 100)
	if abs(score) > 1000 {
		t.Errorf("Equal space should give near-zero score, got %d", score)
	}

	// Significant advantage
	score = evaluateEndgame(myAgent, oppAgent, 150, 50)
	if score <= 0 {
		t.Errorf("Large space advantage should give positive score, got %d", score)
	}

	// Significant disadvantage
	score = evaluateEndgame(myAgent, oppAgent, 50, 150)
	if score >= 0 {
		t.Errorf("Large space disadvantage should give negative score, got %d", score)
	}
}

func TestShouldUseEndgameMode(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	// Early game
	myAgent := &Agent{
		AgentID: 1,
		Trail:   make([]Position, 20),
		Board:   board,
		Alive:   true,
	}

	oppAgent := &Agent{
		AgentID: 2,
		Trail:   make([]Position, 20),
		Board:   board,
		Alive:   true,
	}

	for i := 0; i < 20; i++ {
		myAgent.Trail[i] = Position{X: i, Y: 0}
		oppAgent.Trail[i] = Position{X: i, Y: 1}
		board.SetCellState(myAgent.Trail[i], AGENT)
		board.SetCellState(oppAgent.Trail[i], AGENT)
	}

	shouldUse := shouldUseEndgameMode(myAgent, oppAgent)
	if shouldUse {
		t.Error("Should not use endgame mode with few occupied cells")
	}

	// Fill most of the board
	for y := 0; y < BOARD_HEIGHT-2; y++ {
		for x := 0; x < BOARD_WIDTH; x++ {
			board.SetCellState(Position{X: x, Y: y}, AGENT)
		}
	}

	shouldUse = shouldUseEndgameMode(myAgent, oppAgent)
	if !shouldUse {
		t.Error("Should use endgame mode when board is mostly full")
	}
}

func TestArticulationAwarePath(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	agent := &Agent{
		AgentID:   1,
		Trail:     []Position{{X: 10, Y: 10}},
		TrailSet:  make(map[Position]bool),
		Direction: RIGHT,
		Board:     board,
		Alive:     true,
	}

	// No articulation points
	aps := []Position{}
	dir := articulationAwarePath(agent, aps)
	if dir.DX == 0 && dir.DY == 0 {
		t.Error("Should return a valid direction")
	}

	// With articulation point
	aps = []Position{{X: 12, Y: 10}}
	dir = articulationAwarePath(agent, aps)
	// Should move towards the articulation point (RIGHT in this case)
	// But we just check it returns a valid direction
	if dir.DX == 0 && dir.DY == 0 {
		t.Error("Should return a valid direction")
	}
}

func TestSpaceFillingSuggestion(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	agent := &Agent{
		AgentID:   1,
		Trail:     []Position{{X: 10, Y: 10}},
		TrailSet:  make(map[Position]bool),
		Direction: RIGHT,
		Board:     board,
		Alive:     true,
	}

	dir := spaceFillingSuggestion(agent)

	// Should return a valid direction
	validDirs := []Direction{UP, DOWN, LEFT, RIGHT}
	found := false
	for _, vd := range validDirs {
		if dir.DX == vd.DX && dir.DY == vd.DY {
			found = true
			break
		}
	}

	if !found {
		t.Errorf("Should return a valid direction, got DX=%d DY=%d", dir.DX, dir.DY)
	}
}
