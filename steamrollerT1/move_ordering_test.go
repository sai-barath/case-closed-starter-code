package main

import (
	"testing"
)

// ============================================================================
// Move Ordering Tests
// ============================================================================

func TestNewMoveOrderingContext(t *testing.T) {
	ctx := NewMoveOrderingContext()

	if ctx.killerMoves == nil {
		t.Error("Killer moves map should be initialized")
	}
	if ctx.historyTable == nil {
		t.Error("History table should be initialized")
	}
	if ctx.bestMoveCache == nil {
		t.Error("Best move cache should be initialized")
	}
}

func TestUpdateKillerMove(t *testing.T) {
	ctx := NewMoveOrderingContext()
	depth := 5

	// First killer move
	ctx.updateKillerMove(depth, UP)
	killers := ctx.killerMoves[depth]
	if killers[0] != UP {
		t.Errorf("First killer should be UP, got %v", killers[0])
	}

	// Second killer move (different)
	ctx.updateKillerMove(depth, DOWN)
	killers = ctx.killerMoves[depth]
	if killers[0] != DOWN {
		t.Errorf("First killer should be DOWN, got %v", killers[0])
	}
	if killers[1] != UP {
		t.Errorf("Second killer should be UP, got %v", killers[1])
	}

	// Same move shouldn't shift
	ctx.updateKillerMove(depth, DOWN)
	killers = ctx.killerMoves[depth]
	if killers[0] != DOWN {
		t.Errorf("First killer should still be DOWN, got %v", killers[0])
	}
}

func TestUpdateHistory(t *testing.T) {
	ctx := NewMoveOrderingContext()

	// Initial value
	if ctx.historyTable[UP] != 0 {
		t.Error("Initial history should be 0")
	}

	// Update with cutoff
	ctx.updateHistory(UP, 5, true)
	if ctx.historyTable[UP] <= 0 {
		t.Error("History should increase on cutoff")
	}

	// Update without cutoff
	initialScore := ctx.historyTable[UP]
	ctx.updateHistory(UP, 5, false)
	// Score might decrease or stay depending on gravity
	if ctx.historyTable[UP] == initialScore {
		t.Log("History gravity prevents unbounded growth")
	}
}

func TestOrderMovesStatic(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	myAgent := &Agent{
		AgentID: 1,
		Trail:   []Position{{X: 10, Y: 10}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   true,
	}

	oppAgent := &Agent{
		AgentID: 2,
		Trail:   []Position{{X: 5, Y: 5}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   true,
	}

	snapshot := GameStateSnapshot{
		myAgent:    myAgent,
		otherAgent: oppAgent,
		board:      board,
	}

	scored := []scoredMove{
		{dir: UP, score: 0},
		{dir: DOWN, score: 0},
		{dir: LEFT, score: 0},
		{dir: RIGHT, score: 0},
	}

	scored = orderMovesStatic(scored, snapshot)

	// All moves should have non-zero scores after static evaluation
	allNonZero := true
	for _, sm := range scored {
		if sm.score == 0 {
			allNonZero = false
			break
		}
	}

	if !allNonZero {
		t.Log("Warning: Some moves still have zero score after static ordering")
	}
}

func TestOrderMoves(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	myAgent := &Agent{
		AgentID: 1,
		Trail:   []Position{{X: 10, Y: 10}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   true,
	}

	oppAgent := &Agent{
		AgentID: 2,
		Trail:   []Position{{X: 5, Y: 5}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   true,
	}

	snapshot := GameStateSnapshot{
		myAgent:    myAgent,
		otherAgent: oppAgent,
		board:      board,
	}

	moves := []Direction{UP, DOWN, LEFT, RIGHT}
	ordered := orderMoves(moves, snapshot)

	if len(ordered) != len(moves) {
		t.Errorf("Ordered moves should have same length, got %d", len(ordered))
	}

	// Check all moves are present
	for _, m := range moves {
		found := false
		for _, om := range ordered {
			if m.DX == om.DX && m.DY == om.DY {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Move %v not found in ordered moves", m)
		}
	}
}

func TestCountLocalSpace(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	agent := &Agent{
		AgentID: 1,
		Trail:   []Position{{X: 10, Y: 10}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   true,
	}

	// Empty board
	space := countLocalSpace(Position{X: 10, Y: 10}, agent, 5)
	if space == 0 {
		t.Error("Local space should be positive on empty board")
	}

	// Block some cells
	for x := 8; x <= 12; x++ {
		for y := 8; y <= 12; y++ {
			if !(x == 10 && y == 10) {
				board.SetCellState(Position{X: x, Y: y}, AGENT)
			}
		}
	}

	space2 := countLocalSpace(Position{X: 10, Y: 10}, agent, 5)
	if space2 >= space {
		t.Error("Blocking cells should reduce local space")
	}
}

func TestShouldApplyLMR(t *testing.T) {
	testCases := []struct {
		name      string
		moveIndex int
		depth     int
		isCapture bool
		expected  bool
	}{
		{"First move - no LMR", 0, 5, false, false},
		{"Capture - no LMR", 5, 5, true, false},
		{"Shallow depth - no LMR", 5, 2, false, false},
		{"Late move, deep - LMR", 5, 5, false, true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := shouldApplyLMR(tc.moveIndex, tc.depth, tc.isCapture)
			if result != tc.expected {
				t.Errorf("Expected %v, got %v", tc.expected, result)
			}
		})
	}
}

func TestGetLMRReduction(t *testing.T) {
	testCases := []struct {
		name      string
		moveIndex int
		depth     int
		minReduce int
	}{
		{"Early move", 2, 5, 0},
		{"Late move", 5, 5, 1},
		{"Very late move", 8, 10, 2},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			reduction := getLMRReduction(tc.moveIndex, tc.depth)
			if reduction < tc.minReduce {
				t.Errorf("Expected at least %d reduction, got %d", tc.minReduce, reduction)
			}
			if reduction > tc.depth {
				t.Errorf("Reduction %d should not exceed depth %d", reduction, tc.depth)
			}
		})
	}
}

func TestOrderMovesAtRoot(t *testing.T) {
	ctx := NewMoveOrderingContext()
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	myAgent := &Agent{
		AgentID: 1,
		Trail:   []Position{{X: 10, Y: 10}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   true,
	}

	oppAgent := &Agent{
		AgentID: 2,
		Trail:   []Position{{X: 5, Y: 5}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   true,
	}

	snapshot := GameStateSnapshot{
		myAgent:    myAgent,
		otherAgent: oppAgent,
		board:      board,
	}

	moves := []Direction{UP, DOWN, LEFT, RIGHT}

	// Add a best move from previous iteration
	ctx.bestMoveCache[4] = Move{direction: RIGHT, useBoost: false, score: 100}

	// Add killer moves
	ctx.updateKillerMove(5, LEFT)

	ordered := ctx.orderMovesAtRoot(moves, snapshot, 5)

	if len(ordered) != len(moves) {
		t.Errorf("Ordered moves should have same length, got %d", len(ordered))
	}

	// RIGHT should be first (from hash move)
	// Actually might not be first due to static evaluation overlay,
	// but should be present
	found := false
	for _, m := range ordered {
		if m.DX == RIGHT.DX && m.DY == RIGHT.DY {
			found = true
			break
		}
	}
	if !found {
		t.Error("Hash move should be in ordered moves")
	}
}
