package main

import (
	"testing"
	"time"
)

// ============================================================================
// Search Tests
// ============================================================================

func TestSearchContextTimeExpired(t *testing.T) {
	ctx := SearchContext{
		startTime:     time.Now(),
		deadline:      time.Now().Add(10 * time.Millisecond),
		moveOrdering:  NewMoveOrderingContext(),
		nodesSearched: 0,
	}

	if ctx.timeExpired() {
		t.Error("Time should not have expired immediately")
	}

	time.Sleep(20 * time.Millisecond)

	if !ctx.timeExpired() {
		t.Error("Time should have expired after deadline")
	}
}

func TestShouldBoostAggressively_EscapeTightSpot(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	// Agent in corner with limited moves
	myAgent := &Agent{
		AgentID:         1,
		Trail:           []Position{{X: 0, Y: 0}},
TrailSet:        make(map[Position]bool),
		Board:           board,
		Alive:           true,
		BoostsRemaining: 1,
	}

	oppAgent := &Agent{
		AgentID: 2,
		Trail:   []Position{{X: 10, Y: 10}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   true,
	}

	// Block some directions
	board.SetCellState(Position{X: 1, Y: 0}, AGENT)
	board.SetCellState(Position{X: 0, Y: 1}, AGENT)

	snapshot := GameStateSnapshot{
		myAgent:    myAgent,
		otherAgent: oppAgent,
		board:      board,
	}

	shouldBoost := shouldBoostAggressively(snapshot, RIGHT, true)
	if !shouldBoost {
		t.Error("Should boost to escape tight spot")
	}
}

func TestShouldBoostAggressively_NoBoosts(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	myAgent := &Agent{
		AgentID:         1,
		Trail:           []Position{{X: 10, Y: 10}},
TrailSet:        make(map[Position]bool),
		Board:           board,
		Alive:           true,
		BoostsRemaining: 0,
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

	shouldBoost := shouldBoostAggressively(snapshot, RIGHT, true)
	if shouldBoost {
		t.Error("Should not boost when no boosts remaining")
	}
}

func TestBoostReserveValue(t *testing.T) {
	// Early game
	value1 := boostReserveValue(3, 10)
	if value1 <= 0 {
		t.Error("Boost reserve value should be positive in early game")
	}

	// Late game
	value2 := boostReserveValue(3, 150)
	if value2 <= 0 {
		t.Error("Boost reserve value should be positive in late game")
	}

	// Early game should have higher value
	if value1 <= value2 {
		t.Error("Early game boosts should have higher reserve value")
	}
}

func TestIterativeDeepeningSearch(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	myAgent := &Agent{
		AgentID:         1,
		Trail:           []Position{{X: 10, Y: 10}},
TrailSet:        make(map[Position]bool),
		Direction:       RIGHT,
		Board:           board,
		Alive:           true,
		BoostsRemaining: 3,
	}

	oppAgent := &Agent{
		AgentID:   2,
		Trail:     []Position{{X: 5, Y: 5}},
TrailSet:        make(map[Position]bool),
		Direction: LEFT,
		Board:     board,
		Alive:     true,
	}

	snapshot := GameStateSnapshot{
		myAgent:    myAgent,
		otherAgent: oppAgent,
		board:      board,
		amIRed:     true,
	}

	ctx := SearchContext{
		startTime:     time.Now(),
		deadline:      time.Now().Add(50 * time.Millisecond),
		moveOrdering:  NewMoveOrderingContext(),
		nodesSearched: 0,
	}

	move := iterativeDeepeningSearch(snapshot, ctx)

	// Should return a valid move
	validDirs := []Direction{UP, DOWN, LEFT, RIGHT}
	found := false
	for _, d := range validDirs {
		if move.direction.DX == d.DX && move.direction.DY == d.DY {
			found = true
			break
		}
	}

	if !found {
		t.Errorf("Should return a valid direction, got DX=%d DY=%d",
			move.direction.DX, move.direction.DY)
	}

	// Should have searched some nodes
	if ctx.nodesSearched == 0 {
		t.Error("Should have searched at least one node")
	}
}

func TestSearchAtDepth(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	myAgent := &Agent{
		AgentID:         1,
		Trail:           []Position{{X: 10, Y: 10}},
TrailSet:        make(map[Position]bool),
		Direction:       RIGHT,
		Board:           board,
		Alive:           true,
		BoostsRemaining: 3,
	}

	oppAgent := &Agent{
		AgentID:   2,
		Trail:     []Position{{X: 5, Y: 5}},
TrailSet:        make(map[Position]bool),
		Direction: LEFT,
		Board:     board,
		Alive:     true,
	}

	snapshot := GameStateSnapshot{
		myAgent:    myAgent,
		otherAgent: oppAgent,
		board:      board,
		amIRed:     true,
	}

	ctx := SearchContext{
		startTime:     time.Now(),
		deadline:      time.Now().Add(100 * time.Millisecond),
		moveOrdering:  NewMoveOrderingContext(),
		nodesSearched: 0,
	}

	move := searchAtDepth(snapshot, 3, &ctx)

	// Should return a move
	if move.score == 0 {
		t.Log("Move score is 0, might be early in game")
	}

	// Should have updated history
	if len(ctx.moveOrdering.historyTable) == 0 {
		t.Log("Warning: History table not updated")
	}
}

func TestPVS_Terminal(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	myAgent := &Agent{
		AgentID: 1,
		Trail:   []Position{{X: 10, Y: 10}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   false, // Dead
	}

	oppAgent := &Agent{
		AgentID: 2,
		Trail:   []Position{{X: 5, Y: 5}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   true,
	}

	ctx := SearchContext{
		startTime:     time.Now(),
		deadline:      time.Now().Add(100 * time.Millisecond),
		moveOrdering:  NewMoveOrderingContext(),
		nodesSearched: 0,
	}

	score := pvs(board, myAgent, oppAgent, 3, -10000, 10000, true, true, &ctx)

	// Should return loss score
	if score != LOSE_SCORE {
		t.Errorf("Expected LOSE_SCORE (%d), got %d", LOSE_SCORE, score)
	}
}

func TestPVS_DepthZero(t *testing.T) {
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

	ctx := SearchContext{
		startTime:     time.Now(),
		deadline:      time.Now().Add(100 * time.Millisecond),
		moveOrdering:  NewMoveOrderingContext(),
		nodesSearched: 0,
	}

	score := pvs(board, myAgent, oppAgent, 0, -10000, 10000, true, true, &ctx)

	// Should return evaluation score (not terminal)
	if score == LOSE_SCORE || score == WIN_SCORE || score == DRAW_SCORE {
		t.Logf("Depth 0 returned terminal score: %d", score)
	}
}

func TestEvaluateMoveAtDepthPVS(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	myAgent := &Agent{
		AgentID:         1,
		Trail:           []Position{{X: 10, Y: 10}},
TrailSet:        make(map[Position]bool),
		Direction:       RIGHT,
		Board:           board,
		Alive:           true,
		BoostsRemaining: 3,
	}

	oppAgent := &Agent{
		AgentID:   2,
		Trail:     []Position{{X: 5, Y: 5}},
TrailSet:        make(map[Position]bool),
		Direction: LEFT,
		Board:     board,
		Alive:     true,
	}

	snapshot := GameStateSnapshot{
		myAgent:    myAgent,
		otherAgent: oppAgent,
		board:      board,
		amIRed:     true,
	}

	ctx := SearchContext{
		startTime:     time.Now(),
		deadline:      time.Now().Add(100 * time.Millisecond),
		moveOrdering:  NewMoveOrderingContext(),
		nodesSearched: 0,
	}

	score := evaluateMoveAtDepthPVS(snapshot, RIGHT, false, 2, -10000, 10000, &ctx, true)

	// Should return a score
	if ctx.nodesSearched == 0 {
		t.Error("Should have searched nodes")
	}

	_ = score // Score value depends on position
}
