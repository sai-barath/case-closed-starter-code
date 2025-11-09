package main

import (
	"testing"
)

func TestChamberTreeSimpleChamber(t *testing.T) {
	board := NewGameBoard(10, 10)

	// Create box boundary to prevent torus wraparound
	for x := 0; x < 10; x++ {
		board.SetCellState(Position{X: x, Y: 0}, AGENT)
		board.SetCellState(Position{X: x, Y: 9}, AGENT)
	}
	for y := 1; y < 9; y++ {
		board.SetCellState(Position{X: 0, Y: y}, AGENT)
		board.SetCellState(Position{X: 9, Y: y}, AGENT)
	}

	// Horizontal wall at Y=5
	for x := 1; x < 9; x++ {
		board.SetCellState(Position{X: x, Y: 5}, AGENT)
	}

	myHead := Position{X: 5, Y: 3}
	oppHead := Position{X: 5, Y: 7}

	ct := NewChamberTree(board)
	score := ct.EvaluateChamberTree(myHead, oppHead)

	t.Logf("Chamber score with horizontal wall: %d", score)

	// Should have non-zero score since chambers are separated
	if score == 0 {
		t.Error("Expected non-zero chamber score")
	}
}

func TestChamberTreeNoChamber(t *testing.T) {
	board := NewGameBoard(10, 10)

	myHead := Position{X: 2, Y: 2}
	oppHead := Position{X: 7, Y: 7}

	ct := NewChamberTree(board)
	score := ct.EvaluateChamberTree(myHead, oppHead)

	t.Logf("Chamber score with no articulation points: %d", score)
}

func TestChamberTreeArticulationPoint(t *testing.T) {
	board := NewGameBoard(10, 10)

	// Create a box with walls, with single articulation point
	for x := 0; x < 10; x++ {
		board.SetCellState(Position{X: x, Y: 0}, AGENT)
		board.SetCellState(Position{X: x, Y: 9}, AGENT)
	}
	for y := 1; y < 9; y++ {
		board.SetCellState(Position{X: 0, Y: y}, AGENT)
		board.SetCellState(Position{X: 9, Y: y}, AGENT)
	}

	// Vertical wall in middle with gap at (5,5)
	for y := 1; y < 9; y++ {
		board.SetCellState(Position{X: 5, Y: y}, AGENT)
	}
	board.SetCellState(Position{X: 5, Y: 5}, EMPTY)

	myHead := Position{X: 3, Y: 5}
	oppHead := Position{X: 7, Y: 5}

	apf := NewArticulationPointFinder(board)
	aps := apf.FindArticulationPoints()

	if !aps[Position{X: 5, Y: 5}] {
		t.Logf("Articulation points found: %d", len(aps))
		for ap := range aps {
			t.Logf("  AP at (%d,%d)", ap.X, ap.Y)
		}
		t.Error("Expected (5,5) to be articulation point")
	}

	ct := NewChamberTree(board)
	score := ct.EvaluateChamberTree(myHead, oppHead)

	t.Logf("Chamber score with central articulation point: %d", score)
}

func TestChamberTreeExploreChamber(t *testing.T) {
	board := NewGameBoard(10, 10)

	for x := 0; x < 10; x++ {
		board.SetCellState(Position{X: x, Y: 0}, AGENT)
		board.SetCellState(Position{X: x, Y: 9}, AGENT)
	}
	for y := 1; y < 9; y++ {
		board.SetCellState(Position{X: 0, Y: y}, AGENT)
		board.SetCellState(Position{X: 9, Y: y}, AGENT)
	}

	for x := 1; x < 9; x++ {
		board.SetCellState(Position{X: x, Y: 5}, AGENT)
	}
	board.SetCellState(Position{X: 5, Y: 5}, EMPTY)

	ct := NewChamberTree(board)

	// Debug: Check what's marked as articulation point
	t.Logf("Articulation points: %d", len(ct.articulationPoints))
	for ap := range ct.articulationPoints {
		t.Logf("  AP: (%d,%d)", ap.X, ap.Y)
	}

	visited := make(map[Position]bool)
	upperChamberSize := ct.exploreChamber(Position{X: 5, Y: 3}, visited)

	t.Logf("Explored %d positions in upper chamber, visited map has %d entries", upperChamberSize, len(visited))

	// Upper chamber: rows 1-3 (3 rows) + partial row 4 (7 cells, excluding AP at 5,4) = 3*8 + 7 = 31
	expectedSize := 31
	if upperChamberSize != expectedSize {
		t.Errorf("Expected upper chamber size %d, got %d", expectedSize, upperChamberSize)
	}

	visited2 := make(map[Position]bool)
	lowerChamberSize := ct.exploreChamber(Position{X: 5, Y: 7}, visited2)

	t.Logf("Explored %d positions in lower chamber, visited map has %d entries", lowerChamberSize, len(visited2))

	// Lower chamber: partial row 6 (7 cells, excluding AP at 5,6) + rows 7-8 (2 rows) = 7 + 2*8 = 23
	expectedSize2 := 23
	if lowerChamberSize != expectedSize2 {
		t.Errorf("Expected lower chamber size %d, got %d", expectedSize2, lowerChamberSize)
	}
}

func TestBoostSafety(t *testing.T) {
	board := NewGameBoard(18, 20)

	myTrail := []Position{
		{X: 10, Y: 10},
		{X: 11, Y: 10},
	}
	myAgent := &Agent{
		AgentID:         1,
		Trail:           myTrail,
		TrailSet:        map[Position]bool{{X: 10, Y: 10}: true, {X: 11, Y: 10}: true},
		Direction:       RIGHT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 1,
	}

	for _, pos := range myTrail {
		board.SetCellState(pos, AGENT)
	}

	board.SetCellState(Position{X: 13, Y: 10}, AGENT)

	snapshot := GameStateSnapshot{
		myAgent:    myAgent,
		otherAgent: myAgent,
		board:      board,
		amIRed:     true,
	}

	if isBoostSafe(snapshot, RIGHT) {
		t.Error("Boost to RIGHT should be unsafe (obstacle at 13,10)")
	}

	board.SetCellState(Position{X: 13, Y: 10}, EMPTY)

	if !isBoostSafe(snapshot, RIGHT) {
		t.Error("Boost to RIGHT should be safe now")
	}
}

func TestBoostSafetyImmediateCollision(t *testing.T) {
	board := NewGameBoard(18, 20)

	myTrail := []Position{
		{X: 10, Y: 10},
		{X: 11, Y: 10},
	}
	myAgent := &Agent{
		AgentID:         1,
		Trail:           myTrail,
		TrailSet:        map[Position]bool{{X: 10, Y: 10}: true, {X: 11, Y: 10}: true},
		Direction:       RIGHT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 1,
	}

	for _, pos := range myTrail {
		board.SetCellState(pos, AGENT)
	}

	board.SetCellState(Position{X: 12, Y: 10}, AGENT)

	snapshot := GameStateSnapshot{
		myAgent:    myAgent,
		otherAgent: myAgent,
		board:      board,
		amIRed:     true,
	}

	if isBoostSafe(snapshot, RIGHT) {
		t.Error("Boost to RIGHT should be unsafe (immediate collision at 12,10)")
	}
}

func TestVoronoiChamberTest(t *testing.T) {
	board := NewGameBoard(10, 10)

	myAgent := &Agent{
		AgentID:         1,
		Trail:           []Position{{X: 2, Y: 5}},
		TrailSet:        map[Position]bool{{X: 2, Y: 5}: true},
		Direction:       RIGHT,
		Board:           board,
		Alive:           true,
		Length:          1,
		BoostsRemaining: 3,
	}

	otherAgent := &Agent{
		AgentID:         2,
		Trail:           []Position{{X: 7, Y: 5}},
		TrailSet:        map[Position]bool{{X: 7, Y: 5}: true},
		Direction:       LEFT,
		Board:           board,
		Alive:           true,
		Length:          1,
		BoostsRemaining: 3,
	}

	board.SetCellState(Position{X: 2, Y: 5}, AGENT)
	board.SetCellState(Position{X: 7, Y: 5}, AGENT)

	myTerritory, oppTerritory, _ := calculateVoronoiControl(myAgent, otherAgent)

	t.Logf("My territory: %d, Opponent territory: %d", myTerritory, oppTerritory)

	totalCells := myTerritory + oppTerritory
	if totalCells > 100 {
		t.Errorf("Total territory %d exceeds board size 100", totalCells)
	}

	if myTerritory < 40 || myTerritory > 60 {
		t.Errorf("Expected roughly equal territory, got %d vs %d", myTerritory, oppTerritory)
	}
}
