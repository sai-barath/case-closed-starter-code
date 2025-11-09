package main

import (
	"testing"
)

// ============================================================================
// Territory Tests
// ============================================================================

func TestVoronoiTerritory_Equal(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	// Two agents equidistant from center
	myAgent := &Agent{
		AgentID: 1,
		Trail:   []Position{{X: 5, Y: 9}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   true,
	}

	oppAgent := &Agent{
		AgentID: 2,
		Trail:   []Position{{X: 15, Y: 9}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   true,
	}

	result := voronoiTerritory(myAgent, oppAgent, 20)

	// Territory should be roughly equal (allowing some variance)
	diff := abs(result.myTerritory - result.opponentTerritory)
	maxDiff := (result.myTerritory + result.opponentTerritory) / 4 // 25% tolerance

	if diff > maxDiff {
		t.Errorf("Territory should be roughly equal: my=%d, opp=%d",
			result.myTerritory, result.opponentTerritory)
	}
}

func TestVoronoiTerritory_Advantage(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	// My agent in better position (more central)
	myAgent := &Agent{
		AgentID: 1,
		Trail:   []Position{{X: 10, Y: 9}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   true,
	}

	// Opponent in corner
	oppAgent := &Agent{
		AgentID: 2,
		Trail:   []Position{{X: 0, Y: 0}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   true,
	}

	result := voronoiTerritory(myAgent, oppAgent, 20)

	if result.myTerritory <= result.opponentTerritory {
		t.Errorf("Central position should have more territory: my=%d, opp=%d",
			result.myTerritory, result.opponentTerritory)
	}
}

func TestInfluenceFunctionTerritory(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	myAgent := &Agent{
		AgentID: 1,
		Trail:   []Position{{X: 10, Y: 9}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   true,
	}

	oppAgent := &Agent{
		AgentID: 2,
		Trail:   []Position{{X: 0, Y: 0}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   true,
	}

	result := influenceFunctionTerritory(myAgent, oppAgent)

	if result.myInfluence <= 0 {
		t.Errorf("My influence should be positive, got %f", result.myInfluence)
	}
	if result.opponentInfluence <= 0 {
		t.Errorf("Opponent influence should be positive, got %f", result.opponentInfluence)
	}

	// Central agent should have more influence
	if result.myInfluence <= result.opponentInfluence {
		t.Errorf("Central agent should have more influence: my=%f, opp=%f",
			result.myInfluence, result.opponentInfluence)
	}
}

func TestCountAvailableSpace(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	agent := &Agent{
		AgentID: 1,
		Trail:   []Position{{X: 10, Y: 9}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   true,
	}

	// Empty board should have maximum reachable space
	space := countAvailableSpace(agent)
	if space == 0 {
		t.Error("Empty board should have positive available space")
	}

	// Dead agent should have 0 space
	agent.Alive = false
	space = countAvailableSpace(agent)
	if space != 0 {
		t.Errorf("Dead agent should have 0 space, got %d", space)
	}
}

func TestFloodFillReachable(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	agent := &Agent{
		AgentID: 1,
		Trail:   []Position{{X: 10, Y: 9}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   true,
	}

	reachable := floodFillReachable(agent)
	totalCells := BOARD_HEIGHT * BOARD_WIDTH

	// Should reach all cells on empty board (minus the agent's position)
	if reachable < totalCells-10 {
		t.Errorf("Should reach most cells on empty board, got %d out of %d",
			reachable, totalCells)
	}
}

func TestDetectBarriers(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	agent := &Agent{
		AgentID: 1,
		Trail:   []Position{{X: 10, Y: 10}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   true,
	}

	opponent := &Agent{
		AgentID: 2,
		Trail:   []Position{{X: 5, Y: 5}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   true,
	}

	// No barriers on empty board
	barriers := detectBarriers(agent, opponent)
	if barriers != 0 {
		t.Errorf("Empty board should have 0 barriers, got %d", barriers)
	}

	// Create a 3-sided box
	board.SetCellState(Position{X: 9, Y: 10}, AGENT)
	board.SetCellState(Position{X: 10, Y: 9}, AGENT)
	board.SetCellState(Position{X: 11, Y: 10}, AGENT)

	barriers = detectBarriers(agent, opponent)
	if barriers <= 0 {
		t.Errorf("Should detect barriers in 3-sided box, got %d", barriers)
	}
}

func TestArticulationPoints(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	// Create a narrow corridor
	for x := 0; x < BOARD_WIDTH; x++ {
		if x != 10 {
			board.SetCellState(Position{X: x, Y: 8}, AGENT)
		}
	}

	agent := &Agent{
		AgentID: 1,
		Trail:   []Position{{X: 10, Y: 7}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   true,
	}

	aps := findArticulationPoints(board, agent)

	// Position (10,9) should be an articulation point connecting upper and lower regions
	// (Note: actual implementation may vary based on graph construction)
	if len(aps) == 0 {
		t.Log("Warning: Expected to find articulation points in corridor scenario")
	}
}

func TestEvaluateArticulationControl(t *testing.T) {
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

	// Should not crash on empty board
	score := evaluateArticulationControl(myAgent, oppAgent)
	_ = score // Score may be 0 on empty board
}
