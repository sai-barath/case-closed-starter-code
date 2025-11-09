package main

import (
	"testing"
)

// ============================================================================
// Heuristics Tests
// ============================================================================

func TestDetectGamePhase(t *testing.T) {
	testCases := []struct {
		name          string
		myTrailLen    int
		otherTrailLen int
		expected      GamePhase
	}{
		{"Opening - Few cells occupied", 10, 10, Opening},
		{"Midgame - Moderate occupation", 60, 60, Midgame},
		{"Endgame - Heavy occupation", 100, 100, Endgame},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

			// Create trails
			myTrail := make([]Position, tc.myTrailLen)
			for i := 0; i < tc.myTrailLen; i++ {
				pos := Position{X: i % BOARD_WIDTH, Y: i / BOARD_WIDTH}
				myTrail[i] = pos
				board.SetCellState(pos, AGENT)
			}

			otherTrail := make([]Position, tc.otherTrailLen)
			for i := 0; i < tc.otherTrailLen; i++ {
				pos := Position{X: BOARD_WIDTH - 1 - (i % BOARD_WIDTH), Y: i / BOARD_WIDTH}
				otherTrail[i] = pos
				board.SetCellState(pos, AGENT)
			}

			myAgent := &Agent{
				AgentID:  1,
				Trail:    myTrail,
				TrailSet: make(map[Position]bool),
				Board:    board,
				Alive:    true,
				Length:   tc.myTrailLen,
			}

			otherAgent := &Agent{
				AgentID:  2,
				Trail:    otherTrail,
				TrailSet: make(map[Position]bool),
				Board:    board,
				Alive:    true,
				Length:   tc.otherTrailLen,
			}

			snapshot := GameStateSnapshot{
				myAgent:    myAgent,
				otherAgent: otherAgent,
				board:      board,
			}

			phase := detectGamePhase(snapshot)
			if phase != tc.expected {
				t.Errorf("Expected %v, got %v", tc.expected, phase)
			}
		})
	}
}

func TestEvaluatePosition_Terminal(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	aliveAgent := &Agent{
		AgentID: 1,
		Trail:   []Position{{X: 5, Y: 5}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   true,
		Length:  1,
	}

	deadAgent := &Agent{
		AgentID: 2,
		Trail:   []Position{{X: 10, Y: 10}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   false,
		Length:  1,
	}

	// Both dead -> draw
	deadAgent2 := &Agent{
		AgentID: 3,
		Trail:   []Position{{X: 15, Y: 15}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   false,
		Length:  1,
	}

	t.Run("Both dead - draw", func(t *testing.T) {
		score := evaluatePosition(deadAgent, deadAgent2)
		if score != DRAW_SCORE {
			t.Errorf("Expected DRAW_SCORE (%d), got %d", DRAW_SCORE, score)
		}
	})

	t.Run("I'm dead - loss", func(t *testing.T) {
		score := evaluatePosition(deadAgent, aliveAgent)
		if score != LOSE_SCORE {
			t.Errorf("Expected LOSE_SCORE (%d), got %d", LOSE_SCORE, score)
		}
	})

	t.Run("Opponent dead - win", func(t *testing.T) {
		score := evaluatePosition(aliveAgent, deadAgent)
		if score != WIN_SCORE {
			t.Errorf("Expected WIN_SCORE (%d), got %d", WIN_SCORE, score)
		}
	})
}

func TestManhattanDistanceTorusWrap(t *testing.T) {
	testCases := []struct {
		name     string
		x1, y1   int
		x2, y2   int
		expected int
	}{
		{"Simple case", 0, 0, 3, 4, 7},
		{"Wrap X", 0, 0, BOARD_WIDTH - 1, 0, 1},  // Wraps around
		{"Wrap Y", 0, 0, 0, BOARD_HEIGHT - 1, 1}, // Wraps around
		{"Both wrap", 0, 0, BOARD_WIDTH - 1, BOARD_HEIGHT - 1, 2},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			dist := manhattanDistanceRaw(tc.x1, tc.y1, tc.x2, tc.y2)
			if dist != tc.expected {
				t.Errorf("Expected %d, got %d", tc.expected, dist)
			}
		})
	}
}

func TestCountFreedomDegree(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	// Agent in center with all neighbors free
	centerAgent := &Agent{
		AgentID: 1,
		Trail:   []Position{{X: 10, Y: 10}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   true,
	}

	freedom := countFreedomDegree(centerAgent)
	if freedom != 4 {
		t.Errorf("Center agent should have 4 freedom, got %d", freedom)
	}

	// Block one side
	board.SetCellState(Position{X: 10, Y: 9}, AGENT)
	freedom = countFreedomDegree(centerAgent)
	if freedom != 3 {
		t.Errorf("Agent with one blocked side should have 3 freedom, got %d", freedom)
	}
}

func TestMeasureTightness(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	agent := &Agent{
		AgentID: 1,
		Trail:   []Position{{X: 10, Y: 10}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   true,
	}

	// No surrounding cells
	tightness := measureTightness(agent)
	if tightness != 0 {
		t.Errorf("Empty board should have 0 tightness, got %d", tightness)
	}

	// Add some surrounding cells
	board.SetCellState(Position{X: 9, Y: 9}, AGENT)
	board.SetCellState(Position{X: 11, Y: 11}, AGENT)
	tightness = measureTightness(agent)
	if tightness != 2 {
		t.Errorf("Expected tightness 2, got %d", tightness)
	}
}

func TestCornerProximityPenalty(t *testing.T) {
	testCases := []struct {
		name     string
		pos      Position
		expected int
	}{
		{"At corner", Position{X: 0, Y: 0}, 50},
		{"One away from corner", Position{X: 1, Y: 0}, 40},
		{"Two away from corner", Position{X: 2, Y: 0}, 30},
		{"Far from corner", Position{X: 10, Y: 10}, 0},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			penalty := cornerProximityPenalty(tc.pos)
			if penalty != tc.expected {
				t.Errorf("Expected %d, got %d", tc.expected, penalty)
			}
		})
	}
}

func TestEvaluateForcingMoves(t *testing.T) {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	myAgent := &Agent{
		AgentID: 1,
		Trail:   []Position{{X: 5, Y: 5}},
TrailSet:        make(map[Position]bool),
		Board:   board,
		Alive:   true,
	}

	// Opponent with only one valid move
	oppAgent := &Agent{
		AgentID:   2,
		Trail:     []Position{{X: 1, Y: 1}},
TrailSet:        make(map[Position]bool),
		Direction: RIGHT,
		Board:     board,
		Alive:     true,
	}

	// Block 3 sides
	board.SetCellState(Position{X: 1, Y: 0}, AGENT) // UP
	board.SetCellState(Position{X: 0, Y: 1}, AGENT) // LEFT
	board.SetCellState(Position{X: 1, Y: 2}, AGENT) // DOWN
	// RIGHT is the only free direction

	score := evaluateForcingMoves(myAgent, oppAgent)
	if score <= 0 {
		t.Errorf("Should have positive forcing score when opponent has limited moves, got %d", score)
	}
}
