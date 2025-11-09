package main

import (
	"math"
	"testing"
	"time"
)

func TestDecideMoveEmptyTrail(t *testing.T) {
	myTrail := [][]int{}
	otherTrail := [][]int{{10, 10}, {11, 10}}

	move := DecideMove(myTrail, otherTrail, 0, 3, 1)

	if move != "RIGHT" {
		t.Errorf("Expected RIGHT for empty trail, got %s", move)
	}
}

func TestDecideMoveBasic(t *testing.T) {
	myTrail := [][]int{{5, 5}, {6, 5}}
	otherTrail := [][]int{{15, 15}, {14, 15}}

	move := DecideMove(myTrail, otherTrail, 1, 3, 1)

	if move == "" {
		t.Error("Expected non-empty move")
	}
}

func TestBuildGameSnapshot(t *testing.T) {
	myTrail := [][]int{{1, 2}, {2, 2}}
	otherTrail := [][]int{{17, 15}, {16, 15}}

	snapshot := buildGameSnapshot(myTrail, otherTrail, 3, 1)

	if snapshot.myAgent == nil {
		t.Error("Expected myAgent to be non-nil")
	}
	if snapshot.otherAgent == nil {
		t.Error("Expected otherAgent to be non-nil")
	}
	if !snapshot.myAgent.Alive {
		t.Error("Expected myAgent to be alive")
	}
	if !snapshot.otherAgent.Alive {
		t.Error("Expected otherAgent to be alive")
	}
	if snapshot.myAgent.Length != 2 {
		t.Errorf("Expected myAgent length 2, got %d", snapshot.myAgent.Length)
	}
}

func TestCreateAgentFromTrail(t *testing.T) {
	board := NewGameBoard(18, 20)
	trail := [][]int{{5, 5}, {6, 5}, {7, 5}}

	agent := createAgentFromTrail(1, trail, RIGHT, 3, board)

	if agent == nil {
		t.Fatal("Expected agent to be non-nil")
	}
	if agent.Length != 3 {
		t.Errorf("Expected length 3, got %d", agent.Length)
	}
	if agent.BoostsRemaining != 3 {
		t.Errorf("Expected 3 boosts, got %d", agent.BoostsRemaining)
	}
	if !agent.Alive {
		t.Error("Expected agent to be alive")
	}
}

func TestInferDirection(t *testing.T) {
	tests := []struct {
		name     string
		trail    [][]int
		expected Direction
	}{
		{"single position", [][]int{{5, 5}}, RIGHT},
		{"moving right", [][]int{{5, 5}, {6, 5}}, RIGHT},
		{"moving left", [][]int{{5, 5}, {4, 5}}, LEFT},
		{"moving down", [][]int{{5, 5}, {5, 6}}, DOWN},
		{"moving up", [][]int{{5, 5}, {5, 4}}, UP},
		{"torus wrap right", [][]int{{19, 5}, {0, 5}}, RIGHT},
		{"torus wrap left", [][]int{{0, 5}, {19, 5}}, LEFT},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := inferDirection(tt.trail)
			if result != tt.expected {
				t.Errorf("Expected %+v, got %+v", tt.expected, result)
			}
		})
	}
}

func TestDirectionToString(t *testing.T) {
	tests := []struct {
		dir      Direction
		expected string
	}{
		{UP, "UP"},
		{DOWN, "DOWN"},
		{LEFT, "LEFT"},
		{RIGHT, "RIGHT"},
	}

	for _, tt := range tests {
		result := directionToString(tt.dir)
		if result != tt.expected {
			t.Errorf("Expected %s, got %s", tt.expected, result)
		}
	}
}

func TestCountAvailableSpace(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)

	space := countAvailableSpace(agent)

	if space <= 0 {
		t.Errorf("Expected positive space count, got %d", space)
	}
	if space > 150 {
		t.Errorf("Expected space count <= 150, got %d", space)
	}
}

func TestCountAvailableSpaceDeadAgent(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)
	agent.Alive = false

	space := countAvailableSpace(agent)

	if space != 0 {
		t.Errorf("Expected 0 space for dead agent, got %d", space)
	}
}

func TestEvaluatePositionBothDead(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent1 := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)
	agent2 := NewAgent(2, Position{X: 10, Y: 10}, LEFT, board)

	agent1.Alive = false
	agent2.Alive = false

	score := evaluatePosition(agent1, agent2)

	if score != DRAW_SCORE {
		t.Errorf("Expected DRAW_SCORE (%d), got %d", DRAW_SCORE, score)
	}
}

func TestEvaluatePositionMyAgentDead(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent1 := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)
	agent2 := NewAgent(2, Position{X: 10, Y: 10}, LEFT, board)

	agent1.Alive = false

	score := evaluatePosition(agent1, agent2)

	if score != LOSE_SCORE {
		t.Errorf("Expected LOSE_SCORE (%d), got %d", LOSE_SCORE, score)
	}
}

func TestEvaluatePositionOtherAgentDead(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent1 := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)
	agent2 := NewAgent(2, Position{X: 10, Y: 10}, LEFT, board)

	agent2.Alive = false

	score := evaluatePosition(agent1, agent2)

	if score != WIN_SCORE {
		t.Errorf("Expected WIN_SCORE (%d), got %d", WIN_SCORE, score)
	}
}

func TestAlphabetaBasic(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent1 := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)
	agent2 := NewAgent(2, Position{X: 10, Y: 10}, LEFT, board)

	ctx := &SearchContext{
		startTime: time.Now(),
		deadline:  time.Now().Add(100 * time.Millisecond),
	}

	score := alphabeta(agent1, agent2, 1, math.MinInt32, math.MaxInt32, true, ctx)

	if score == 0 {
		t.Log("Score is 0 - this is acceptable for symmetric positions")
	}
}

func TestSearchAtDepthBasic(t *testing.T) {
	myTrail := [][]int{{5, 5}, {6, 5}}
	otherTrail := [][]int{{15, 15}, {14, 15}}

	snapshot := buildGameSnapshot(myTrail, otherTrail, 3, 1)

	ctx := &SearchContext{
		startTime: time.Now(),
		deadline:  time.Now().Add(100 * time.Millisecond),
	}

	move := searchAtDepth(snapshot, 2, ctx)

	if move.direction == (Direction{}) {
		t.Error("Expected valid direction")
	}
}

func TestMaxFunction(t *testing.T) {
	if max(5, 10) != 10 {
		t.Error("max(5, 10) should be 10")
	}
	if max(10, 5) != 10 {
		t.Error("max(10, 5) should be 10")
	}
	if max(-5, -10) != -5 {
		t.Error("max(-5, -10) should be -5")
	}
}

func TestAbsFunction(t *testing.T) {
	if abs(5) != 5 {
		t.Error("abs(5) should be 5")
	}
	if abs(-5) != 5 {
		t.Error("abs(-5) should be 5")
	}
	if abs(0) != 0 {
		t.Error("abs(0) should be 0")
	}
}
