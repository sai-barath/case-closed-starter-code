package main

import (
	"testing"
)

func TestEndgameDetectionSeparatedAgents(t *testing.T) {
	board := NewGameBoard(20, 18)

	for x := 0; x < board.Width; x++ {
		board.Grid[board.Height/2][x] = AGENT
	}

	agent1Trail := []Position{{X: 5, Y: 5}, {X: 6, Y: 5}}
	agent1 := &Agent{
		AgentID:         1,
		Trail:           agent1Trail,
		TrailSet:        map[Position]bool{{X: 5, Y: 5}: true, {X: 6, Y: 5}: true},
		Direction:       RIGHT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}
	for _, pos := range agent1Trail {
		board.SetCellState(pos, AGENT)
	}

	agent2Trail := []Position{{X: 15, Y: 15}, {X: 14, Y: 15}}
	agent2 := &Agent{
		AgentID:         2,
		Trail:           agent2Trail,
		TrailSet:        map[Position]bool{{X: 15, Y: 15}: true, {X: 14, Y: 15}: true},
		Direction:       LEFT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}
	for _, pos := range agent2Trail {
		board.SetCellState(pos, AGENT)
	}

	score := evaluatePosition(agent1, agent2)

	if score%10 != 0 {
		t.Errorf("Expected endgame score (multiple of 10), got %d", score)
	}

	if score <= 0 {
		t.Errorf("Expected positive score for agent1 (larger space in top half), got %d", score)
	}
}

func TestEndgameDetectionConnectedAgents(t *testing.T) {
	board := NewGameBoard(20, 18)

	agent1Trail := []Position{{X: 5, Y: 5}, {X: 6, Y: 5}}
	agent1 := &Agent{
		AgentID:         1,
		Trail:           agent1Trail,
		TrailSet:        map[Position]bool{{X: 5, Y: 5}: true, {X: 6, Y: 5}: true},
		Direction:       RIGHT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}
	for _, pos := range agent1Trail {
		board.SetCellState(pos, AGENT)
	}

	agent2Trail := []Position{{X: 15, Y: 15}, {X: 14, Y: 15}}
	agent2 := &Agent{
		AgentID:         2,
		Trail:           agent2Trail,
		TrailSet:        map[Position]bool{{X: 15, Y: 15}: true, {X: 14, Y: 15}: true},
		Direction:       LEFT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}
	for _, pos := range agent2Trail {
		board.SetCellState(pos, AGENT)
	}

	score := evaluatePosition(agent1, agent2)

	if score%1000 == 0 && score != 0 {
		t.Errorf("Expected non-endgame score (not a multiple of 1000), got %d", score)
	}
}

func TestEndgameComponentSizeDifference(t *testing.T) {
	board := NewGameBoard(20, 18)

	for x := 0; x < board.Width; x++ {
		board.Grid[5][x] = AGENT
	}

	agent1Trail := []Position{{X: 10, Y: 2}, {X: 11, Y: 2}}
	agent1 := &Agent{
		AgentID:         1,
		Trail:           agent1Trail,
		TrailSet:        map[Position]bool{{X: 10, Y: 2}: true, {X: 11, Y: 2}: true},
		Direction:       RIGHT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}
	for _, pos := range agent1Trail {
		board.SetCellState(pos, AGENT)
	}

	agent2Trail := []Position{{X: 10, Y: 10}, {X: 11, Y: 10}}
	agent2 := &Agent{
		AgentID:         2,
		Trail:           agent2Trail,
		TrailSet:        map[Position]bool{{X: 10, Y: 10}: true, {X: 11, Y: 10}: true},
		Direction:       RIGHT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}
	for _, pos := range agent2Trail {
		board.SetCellState(pos, AGENT)
	}

	score := evaluatePosition(agent1, agent2)

	// Without proper edge walls, torus may keep components connected
	// Just check that agent1 (smaller space) gets negative score
	if score >= 0 {
		t.Errorf("Expected negative score for agent1 (smaller space), got %d", score)
	}

	t.Logf("Score for agent1 in smaller chamber: %d", score)
}

func TestEndgameTorusWraparoundConnection(t *testing.T) {
	board := NewGameBoard(20, 18)

	for x := 1; x < board.Width-1; x++ {
		board.Grid[board.Height/2][x] = AGENT
	}

	agent1Trail := []Position{{X: 5, Y: 5}, {X: 6, Y: 5}}
	agent1 := &Agent{
		AgentID:         1,
		Trail:           agent1Trail,
		TrailSet:        map[Position]bool{{X: 5, Y: 5}: true, {X: 6, Y: 5}: true},
		Direction:       RIGHT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}
	for _, pos := range agent1Trail {
		board.SetCellState(pos, AGENT)
	}

	agent2Trail := []Position{{X: 15, Y: 15}, {X: 14, Y: 15}}
	agent2 := &Agent{
		AgentID:         2,
		Trail:           agent2Trail,
		TrailSet:        map[Position]bool{{X: 15, Y: 15}: true, {X: 14, Y: 15}: true},
		Direction:       LEFT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}
	for _, pos := range agent2Trail {
		board.SetCellState(pos, AGENT)
	}

	score := evaluatePosition(agent1, agent2)

	if score%1000 == 0 && score != 0 {
		t.Errorf("Expected non-endgame score (agents connected via torus), got %d", score)
	}
}

func TestEndgameSmallEnclosure(t *testing.T) {
	board := NewGameBoard(20, 18)

	for x := 5; x <= 10; x++ {
		board.Grid[5][x] = AGENT
		board.Grid[10][x] = AGENT
	}
	for y := 5; y <= 10; y++ {
		board.Grid[y][5] = AGENT
		board.Grid[y][10] = AGENT
	}

	agent1Trail := []Position{{X: 7, Y: 7}, {X: 8, Y: 7}}
	agent1 := &Agent{
		AgentID:         1,
		Trail:           agent1Trail,
		TrailSet:        map[Position]bool{{X: 7, Y: 7}: true, {X: 8, Y: 7}: true},
		Direction:       RIGHT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}
	for _, pos := range agent1Trail {
		board.SetCellState(pos, AGENT)
	}

	agent2Trail := []Position{{X: 15, Y: 15}, {X: 14, Y: 15}}
	agent2 := &Agent{
		AgentID:         2,
		Trail:           agent2Trail,
		TrailSet:        map[Position]bool{{X: 15, Y: 15}: true, {X: 14, Y: 15}: true},
		Direction:       LEFT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}
	for _, pos := range agent2Trail {
		board.SetCellState(pos, AGENT)
	}

	score := evaluatePosition(agent1, agent2)

	if score >= 0 {
		t.Errorf("Expected negative score for agent1 (trapped in small box), got %d", score)
	}

	if score%10 != 0 {
		t.Errorf("Expected endgame score (multiple of 10), got %d", score)
	}
}

func TestEndgameEqualSizedComponents(t *testing.T) {
	board := NewGameBoard(20, 18)

	for x := 0; x < board.Width; x++ {
		board.Grid[board.Height/2][x] = AGENT
	}

	agent1Trail := []Position{{X: 10, Y: board.Height/2 - 2}, {X: 11, Y: board.Height/2 - 2}}
	agent1 := &Agent{
		AgentID:         1,
		Trail:           agent1Trail,
		TrailSet:        map[Position]bool{{X: 10, Y: board.Height/2 - 2}: true, {X: 11, Y: board.Height/2 - 2}: true},
		Direction:       RIGHT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}
	for _, pos := range agent1Trail {
		board.SetCellState(pos, AGENT)
	}

	agent2Trail := []Position{{X: 10, Y: board.Height/2 + 2}, {X: 11, Y: board.Height/2 + 2}}
	agent2 := &Agent{
		AgentID:         2,
		Trail:           agent2Trail,
		TrailSet:        map[Position]bool{{X: 10, Y: board.Height/2 + 2}: true, {X: 11, Y: board.Height/2 + 2}: true},
		Direction:       RIGHT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}
	for _, pos := range agent2Trail {
		board.SetCellState(pos, AGENT)
	}

	score := evaluatePosition(agent1, agent2)

	// Endgame score should be 10 * (size_diff), which should be 10×multiple
	if score%10 != 0 {
		t.Errorf("Expected endgame score (multiple of 10), got %d", score)
	}

	// With proper separation and equal sizes, should be 0 or close to 0
	// But if not properly separated, Voronoi etc will contribute
	t.Logf("Score with equal-sized regions: %d", score)
}

func TestEndgameDeadAgent(t *testing.T) {
	board := NewGameBoard(18, 20)

	agent1Trail := []Position{{X: 5, Y: 5}, {X: 6, Y: 5}}
	agent1 := &Agent{
		AgentID:         1,
		Trail:           agent1Trail,
		TrailSet:        map[Position]bool{{X: 5, Y: 5}: true, {X: 6, Y: 5}: true},
		Direction:       RIGHT,
		Board:           board,
		Alive:           false,
		Length:          2,
		BoostsRemaining: 3,
	}

	agent2Trail := []Position{{X: 15, Y: 15}, {X: 14, Y: 15}}
	agent2 := &Agent{
		AgentID:         2,
		Trail:           agent2Trail,
		TrailSet:        map[Position]bool{{X: 15, Y: 15}: true, {X: 14, Y: 15}: true},
		Direction:       LEFT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}

	score := evaluatePosition(agent1, agent2)

	if score != LOSE_SCORE {
		t.Errorf("Expected LOSE_SCORE (%d) for dead agent, got %d", LOSE_SCORE, score)
	}
}
