package main

import (
	"testing"
)

func TestCalculateSecondOrderVoronoi(t *testing.T) {
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

	score := calculateSecondOrderVoronoi(agent1, agent2)
	t.Logf("Second-order Voronoi score: %d", score)
}

func TestCalculatePotentialMobility(t *testing.T) {
	board := NewGameBoard(20, 18)
	
	agent1Trail := []Position{{X: 10, Y: 10}, {X: 11, Y: 10}}
	agent1 := &Agent{
		AgentID:         1,
		Trail:           agent1Trail,
		TrailSet:        map[Position]bool{{X: 10, Y: 10}: true, {X: 11, Y: 10}: true},
		Direction:       RIGHT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}
	for _, pos := range agent1Trail {
		board.SetCellState(pos, AGENT)
	}

	agent2Trail := []Position{{X: 5, Y: 5}, {X: 6, Y: 5}}
	agent2 := &Agent{
		AgentID:         2,
		Trail:           agent2Trail,
		TrailSet:        map[Position]bool{{X: 5, Y: 5}: true, {X: 6, Y: 5}: true},
		Direction:       LEFT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}
	for _, pos := range agent2Trail {
		board.SetCellState(pos, AGENT)
	}

	mobility := calculatePotentialMobility(agent1, agent2)
	
	if mobility < 0 {
		t.Errorf("Expected non-negative mobility, got %d", mobility)
	}
	
	t.Logf("Potential mobility score: %d", mobility)
}

func TestCalculateTrailThreat(t *testing.T) {
	board := NewGameBoard(20, 18)
	
	agent1Trail := []Position{{X: 10, Y: 10}, {X: 11, Y: 10}}
	agent1 := &Agent{
		AgentID:         1,
		Trail:           agent1Trail,
		TrailSet:        map[Position]bool{{X: 10, Y: 10}: true, {X: 11, Y: 10}: true},
		Direction:       RIGHT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}
	for _, pos := range agent1Trail {
		board.SetCellState(pos, AGENT)
	}

	agent2Trail := []Position{{X: 12, Y: 10}, {X: 13, Y: 10}}
	agent2 := &Agent{
		AgentID:         2,
		Trail:           agent2Trail,
		TrailSet:        map[Position]bool{{X: 12, Y: 10}: true, {X: 13, Y: 10}: true},
		Direction:       LEFT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}
	for _, pos := range agent2Trail {
		board.SetCellState(pos, AGENT)
	}

	threat := calculateTrailThreat(agent1, agent2)
	
	if threat < 0 {
		t.Errorf("Expected non-negative threat, got %d", threat)
	}
	
	if threat == 0 {
		t.Error("Expected positive threat (agents adjacent), got 0")
	}
	
	t.Logf("Trail threat score (adjacent agents): %d", threat)
}

func TestCalculateInfluence(t *testing.T) {
	board := NewGameBoard(20, 18)
	
	agent1Trail := []Position{{X: 10, Y: 9}, {X: 10, Y: 10}}
	agent1 := &Agent{
		AgentID:         1,
		Trail:           agent1Trail,
		TrailSet:        map[Position]bool{{X: 10, Y: 9}: true, {X: 10, Y: 10}: true},
		Direction:       UP,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}
	for _, pos := range agent1Trail {
		board.SetCellState(pos, AGENT)
	}

	agent2Trail := []Position{{X: 2, Y: 2}, {X: 3, Y: 2}}
	agent2 := &Agent{
		AgentID:         2,
		Trail:           agent2Trail,
		TrailSet:        map[Position]bool{{X: 2, Y: 2}: true, {X: 3, Y: 2}: true},
		Direction:       RIGHT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}
	for _, pos := range agent2Trail {
		board.SetCellState(pos, AGENT)
	}

	influence := calculateInfluence(agent1, agent2)
	
	if influence <= 0 {
		t.Errorf("Expected positive influence for center agent, got %d", influence)
	}
	
	t.Logf("Influence score (center vs corner): %d", influence)
}

func TestCalculateWallPenalty(t *testing.T) {
	board := NewGameBoard(20, 18)
	
	for x := 8; x <= 12; x++ {
		board.Grid[9][x] = AGENT
	}
	
	agent1Trail := []Position{{X: 10, Y: 10}, {X: 10, Y: 11}}
	agent1 := &Agent{
		AgentID:         1,
		Trail:           agent1Trail,
		TrailSet:        map[Position]bool{{X: 10, Y: 10}: true, {X: 10, Y: 11}: true},
		Direction:       DOWN,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}
	for _, pos := range agent1Trail {
		board.SetCellState(pos, AGENT)
	}

	penalty := calculateWallPenalty(agent1)
	
	if penalty <= 0 {
		t.Error("Expected positive penalty for agent near wall, got 0 or negative")
	}
	
	t.Logf("Wall penalty (agent near wall): %d", penalty)
}

func TestCalculateEscapeRoutes(t *testing.T) {
	board := NewGameBoard(20, 18)
	
	agent1Trail := []Position{{X: 10, Y: 10}, {X: 11, Y: 10}}
	agent1 := &Agent{
		AgentID:         1,
		Trail:           agent1Trail,
		TrailSet:        map[Position]bool{{X: 10, Y: 10}: true, {X: 11, Y: 10}: true},
		Direction:       RIGHT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}
	for _, pos := range agent1Trail {
		board.SetCellState(pos, AGENT)
	}

	routes := calculateEscapeRoutes(agent1)
	
	if routes < 0 {
		t.Errorf("Expected non-negative escape routes, got %d", routes)
	}
	
	if routes > 4 {
		t.Errorf("Expected at most 4 escape routes, got %d", routes)
	}
	
	t.Logf("Escape routes: %d", routes)
}

func TestCalculateCenterControl(t *testing.T) {
	board := NewGameBoard(20, 18)
	
	agent1Trail := []Position{{X: 10, Y: 9}, {X: 10, Y: 10}}
	agent1 := &Agent{
		AgentID:         1,
		Trail:           agent1Trail,
		TrailSet:        map[Position]bool{{X: 10, Y: 9}: true, {X: 10, Y: 10}: true},
		Direction:       DOWN,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}
	for _, pos := range agent1Trail {
		board.SetCellState(pos, AGENT)
	}

	agent2Trail := []Position{{X: 1, Y: 1}, {X: 2, Y: 1}}
	agent2 := &Agent{
		AgentID:         2,
		Trail:           agent2Trail,
		TrailSet:        map[Position]bool{{X: 1, Y: 1}: true, {X: 2, Y: 1}: true},
		Direction:       RIGHT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}
	for _, pos := range agent2Trail {
		board.SetCellState(pos, AGENT)
	}

	centerControl1 := calculateCenterControl(agent1)
	centerControl2 := calculateCenterControl(agent2)
	
	if centerControl1 <= centerControl2 {
		t.Errorf("Expected center agent to have higher control (%d) than corner agent (%d)", centerControl1, centerControl2)
	}
	
	t.Logf("Center control - center agent: %d, corner agent: %d", centerControl1, centerControl2)
}

func TestHeuristicsWithDeadAgent(t *testing.T) {
	board := NewGameBoard(20, 18)
	
	agent1Trail := []Position{{X: 10, Y: 10}, {X: 11, Y: 10}}
	agent1 := &Agent{
		AgentID:         1,
		Trail:           agent1Trail,
		TrailSet:        map[Position]bool{{X: 10, Y: 10}: true, {X: 11, Y: 10}: true},
		Direction:       RIGHT,
		Board:           board,
		Alive:           false,
		Length:          2,
		BoostsRemaining: 3,
	}

	agent2Trail := []Position{{X: 5, Y: 5}, {X: 6, Y: 5}}
	agent2 := &Agent{
		AgentID:         2,
		Trail:           agent2Trail,
		TrailSet:        map[Position]bool{{X: 5, Y: 5}: true, {X: 6, Y: 5}: true},
		Direction:       LEFT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}

	voronoi := calculateSecondOrderVoronoi(agent1, agent2)
	mobility := calculatePotentialMobility(agent1, agent2)
	
	if voronoi != 0 {
		t.Errorf("Expected 0 voronoi for dead agent, got %d", voronoi)
	}
	
	if mobility != 0 {
		t.Errorf("Expected 0 mobility for dead agent, got %d", mobility)
	}
}

func TestHeuristicsTorusWraparound(t *testing.T) {
	board := NewGameBoard(20, 18)
	
	agent1Trail := []Position{{X: 0, Y: 0}, {X: 1, Y: 0}}
	agent1 := &Agent{
		AgentID:         1,
		Trail:           agent1Trail,
		TrailSet:        map[Position]bool{{X: 0, Y: 0}: true, {X: 1, Y: 0}: true},
		Direction:       RIGHT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}
	for _, pos := range agent1Trail {
		board.SetCellState(pos, AGENT)
	}

	agent2Trail := []Position{{X: 19, Y: 17}, {X: 18, Y: 17}}
	agent2 := &Agent{
		AgentID:         2,
		Trail:           agent2Trail,
		TrailSet:        map[Position]bool{{X: 19, Y: 17}: true, {X: 18, Y: 17}: true},
		Direction:       LEFT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}
	for _, pos := range agent2Trail {
		board.SetCellState(pos, AGENT)
	}

	threat := calculateTrailThreat(agent1, agent2)
	spacing := calculateDefensiveSpacing(agent1, agent2)
	
	if threat == 0 {
		t.Error("Expected some threat due to torus wraparound proximity")
	}
	
	t.Logf("Trail threat with torus wraparound: %d", threat)
	t.Logf("Defensive spacing with torus wraparound: %d", spacing)
}
