package main

import (
	"testing"
)

func TestVoronoiBasic(t *testing.T) {
	board := NewGameBoard(18, 20)

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

	board.SetCellState(Position{X: 5, Y: 5}, AGENT)
	board.SetCellState(Position{X: 6, Y: 5}, AGENT)
	board.SetCellState(Position{X: 15, Y: 15}, AGENT)
	board.SetCellState(Position{X: 14, Y: 15}, AGENT)

	my, opp, control := calculateVoronoiControl(agent1, agent2)

	if my+opp > BOARD_HEIGHT*BOARD_WIDTH {
		t.Errorf("Territory sum exceeds board size: %d + %d = %d > %d", my, opp, my+opp, BOARD_HEIGHT*BOARD_WIDTH)
	}

	if my == 0 || opp == 0 {
		t.Errorf("One player has no territory: my=%d, opp=%d", my, opp)
	}

	myHead := agent1.GetHead()
	if control[myHead.Y][myHead.X] != -1 {
		t.Errorf("My head position should be marked as wall (-1), got %d", control[myHead.Y][myHead.X])
	}

	oppHead := agent2.GetHead()
	if control[oppHead.Y][oppHead.X] != -1 {
		t.Errorf("Opponent head position should be marked as wall (-1), got %d", control[oppHead.Y][oppHead.X])
	}

	t.Logf("Territory distribution: mine=%d, opponent=%d, total=%d/%d", my, opp, my+opp, BOARD_HEIGHT*BOARD_WIDTH)
}

func TestVoronoiSymmetric(t *testing.T) {
	board := NewGameBoard(18, 20)

	agent1Trail := []Position{{X: 5, Y: 9}, {X: 6, Y: 9}}
	agent1 := &Agent{
		AgentID:         1,
		Trail:           agent1Trail,
		TrailSet:        map[Position]bool{{X: 5, Y: 9}: true, {X: 6, Y: 9}: true},
		Direction:       RIGHT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}

	agent2Trail := []Position{{X: 14, Y: 9}, {X: 13, Y: 9}}
	agent2 := &Agent{
		AgentID:         2,
		Trail:           agent2Trail,
		TrailSet:        map[Position]bool{{X: 14, Y: 9}: true, {X: 13, Y: 9}: true},
		Direction:       LEFT,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}

	board.SetCellState(Position{X: 5, Y: 9}, AGENT)
	board.SetCellState(Position{X: 6, Y: 9}, AGENT)
	board.SetCellState(Position{X: 14, Y: 9}, AGENT)
	board.SetCellState(Position{X: 13, Y: 9}, AGENT)

	my, opp, _ := calculateVoronoiControl(agent1, agent2)

	diff := my - opp
	if diff < 0 {
		diff = -diff
	}

	if diff > 10 {
		t.Errorf("Symmetric positions should have similar territory: my=%d, opp=%d, diff=%d", my, opp, diff)
	}

	t.Logf("Symmetric territory: mine=%d, opponent=%d, diff=%d", my, opp, diff)
}

func TestVoronoiAdjacentEmpty(t *testing.T) {
	board := NewGameBoard(18, 20)

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

	agent2Trail := []Position{{X: 10, Y: 8}, {X: 10, Y: 7}}
	agent2 := &Agent{
		AgentID:         2,
		Trail:           agent2Trail,
		TrailSet:        map[Position]bool{{X: 10, Y: 8}: true, {X: 10, Y: 7}: true},
		Direction:       UP,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}

	board.SetCellState(Position{X: 10, Y: 9}, AGENT)
	board.SetCellState(Position{X: 10, Y: 10}, AGENT)
	board.SetCellState(Position{X: 10, Y: 8}, AGENT)
	board.SetCellState(Position{X: 10, Y: 7}, AGENT)

	my, opp, control := calculateVoronoiControl(agent1, agent2)

	emptySpace := Position{X: 11, Y: 10}
	if control[emptySpace.Y][emptySpace.X] != 1 {
		t.Errorf("Space next to my head should be mine, got %d", control[emptySpace.Y][emptySpace.X])
	}

	emptySpaceOpp := Position{X: 11, Y: 7}
	if control[emptySpaceOpp.Y][emptySpaceOpp.X] != 2 {
		t.Errorf("Space next to opponent head should be theirs, got %d", control[emptySpaceOpp.Y][emptySpaceOpp.X])
	}

	t.Logf("Adjacent test: mine=%d, opponent=%d", my, opp)
}

func TestVoronoiWithWalls(t *testing.T) {
	board := NewGameBoard(18, 20)

	for x := 8; x <= 12; x++ {
		board.SetCellState(Position{X: x, Y: 9}, AGENT)
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

	board.SetCellState(Position{X: 5, Y: 5}, AGENT)
	board.SetCellState(Position{X: 6, Y: 5}, AGENT)
	board.SetCellState(Position{X: 15, Y: 15}, AGENT)
	board.SetCellState(Position{X: 14, Y: 15}, AGENT)

	my, opp, control := calculateVoronoiControl(agent1, agent2)

	for x := 8; x <= 12; x++ {
		if control[9][x] != -1 {
			t.Errorf("Wall position (%d, 9) should be marked as -1, got %d", x, control[9][x])
		}
	}

	if my+opp > BOARD_HEIGHT*BOARD_WIDTH {
		t.Errorf("Territory sum exceeds board size with walls: %d + %d = %d > %d", my, opp, my+opp, BOARD_HEIGHT*BOARD_WIDTH)
	}

	t.Logf("With walls: mine=%d, opponent=%d", my, opp)
}

func TestVoronoiOneAgentDead(t *testing.T) {
	board := NewGameBoard(18, 20)

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

	agent2Trail := []Position{{X: 15, Y: 15}, {X: 14, Y: 15}}
	agent2 := &Agent{
		AgentID:         2,
		Trail:           agent2Trail,
		TrailSet:        map[Position]bool{{X: 15, Y: 15}: true, {X: 14, Y: 15}: true},
		Direction:       LEFT,
		Board:           board,
		Alive:           false,
		Length:          2,
		BoostsRemaining: 3,
	}

	board.SetCellState(Position{X: 5, Y: 5}, AGENT)
	board.SetCellState(Position{X: 6, Y: 5}, AGENT)

	my, opp, _ := calculateVoronoiControl(agent1, agent2)

	if my != BOARD_HEIGHT*BOARD_WIDTH {
		t.Errorf("Alive agent should control entire board when opponent is dead: got my=%d, expected %d", my, BOARD_HEIGHT*BOARD_WIDTH)
	}

	if opp != 0 {
		t.Errorf("Dead agent should control no territory: got %d", opp)
	}

	t.Logf("One dead: mine=%d, opponent=%d", my, opp)
}
