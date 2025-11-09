package main

import (
	"testing"
)

func TestUndoMove_SingleMove(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)

	originalTrailLen := len(agent.Trail)
	originalLength := agent.Length
	originalDir := agent.Direction

	success, state := agent.UndoableMove(UP, nil, false)

	if !success {
		t.Fatal("Move should succeed")
	}

	if len(agent.Trail) != originalTrailLen+1 {
		t.Errorf("Trail length should be %d, got %d", originalTrailLen+1, len(agent.Trail))
	}

	newHead := agent.GetHead()
	if !agent.TrailSet[newHead] {
		t.Error("New head should be in TrailSet")
	}

	if board.GetCellState(newHead) != AGENT {
		t.Error("New head position should be marked AGENT on board")
	}

	agent.UndoMove(state, nil)

	if len(agent.Trail) != originalTrailLen {
		t.Errorf("After undo, trail length should be %d, got %d", originalTrailLen, len(agent.Trail))
	}

	if agent.Length != originalLength {
		t.Errorf("After undo, length should be %d, got %d", originalLength, agent.Length)
	}

	if agent.Direction != originalDir {
		t.Errorf("After undo, direction should be %v, got %v", originalDir, agent.Direction)
	}

	if agent.TrailSet[newHead] {
		t.Error("After undo, new head should NOT be in TrailSet")
	}

	if board.GetCellState(newHead) != EMPTY {
		t.Error("After undo, new head position should be EMPTY on board")
	}
}

func TestUndoMove_WithBoost(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)

	originalTrailLen := len(agent.Trail)
	originalBoosts := agent.BoostsRemaining

	success, state := agent.UndoableMove(UP, nil, true)

	if !success {
		t.Fatal("Move should succeed")
	}

	if len(agent.Trail) != originalTrailLen+2 {
		t.Errorf("Trail length with boost should be %d, got %d", originalTrailLen+2, len(agent.Trail))
	}

	if agent.BoostsRemaining != originalBoosts-1 {
		t.Errorf("Boosts should be %d, got %d", originalBoosts-1, agent.BoostsRemaining)
	}

	pos1 := agent.Trail[len(agent.Trail)-2]
	pos2 := agent.Trail[len(agent.Trail)-1]

	agent.UndoMove(state, nil)

	if len(agent.Trail) != originalTrailLen {
		t.Errorf("After undo, trail length should be %d, got %d", originalTrailLen, len(agent.Trail))
	}

	if agent.BoostsRemaining != originalBoosts {
		t.Errorf("After undo, boosts should be %d, got %d", originalBoosts, agent.BoostsRemaining)
	}

	if agent.TrailSet[pos1] {
		t.Error("After undo, first boost position should NOT be in TrailSet")
	}

	if agent.TrailSet[pos2] {
		t.Error("After undo, second boost position should NOT be in TrailSet")
	}

	if board.GetCellState(pos1) != EMPTY {
		t.Error("After undo, first boost position should be EMPTY")
	}

	if board.GetCellState(pos2) != EMPTY {
		t.Error("After undo, second boost position should be EMPTY")
	}
}

func TestUndoMove_MultipleSequentialMoves(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 10, Y: 10}, RIGHT, board)

	originalTrailLen := len(agent.Trail)

	var states []MoveState
	moves := []Direction{UP, UP, RIGHT, RIGHT}

	for _, dir := range moves {
		success, state := agent.UndoableMove(dir, nil, false)
		if !success {
			t.Fatalf("Move %v should succeed", dir)
		}
		states = append(states, state)
	}

	if len(agent.Trail) != originalTrailLen+4 {
		t.Errorf("After 4 moves, trail length should be %d, got %d", originalTrailLen+4, len(agent.Trail))
	}

	for i := len(states) - 1; i >= 0; i-- {
		agent.UndoMove(states[i], nil)
	}

	if len(agent.Trail) != originalTrailLen {
		t.Errorf("After undoing all moves, trail length should be %d, got %d", originalTrailLen, len(agent.Trail))
	}

	finalHead := agent.GetHead()
	originalHead := Position{X: 11, Y: 10}

	if finalHead.X != originalHead.X || finalHead.Y != originalHead.Y {
		t.Errorf("After undoing all moves, head should be at %v, got %v", originalHead, finalHead)
	}
}

func TestUndoMove_CollisionWithOwnTrail(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 10, Y: 10}, RIGHT, board)

	agent.UndoableMove(UP, nil, false)
	agent.UndoableMove(UP, nil, false)
	agent.UndoableMove(UP, nil, false)
	agent.UndoableMove(LEFT, nil, false)
	agent.UndoableMove(LEFT, nil, false)
	agent.UndoableMove(DOWN, nil, false)
	agent.UndoableMove(DOWN, nil, false)
	agent.UndoableMove(DOWN, nil, false)

	success, state := agent.UndoableMove(RIGHT, nil, false)

	if success {
		t.Error("Move should fail - collision with own trail")
	}

	if agent.Alive {
		t.Error("Agent should be dead after collision")
	}

	if !state.MyAliveChanged {
		t.Error("State should indicate alive status changed")
	}

	agent.UndoMove(state, nil)

	if !agent.Alive {
		t.Error("After undo, agent should be alive again")
	}
}

func TestUndoMove_CollisionWithOtherAgent(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent1 := NewAgent(1, Position{X: 5, Y: 10}, RIGHT, board)
	agent2 := NewAgent(2, Position{X: 14, Y: 10}, LEFT, board)

	_, state1 := agent1.UndoableMove(RIGHT, agent2, false)
	_, state2 := agent1.UndoableMove(RIGHT, agent2, false)
	_, state3 := agent1.UndoableMove(RIGHT, agent2, false)

	_, state4 := agent2.UndoableMove(LEFT, agent1, false)
	_, state5 := agent2.UndoableMove(LEFT, agent1, false)
	_, state6 := agent2.UndoableMove(LEFT, agent1, false)

	t.Logf("Agent1 head: %v, trail: %v", agent1.GetHead(), agent1.Trail)
	t.Logf("Agent2 head: %v, trail: %v", agent2.GetHead(), agent2.Trail)
	t.Logf("Before collision move: agent1.Alive=%v, agent2.Alive=%v", agent1.Alive, agent2.Alive)

	if !agent1.Alive || !agent2.Alive {
		agent2.UndoMove(state6, agent1)
		agent2.UndoMove(state5, agent1)
		agent2.UndoMove(state4, agent1)
		agent1.UndoMove(state3, agent2)
		agent1.UndoMove(state2, agent2)
		agent1.UndoMove(state1, agent2)
		t.Fatal("Agents died during setup moves, test setup is incorrect")
	}

	success, state := agent1.UndoableMove(RIGHT, agent2, false)

	t.Logf("Move result: success=%v, agent1.alive=%v, agent2.alive=%v", success, agent1.Alive, agent2.Alive)
	t.Logf("State: myAliveChanged=%v, otherAliveChanged=%v, oldMyAlive=%v, oldOtherAlive=%v", state.MyAliveChanged, state.OtherAliveChanged, state.OldMyAlive, state.OldOtherAlive)

	if success {
		t.Error("Agent1 move should fail - head-on collision")
	}

	if agent1.Alive {
		t.Error("Agent1 should be dead after collision")
	}

	if agent2.Alive {
		t.Error("Agent2 should be dead after head-on collision")
	}

	agent1.UndoMove(state, agent2)

	t.Logf("After undo: agent1.alive=%v, agent2.alive=%v", agent1.Alive, agent2.Alive)

	if !agent1.Alive {
		t.Error("After undo, agent1 should be alive again")
	}

	if !agent2.Alive {
		t.Error("After undo, agent2 should be alive again")
	}
}

func TestUndoMove_ConsistencyCheck(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 10, Y: 10}, RIGHT, board)

	for i := 0; i < 10; i++ {
		success, state := agent.UndoableMove(UP, nil, false)
		if !success {
			t.Fatalf("Move %d should succeed", i)
		}

		head := agent.GetHead()

		if !agent.TrailSet[head] {
			t.Errorf("Move %d: Head %v should be in TrailSet", i, head)
		}

		if board.GetCellState(head) != AGENT {
			t.Errorf("Move %d: Head %v should be marked AGENT on board", i, head)
		}

		found := false
		for _, pos := range agent.Trail {
			if pos.X == head.X && pos.Y == head.Y {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Move %d: Head %v should be in Trail array", i, head)
		}

		agent.UndoMove(state, nil)

		if agent.TrailSet[head] {
			t.Errorf("Move %d undo: Head %v should NOT be in TrailSet after undo", i, head)
		}

		if board.GetCellState(head) != EMPTY {
			t.Errorf("Move %d undo: Head %v should be EMPTY on board after undo", i, head)
		}

		for _, pos := range agent.Trail {
			if pos.X == head.X && pos.Y == head.Y {
				t.Errorf("Move %d undo: Head %v should NOT be in Trail array after undo", i, head)
			}
		}
	}
}

func TestGetValidMoves_AfterUndoMove(t *testing.T) {
	board := NewGameBoard(18, 20)
	agent := NewAgent(1, Position{X: 10, Y: 10}, RIGHT, board)

	originalValidMoves := len(agent.GetValidMoves())

	success, state := agent.UndoableMove(UP, nil, false)
	if !success {
		t.Fatal("Move should succeed")
	}

	agent.UndoMove(state, nil)

	afterUndoValidMoves := len(agent.GetValidMoves())

	if originalValidMoves != afterUndoValidMoves {
		t.Errorf("Valid moves count should be same after undo. Before: %d, After: %d",
			originalValidMoves, afterUndoValidMoves)
	}

	validMoves := agent.GetValidMoves()
	for _, dir := range validMoves {
		head := agent.GetHead()
		nextPos := Position{X: head.X + dir.DX, Y: head.Y + dir.DY}
		nextPos = board.TorusCheck(nextPos)

		if agent.ContainsPosition(nextPos) {
			t.Errorf("Valid move %v leads to position %v which is in our own trail", dir, nextPos)
		}

		if board.GetCellState(nextPos) == AGENT {
			t.Errorf("Valid move %v leads to position %v which is marked AGENT", dir, nextPos)
		}
	}
}
