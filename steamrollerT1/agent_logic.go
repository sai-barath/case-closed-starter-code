package main

import (
	"time"
)

// ============================================================================
// Data structures for search
// ============================================================================

type GameStateSnapshot struct {
	myAgent    *Agent
	otherAgent *Agent
	board      *GameBoard
	amIRed     bool
}

type Move struct {
	direction Direction
	useBoost  bool
	score     int
}

type scoredMove struct {
	dir   Direction
	score int
}

// ============================================================================
// Main entry point - called every turn
// ============================================================================

func DecideMove(myTrail, otherTrail [][]int, turnCount, myBoosts, playerNumber int) string {
	logDebug("Turn %d: Starting iterative deepening search with %d boosts (Player %d)", turnCount, myBoosts, playerNumber)

	if len(myTrail) == 0 {
		return "RIGHT"
	}

	snapshot := buildGameSnapshot(myTrail, otherTrail, myBoosts, playerNumber)

	if !snapshot.myAgent.Alive {
		return "RIGHT"
	}

	transpositionTable = make(map[uint64]TTEntry)

	ctx := SearchContext{
		startTime:     time.Now(),
		deadline:      time.Now().Add(SEARCH_TIME_LIMIT),
		moveOrdering:  NewMoveOrderingContext(),
		nodesSearched: 0,
	}

	bestMove := iterativeDeepeningSearch(snapshot, ctx)

	elapsed := time.Since(ctx.startTime)

	moveStr := directionToString(bestMove.direction)
	if bestMove.useBoost {
		moveStr += ":BOOST"
	}

	logDebug("Selected move: %s (score: %d, time: %v, nodes: %d)", moveStr, bestMove.score, elapsed, ctx.nodesSearched)
	return moveStr
}

// ============================================================================
// Boilerplate: Build game state from trail data
// ============================================================================

func buildGameSnapshot(myTrail, otherTrail [][]int, myBoosts, playerNumber int) GameStateSnapshot {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	// Mark all trail positions as occupied
	for _, pos := range myTrail {
		board.SetCellState(Position{X: pos[0], Y: pos[1]}, AGENT)
	}
	for _, pos := range otherTrail {
		board.SetCellState(Position{X: pos[0], Y: pos[1]}, AGENT)
	}

	// Create agent objects
	myStartPos := Position{X: myTrail[0][0], Y: myTrail[0][1]}
	myDir := inferDirection(myTrail)
	myAgent := createAgentFromTrail(1, myTrail, myDir, myBoosts, board)

	otherStartPos := Position{X: otherTrail[0][0], Y: otherTrail[0][1]}
	otherDir := inferDirection(otherTrail)
	otherAgent := createAgentFromTrail(2, otherTrail, otherDir, 3, board)

	logDebug("Built snapshot: my pos (%d,%d), other pos (%d,%d)",
		myStartPos.X, myStartPos.Y, otherStartPos.X, otherStartPos.Y)

	return GameStateSnapshot{
		myAgent:    myAgent,
		otherAgent: otherAgent,
		board:      board,
		amIRed:     playerNumber == 1,
	}
}

func createAgentFromTrail(agentID int, trail [][]int, dir Direction, boosts int, board *GameBoard) *Agent {
	if len(trail) == 0 {
		return nil
	}

	trailPositions := make([]Position, len(trail))
	trailSet := make(map[Position]bool)

	for i, pos := range trail {
		p := Position{X: pos[0], Y: pos[1]}
		trailPositions[i] = p
		trailSet[p] = true
	}

	return &Agent{
		AgentID:         agentID,
		Trail:           trailPositions,
		TrailSet:        trailSet,
		Direction:       dir,
		Board:           board,
		Alive:           true,
		Length:          len(trail),
		BoostsRemaining: boosts,
	}
}

func inferDirection(trail [][]int) Direction {
	if len(trail) < 2 {
		return RIGHT
	}

	head := trail[len(trail)-1]
	prev := trail[len(trail)-2]

	dx := head[0] - prev[0]
	dy := head[1] - prev[1]

	// Handle torus wraparound
	if abs(dx) > 1 {
		if dx > 0 {
			dx = -1
		} else {
			dx = 1
		}
	}
	if abs(dy) > 1 {
		if dy > 0 {
			dy = -1
		} else {
			dy = 1
		}
	}

	if dx == 1 {
		return RIGHT
	} else if dx == -1 {
		return LEFT
	} else if dy == 1 {
		return DOWN
	} else if dy == -1 {
		return UP
	}

	return RIGHT
}

func directionToString(dir Direction) string {
	switch dir {
	case UP:
		return "UP"
	case DOWN:
		return "DOWN"
	case LEFT:
		return "LEFT"
	case RIGHT:
		return "RIGHT"
	default:
		return "RIGHT"
	}
}
