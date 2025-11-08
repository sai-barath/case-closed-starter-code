package main

import (
	"fmt"
	"math"
	"os"
	"time"
)

// ============================================================================
// BOILERPLATE: Configuration and utilities
// ============================================================================

var debugMode = os.Getenv("DEBUG") == "1"

const (
	SEARCH_TIME_LIMIT = 3500 * time.Millisecond // **CUSTOMIZE**: Time budget per move
	WIN_SCORE         = 10000                   // Terminal score for winning
	LOSE_SCORE        = -10000                  // Terminal score for losing
	DRAW_SCORE        = 0                       // Terminal score for draw
	BOARD_HEIGHT      = 18
	BOARD_WIDTH       = 20
)

func logDebug(format string, args ...interface{}) {
	if debugMode {
		fmt.Printf("[DEBUG] "+format+"\n", args...)
	}
}

// ============================================================================
// BOILERPLATE: Data structures for search
// ============================================================================

type GameStateSnapshot struct {
	myAgent    *Agent
	otherAgent *Agent
	board      *GameBoard
}

type Move struct {
	direction Direction
	useBoost  bool
	score     int
}

type SearchContext struct {
	startTime time.Time
	deadline  time.Time
}

// ============================================================================
// YOUR LOGIC: Main entry point - this is called every turn
// ============================================================================

func DecideMove(myTrail, otherTrail [][]int, turnCount, myBoosts int) string {
	logDebug("Turn %d: Starting iterative deepening search with %d boosts", turnCount, myBoosts)
	
	// Fallback for first turn
	if len(myTrail) == 0 {
		return "RIGHT"
	}
	
	// BOILERPLATE: Build game state from trail data
	snapshot := buildGameSnapshot(myTrail, otherTrail, myBoosts)
	
	// BOILERPLATE: Set up search timer
	ctx := SearchContext{
		startTime: time.Now(),
		deadline:  time.Now().Add(SEARCH_TIME_LIMIT),
	}
	
	// **YOUR LOGIC**: This runs the search algorithm (see below)
	bestMove := iterativeDeepeningSearch(snapshot, ctx)
	
	elapsed := time.Since(ctx.startTime)
	
	// BOILERPLATE: Format move string
	moveStr := directionToString(bestMove.direction)
	if bestMove.useBoost {
		moveStr += ":BOOST"
	}
	
	logDebug("Selected move: %s (score: %d, time: %v)", moveStr, bestMove.score, elapsed)
	return moveStr
}

// ============================================================================
// BOILERPLATE: Timer check
// ============================================================================

func (ctx *SearchContext) timeExpired() bool {
	return time.Now().After(ctx.deadline)
}

// ============================================================================
// YOUR LOGIC: Iterative deepening search
// **CUSTOMIZE**: Can modify depth strategy or stopping conditions
// ============================================================================

func iterativeDeepeningSearch(snapshot GameStateSnapshot, ctx SearchContext) Move {
	validMoves := snapshot.myAgent.GetValidMoves()
	
	if len(validMoves) == 0 {
		return Move{direction: RIGHT, useBoost: false, score: LOSE_SCORE}
	}
	
	bestMove := Move{direction: validMoves[0], useBoost: false, score: math.MinInt32}
	
	// Iterative deepening: search depth 1, 2, 3... until time runs out
	depth := 1
	for !ctx.timeExpired() {
		logDebug("Searching at depth %d...", depth)
		
		depthBestMove := searchAtDepth(snapshot, depth, ctx)
		
		if ctx.timeExpired() {
			logDebug("Time expired during depth %d, using previous result", depth)
			break
		}
		
		bestMove = depthBestMove
		logDebug("Completed depth %d: best move %s (boost=%v) with score %d", 
			depth, directionToString(bestMove.direction), bestMove.useBoost, bestMove.score)
		
		depth++
		
		// **CUSTOMIZE**: Can add more stopping conditions here
		if bestMove.score >= WIN_SCORE || bestMove.score <= LOSE_SCORE {
			logDebug("Found terminal score, stopping search")
			break
		}
	}
	
	logDebug("Search completed: reached depth %d", depth-1)
	return bestMove
}

// ============================================================================
// YOUR LOGIC: Search at a specific depth
// **CUSTOMIZE**: Can add alpha-beta pruning or move ordering here
// ============================================================================

func searchAtDepth(snapshot GameStateSnapshot, maxDepth int, ctx SearchContext) Move {
	validMoves := snapshot.myAgent.GetValidMoves()
	
	bestMove := Move{direction: validMoves[0], useBoost: false, score: math.MinInt32}
	
	// Try each direction
	for _, dir := range validMoves {
		if ctx.timeExpired() {
			return bestMove
		}
		
		// Try with and without boost
		for _, useBoost := range []bool{false, true} {
			if useBoost && snapshot.myAgent.BoostsRemaining <= 0 {
				continue
			}
			
			if ctx.timeExpired() {
				return bestMove
			}
			
			// Evaluate this move by looking ahead
			score := evaluateMoveAtDepth(snapshot, dir, useBoost, maxDepth, ctx)
			
			if score > bestMove.score {
				bestMove = Move{
					direction: dir,
					useBoost:  useBoost,
					score:     score,
				}
			}
		}
	}
	
	return bestMove
}

// ============================================================================
// BOILERPLATE: Build game state from trail data (from judge)
// ============================================================================

func buildGameSnapshot(myTrail, otherTrail [][]int, myBoosts int) GameStateSnapshot {
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

// ============================================================================
// BOILERPLATE: Simulate a move and evaluate it
// Uses undo pattern instead of cloning for better performance
// ============================================================================

func evaluateMoveAtDepth(snapshot GameStateSnapshot, dir Direction, useBoost bool, maxDepth int, ctx SearchContext) int {
	// Make the move using undo pattern (much faster than Clone)
	_, myState := snapshot.myAgent.UndoableMove(dir, snapshot.otherAgent, useBoost)
	
	// Use minimax to evaluate the resulting position
	score := minimax(snapshot.board, snapshot.myAgent, snapshot.otherAgent, maxDepth, true, ctx)
	
	// Undo the move to restore original state
	snapshot.myAgent.UndoMove(myState, snapshot.otherAgent)
	
	return score
}

// ============================================================================
// BOILERPLATE: Minimax algorithm (assumes opponent plays optimally)
// Uses undo pattern for efficiency - no expensive cloning!
// **CUSTOMIZE**: Can add alpha-beta pruning here for even better performance
// ============================================================================

func minimax(board *GameBoard, myAgent *Agent, otherAgent *Agent, depth int, isMaximizing bool, ctx SearchContext) int {
	// Stop if time expired
	if ctx.timeExpired() {
		return evaluatePosition(myAgent, otherAgent)
	}
	
	// Base case: reached leaf node or game over
	if depth == 0 || !myAgent.Alive || !otherAgent.Alive {
		return evaluatePosition(myAgent, otherAgent)
	}
	
	if isMaximizing {
		// Maximizing player (us): pick best move
		maxScore := math.MinInt32
		
		for _, dir := range myAgent.GetValidMoves() {
			if ctx.timeExpired() {
				return maxScore
			}
			
			// Simulate our move using undo pattern
			_, myState := myAgent.UndoableMove(dir, otherAgent, false)
			
			// Recurse: opponent's turn
			score := minimax(board, myAgent, otherAgent, depth-1, false, ctx)
			
			// Undo the move
			myAgent.UndoMove(myState, otherAgent)
			
			if score > maxScore {
				maxScore = score
			}
		}
		
		return maxScore
	} else {
		// Minimizing player (opponent): assumes they play optimally
		minScore := math.MaxInt32
		
		for _, dir := range otherAgent.GetValidMoves() {
			if ctx.timeExpired() {
				return minScore
			}
			
			// Simulate opponent's move using undo pattern
			_, otherState := otherAgent.UndoableMove(dir, myAgent, false)
			
			// Recurse: our turn
			score := minimax(board, myAgent, otherAgent, depth-1, true, ctx)
			
			// Undo the move
			otherAgent.UndoMove(otherState, myAgent)
			
			if score < minScore {
				minScore = score
			}
		}
		
		return minScore
	}
}

// ============================================================================
// **YOUR LOGIC**: Heuristic evaluation function - THIS IS THE KEY FUNCTION!
// **CUSTOMIZE**: This is where you define what makes a position "good"
// ============================================================================

func evaluatePosition(myAgent *Agent, otherAgent *Agent) int {
	// Terminal states (game over)
	if !myAgent.Alive && !otherAgent.Alive {
		return DRAW_SCORE
	}
	
	if !myAgent.Alive {
		return LOSE_SCORE
	}
	
	if !otherAgent.Alive {
		return WIN_SCORE
	}
	
	// Non-terminal state: use heuristics to evaluate position
	score := 0
	
	// **CUSTOMIZE**: Current heuristic is simple:
	// 1. Prefer longer trail (10 points per cell)
	score += myAgent.Length * 10
	score -= otherAgent.Length * 10
	
	// 2. Prefer more available space (5 points per reachable cell)
	mySpace := countAvailableSpace(myAgent)
	otherSpace := countAvailableSpace(otherAgent)
	score += (mySpace - otherSpace) * 5
	
	// **TODO**: You can add more heuristics here:
	// - Distance to opponent
	// - Control of center
	// - Number of escape routes
	// - Wall proximity
	// - Territory control
	
	return score
}

// ============================================================================
// BOILERPLATE: Count reachable empty cells (used by heuristic)
// ============================================================================

func countAvailableSpace(agent *Agent) int {
	if !agent.Alive {
		return 0
	}
	
	head := agent.GetHead()
	visited := make(map[Position]bool)
	queue := []Position{head}
	visited[head] = true
	count := 0
	
	// BFS to count reachable cells (capped at 50 for performance)
	for len(queue) > 0 && count < 50 {
		current := queue[0]
		queue = queue[1:]
		count++
		
		for _, dir := range AllDirections {
			next := Position{
				X: current.X + dir.DX,
				Y: current.Y + dir.DY,
			}
			next = agent.Board.TorusCheck(next)
			
			if !visited[next] && agent.Board.GetCellState(next) == EMPTY {
				visited[next] = true
				queue = append(queue, next)
			}
		}
	}
	
	return count
}

// ============================================================================
// BOILERPLATE: Helper functions (direction inference, conversions)
// ============================================================================

func inferDirection(trail [][]int) Direction {
	if len(trail) < 2 {
		return RIGHT
	}
	
	head := trail[len(trail)-1]
	prev := trail[len(trail)-2]
	
	dx := head[0] - prev[0]
	dy := head[1] - prev[1]
	
	// Handle torus wraparound
	if math.Abs(float64(dx)) > 1 {
		if dx > 0 {
			dx = -1
		} else {
			dx = 1
		}
	}
	if math.Abs(float64(dy)) > 1 {
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
