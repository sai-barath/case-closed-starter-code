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
	// SEARCH_TIME_LIMIT = 3500 * time.Millisecond // **CUSTOMIZE**: Time budget per move
	SEARCH_TIME_LIMIT = 100 * time.Millisecond // **CUSTOMIZE**: Time budget per move
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
// YOUR LOGIC: Search at a specific depth with alpha-beta pruning
// ============================================================================

func searchAtDepth(snapshot GameStateSnapshot, maxDepth int, ctx SearchContext) Move {
	validMoves := snapshot.myAgent.GetValidMoves()
	
	bestMove := Move{direction: validMoves[0], useBoost: false, score: math.MinInt32}
	alpha := math.MinInt32
	beta := math.MaxInt32
	
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
			
			// Evaluate this move by looking ahead with alpha-beta
			score := evaluateMoveAtDepth(snapshot, dir, useBoost, maxDepth, alpha, beta, ctx)
			
			if score > bestMove.score {
				bestMove = Move{
					direction: dir,
					useBoost:  useBoost,
					score:     score,
				}
				alpha = score
			}
			
			// Alpha-beta cutoff
			if alpha >= beta {
				return bestMove
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
// BOILERPLATE: Simulate a move and evaluate it with alpha-beta
// Uses undo pattern instead of cloning for better performance
// ============================================================================

func evaluateMoveAtDepth(snapshot GameStateSnapshot, dir Direction, useBoost bool, maxDepth int, alpha int, beta int, ctx SearchContext) int {
	// Make the move using undo pattern (much faster than Clone)
	_, myState := snapshot.myAgent.UndoableMove(dir, snapshot.otherAgent, useBoost)
	
	// Use minimax with alpha-beta to evaluate the resulting position
	score := alphabeta(snapshot.board, snapshot.myAgent, snapshot.otherAgent, maxDepth, alpha, beta, true, ctx)
	
	// Undo the move to restore original state
	snapshot.myAgent.UndoMove(myState, snapshot.otherAgent)
	
	return score
}

// ============================================================================
// Alpha-Beta Pruning Algorithm (optimized minimax)
// Uses undo pattern for efficiency - no expensive cloning!
// ============================================================================

func alphabeta(board *GameBoard, myAgent *Agent, otherAgent *Agent, depth int, alpha int, beta int, isMaximizing bool, ctx SearchContext) int {
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
			score := alphabeta(board, myAgent, otherAgent, depth-1, alpha, beta, false, ctx)
			
			// Undo the move
			myAgent.UndoMove(myState, otherAgent)
			
			if score > maxScore {
				maxScore = score
			}
			
			// Alpha-beta pruning
			alpha = max(alpha, score)
			if beta <= alpha {
				break // Beta cutoff
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
			score := alphabeta(board, myAgent, otherAgent, depth-1, alpha, beta, true, ctx)
			
			// Undo the move
			otherAgent.UndoMove(otherState, myAgent)
			
			if score < minScore {
				minScore = score
			}
			
			// Alpha-beta pruning
			beta = min(beta, score)
			if beta <= alpha {
				break // Alpha cutoff
			}
		}
		
		return minScore
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ============================================================================
// BOILERPLATE: Old minimax kept for reference (not used anymore)
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
	if !myAgent.Alive && !otherAgent.Alive {
		return DRAW_SCORE
	}
	
	if !myAgent.Alive {
		return LOSE_SCORE
	}
	
	if !otherAgent.Alive {
		return WIN_SCORE
	}
	
	score := 0
	
	myHead := myAgent.GetHead()
	opponentHead := otherAgent.GetHead()
	
	mySpace := countAvailableSpace(myAgent)
	opponentSpace := countAvailableSpace(otherAgent)
	
	territoryResult := voronoiTerritory(myAgent, otherAgent)
	myTerritory := territoryResult.myTerritory
	opponentTerritory := territoryResult.opponentTerritory
	
	score += (myTerritory - opponentTerritory) * 15
	
	spaceDiff := mySpace - opponentSpace
	if spaceDiff > 50 {
		score += 3000
	} else if spaceDiff < -50 {
		score -= 3000
	} else {
		score += spaceDiff * 25
	}
	
	score += myAgent.Length * 8
	score -= otherAgent.Length * 8
	
	dist := manhattanDistance(myHead, opponentHead)
	if mySpace > opponentSpace {
		score += dist * 3
	} else if mySpace < opponentSpace {
		score -= dist * 5
	}
	
	centerX, centerY := BOARD_WIDTH/2, BOARD_HEIGHT/2
	myCenterDist := manhattanDistanceRaw(myHead.X, myHead.Y, centerX, centerY)
	opponentCenterDist := manhattanDistanceRaw(opponentHead.X, opponentHead.Y, centerX, centerY)
	score += (opponentCenterDist - myCenterDist) * 2
	
	myFreedom := countFreedomDegree(myAgent)
	opponentFreedom := countFreedomDegree(otherAgent)
	score += (myFreedom - opponentFreedom) * 40
	
	myTightness := measureTightness(myAgent)
	opponentTightness := measureTightness(otherAgent)
	score += (opponentTightness - myTightness) * 20
	
	myCornerPenalty := cornerProximityPenalty(myHead)
	opponentCornerPenalty := cornerProximityPenalty(opponentHead)
	score += (opponentCornerPenalty - myCornerPenalty) * 15
	
	if myAgent.BoostsRemaining > 0 && otherAgent.BoostsRemaining == 0 {
		score += 100
	}
	
	myBlockingScore := blockingPositionScore(myAgent, otherAgent)
	opponentBlockingScore := blockingPositionScore(otherAgent, myAgent)
	score += (myBlockingScore - opponentBlockingScore) * 8
	
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
	
	for len(queue) > 0 && count < 150 {
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

type VoronoiResult struct {
	myTerritory       int
	opponentTerritory int
}

func voronoiTerritory(myAgent *Agent, opponentAgent *Agent) VoronoiResult {
	myHead := myAgent.GetHead()
	opponentHead := opponentAgent.GetHead()
	
	myQueue := []Position{myHead}
	opponentQueue := []Position{opponentHead}
	
	myVisited := make(map[Position]int)
	opponentVisited := make(map[Position]int)
	
	myVisited[myHead] = 0
	opponentVisited[opponentHead] = 0
	
	maxSteps := 30
	
	for step := 0; step < maxSteps; step++ {
		nextMyQueue := []Position{}
		for _, pos := range myQueue {
			for _, dir := range AllDirections {
				next := Position{X: pos.X + dir.DX, Y: pos.Y + dir.DY}
				next = myAgent.Board.TorusCheck(next)
				
				if myAgent.Board.GetCellState(next) == EMPTY {
					if _, visited := myVisited[next]; !visited {
						myVisited[next] = step + 1
						nextMyQueue = append(nextMyQueue, next)
					}
				}
			}
		}
		myQueue = nextMyQueue
		
		nextOpponentQueue := []Position{}
		for _, pos := range opponentQueue {
			for _, dir := range AllDirections {
				next := Position{X: pos.X + dir.DX, Y: pos.Y + dir.DY}
				next = opponentAgent.Board.TorusCheck(next)
				
				if opponentAgent.Board.GetCellState(next) == EMPTY {
					if _, visited := opponentVisited[next]; !visited {
						opponentVisited[next] = step + 1
						nextOpponentQueue = append(nextOpponentQueue, next)
					}
				}
			}
		}
		opponentQueue = nextOpponentQueue
	}
	
	myTerritory := 0
	opponentTerritory := 0
	
	for pos, myDist := range myVisited {
		if opponentDist, opponentReached := opponentVisited[pos]; opponentReached {
			if myDist < opponentDist {
				myTerritory++
			} else if opponentDist < myDist {
				opponentTerritory++
			}
		} else {
			myTerritory++
		}
	}
	
	for pos := range opponentVisited {
		if _, myReached := myVisited[pos]; !myReached {
			opponentTerritory++
		}
	}
	
	return VoronoiResult{
		myTerritory:       myTerritory,
		opponentTerritory: opponentTerritory,
	}
}

func countFreedomDegree(agent *Agent) int {
	if !agent.Alive {
		return 0
	}
	
	head := agent.GetHead()
	freeCount := 0
	
	for _, dir := range AllDirections {
		next := Position{X: head.X + dir.DX, Y: head.Y + dir.DY}
		next = agent.Board.TorusCheck(next)
		
		if agent.Board.GetCellState(next) == EMPTY {
			freeCount++
		}
	}
	
	return freeCount
}

func measureTightness(agent *Agent) int {
	if !agent.Alive {
		return 0
	}
	
	head := agent.GetHead()
	tightness := 0
	
	for dx := -2; dx <= 2; dx++ {
		for dy := -2; dy <= 2; dy++ {
			if dx == 0 && dy == 0 {
				continue
			}
			pos := Position{X: head.X + dx, Y: head.Y + dy}
			pos = agent.Board.TorusCheck(pos)
			
			if agent.Board.GetCellState(pos) == AGENT {
				tightness++
			}
		}
	}
	
	return tightness
}

func cornerProximityPenalty(pos Position) int {
	corners := []Position{
		{X: 0, Y: 0},
		{X: BOARD_WIDTH - 1, Y: 0},
		{X: 0, Y: BOARD_HEIGHT - 1},
		{X: BOARD_WIDTH - 1, Y: BOARD_HEIGHT - 1},
	}
	
	minDist := math.MaxInt32
	for _, corner := range corners {
		dist := manhattanDistanceRaw(pos.X, pos.Y, corner.X, corner.Y)
		if dist < minDist {
			minDist = dist
		}
	}
	
	if minDist <= 2 {
		return 50 - minDist*10
	}
	
	return 0
}

func blockingPositionScore(agent *Agent, opponent *Agent) int {
	if !agent.Alive || !opponent.Alive {
		return 0
	}
	
	myHead := agent.GetHead()
	opponentHead := opponent.GetHead()
	
	dist := manhattanDistance(myHead, opponentHead)
	
	if dist <= 5 {
		opponentValidMoves := 0
		for _, dir := range opponent.GetValidMoves() {
			next := Position{X: opponentHead.X + dir.DX, Y: opponentHead.Y + dir.DY}
			next = opponent.Board.TorusCheck(next)
			
			if opponent.Board.GetCellState(next) == EMPTY {
				opponentValidMoves++
			}
		}
		
		if opponentValidMoves <= 1 {
			return 100
		} else if opponentValidMoves == 2 {
			return 30
		}
	}
	
	return 0
}

func manhattanDistance(p1, p2 Position) int {
	return manhattanDistanceRaw(p1.X, p1.Y, p2.X, p2.Y)
}

func manhattanDistanceRaw(x1, y1, x2, y2 int) int {
	dx := abs(x1 - x2)
	dy := abs(y1 - y2)
	
	if dx > BOARD_WIDTH/2 {
		dx = BOARD_WIDTH - dx
	}
	if dy > BOARD_HEIGHT/2 {
		dy = BOARD_HEIGHT - dy
	}
	
	return dx + dy
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
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
