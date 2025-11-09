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
	SEARCH_TIME_LIMIT = 100 * time.Millisecond // **CUSTOMIZE**: Time budget per move
	WIN_SCORE         = 10000                    // Terminal score for winning
	LOSE_SCORE        = -10000                   // Terminal score for losing
	DRAW_SCORE        = 0                        // Terminal score for draw
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

type SearchContext struct {
	startTime      time.Time
	deadline       time.Time
	killerMoves    map[int][2]Direction
	historyTable   map[Direction]int
	bestMoveCache  map[int]Move
}

type TTEntry struct {
	score int
	depth int
	flag  int
}

const (
	EXACT = 0
	LOWER = 1
	UPPER = 2
)

var transpositionTable map[uint64]TTEntry

var (
	cachedPath        []Direction
	cachedPathIndex   int
	partitionDetected bool
)

// ============================================================================
// YOUR LOGIC: Main entry point - this is called every turn
// ============================================================================

func DecideMove(myTrail, otherTrail [][]int, turnCount, myBoosts, playerNumber int) string {
	logDebug("Turn %d: Starting iterative deepening search with %d boosts (Player %d)", turnCount, myBoosts, playerNumber)
	
	if len(myTrail) == 0 {
		partitionDetected = false
		cachedPath = nil
		cachedPathIndex = 0
		return "RIGHT"
	}
	
	snapshot := buildGameSnapshot(myTrail, otherTrail, myBoosts, playerNumber)
	
	if !snapshot.myAgent.Alive {
		partitionDetected = false
		cachedPath = nil
		cachedPathIndex = 0
		return "RIGHT"
	}
	
	if partitionDetected && cachedPathIndex < len(cachedPath) {
		validMoves := snapshot.myAgent.GetValidMoves()
		if len(validMoves) == 0 {
			partitionDetected = false
			cachedPath = nil
			cachedPathIndex = 0
			return "RIGHT"
		}
		
		move := cachedPath[cachedPathIndex]
		cachedPathIndex++
		logDebug("Following cached path: move %d/%d - %s", cachedPathIndex, len(cachedPath), directionToString(move))
		return directionToString(move)
	}
	
	if !partitionDetected {
		myReachable := floodFillReachable(snapshot.myAgent)
		oppReachable := floodFillReachable(snapshot.otherAgent)
		
		if myReachable > oppReachable + 10 {
			logDebug("SPACE PARTITION DETECTED! My reachable: %d, Opponent: %d", myReachable, oppReachable)
			partitionDetected = true
			cachedPath = computeLongestPath(snapshot)
			cachedPathIndex = 0
			
			if len(cachedPath) > 0 {
				move := cachedPath[cachedPathIndex]
				cachedPathIndex++
				logDebug("Computed path with %d moves, following move 1: %s", len(cachedPath), directionToString(move))
				return directionToString(move)
			}
		}
	}
	
	transpositionTable = make(map[uint64]TTEntry)
	
	ctx := SearchContext{
		startTime:     time.Now(),
		deadline:      time.Now().Add(SEARCH_TIME_LIMIT),
		killerMoves:   make(map[int][2]Direction),
		historyTable:  make(map[Direction]int),
		bestMoveCache: make(map[int]Move),
	}
	
	bestMove := iterativeDeepeningSearch(snapshot, ctx)
	
	elapsed := time.Since(ctx.startTime)
	
	moveStr := directionToString(bestMove.direction)
	if bestMove.useBoost {
		moveStr += ":BOOST"
	}
	
	logDebug("Selected move: %s (score: %d, time: %v)", moveStr, bestMove.score, elapsed)
	return moveStr
}

// ============================================================================
// ENDGAME SPACE-FILLING: Compute longest path through available space
// Called once when partition detected, then cached for rest of game
// ============================================================================

func computeLongestPath(snapshot GameStateSnapshot) []Direction {
	path := []Direction{}
	visited := make(map[Position]bool)
	
	head := snapshot.myAgent.GetHead()
	reachable := floodFillReachable(snapshot.myAgent)
	maxDepth := min(reachable, 50)
	
	visitCount := 0
	maxVisits := 10000
	
	var dfs func(pos Position, currentPath []Direction, depth int) []Direction
	dfs = func(pos Position, currentPath []Direction, depth int) []Direction {
		visitCount++
		if visitCount > maxVisits || depth >= maxDepth {
			return currentPath
		}
		
		visited[pos] = true
		
		bestPath := make([]Direction, len(currentPath))
		copy(bestPath, currentPath)
		
		for _, dir := range AllDirections {
			nextPos := Position{X: pos.X + dir.DX, Y: pos.Y + dir.DY}
			nextPos = snapshot.myAgent.Board.TorusCheck(nextPos)
			
			if !visited[nextPos] && snapshot.myAgent.Board.GetCellState(nextPos) == EMPTY {
				newPath := append(currentPath, dir)
				result := dfs(nextPos, newPath, depth+1)
				if len(result) > len(bestPath) {
					bestPath = result
				}
			}
		}
		
		visited[pos] = false
		return bestPath
	}
	
	path = dfs(head, []Direction{}, 0)
	
	if len(path) == 0 {
		validMoves := snapshot.myAgent.GetValidMoves()
		if len(validMoves) > 0 {
			path = []Direction{validMoves[0]}
		}
	}
	
	logDebug("Computed longest path with %d moves (visited %d nodes, reachable %d)", len(path), visitCount, reachable)
	return path
}

// ============================================================================
// BOILERPLATE: Timer check
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
	
	depth := 1
	for !ctx.timeExpired() {
		logDebug("Searching at depth %d...", depth)
		
		depthBestMove := searchAtDepth(snapshot, depth, &ctx)
		
		if ctx.timeExpired() {
			logDebug("Time expired during depth %d, using previous result", depth)
			break
		}
		
		bestMove = depthBestMove
		ctx.bestMoveCache[depth] = bestMove
		
		logDebug("Completed depth %d: best move %s (boost=%v) with score %d", 
			depth, directionToString(bestMove.direction), bestMove.useBoost, bestMove.score)
		
		depth++
		
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

func searchAtDepth(snapshot GameStateSnapshot, maxDepth int, ctx *SearchContext) Move {
	validMoves := snapshot.myAgent.GetValidMoves()
	
	if len(validMoves) == 0 {
		return Move{direction: RIGHT, useBoost: false, score: LOSE_SCORE}
	}
	
	bestMove := Move{direction: validMoves[0], useBoost: false, score: math.MinInt32}
	alpha := math.MinInt32
	beta := math.MaxInt32
	
	orderedMoves := orderMovesAtRoot(validMoves, snapshot, maxDepth, ctx)
	
	for _, dir := range orderedMoves {
		if ctx.timeExpired() {
			return bestMove
		}
		
		for _, useBoost := range []bool{false, true} {
			if useBoost && snapshot.myAgent.BoostsRemaining <= 0 {
				continue
			}
			
			if ctx.timeExpired() {
				return bestMove
			}
			
			score := evaluateMoveAtDepth(snapshot, dir, useBoost, maxDepth, alpha, beta, ctx)
			
			if shouldBoostAggressively(snapshot, dir, useBoost) {
				score += 500
			}
			
			if score > bestMove.score {
				bestMove = Move{
					direction: dir,
					useBoost:  useBoost,
					score:     score,
				}
				alpha = score
				
				ctx.historyTable[dir] += maxDepth * maxDepth
			}
			
			if alpha >= beta {
				killers := ctx.killerMoves[maxDepth]
				if killers[0] != dir {
					killers[1] = killers[0]
					killers[0] = dir
					ctx.killerMoves[maxDepth] = killers
				}
				return bestMove
			}
		}
	}
	
	return bestMove
}

func orderMovesAtRoot(moves []Direction, snapshot GameStateSnapshot, depth int, ctx *SearchContext) []Direction {
	if len(moves) == 0 {
		return moves
	}
	
	scored := make([]scoredMove, 0, len(moves))
	
	for _, dir := range moves {
		moveScore := 0
		
		if prevBest, exists := ctx.bestMoveCache[depth-1]; exists && prevBest.direction == dir {
			moveScore += 100000
		}
		
		killers := ctx.killerMoves[depth]
		if killers[0] == dir {
			moveScore += 10000
		} else if killers[1] == dir {
			moveScore += 5000
		}
		
		moveScore += ctx.historyTable[dir]
		
		scored = append(scored, scoredMove{dir: dir, score: moveScore})
	}
	
	scored = orderMovesStatic(scored, snapshot)
	
	for i := 0; i < len(scored)-1; i++ {
		for j := i + 1; j < len(scored); j++ {
			if scored[j].score > scored[i].score {
				scored[i], scored[j] = scored[j], scored[i]
			}
		}
	}
	
	ordered := make([]Direction, len(scored))
	for i, sm := range scored {
		ordered[i] = sm.dir
	}
	
	return ordered
}

func orderMovesStatic(scored []scoredMove, snapshot GameStateSnapshot) []scoredMove {
	myHead := snapshot.myAgent.GetHead()
	centerX, centerY := BOARD_WIDTH/2, BOARD_HEIGHT/2
	
	for i := range scored {
		dir := scored[i].dir
		nextPos := Position{X: myHead.X + dir.DX, Y: myHead.Y + dir.DY}
		nextPos = snapshot.myAgent.Board.TorusCheck(nextPos)
		
		centerDist := manhattanDistanceRaw(nextPos.X, nextPos.Y, centerX, centerY)
		scored[i].score -= centerDist * 5
		
		freeNeighbors := 0
		for _, d := range AllDirections {
			neighbor := Position{X: nextPos.X + d.DX, Y: nextPos.Y + d.DY}
			neighbor = snapshot.myAgent.Board.TorusCheck(neighbor)
			if snapshot.myAgent.Board.GetCellState(neighbor) == EMPTY {
				freeNeighbors++
			}
		}
		scored[i].score += freeNeighbors * 30
	}
	
	return scored
}

func orderMoves(moves []Direction, snapshot GameStateSnapshot) []Direction {
	scored := make([]scoredMove, 0, len(moves))
	myHead := snapshot.myAgent.GetHead()
	centerX, centerY := BOARD_WIDTH/2, BOARD_HEIGHT/2
	
	for _, dir := range moves {
		nextPos := Position{X: myHead.X + dir.DX, Y: myHead.Y + dir.DY}
		nextPos = snapshot.myAgent.Board.TorusCheck(nextPos)
		
		moveScore := 0
		
		centerDist := manhattanDistanceRaw(nextPos.X, nextPos.Y, centerX, centerY)
		moveScore -= centerDist * 5
		
		freeNeighbors := 0
		for _, d := range AllDirections {
			neighbor := Position{X: nextPos.X + d.DX, Y: nextPos.Y + d.DY}
			neighbor = snapshot.myAgent.Board.TorusCheck(neighbor)
			if snapshot.myAgent.Board.GetCellState(neighbor) == EMPTY {
				freeNeighbors++
			}
		}
		moveScore += freeNeighbors * 30
		
		scored = append(scored, scoredMove{dir: dir, score: moveScore})
	}
	
	for i := 0; i < len(scored)-1; i++ {
		for j := i + 1; j < len(scored); j++ {
			if scored[j].score > scored[i].score {
				scored[i], scored[j] = scored[j], scored[i]
			}
		}
	}
	
	ordered := make([]Direction, len(scored))
	for i, sm := range scored {
		ordered[i] = sm.dir
	}
	
	return ordered
}

func countLocalSpace(pos Position, agent *Agent, radius int) int {
	visited := make(map[Position]bool)
	queue := []Position{pos}
	visited[pos] = true
	count := 0
	
	for len(queue) > 0 && count < radius*4 {
		current := queue[0]
		queue = queue[1:]
		count++
		
		for _, dir := range AllDirections {
			next := Position{X: current.X + dir.DX, Y: current.Y + dir.DY}
			next = agent.Board.TorusCheck(next)
			
			if !visited[next] && agent.Board.GetCellState(next) == EMPTY {
				dist := manhattanDistanceRaw(next.X, next.Y, pos.X, pos.Y)
				if dist <= radius {
					visited[next] = true
					queue = append(queue, next)
				}
			}
		}
	}
	
	return count
}

// ============================================================================
// BOILERPLATE: Build game state from trail data (from judge)
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

// ============================================================================
// BOILERPLATE: Simulate a move and evaluate it with alpha-beta
// Uses undo pattern instead of cloning for better performance
// ============================================================================

func evaluateMoveAtDepth(snapshot GameStateSnapshot, dir Direction, useBoost bool, maxDepth int, alpha int, beta int, ctx *SearchContext) int {
	bestScore := math.MinInt32
	
	oppValidMoves := snapshot.otherAgent.GetValidMoves()
	orderedOppMoves := orderMoves(oppValidMoves, GameStateSnapshot{
		myAgent:    snapshot.otherAgent,
		otherAgent: snapshot.myAgent,
		board:      snapshot.board,
		amIRed:     !snapshot.amIRed,
	})
	
	for _, oppDir := range orderedOppMoves {
		if ctx.timeExpired() {
			return bestScore
		}
		
		var score int
		if snapshot.amIRed {
			_, myState := snapshot.myAgent.UndoableMove(dir, snapshot.otherAgent, useBoost)
			_, oppState := snapshot.otherAgent.UndoableMove(oppDir, snapshot.myAgent, false)
			score = alphabeta(snapshot.board, snapshot.myAgent, snapshot.otherAgent, maxDepth-1, alpha, beta, true, snapshot.amIRed, ctx)
			snapshot.otherAgent.UndoMove(oppState, snapshot.myAgent)
			snapshot.myAgent.UndoMove(myState, snapshot.otherAgent)
		} else {
			_, oppState := snapshot.otherAgent.UndoableMove(oppDir, snapshot.myAgent, false)
			_, myState := snapshot.myAgent.UndoableMove(dir, snapshot.otherAgent, useBoost)
			score = alphabeta(snapshot.board, snapshot.myAgent, snapshot.otherAgent, maxDepth-1, alpha, beta, true, snapshot.amIRed, ctx)
			snapshot.myAgent.UndoMove(myState, snapshot.otherAgent)
			snapshot.otherAgent.UndoMove(oppState, snapshot.myAgent)
		}
		
		if score > bestScore {
			bestScore = score
		}
		
		if bestScore > alpha {
			alpha = bestScore
		}
		
		if alpha >= beta {
			break
		}
	}
	
	return bestScore
}

// ============================================================================
// Alpha-Beta Pruning Algorithm (optimized minimax)
// Uses undo pattern for efficiency - no expensive cloning!
// ============================================================================

func shouldBoostAggressively(snapshot GameStateSnapshot, dir Direction, useBoost bool) bool {
	if !useBoost || snapshot.myAgent.BoostsRemaining == 0 {
		return false
	}
	
	myHead := snapshot.myAgent.GetHead()
	opponentHead := snapshot.otherAgent.GetHead()
	
	myValidMoves := len(snapshot.myAgent.GetValidMoves())
	if myValidMoves <= 2 {
		return true
	}
	
	mySpace := countAvailableSpace(snapshot.myAgent)
	opponentSpace := countAvailableSpace(snapshot.otherAgent)
	
	if mySpace < opponentSpace-30 {
		centerX, centerY := BOARD_WIDTH/2, BOARD_HEIGHT/2
		distToCenter := manhattanDistanceRaw(myHead.X, myHead.Y, centerX, centerY)
		if distToCenter > 5 {
			return true
		}
	}
	
	dist := manhattanDistance(myHead, opponentHead)
	if dist >= 3 && dist <= 6 && mySpace > opponentSpace+20 {
		return true
	}
	
	return false
}

func alphabeta(board *GameBoard, myAgent *Agent, otherAgent *Agent, depth int, alpha int, beta int, isMaximizing bool, amIRed bool, ctx *SearchContext) int {
	if ctx.timeExpired() {
		return evaluatePositionWithBias(myAgent, otherAgent, amIRed)
	}
	
	if depth == 0 || !myAgent.Alive || !otherAgent.Alive {
		return evaluatePositionWithBias(myAgent, otherAgent, amIRed)
	}
	
	if isMaximizing {
		maxScore := math.MinInt32
		
		for _, myDir := range myAgent.GetValidMoves() {
			if ctx.timeExpired() {
				return maxScore
			}
			
			boostOptions := []bool{false}
			if myAgent.BoostsRemaining > 0 && depth >= 3 {
				boostOptions = append(boostOptions, true)
			}
			
			for _, myBoost := range boostOptions {
				minOpponentScore := math.MaxInt32
				
				for _, oppDir := range otherAgent.GetValidMoves() {
					if ctx.timeExpired() {
						break
					}
					
					oppBoost := false
					
					var score int
					if amIRed {
						_, myState := myAgent.UndoableMove(myDir, otherAgent, myBoost)
						_, oppState := otherAgent.UndoableMove(oppDir, myAgent, oppBoost)
						score = alphabeta(board, myAgent, otherAgent, depth-1, alpha, beta, true, amIRed, ctx)
						otherAgent.UndoMove(oppState, myAgent)
						myAgent.UndoMove(myState, otherAgent)
					} else {
						_, oppState := otherAgent.UndoableMove(oppDir, myAgent, oppBoost)
						_, myState := myAgent.UndoableMove(myDir, otherAgent, myBoost)
						score = alphabeta(board, myAgent, otherAgent, depth-1, alpha, beta, true, amIRed, ctx)
						myAgent.UndoMove(myState, otherAgent)
						otherAgent.UndoMove(oppState, myAgent)
					}
					
					if score < minOpponentScore {
						minOpponentScore = score
					}
					
					if minOpponentScore <= alpha {
						break
					}
				}
				
				if minOpponentScore > maxScore {
					maxScore = minOpponentScore
				}
				
				alpha = max(alpha, minOpponentScore)
				if beta <= alpha {
					return maxScore
				}
			}
		}
		
		return maxScore
	} else {
		return evaluatePositionWithBias(myAgent, otherAgent, amIRed)
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

func minimax(board *GameBoard, myAgent *Agent, otherAgent *Agent, depth int, isMaximizing bool, ctx *SearchContext) int {
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
	
	myValidMoves := myAgent.GetValidMoves()
	opponentValidMoves := otherAgent.GetValidMoves()
	
	if len(myValidMoves) == 0 {
		return LOSE_SCORE
	}
	if len(opponentValidMoves) == 0 {
		return WIN_SCORE
	}
	
	mySpace := countAvailableSpace(myAgent)
	opponentSpace := countAvailableSpace(otherAgent)
	totalEmptySpace := mySpace + opponentSpace
	
	if totalEmptySpace < 80 {
		endgameResult := evaluateEndgame(myAgent, otherAgent, mySpace, opponentSpace)
		if endgameResult != 0 {
			return endgameResult
		}
	}
	
	score := 0
	
	myHead := myAgent.GetHead()
	opponentHead := otherAgent.GetHead()
	
	territoryResult := voronoiTerritory(myAgent, otherAgent)
	myTerritory := territoryResult.myTerritory
	opponentTerritory := territoryResult.opponentTerritory
	
	score += (myTerritory - opponentTerritory) * 15
	
	spaceDiff := mySpace - opponentSpace
	if spaceDiff > 50 {
		score += 3000
		
		score += evaluateTerritoryCutting(myAgent, otherAgent) * 25
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
	
	forcingBonus := evaluateForcingMoves(myAgent, otherAgent)
	score += forcingBonus * 30
	
	return score
}

func evaluatePositionWithBias(myAgent *Agent, otherAgent *Agent, amIRed bool) int {
	baseScore := evaluatePosition(myAgent, otherAgent)
	
	if !myAgent.Alive || !otherAgent.Alive {
		return baseScore
	}
	
	myHead := myAgent.GetHead()
	opponentHead := otherAgent.GetHead()
	dist := manhattanDistance(myHead, opponentHead)
	
	if dist <= 4 {
		if amIRed {
			baseScore += 150
		} else {
			baseScore -= 200
		}
	}
	
	if !amIRed && dist <= 3 {
		for _, dir := range myAgent.GetValidMoves() {
			nextPos := Position{X: myHead.X + dir.DX, Y: myHead.Y + dir.DY}
			nextPos = myAgent.Board.TorusCheck(nextPos)
			
			distToOpp := manhattanDistance(nextPos, opponentHead)
			if distToOpp > dist {
				baseScore += 80
				break
			}
		}
	}
	
	return baseScore
}

func evaluateForcingMoves(myAgent *Agent, opponentAgent *Agent) int {
	if !myAgent.Alive || !opponentAgent.Alive {
		return 0
	}
	
	opponentValidMoves := opponentAgent.GetValidMoves()
	
	if len(opponentValidMoves) == 1 {
		return 80
	} else if len(opponentValidMoves) == 2 {
		opponentHead := opponentAgent.GetHead()
		
		badMoves := 0
		for _, dir := range opponentValidMoves {
			nextPos := Position{X: opponentHead.X + dir.DX, Y: opponentHead.Y + dir.DY}
			nextPos = opponentAgent.Board.TorusCheck(nextPos)
			
			freeNeighbors := 0
			for _, d := range AllDirections {
				neighbor := Position{X: nextPos.X + d.DX, Y: nextPos.Y + d.DY}
				neighbor = opponentAgent.Board.TorusCheck(neighbor)
				if opponentAgent.Board.GetCellState(neighbor) == EMPTY {
					freeNeighbors++
				}
			}
			
			if freeNeighbors <= 2 {
				badMoves++
			}
		}
		
		if badMoves >= 1 {
			return 40
		}
		return 20
	}
	
	return 0
}

func evaluateTerritoryCutting(myAgent *Agent, opponentAgent *Agent) int {
	if !myAgent.Alive || !opponentAgent.Alive {
		return 0
	}
	
	myHead := myAgent.GetHead()
	opponentHead := opponentAgent.GetHead()
	
	dist := manhattanDistance(myHead, opponentHead)
	
	if dist > 8 {
		return 0
	}
	
	cuttingScore := 0
	
	midX := (myHead.X + opponentHead.X) / 2
	midY := (myHead.Y + opponentHead.Y) / 2
	
	myDistToMid := manhattanDistanceRaw(myHead.X, myHead.Y, midX, midY)
	oppDistToMid := manhattanDistanceRaw(opponentHead.X, opponentHead.Y, midX, midY)
	
	if myDistToMid < oppDistToMid {
		cuttingScore += 50
	}
	
	opponentValidMoves := len(opponentAgent.GetValidMoves())
	if opponentValidMoves <= 2 {
		cuttingScore += 100
	} else if opponentValidMoves == 3 {
		cuttingScore += 30
	}
	
	return cuttingScore
}

func evaluateEndgame(myAgent *Agent, opponentAgent *Agent, mySpace int, opponentSpace int) int {
	spaceDiff := mySpace - opponentSpace
	
	if spaceDiff > 15 {
		return WIN_SCORE / 2
	} else if spaceDiff < -15 {
		return LOSE_SCORE / 2
	}
	
	myReachable := floodFillReachable(myAgent)
	oppReachable := floodFillReachable(opponentAgent)
	
	reachableDiff := myReachable - oppReachable
	
	if reachableDiff > 10 {
		return WIN_SCORE / 3
	} else if reachableDiff < -10 {
		return LOSE_SCORE / 3
	}
	
	return reachableDiff * 200
}

func floodFillReachable(agent *Agent) int {
	if !agent.Alive {
		return 0
	}
	
	head := agent.GetHead()
	visited := make(map[Position]bool)
	queue := []Position{head}
	visited[head] = true
	count := 0
	
	for len(queue) > 0 {
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
