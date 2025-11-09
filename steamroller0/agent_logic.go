package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"
)

var debugMode = os.Getenv("DEBUG") == "1"

const (
	SEARCH_TIME_LIMIT = 100 * time.Millisecond
	WIN_SCORE         = 10000
	LOSE_SCORE        = -10000
	DRAW_SCORE        = 0
	BOARD_HEIGHT      = 18
	BOARD_WIDTH       = 20
)

// Tunable parameters for evolutionary training
var (
	WEIGHT_TERRITORY      = 50
	WEIGHT_FREEDOM        = 150
	WEIGHT_REACHABLE      = 100
	WEIGHT_BOOST          = 20
	WEIGHT_CHAMBER        = 30
	WEIGHT_EDGE           = 15
	WEIGHT_COMPACTNESS    = 25
	WEIGHT_CUTOFF         = 40
	WEIGHT_GROWTH         = 30
	PENALTY_CORRIDOR_BASE = 500
	PENALTY_HEAD_DISTANCE = 200
)

func logDebug(format string, args ...interface{}) {
	if debugMode {
		fmt.Printf("[DEBUG] "+format+"\n", args...)
	}
}

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

type SearchContext struct {
	startTime time.Time
	deadline  time.Time
}

func (ctx *SearchContext) timeExpired() bool {
	return time.Now().After(ctx.deadline)
}

func DecideMove(myTrail, otherTrail [][]int, turnCount, myBoosts, playerNumber int) string {
	logDebug("Turn %d: Starting search with %d boosts (Player %d)", turnCount, myBoosts, playerNumber)

	// Seed randomness based on turn and player
	rand.Seed(time.Now().UnixNano() + int64(turnCount)*1000 + int64(playerNumber))

	if len(myTrail) == 0 {
		return "RIGHT"
	}

	snapshot := buildGameSnapshot(myTrail, otherTrail, myBoosts, playerNumber)

	if snapshot.myAgent == nil || !snapshot.myAgent.Alive {
		return "RIGHT"
	}

	if snapshot.otherAgent == nil || !snapshot.otherAgent.Alive {
		validMoves := snapshot.myAgent.GetValidMoves()
		if len(validMoves) > 0 {
			return directionToString(validMoves[0])
		}
		return "RIGHT"
	}

	ctx := SearchContext{
		startTime: time.Now(),
		deadline:  time.Now().Add(SEARCH_TIME_LIMIT),
	}

	bestMove := iterativeDeepeningSearch(snapshot, &ctx)

	if debugMode {
		myVoronoi, oppVoronoi, _ := calculateVoronoiControl(snapshot.myAgent, snapshot.otherAgent)
		logDebug("Voronoi control - Me: %d, Opp: %d, Neutral: %d", myVoronoi, oppVoronoi, BOARD_HEIGHT*BOARD_WIDTH-myVoronoi-oppVoronoi)
	}

	elapsed := time.Since(ctx.startTime)

	moveStr := directionToString(bestMove.direction)
	if bestMove.useBoost {
		moveStr += ":BOOST"
	}

	logDebug("Selected move: %s (score: %d, time: %v)", moveStr, bestMove.score, elapsed)
	return moveStr
}

func iterativeDeepeningSearch(snapshot GameStateSnapshot, ctx *SearchContext) Move {
	validMoves := snapshot.myAgent.GetValidMoves()

	if len(validMoves) == 0 {
		return Move{direction: RIGHT, useBoost: false, score: LOSE_SCORE}
	}

	bestMove := Move{direction: validMoves[0], useBoost: false, score: math.MinInt32}

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

		if bestMove.score >= WIN_SCORE || bestMove.score <= LOSE_SCORE {
			logDebug("Found terminal score, stopping search")
			break
		}
	}

	logDebug("Search completed: reached depth %d", depth-1)
	return bestMove
}

func searchAtDepth(snapshot GameStateSnapshot, maxDepth int, ctx *SearchContext) Move {
	validMoves := snapshot.myAgent.GetValidMoves()

	if len(validMoves) == 0 {
		return Move{direction: RIGHT, useBoost: false, score: LOSE_SCORE}
	}

	// Shuffle move order for variety in exploration
	rand.Shuffle(len(validMoves), func(i, j int) {
		validMoves[i], validMoves[j] = validMoves[j], validMoves[i]
	})

	bestMove := Move{direction: validMoves[0], useBoost: false, score: math.MinInt32}
	alpha := math.MinInt32
	beta := math.MaxInt32

	for _, dir := range validMoves {
		if ctx.timeExpired() {
			return bestMove
		}

		for _, useBoost := range []bool{false, true} {
			if useBoost && snapshot.myAgent.BoostsRemaining <= 0 {
				continue
			}

			// Safety check + tactical check
			if useBoost && (!isBoostSafe(snapshot, dir) || !shouldUseBoost(snapshot, dir)) {
				continue
			}

			if ctx.timeExpired() {
				return bestMove
			}

			score := evaluateMoveAtDepth(snapshot, dir, useBoost, maxDepth, alpha, beta, ctx)

			if score > bestMove.score {
				bestMove = Move{
					direction: dir,
					useBoost:  useBoost,
					score:     score,
				}
				alpha = score
			}

			if alpha >= beta {
				return bestMove
			}
		}
	}

	return bestMove
}

func evaluateMoveAtDepth(snapshot GameStateSnapshot, dir Direction, useBoost bool, maxDepth int, alpha int, beta int, ctx *SearchContext) int {
	bestScore := math.MinInt32

	oppValidMoves := snapshot.otherAgent.GetValidMoves()

	for _, oppDir := range oppValidMoves {
		if ctx.timeExpired() {
			return bestScore
		}

		// Consider opponent boost options
		for _, oppUseBoost := range []bool{false, true} {
			if oppUseBoost && snapshot.otherAgent.BoostsRemaining <= 0 {
				continue
			}

			if ctx.timeExpired() {
				return bestScore
			}

			var score int
			if snapshot.amIRed {
				// P1 moves first in reality, so opponent sees my move and reacts
				// Simulate my move first, then opponent's reaction
				_, myState := snapshot.myAgent.UndoableMove(dir, snapshot.otherAgent, useBoost)
				_, oppState := snapshot.otherAgent.UndoableMove(oppDir, snapshot.myAgent, oppUseBoost)
				score = alphabeta(snapshot.myAgent, snapshot.otherAgent, maxDepth-1, alpha, beta, true, ctx)
				snapshot.otherAgent.UndoMove(oppState, snapshot.myAgent)
				snapshot.myAgent.UndoMove(myState, snapshot.otherAgent)
			} else {
				// P2 sees P1's move and can react to it
				// Simulate opponent's move first, then my reaction
				_, oppState := snapshot.otherAgent.UndoableMove(oppDir, snapshot.myAgent, oppUseBoost)
				_, myState := snapshot.myAgent.UndoableMove(dir, snapshot.otherAgent, useBoost)
				score = alphabeta(snapshot.myAgent, snapshot.otherAgent, maxDepth-1, alpha, beta, true, ctx)
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
	}

	return bestScore
}

func alphabeta(myAgent *Agent, otherAgent *Agent, depth int, alpha int, beta int, isMaximizing bool, ctx *SearchContext) int {
	if ctx.timeExpired() {
		return evaluatePosition(myAgent, otherAgent)
	}

	if depth == 0 || !myAgent.Alive || !otherAgent.Alive {
		return evaluatePosition(myAgent, otherAgent)
	}

	if isMaximizing {
		maxScore := math.MinInt32

		for _, myDir := range myAgent.GetValidMoves() {
			if ctx.timeExpired() {
				return maxScore
			}

			minOpponentScore := math.MaxInt32

			for _, oppDir := range otherAgent.GetValidMoves() {
				if ctx.timeExpired() {
					break
				}

				_, myState := myAgent.UndoableMove(myDir, otherAgent, false)
				_, oppState := otherAgent.UndoableMove(oppDir, myAgent, false)
				score := alphabeta(myAgent, otherAgent, depth-1, alpha, beta, true, ctx)
				otherAgent.UndoMove(oppState, myAgent)
				myAgent.UndoMove(myState, otherAgent)

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

		return maxScore
	}

	return evaluatePosition(myAgent, otherAgent)
}

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

	myHead := myAgent.GetHead()
	oppHead := otherAgent.GetHead()

	myHeadState := myAgent.Board.GetCellState(myHead)
	oppHeadState := otherAgent.Board.GetCellState(oppHead)
	myAgent.Board.SetCellState(myHead, EMPTY)
	otherAgent.Board.SetCellState(oppHead, EMPTY)

	cc := NewConnectedComponents(myAgent.Board)
	cc.Calculate()

	myAgent.Board.SetCellState(myHead, myHeadState)
	otherAgent.Board.SetCellState(oppHead, oppHeadState)

	if !cc.AreConnected(myHead, oppHead) {
		myComponentID := cc.GetComponentID(myHead)
		oppComponentID := cc.GetComponentID(oppHead)

		myComponentSize := cc.GetComponentSize(myComponentID)
		oppComponentSize := cc.GetComponentSize(oppComponentID)

		endgameScore := 10 * (myComponentSize - oppComponentSize)

		logDebug("ENDGAME MODE: Players separated! My component: %d, Opp component: %d, Score: %d",
			myComponentSize, oppComponentSize, endgameScore)

		return endgameScore
	}

	score := 0

	// Core metrics (same as steamroller03)
	myTerritory, oppTerritory, control := calculateVoronoiControl(myAgent, otherAgent)
	territoryDiff := myTerritory - oppTerritory

	myFreedom := len(myValidMoves)
	opponentFreedom := len(opponentValidMoves)

	myLocalSpace := countReachableSpace(myAgent, 15)
	oppLocalSpace := countReachableSpace(otherAgent, 15)

	boostDiff := myAgent.BoostsRemaining - otherAgent.BoostsRemaining

	// Base scoring (similar to steamroller03, but keep territory at 50 to match their speed)
	score += territoryDiff * WEIGHT_TERRITORY
	score += (myFreedom - opponentFreedom) * WEIGHT_FREEDOM
	score += (myLocalSpace - oppLocalSpace) * WEIGHT_REACHABLE
	score += boostDiff * WEIGHT_BOOST

	// OUR STRATEGIC ADVANTAGES (kept but with lighter weights)
	// Chamber tree detects articulation points - tactical advantage
	ct := NewChamberTree(myAgent.Board)
	chamberScore := ct.EvaluateChamberTree(myHead, oppHead)
	score += chamberScore * WEIGHT_CHAMBER

	// Edge bonus - positions near walls are safer
	myEdgeBonus := calculateEdgeBonus(myAgent.Board, control, 1)
	oppEdgeBonus := calculateEdgeBonus(otherAgent.Board, control, 2)
	edgeDiff := myEdgeBonus - oppEdgeBonus
	score += edgeDiff * WEIGHT_EDGE

	// NEW: Space-filling efficiency - penalize leaving gaps in our territory
	// Count the "compactness" of each agent's position
	myCompactness := evaluateCompactness(myAgent, control, 1)
	oppCompactness := evaluateCompactness(otherAgent, control, 2)
	score += (myCompactness - oppCompactness) * WEIGHT_COMPACTNESS

	// NEW: Corridor trap detection - heavily penalize dead-end corridors
	myTrapPenalty := detectCorridorTraps(myAgent, otherAgent)
	oppTrapPenalty := detectCorridorTraps(otherAgent, myAgent)
	score += (oppTrapPenalty - myTrapPenalty) // Penalize us being trapped, reward opponent trapped

	// NEW: Head-to-head distance - maintain safe distance from opponent
	headDistance := torusDistance(myHead, oppHead, myAgent.Board)
	if headDistance <= 2 {
		score -= PENALTY_HEAD_DISTANCE
	} else if headDistance <= 4 {
		score -= PENALTY_HEAD_DISTANCE / 4
	}

	// NEW: Cut-off opportunities - detect if we can trap opponent
	cutoffScore := evaluateCutoffOpportunities(myAgent, otherAgent, control)
	score += cutoffScore * WEIGHT_CUTOFF

	// NEW: Future space advantage - look ahead at space growth potential
	myGrowthPotential := evaluateSpaceGrowth(myAgent, otherAgent, control, 1)
	oppGrowthPotential := evaluateSpaceGrowth(otherAgent, myAgent, control, 2)
	score += (myGrowthPotential - oppGrowthPotential) * WEIGHT_GROWTH

	return score
}

// detectCorridorTraps simulates future moves to detect if agent is in a narrow corridor with no escape
// Returns penalty score: higher = more trapped
func detectCorridorTraps(agent *Agent, opponent *Agent) int {
	if !agent.Alive {
		return 0
	}

	// Quick check: if we currently have 3-4 moves, we're probably not in a corridor
	currentMoves := agent.GetValidMoves()
	if len(currentMoves) >= 3 {
		return 0 // Not in a corridor
	}

	if len(currentMoves) == 0 {
		return PENALTY_CORRIDOR_BASE * 10 // Already trapped
	}

	// Simulate 3 moves ahead checking mobility at each step
	penalty := 0
	testBoard := agent.Board.Clone()
	testAgent := agent.Clone(testBoard)
	testOpponent := opponent.Clone(testBoard)

	// Try all possible move sequences (limited depth to save performance)
	minFutureMobility := 4 // Best case: 4 valid moves
	validMovesAtDepth := []int{}

	// Depth 1
	for _, move1 := range currentMoves {
		// Clone for each branch
		b1 := testBoard.Clone()
		a1 := testAgent.Clone(b1)
		o1 := testOpponent.Clone(b1)

		success := a1.Move(move1, o1, false)
		if !success || !a1.Alive {
			continue
		}

		moves1 := a1.GetValidMoves()
		validMovesAtDepth = append(validMovesAtDepth, len(moves1))
		if len(moves1) < minFutureMobility {
			minFutureMobility = len(moves1)
		}

		if len(moves1) == 0 {
			continue // Dead end after 1 move
		}

		// Depth 2 - only explore up to 2 moves at this level to save time
		maxMovesToCheck := 2
		if len(moves1) < maxMovesToCheck {
			maxMovesToCheck = len(moves1)
		}

		for i := 0; i < maxMovesToCheck; i++ {
			move2 := moves1[i]
			b2 := b1.Clone()
			a2 := a1.Clone(b2)
			o2 := o1.Clone(b2)

			success := a2.Move(move2, o2, false)
			if !success || !a2.Alive {
				continue
			}

			moves2 := a2.GetValidMoves()
			validMovesAtDepth = append(validMovesAtDepth, len(moves2))
			if len(moves2) < minFutureMobility {
				minFutureMobility = len(moves2)
			}

			if len(moves2) == 0 {
				continue // Dead end after 2 moves
			}

			// Depth 3 - only check first move
			if len(moves2) > 0 {
				b3 := b2.Clone()
				a3 := a2.Clone(b3)
				o3 := o2.Clone(b3)

				success := a3.Move(moves2[0], o3, false)
				if success && a3.Alive {
					moves3 := a3.GetValidMoves()
					validMovesAtDepth = append(validMovesAtDepth, len(moves3))
					if len(moves3) < minFutureMobility {
						minFutureMobility = len(moves3)
					}
				}
			}
		}
	}

	// Analyze mobility trend
	// If future mobility is consistently low (1-2 moves), we're in a corridor
	avgFutureMobility := 0
	if len(validMovesAtDepth) > 0 {
		sum := 0
		for _, m := range validMovesAtDepth {
			sum += m
		}
		avgFutureMobility = sum / len(validMovesAtDepth)
	}

	// Penalty based on how constrained we become
	if minFutureMobility <= 1 {
		penalty += PENALTY_CORRIDOR_BASE * 6 // Very bad: will have ≤1 move soon
	} else if minFutureMobility == 2 && avgFutureMobility*2 < 5 { // avgFutureMobility < 2.5
		penalty += PENALTY_CORRIDOR_BASE * 3 // Bad: consistently low mobility
	} else if avgFutureMobility*10 < 20 { // avgFutureMobility < 2.0
		penalty += PENALTY_CORRIDOR_BASE // Warning: entering narrow area
	}

	return penalty
}

// torusDistance calculates Manhattan distance on a torus board
func torusDistance(p1, p2 Position, board *GameBoard) int {
	dx := abs(p1.X - p2.X)
	dy := abs(p1.Y - p2.Y)

	// Torus wraparound - check if going the other way is shorter
	if dx > board.Width/2 {
		dx = board.Width - dx
	}
	if dy > board.Height/2 {
		dy = board.Height - dy
	}

	return dx + dy
}

// evaluateCutoffOpportunities detects if we can trap opponent by cutting them off
func evaluateCutoffOpportunities(myAgent *Agent, otherAgent *Agent, control [][]int) int {
	if !myAgent.Alive || !otherAgent.Alive {
		return 0
	}

	oppHead := otherAgent.GetHead()

	// Check if opponent is in a region we can potentially seal off
	// Count how many cells near opponent we control vs they control
	oppRegionSize := 0
	myControlNearOpp := 0
	oppControlNearOpp := 0

	// BFS from opponent head, limited range
	visited := make(map[Position]bool)
	queue := []Position{oppHead}
	visited[oppHead] = true
	depth := 0
	maxDepth := 8

	for len(queue) > 0 && depth < maxDepth {
		levelSize := len(queue)
		for i := 0; i < levelSize; i++ {
			current := queue[0]
			queue = queue[1:]
			oppRegionSize++

			for _, dir := range AllDirections {
				next := Position{
					X: current.X + dir.DX,
					Y: current.Y + dir.DY,
				}
				next = myAgent.Board.TorusCheck(next)

				state := myAgent.Board.GetCellState(next)

				if state == EMPTY {
					if !visited[next] {
						visited[next] = true
						queue = append(queue, next)
					}

					// Check who controls the border
					if control[next.Y][next.X] == 1 {
						myControlNearOpp++
					} else if control[next.Y][next.X] == 2 {
						oppControlNearOpp++
					}
				} else if myAgent.ContainsPosition(next) {
					// Our trail is a barrier
					myControlNearOpp += 2
				}
			}
		}
		depth++
	}

	// If we control more of the border around opponent, we have a cutoff opportunity
	if myControlNearOpp > oppControlNearOpp*2 && oppRegionSize < 30 {
		return 5 // Good cutoff opportunity
	} else if myControlNearOpp > oppControlNearOpp {
		return 2 // Moderate cutoff potential
	}

	return 0
}

// evaluateSpaceGrowth estimates how much space an agent can expand into
// This is different from countReachableSpace - it considers growth *direction*
func evaluateSpaceGrowth(agent *Agent, opponent *Agent, control [][]int, playerID int) int {
	if !agent.Alive {
		return 0
	}

	head := agent.GetHead()
	growthScore := 0

	// Check each direction from head
	for _, dir := range AllDirections {
		next := Position{
			X: head.X + dir.DX,
			Y: head.Y + dir.DY,
		}
		next = agent.Board.TorusCheck(next)

		if agent.Board.GetCellState(next) != EMPTY {
			continue
		}

		// BFS in this direction to see how much space opens up
		visited := make(map[Position]bool)
		queue := []Position{next}
		visited[next] = true
		directionSpace := 0
		controlledSpace := 0
		maxDepth := 10

		for len(queue) > 0 && directionSpace < maxDepth {
			current := queue[0]
			queue = queue[1:]
			directionSpace++

			if control[current.Y][current.X] == playerID {
				controlledSpace++
			}

			for _, d := range AllDirections {
				nextPos := Position{
					X: current.X + d.DX,
					Y: current.Y + d.DY,
				}
				nextPos = agent.Board.TorusCheck(nextPos)

				if !visited[nextPos] && agent.Board.GetCellState(nextPos) == EMPTY {
					visited[nextPos] = true
					queue = append(queue, nextPos)
				}
			}
		}

		// Reward directions that lead to open space we control
		if directionSpace > 5 && controlledSpace > directionSpace/2 {
			growthScore += 3
		} else if directionSpace > 3 {
			growthScore += 1
		}
	}

	return growthScore
}

// evaluateCompactness measures how efficiently an agent is filling its controlled territory
// Higher scores = fewer gaps, more efficient space usage
func evaluateCompactness(agent *Agent, control [][]int, playerID int) int {
	if !agent.Alive {
		return 0
	}

	head := agent.GetHead()
	compactness := 0

	// BFS from head to find nearby cells we control
	visited := make(map[Position]bool)
	queue := []Position{head}
	visited[head] = true
	depth := 0
	maxDepth := 8 // Look ahead 8 cells

	for len(queue) > 0 && depth < maxDepth {
		levelSize := len(queue)
		for i := 0; i < levelSize; i++ {
			current := queue[0]
			queue = queue[1:]

			// Count neighbors in our controlled territory
			controlledNeighbors := 0
			emptyNeighbors := 0

			for _, dir := range AllDirections {
				next := Position{
					X: current.X + dir.DX,
					Y: current.Y + dir.DY,
				}
				next = agent.Board.TorusCheck(next)

				state := agent.Board.GetCellState(next)

				if state == EMPTY {
					if control[next.Y][next.X] == playerID {
						controlledNeighbors++
						if !visited[next] {
							visited[next] = true
							queue = append(queue, next)
						}
					}
					emptyNeighbors++
				} else if agent.ContainsPosition(next) {
					// Neighbor is our own trail
					controlledNeighbors++
				}
			}

			// Reward having many controlled neighbors (compact filling)
			// Penalize having many empty neighbors we don't control (gaps)
			if emptyNeighbors > 0 {
				compactness += (controlledNeighbors * 10) / emptyNeighbors
			} else {
				compactness += controlledNeighbors * 10
			}
		}
		depth++
	}

	return compactness
}

func countReachableSpace(agent *Agent, maxDepth int) int {
	if !agent.Alive {
		return 0
	}

	head := agent.GetHead()
	visited := make(map[Position]bool)
	queue := []Position{head}
	visited[head] = true
	count := 0

	for len(queue) > 0 && count < maxDepth {
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

func shouldUseBoost(snapshot GameStateSnapshot, dir Direction) bool {
	// Conservative baseline (from steamroller03)
	myTerritory, oppTerritory := calculateVoronoiTerritory(snapshot.myAgent, snapshot.otherAgent)

	if myTerritory > oppTerritory+20 {
		return true
	}

	myLocalSpace := countReachableSpace(snapshot.myAgent, 10)
	oppLocalSpace := countReachableSpace(snapshot.otherAgent, 10)

	if myLocalSpace > oppLocalSpace+15 {
		return true
	}

	// NEW: Aggressive boost - use when we can cut opponent off
	testBoard := snapshot.board.Clone()
	testMyAgent := snapshot.myAgent.Clone(testBoard)
	testOtherAgent := snapshot.otherAgent.Clone(testBoard)

	// Try the boost move
	success := testMyAgent.Move(dir, testOtherAgent, true)
	if !success || !testMyAgent.Alive {
		return false
	}

	// Check if boost creates a chamber advantage
	ct := NewChamberTree(testMyAgent.Board)
	myHead := testMyAgent.GetHead()
	oppHead := testOtherAgent.GetHead()
	chamberScore := ct.EvaluateChamberTree(myHead, oppHead)

	// Use boost if it creates a significant chamber advantage (trap opponent)
	if chamberScore > 3 {
		return true
	}

	// NEW: Use boost to get closer and cut off opponent when we're ahead
	myHeadBefore := snapshot.myAgent.GetHead()
	distBefore := torusDistance(myHeadBefore, oppHead, snapshot.board)
	distAfter := torusDistance(myHead, oppHead, testMyAgent.Board)

	// If we're ahead and boost gets us significantly closer, use it aggressively
	if myTerritory > oppTerritory && distBefore > 5 && distAfter < distBefore-2 {
		return true
	}

	return false
}

func isBoostSafe(snapshot GameStateSnapshot, dir Direction) bool {
	head := snapshot.myAgent.GetHead()

	firstPos := Position{
		X: head.X + dir.DX,
		Y: head.Y + dir.DY,
	}
	firstPos = snapshot.myAgent.Board.TorusCheck(firstPos)

	if snapshot.myAgent.Board.GetCellState(firstPos) != EMPTY {
		return false
	}

	secondPos := Position{
		X: firstPos.X + dir.DX,
		Y: firstPos.Y + dir.DY,
	}
	secondPos = snapshot.myAgent.Board.TorusCheck(secondPos)

	if snapshot.myAgent.Board.GetCellState(secondPos) != EMPTY {
		return false
	}

	return true
}

func calculateVoronoiTerritory(myAgent *Agent, otherAgent *Agent) (int, int) {
	myCount, oppCount, _ := calculateVoronoiControl(myAgent, otherAgent)
	return myCount, oppCount
}

func calculateVoronoiControl(myAgent *Agent, otherAgent *Agent) (int, int, [][]int) {
	if !myAgent.Alive {
		control := make([][]int, BOARD_HEIGHT)
		for y := 0; y < BOARD_HEIGHT; y++ {
			row := make([]int, BOARD_WIDTH)
			for x := 0; x < BOARD_WIDTH; x++ {
				row[x] = 2
			}
			control[y] = row
		}
		return 0, BOARD_HEIGHT * BOARD_WIDTH, control
	}
	if !otherAgent.Alive {
		control := make([][]int, BOARD_HEIGHT)
		for y := 0; y < BOARD_HEIGHT; y++ {
			row := make([]int, BOARD_WIDTH)
			for x := 0; x < BOARD_WIDTH; x++ {
				row[x] = 1
			}
			control[y] = row
		}
		return BOARD_HEIGHT * BOARD_WIDTH, 0, control
	}

	type QueueItem struct {
		pos   Position
		owner int
	}

	control := make([][]int, myAgent.Board.Height)
	for y := 0; y < myAgent.Board.Height; y++ {
		control[y] = make([]int, myAgent.Board.Width)
	}

	visited := make(map[Position]int)
	queue := []QueueItem{}

	myHead := myAgent.GetHead()
	oppHead := otherAgent.GetHead()

	queue = append(queue, QueueItem{pos: myHead, owner: 1})
	queue = append(queue, QueueItem{pos: oppHead, owner: 2})
	visited[myHead] = 1
	visited[oppHead] = 2

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		for _, dir := range AllDirections {
			next := Position{
				X: current.pos.X + dir.DX,
				Y: current.pos.Y + dir.DY,
			}
			next = myAgent.Board.TorusCheck(next)

			if myAgent.Board.GetCellState(next) != EMPTY {
				continue
			}

			if _, seen := visited[next]; !seen {
				visited[next] = current.owner
				queue = append(queue, QueueItem{pos: next, owner: current.owner})
			}
		}
	}

	myTerritory := 0
	oppTerritory := 0

	for pos, owner := range visited {
		if owner == 1 {
			myTerritory++
			control[pos.Y][pos.X] = 1
		} else if owner == 2 {
			oppTerritory++
			control[pos.Y][pos.X] = 2
		}
	}

	for y := 0; y < myAgent.Board.Height; y++ {
		for x := 0; x < myAgent.Board.Width; x++ {
			p := Position{X: x, Y: y}
			if myAgent.Board.GetCellState(p) != EMPTY {
				control[y][x] = -1
			} else if control[y][x] == 0 {
				control[y][x] = 0
			}
		}
	}

	return myTerritory, oppTerritory, control
}

func calculateEdgeBonus(board *GameBoard, control [][]int, owner int) int {
	bonus := 0

	for y := 0; y < board.Height; y++ {
		for x := 0; x < board.Width; x++ {
			if control[y][x] != owner {
				continue
			}

			pos := Position{X: x, Y: y}
			cellState := board.GetCellState(pos)

			if cellState != EMPTY {
				continue
			}

			for _, dir := range AllDirections {
				neighbor := Position{X: pos.X + dir.DX, Y: pos.Y + dir.DY}
				neighbor = board.TorusCheck(neighbor)

				if board.GetCellState(neighbor) == EMPTY {
					bonus++
				}
			}
		}
	}

	return bonus
}

func printVoronoiMap(myAgent *Agent, otherAgent *Agent, control [][]int, bestMove Move) {
	if !debugMode {
		return
	}

	fmt.Println("Voronoi map (.: empty, #: wall, M: me, O: opp, m: mine, o: opp, =: neutral)")
	fmt.Printf("Best move: %s (boost=%v, score=%d)\n", directionToString(bestMove.direction), bestMove.useBoost, bestMove.score)

	for y := 0; y < BOARD_HEIGHT; y++ {
		for x := 0; x < BOARD_WIDTH; x++ {
			p := Position{X: x, Y: y}
			cell := myAgent.Board.GetCellState(p)

			if myAgent.IsHead(p) {
				fmt.Print("M")
			} else if otherAgent.IsHead(p) {
				fmt.Print("O")
			} else if cell == AGENT {
				fmt.Print("#")
			} else {
				owner := control[y][x]
				if owner == 1 {
					fmt.Print("m")
				} else if owner == 2 {
					fmt.Print("o")
				} else {
					fmt.Print("=")
				}
			}
		}
		fmt.Println()
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func buildGameSnapshot(myTrail, otherTrail [][]int, myBoosts, playerNumber int) GameStateSnapshot {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)

	for _, pos := range myTrail {
		board.SetCellState(Position{X: pos[0], Y: pos[1]}, AGENT)
	}
	for _, pos := range otherTrail {
		board.SetCellState(Position{X: pos[0], Y: pos[1]}, AGENT)
	}

	myDir := inferDirection(myTrail)
	myAgent := createAgentFromTrail(1, myTrail, myDir, myBoosts, board)

	otherDir := inferDirection(otherTrail)
	otherAgent := createAgentFromTrail(2, otherTrail, otherDir, 3, board)

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
