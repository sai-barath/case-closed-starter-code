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
	SEARCH_TIME_LIMIT         = 3500 * time.Millisecond
	WIN_SCORE                 = 10000000
	LOSE_SCORE                = -10000000
	DRAW_SCORE                = 0
	BOARD_HEIGHT              = 18
	BOARD_WIDTH               = 20
	TT_SIZE                   = 1 << 20
	ASPIRATION_WINDOW_INIT    = 50
	NULL_MOVE_R_SHALLOW       = 2
	NULL_MOVE_R_DEEP          = 3
	LMR_BASE                  = 2.5
	QSEARCH_MAX_DEPTH         = 8
	KILLER_MOVES_PER_PLY      = 2
	MAX_SEARCH_DEPTH          = 50
	MVV_LVA_MULTIPLIER        = 100
	ADAPTIVE_DEPTH_MULTIPLIER = 0.4
	ADAPTIVE_DEPTH_BASE       = 3
)

// Tunable parameters for evolutionary training
var (
	// Base heuristics
	// WEIGHT_TERRITORY      = 140
	// WEIGHT_FREEDOM        = 105
	// WEIGHT_REACHABLE      = 110
	// WEIGHT_BOOST          = 19
	// WEIGHT_CHAMBER        = 9
	// WEIGHT_EDGE           = 37
	// WEIGHT_COMPACTNESS    = -18
	// WEIGHT_CUTOFF         = 2
	// WEIGHT_GROWTH         = 9
	// PENALTY_CORRIDOR_BASE = 677
	// PENALTY_HEAD_DISTANCE = 94

	// // Advanced heuristics
	// WEIGHT_VORONOI_SECOND_ORDER = 80
	// WEIGHT_POTENTIAL_MOBILITY   = 90
	// WEIGHT_TRAIL_THREAT         = 120
	// WEIGHT_INFLUENCE            = 60
	// WEIGHT_WALL_PENALTY         = 200
	// WEIGHT_TERRITORY_DENSITY    = 40
	// WEIGHT_ESCAPE_ROUTES        = 150
	// WEIGHT_OPPONENT_MOBILITY    = 70
	// WEIGHT_LOOKAHEAD_CONTROL    = 85
	// WEIGHT_SPACE_EFFICIENCY     = 55
	// WEIGHT_AGGRESSIVE_CUTOFF    = 110
	// WEIGHT_DEFENSIVE_SPACING    = 75
	// WEIGHT_CENTER_CONTROL       = 45
	// WEIGHT_FUTURE_TERRITORY     = 95
	// WEIGHT_MOBILITY_PROJECTION  = 65
	// WEIGHT_CHOKE_POINT          = 50

	EARLY_GAME_EXPANSION        = 2.0
	MID_GAME_BALANCE            = 1.0
	LATE_GAME_TERRITORY         = 1.5
	ENDGAME_SURVIVAL            = 3.0
	WEIGHT_TERRITORY            = 153
	WEIGHT_FREEDOM              = 8
	WEIGHT_REACHABLE            = 165
	WEIGHT_BOOST                = 11
	WEIGHT_CHAMBER              = 48
	WEIGHT_EDGE                 = -12
	WEIGHT_COMPACTNESS          = 18
	WEIGHT_CUTOFF               = 12
	WEIGHT_GROWTH               = 93
	PENALTY_CORRIDOR_BASE       = 452
	PENALTY_HEAD_DISTANCE       = 179
	WEIGHT_VORONOI_SECOND_ORDER = 115
	WEIGHT_POTENTIAL_MOBILITY   = 120
	WEIGHT_TRAIL_THREAT         = 9
	WEIGHT_INFLUENCE            = 24
	WEIGHT_WALL_PENALTY         = 208
	WEIGHT_TERRITORY_DENSITY    = 84
	WEIGHT_ESCAPE_ROUTES        = 111
	WEIGHT_OPPONENT_MOBILITY    = 17
	WEIGHT_LOOKAHEAD_CONTROL    = 29
	WEIGHT_SPACE_EFFICIENCY     = 15
	WEIGHT_AGGRESSIVE_CUTOFF    = 123
	WEIGHT_DEFENSIVE_SPACING    = 109
	WEIGHT_CENTER_CONTROL       = 67
	WEIGHT_FUTURE_TERRITORY     = 3
	WEIGHT_MOBILITY_PROJECTION  = 36
	WEIGHT_CHOKE_POINT          = 30

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
	turnCount  int
}

type Move struct {
	direction Direction
	useBoost  bool
	score     int
}

type BoundType int

const (
	EXACT_BOUND BoundType = iota
	ALPHA_BOUND
	BETA_BOUND
)

type TranspositionEntry struct {
	zobristKey uint64
	depth      int
	score      int
	bestMove   Direction
	useBoost   bool
	boundType  BoundType
}

type KillerMoves struct {
	moves [MAX_SEARCH_DEPTH][KILLER_MOVES_PER_PLY]Direction
}

type HistoryTable struct {
	scores [4][BOARD_HEIGHT][BOARD_WIDTH]int
}

type SearchContext struct {
	startTime     time.Time
	deadline      time.Time
	tt            map[uint64]*TranspositionEntry
	killers       *KillerMoves
	history       *HistoryTable
	nodesSearched int
	ttHits        int
	ttCutoffs     int
}

func (ctx *SearchContext) timeExpired() bool {
	return time.Now().After(ctx.deadline)
}

func (km *KillerMoves) add(depth int, move Direction) {
	if depth >= MAX_SEARCH_DEPTH || depth < 0 {
		return
	}
	if km.moves[depth][0] != move {
		km.moves[depth][1] = km.moves[depth][0]
		km.moves[depth][0] = move
	}
}

func (km *KillerMoves) isKiller(depth int, move Direction) bool {
	if depth >= MAX_SEARCH_DEPTH || depth < 0 {
		return false
	}
	return km.moves[depth][0] == move || km.moves[depth][1] == move
}

func (ht *HistoryTable) add(dir Direction, depth int, pos Position) {
	if directionToInt(dir) < 4 && pos.X < BOARD_WIDTH && pos.Y < BOARD_HEIGHT && pos.X >= 0 && pos.Y >= 0 {
		ht.scores[directionToInt(dir)][pos.Y][pos.X] += depth * depth
	}
}

func (ht *HistoryTable) get(dir Direction, pos Position) int {
	if directionToInt(dir) < 4 && pos.X < BOARD_WIDTH && pos.Y < BOARD_HEIGHT && pos.X >= 0 && pos.Y >= 0 {
		return ht.scores[directionToInt(dir)][pos.Y][pos.X]
	}
	return 0
}

func directionToInt(dir Direction) int {
	if dir == UP {
		return 0
	} else if dir == DOWN {
		return 1
	} else if dir == LEFT {
		return 2
	} else if dir == RIGHT {
		return 3
	}
	return 0
}

func computeZobrist(myAgent *Agent, otherAgent *Agent) uint64 {
	var hash uint64 = 0
	for _, pos := range myAgent.Trail {
		hash ^= uint64(pos.X*BOARD_WIDTH + pos.Y)
	}
	for _, pos := range otherAgent.Trail {
		hash ^= uint64((pos.X*BOARD_WIDTH + pos.Y) * 31337)
	}
	hash ^= uint64(myAgent.BoostsRemaining) << 32
	hash ^= uint64(otherAgent.BoostsRemaining) << 40
	return hash
}

func DecideMove(myTrail, otherTrail [][]int, turnCount, myBoosts, playerNumber int) string {
	logDebug("Turn %d: Starting search with %d boosts (Player %d)", turnCount, myBoosts, playerNumber)

	// Seed randomness based on turn and player
	rand.Seed(time.Now().UnixNano() + int64(turnCount)*1000 + int64(playerNumber))

	if len(myTrail) == 0 {
		return "RIGHT"
	}

	snapshot := buildGameSnapshot(myTrail, otherTrail, myBoosts, playerNumber, turnCount)

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
		startTime:     time.Now(),
		deadline:      time.Now().Add(SEARCH_TIME_LIMIT),
		tt:            make(map[uint64]*TranspositionEntry, TT_SIZE),
		killers:       &KillerMoves{},
		history:       &HistoryTable{},
		nodesSearched: 0,
		ttHits:        0,
		ttCutoffs:     0,
	}

	bestMove := iterativeDeepeningSearch(snapshot, &ctx)

	if debugMode {
		myVoronoi, oppVoronoi, _ := calculateVoronoiControl(snapshot.myAgent, snapshot.otherAgent)
		logDebug("Voronoi control - Me: %d, Opp: %d, Neutral: %d", myVoronoi, oppVoronoi, BOARD_HEIGHT*BOARD_WIDTH-myVoronoi-oppVoronoi)
		logDebug("Search stats - Nodes: %d, TT hits: %d, TT cutoffs: %d", ctx.nodesSearched, ctx.ttHits, ctx.ttCutoffs)
		if ctx.nodesSearched > 0 {
			logDebug("TT hit rate: %.1f%%", float64(ctx.ttHits)*100.0/float64(ctx.nodesSearched))
		}
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
	prevScore := 0

	depth := 1
	for !ctx.timeExpired() && depth <= MAX_SEARCH_DEPTH {
		logDebug("Searching at depth %d...", depth)

		window := ASPIRATION_WINDOW_INIT
		alpha := prevScore - window
		beta := prevScore + window

		if depth == 1 {
			alpha = math.MinInt32
			beta = math.MaxInt32
		}

		var depthBestMove Move

		for {
			depthBestMove = searchAtDepth(snapshot, depth, alpha, beta, ctx)

			if ctx.timeExpired() {
				logDebug("Time expired during depth %d, using previous result", depth)
				return bestMove
			}

			if depthBestMove.score <= alpha {
				
				alpha = depthBestMove.score - window
				window *= 2
				logDebug("Aspiration window fail-low, widening to [%d, %d]", alpha, beta)
			} else if depthBestMove.score >= beta {
				
				beta = depthBestMove.score + window
				window *= 2
				logDebug("Aspiration window fail-high, widening to [%d, %d]", alpha, beta)
			} else {
				break
			}

			if window > WIN_SCORE {
				alpha = math.MinInt32
				beta = math.MaxInt32
				logDebug("Aspiration window too wide, using full window")
				continue
			}
		}

		bestMove = depthBestMove
		prevScore = bestMove.score

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

func searchAtDepth(snapshot GameStateSnapshot, maxDepth int, alpha int, beta int, ctx *SearchContext) Move {
	validMoves := snapshot.myAgent.GetValidMoves()

	if len(validMoves) == 0 {
		return Move{direction: RIGHT, useBoost: false, score: LOSE_SCORE}
	}

	zobrist := computeZobrist(snapshot.myAgent, snapshot.otherAgent)
	ttMove := UP
	if entry, ok := ctx.tt[zobrist]; ok && entry.depth >= maxDepth {
		ttMove = entry.bestMove
	}

	scoredMoves := make([]struct {
		dir      Direction
		useBoost bool
		score    int
	}, 0, len(validMoves)*2)

	for _, dir := range validMoves {
		for _, useBoost := range []bool{false, true} {
			if useBoost && snapshot.myAgent.BoostsRemaining <= 0 {
				continue
			}
			if useBoost && (!isBoostSafe(snapshot, dir) || !shouldUseBoost(snapshot, dir)) {
				continue
			}

			moveScore := scoreMoveForOrdering(snapshot, dir, useBoost, ttMove, ctx, maxDepth)
			scoredMoves = append(scoredMoves, struct {
				dir      Direction
				useBoost bool
				score    int
			}{dir, useBoost, moveScore})
		}
	}

	sortMoves(scoredMoves)

	bestMove := Move{direction: validMoves[0], useBoost: false, score: math.MinInt32}

	for idx, sm := range scoredMoves {
		if ctx.timeExpired() {
			return bestMove
		}

		reduction := 0
		if idx >= 4 && maxDepth >= 3 && !sm.useBoost {
			reduction = int(math.Log(float64(maxDepth)) * math.Log(float64(idx)) / LMR_BASE)
			if reduction > maxDepth-1 {
				reduction = maxDepth - 1
			}
		}

		score := evaluateMoveAtDepth(snapshot, sm.dir, sm.useBoost, maxDepth-reduction, alpha, beta, ctx, maxDepth)

		if reduction > 0 && score > alpha {
			score = evaluateMoveAtDepth(snapshot, sm.dir, sm.useBoost, maxDepth, alpha, beta, ctx, maxDepth)
		}

		if score > bestMove.score {
			bestMove = Move{
				direction: sm.dir,
				useBoost:  sm.useBoost,
				score:     score,
			}
			alpha = score
		}

		if alpha >= beta {
			if !sm.useBoost {
				ctx.killers.add(maxDepth, sm.dir)
				myHead := snapshot.myAgent.GetHead()
				ctx.history.add(sm.dir, maxDepth, myHead)
			}
			break
		}
	}

	boundType := EXACT_BOUND
	if bestMove.score <= alpha {
		boundType = ALPHA_BOUND
	} else if bestMove.score >= beta {
		boundType = BETA_BOUND
	}

	ctx.tt[zobrist] = &TranspositionEntry{
		zobristKey: zobrist,
		depth:      maxDepth,
		score:      bestMove.score,
		bestMove:   bestMove.direction,
		useBoost:   bestMove.useBoost,
		boundType:  boundType,
	}

	return bestMove
}

func scoreMoveForOrdering(snapshot GameStateSnapshot, dir Direction, useBoost bool, ttMove Direction, ctx *SearchContext, depth int) int {
	score := 0

	if dir == ttMove {
		score += 10000
	}

	if ctx.killers.isKiller(depth, dir) {
		score += 5000
	}

	myHead := snapshot.myAgent.GetHead()
	score += ctx.history.get(dir, myHead)

	if useBoost {
		score += 1000
	}

	return score
}

func sortMoves(moves []struct {
	dir      Direction
	useBoost bool
	score    int
}) {
	for i := 0; i < len(moves); i++ {
		for j := i + 1; j < len(moves); j++ {
			if moves[j].score > moves[i].score {
				moves[i], moves[j] = moves[j], moves[i]
			}
		}
	}
}

func evaluateMoveAtDepth(snapshot GameStateSnapshot, dir Direction, useBoost bool, maxDepth int, alpha int, beta int, ctx *SearchContext, originalDepth int) int {
	bestScore := math.MinInt32

	oppValidMoves := snapshot.otherAgent.GetValidMoves()

	for _, oppDir := range oppValidMoves {
		if ctx.timeExpired() {
			return bestScore
		}

		for _, oppUseBoost := range []bool{false, true} {
			if oppUseBoost && snapshot.otherAgent.BoostsRemaining <= 0 {
				continue
			}

			if ctx.timeExpired() {
				return bestScore
			}

			var score int
			if snapshot.amIRed {
				_, myState := snapshot.myAgent.UndoableMove(dir, snapshot.otherAgent, useBoost)
				_, oppState := snapshot.otherAgent.UndoableMove(oppDir, snapshot.myAgent, oppUseBoost)
				score = alphabeta(snapshot.myAgent, snapshot.otherAgent, maxDepth-1, alpha, beta, true, ctx, snapshot.turnCount, originalDepth)
				snapshot.otherAgent.UndoMove(oppState, snapshot.myAgent)
				snapshot.myAgent.UndoMove(myState, snapshot.otherAgent)
			} else {
				_, oppState := snapshot.otherAgent.UndoableMove(oppDir, snapshot.myAgent, oppUseBoost)
				_, myState := snapshot.myAgent.UndoableMove(dir, snapshot.otherAgent, useBoost)
				score = alphabeta(snapshot.myAgent, snapshot.otherAgent, maxDepth-1, alpha, beta, true, ctx, snapshot.turnCount, originalDepth)
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

func alphabeta(myAgent *Agent, otherAgent *Agent, depth int, alpha int, beta int, isMaximizing bool, ctx *SearchContext, turnCount int, searchDepth int) int {
	ctx.nodesSearched++

	if ctx.timeExpired() {
		return evaluatePositionWithDepth(myAgent, otherAgent, turnCount, depth, searchDepth)
	}

	if depth == 0 || !myAgent.Alive || !otherAgent.Alive {
		return quiescenceSearch(myAgent, otherAgent, alpha, beta, QSEARCH_MAX_DEPTH, ctx, turnCount, searchDepth)
	}

	zobrist := computeZobrist(myAgent, otherAgent)
	if entry, ok := ctx.tt[zobrist]; ok {
		ctx.ttHits++
		if entry.depth >= depth {
			if entry.boundType == EXACT_BOUND {
				ctx.ttCutoffs++
				return entry.score
			} else if entry.boundType == ALPHA_BOUND && entry.score <= alpha {
				ctx.ttCutoffs++
				return entry.score
			} else if entry.boundType == BETA_BOUND && entry.score >= beta {
				ctx.ttCutoffs++
				return entry.score
			}
		}
	}

	if isMaximizing && depth >= 3 {
		staticEval := evaluatePositionWithDepth(myAgent, otherAgent, turnCount, depth, searchDepth)
		if staticEval >= beta && len(myAgent.GetValidMoves()) > 0 {
			R := NULL_MOVE_R_SHALLOW
			if depth > 6 {
				R = NULL_MOVE_R_DEEP
			}

			score := -alphabeta(otherAgent, myAgent, depth-R-1, -beta, -beta+1, true, ctx, turnCount, searchDepth)
			if score >= beta {
				return beta
			}
		}
	}

	if isMaximizing {
		maxScore := math.MinInt32
		bestMove := UP

		validMoves := myAgent.GetValidMoves()
		if len(validMoves) == 0 {
			return evaluatePositionWithDepth(myAgent, otherAgent, turnCount, depth, searchDepth)
		}

		for _, myDir := range validMoves {
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
				score := alphabeta(myAgent, otherAgent, depth-1, alpha, beta, true, ctx, turnCount, searchDepth)
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
				bestMove = myDir
			}

			alpha = max(alpha, minOpponentScore)
			if beta <= alpha {
				break
			}
		}

		boundType := EXACT_BOUND
		if maxScore <= alpha {
			boundType = ALPHA_BOUND
		} else if maxScore >= beta {
			boundType = BETA_BOUND
		}

		ctx.tt[zobrist] = &TranspositionEntry{
			zobristKey: zobrist,
			depth:      depth,
			score:      maxScore,
			bestMove:   bestMove,
			useBoost:   false,
			boundType:  boundType,
		}

		return maxScore
	}

	return evaluatePositionWithDepth(myAgent, otherAgent, turnCount, depth, searchDepth)
}

func quiescenceSearch(myAgent *Agent, otherAgent *Agent, alpha int, beta int, qDepth int, ctx *SearchContext, turnCount int, searchDepth int) int {
	standPat := evaluatePositionWithDepth(myAgent, otherAgent, turnCount, 0, searchDepth)

	if qDepth <= 0 || ctx.timeExpired() {
		return standPat
	}

	if standPat >= beta {
		return beta
	}

	if standPat > alpha {
		alpha = standPat
	}

	return standPat
}

func evaluatePositionWithDepth(myAgent *Agent, otherAgent *Agent, turnCount int, remainingDepth int, searchDepth int) int {
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

	weights := getDynamicWeights(turnCount, searchDepth)

	myTerritory, oppTerritory, control := calculateVoronoiControl(myAgent, otherAgent)
	territoryDiff := myTerritory - oppTerritory

	myFreedom := len(myValidMoves)
	opponentFreedom := len(opponentValidMoves)

	adaptiveDepth := calculateAdaptiveDepth(15, remainingDepth)
	myLocalSpace := countReachableSpace(myAgent, adaptiveDepth)
	oppLocalSpace := countReachableSpace(otherAgent, adaptiveDepth)

	boostDiff := myAgent.BoostsRemaining - otherAgent.BoostsRemaining

	score += territoryDiff * weights.territory
	score += (myFreedom - opponentFreedom) * weights.freedom
	score += (myLocalSpace - oppLocalSpace) * weights.reachable
	score += boostDiff * weights.boost

	ct := NewChamberTree(myAgent.Board)
	chamberScore := ct.EvaluateChamberTree(myHead, oppHead)
	score += chamberScore * weights.chamber

	myEdgeBonus := calculateEdgeBonus(myAgent.Board, control, 1)
	oppEdgeBonus := calculateEdgeBonus(otherAgent.Board, control, 2)
	edgeDiff := myEdgeBonus - oppEdgeBonus
	score += edgeDiff * weights.edge

	myCompactness := evaluateCompactness(myAgent, control, 1)
	oppCompactness := evaluateCompactness(otherAgent, control, 2)
	score += (myCompactness - oppCompactness) * weights.compactness

	adaptiveTrapDepth := calculateAdaptiveDepth(3, remainingDepth)
	myTrapPenalty := detectCorridorTrapsAdaptive(myAgent, otherAgent, adaptiveTrapDepth)
	oppTrapPenalty := detectCorridorTrapsAdaptive(otherAgent, myAgent, adaptiveTrapDepth)
	score += (oppTrapPenalty - myTrapPenalty)

	headDistance := torusDistance(myHead, oppHead, myAgent.Board)
	if headDistance <= 2 {
		score -= PENALTY_HEAD_DISTANCE
	} else if headDistance <= 4 {
		score -= PENALTY_HEAD_DISTANCE / 4
	}

	cutoffScore := evaluateCutoffOpportunities(myAgent, otherAgent, control)
	score += cutoffScore * weights.cutoff

	myGrowthPotential := evaluateSpaceGrowth(myAgent, otherAgent, control, 1)
	oppGrowthPotential := evaluateSpaceGrowth(otherAgent, myAgent, control, 2)
	score += (myGrowthPotential - oppGrowthPotential) * weights.growth

	chokeScore := evaluateChokePoints(myAgent, otherAgent, control)
	score += chokeScore * weights.chokePoint

	secondOrderVoronoi := calculateSecondOrderVoronoi(myAgent, otherAgent)
	score += secondOrderVoronoi * weights.voronoiSecondOrder

	potentialMobility := calculatePotentialMobility(myAgent, otherAgent)
	score += potentialMobility * weights.potentialMobility

	myTrailThreat := calculateTrailThreat(myAgent, otherAgent)
	oppTrailThreat := calculateTrailThreat(otherAgent, myAgent)
	score -= myTrailThreat * weights.trailThreat
	score += oppTrailThreat * weights.trailThreat

	influenceScore := calculateInfluence(myAgent, otherAgent)
	score += influenceScore * weights.influence

	myWallPenalty := calculateWallPenalty(myAgent)
	oppWallPenalty := calculateWallPenalty(otherAgent)
	score -= myWallPenalty * weights.wallPenalty
	score += oppWallPenalty * weights.wallPenalty

	myTerritoryDensity := calculateTerritoryDensity(myAgent, control, 1)
	oppTerritoryDensity := calculateTerritoryDensity(otherAgent, control, 2)
	score += (myTerritoryDensity - oppTerritoryDensity) * weights.territoryDensity

	myEscapeRoutes := calculateEscapeRoutes(myAgent)
	oppEscapeRoutes := calculateEscapeRoutes(otherAgent)
	score += (myEscapeRoutes - oppEscapeRoutes) * weights.escapeRoutes

	oppMobilityRestriction := calculateOpponentMobilityRestriction(myAgent, otherAgent)
	score += oppMobilityRestriction * weights.opponentMobility

	lookaheadControl := calculateLookaheadControl(myAgent, otherAgent)
	score += lookaheadControl * weights.lookaheadControl

	mySpaceEfficiency := calculateSpaceEfficiency(myAgent, otherAgent)
	oppSpaceEfficiency := calculateSpaceEfficiency(otherAgent, myAgent)
	score += (mySpaceEfficiency - oppSpaceEfficiency) * weights.spaceEfficiency

	aggressiveCutoffScore := calculateAggressiveCutoff(myAgent, otherAgent)
	score += aggressiveCutoffScore * weights.aggressiveCutoff

	defensiveSpacing := calculateDefensiveSpacing(myAgent, otherAgent)
	score += defensiveSpacing * weights.defensiveSpacing

	myCenterControl := calculateCenterControl(myAgent)
	oppCenterControl := calculateCenterControl(otherAgent)
	score += (myCenterControl - oppCenterControl) * weights.centerControl

	futureTerritory := calculateFutureTerritory(myAgent, otherAgent)
	score += futureTerritory * weights.futureTerritory

	myMobilityProjection := calculateMobilityProjection(myAgent, otherAgent)
	oppMobilityProjection := calculateMobilityProjection(otherAgent, myAgent)
	score += (myMobilityProjection - oppMobilityProjection) * weights.mobilityProjection

	return score
}

func evaluatePosition(myAgent *Agent, otherAgent *Agent, turnCount int) int {
	return evaluatePositionWithDepth(myAgent, otherAgent, turnCount, 0, 1)
}

func calculateAdaptiveDepth(maxDepth int, remainingDepth int) int {
	adaptiveDepth := int(float64(remainingDepth)*ADAPTIVE_DEPTH_MULTIPLIER) + ADAPTIVE_DEPTH_BASE
	if adaptiveDepth > maxDepth {
		return maxDepth
	}
	if adaptiveDepth < 1 {
		return 1
	}
	return adaptiveDepth
}

func detectCorridorTrapsAdaptive(agent *Agent, opponent *Agent, simDepth int) int {
	if !agent.Alive {
		return 0
	}

	currentMoves := agent.GetValidMoves()
	if len(currentMoves) >= 3 {
		return 0
	}

	if len(currentMoves) == 0 {
		return PENALTY_CORRIDOR_BASE * 10
	}

	if simDepth <= 0 {
		simDepth = 3
	}

	penalty := 0
	testBoard := agent.Board.Clone()
	testAgent := agent.Clone(testBoard)
	testOpponent := opponent.Clone(testBoard)

	minFutureMobility := 4
	validMovesAtDepth := []int{}

	for _, move1 := range currentMoves {
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

		if len(moves1) == 0 || simDepth <= 1 {
			continue
		}

		maxMovesToCheck := 2
		if len(moves1) < maxMovesToCheck {
			maxMovesToCheck = len(moves1)
		}

		for i := 0; i < maxMovesToCheck && i < len(moves1); i++ {
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

			if len(moves2) == 0 || simDepth <= 2 {
				continue
			}

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

	avgFutureMobility := 0
	if len(validMovesAtDepth) > 0 {
		sum := 0
		for _, m := range validMovesAtDepth {
			sum += m
		}
		avgFutureMobility = sum / len(validMovesAtDepth)
	}

	if minFutureMobility <= 1 {
		penalty += PENALTY_CORRIDOR_BASE * 6
	} else if minFutureMobility == 2 && avgFutureMobility*2 < 5 {
		penalty += PENALTY_CORRIDOR_BASE * 3
	} else if avgFutureMobility*10 < 20 {
		penalty += PENALTY_CORRIDOR_BASE
	}

	return penalty
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

func evaluateChokePoints(myAgent *Agent, otherAgent *Agent, control [][]int) int {
	if !myAgent.Alive || !otherAgent.Alive {
		return 0
	}

	apf := NewArticulationPointFinder(myAgent.Board)
	articulationPoints := apf.FindArticulationPoints()

	if len(articulationPoints) == 0 {
		return 0
	}

	myHead := myAgent.GetHead()
	oppHead := otherAgent.GetHead()
	score := 0

	for ap := range articulationPoints {
		myDist := torusDistance(myHead, ap, myAgent.Board)
		oppDist := torusDistance(oppHead, ap, myAgent.Board)

		distAdvantage := oppDist - myDist

		if myDist < oppDist {
			score += distAdvantage * 50

			if myDist <= 2 {
				score += 100
			} else if myDist <= 4 {
				score += 50
			}
		} else if oppDist < myDist {
			score += distAdvantage * 50

			if oppDist <= 2 {
				score -= 100
			}
		}

		chokeValue := evaluateChokePointValue(ap, myAgent.Board)
		if chokeValue > 10 {
			score += distAdvantage * (chokeValue / 10)
		}
	}

	return score
}

func evaluateChokePointValue(chokePoint Position, board *GameBoard) int {
	originalState := board.GetCellState(chokePoint)
	board.SetCellState(chokePoint, AGENT)

	cc := NewConnectedComponents(board)
	cc.Calculate()

	componentSizes := make(map[int]int)
	for y := 0; y < board.Height; y++ {
		for x := 0; x < board.Width; x++ {
			compID := cc.Components[y][x]
			if compID >= 0 {
				componentSizes[compID]++
			}
		}
	}

	board.SetCellState(chokePoint, originalState)

	numComponents := len(componentSizes)
	if numComponents <= 1 {
		return 0
	}

	value := 0
	for _, size := range componentSizes {
		value += size / 10
	}

	value += (numComponents - 1) * 20

	return value
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

type WeightSet struct {
	territory          int
	freedom            int
	reachable          int
	boost              int
	chamber            int
	edge               int
	compactness        int
	cutoff             int
	growth             int
	chokePoint         int
	voronoiSecondOrder int
	potentialMobility  int
	trailThreat        int
	influence          int
	wallPenalty        int
	territoryDensity   int
	escapeRoutes       int
	opponentMobility   int
	lookaheadControl   int
	spaceEfficiency    int
	aggressiveCutoff   int
	defensiveSpacing   int
	centerControl      int
	futureTerritory    int
	mobilityProjection int
}

func getDynamicWeights(turnCount int, searchDepth int) WeightSet {
	weights := WeightSet{}

	depthMultiplierTactical := 1.0
	depthMultiplierStrategic := 1.0

	if searchDepth < 5 {
		depthMultiplierTactical = 1.5
		depthMultiplierStrategic = 0.7
	} else if searchDepth > 8 {
		depthMultiplierTactical = 0.8
		depthMultiplierStrategic = 1.3
	}

	_ = int(float64(20) * (float64(WEIGHT_TERRITORY) / 200.0))

	if turnCount < 50 {
		weights.territory = int(float64(WEIGHT_TERRITORY/2) * depthMultiplierStrategic)
		weights.freedom = int(float64(WEIGHT_FREEDOM*2) * depthMultiplierTactical)
		weights.reachable = int(float64(WEIGHT_REACHABLE*2) * depthMultiplierTactical)
		weights.boost = WEIGHT_BOOST
		weights.chamber = WEIGHT_CHAMBER
		weights.edge = WEIGHT_EDGE / 2
		weights.compactness = WEIGHT_COMPACTNESS / 2
		weights.cutoff = WEIGHT_CUTOFF / 2
		weights.growth = WEIGHT_GROWTH * 2
		weights.chokePoint = 20
		weights.voronoiSecondOrder = WEIGHT_VORONOI_SECOND_ORDER / 2
		weights.potentialMobility = int(float64(WEIGHT_POTENTIAL_MOBILITY) * EARLY_GAME_EXPANSION)
		weights.trailThreat = WEIGHT_TRAIL_THREAT / 2
		weights.influence = WEIGHT_INFLUENCE * 2
		weights.wallPenalty = WEIGHT_WALL_PENALTY
		weights.territoryDensity = WEIGHT_TERRITORY_DENSITY / 2
		weights.escapeRoutes = int(float64(WEIGHT_ESCAPE_ROUTES) * EARLY_GAME_EXPANSION)
		weights.opponentMobility = WEIGHT_OPPONENT_MOBILITY
		weights.lookaheadControl = WEIGHT_LOOKAHEAD_CONTROL * 2
		weights.spaceEfficiency = WEIGHT_SPACE_EFFICIENCY / 2
		weights.aggressiveCutoff = WEIGHT_AGGRESSIVE_CUTOFF / 2
		weights.defensiveSpacing = WEIGHT_DEFENSIVE_SPACING * 2
		weights.centerControl = WEIGHT_CENTER_CONTROL * 2
		weights.futureTerritory = WEIGHT_FUTURE_TERRITORY * 2
		weights.mobilityProjection = int(float64(WEIGHT_MOBILITY_PROJECTION) * EARLY_GAME_EXPANSION)
	} else if turnCount < 150 {
		weights.territory = int(float64(WEIGHT_TERRITORY) * depthMultiplierStrategic)
		weights.freedom = int(float64(WEIGHT_FREEDOM) * depthMultiplierTactical)
		weights.reachable = int(float64(WEIGHT_REACHABLE) * depthMultiplierTactical)
		weights.boost = WEIGHT_BOOST
		weights.chamber = WEIGHT_CHAMBER
		weights.edge = WEIGHT_EDGE
		weights.compactness = WEIGHT_COMPACTNESS
		weights.cutoff = WEIGHT_CUTOFF
		weights.growth = WEIGHT_GROWTH
		weights.chokePoint = 40
		weights.voronoiSecondOrder = WEIGHT_VORONOI_SECOND_ORDER
		weights.potentialMobility = WEIGHT_POTENTIAL_MOBILITY
		weights.trailThreat = WEIGHT_TRAIL_THREAT
		weights.influence = WEIGHT_INFLUENCE
		weights.wallPenalty = WEIGHT_WALL_PENALTY
		weights.territoryDensity = WEIGHT_TERRITORY_DENSITY
		weights.escapeRoutes = WEIGHT_ESCAPE_ROUTES
		weights.opponentMobility = WEIGHT_OPPONENT_MOBILITY
		weights.lookaheadControl = WEIGHT_LOOKAHEAD_CONTROL
		weights.spaceEfficiency = WEIGHT_SPACE_EFFICIENCY
		weights.aggressiveCutoff = WEIGHT_AGGRESSIVE_CUTOFF
		weights.defensiveSpacing = WEIGHT_DEFENSIVE_SPACING
		weights.centerControl = WEIGHT_CENTER_CONTROL
		weights.futureTerritory = WEIGHT_FUTURE_TERRITORY
		weights.mobilityProjection = WEIGHT_MOBILITY_PROJECTION
	} else {
		weights.territory = int(float64(WEIGHT_TERRITORY) * LATE_GAME_TERRITORY * 2 * depthMultiplierStrategic)
		weights.freedom = int(float64(WEIGHT_FREEDOM/2) * depthMultiplierTactical)
		weights.reachable = int(float64(WEIGHT_REACHABLE/2) * depthMultiplierTactical)
		weights.boost = WEIGHT_BOOST
		weights.chamber = WEIGHT_CHAMBER * 2
		weights.edge = WEIGHT_EDGE * 2
		weights.compactness = WEIGHT_COMPACTNESS * 2
		weights.cutoff = WEIGHT_CUTOFF * 2
		weights.growth = WEIGHT_GROWTH / 2
		weights.chokePoint = 60
		weights.voronoiSecondOrder = int(float64(WEIGHT_VORONOI_SECOND_ORDER) * LATE_GAME_TERRITORY)
		weights.potentialMobility = WEIGHT_POTENTIAL_MOBILITY / 2
		weights.trailThreat = int(float64(WEIGHT_TRAIL_THREAT) * ENDGAME_SURVIVAL)
		weights.influence = WEIGHT_INFLUENCE / 2
		weights.wallPenalty = int(float64(WEIGHT_WALL_PENALTY) * ENDGAME_SURVIVAL)
		weights.territoryDensity = int(float64(WEIGHT_TERRITORY_DENSITY) * LATE_GAME_TERRITORY)
		weights.escapeRoutes = int(float64(WEIGHT_ESCAPE_ROUTES) * ENDGAME_SURVIVAL)
		weights.opponentMobility = int(float64(WEIGHT_OPPONENT_MOBILITY) * LATE_GAME_TERRITORY)
		weights.lookaheadControl = WEIGHT_LOOKAHEAD_CONTROL / 2
		weights.spaceEfficiency = int(float64(WEIGHT_SPACE_EFFICIENCY) * LATE_GAME_TERRITORY)
		weights.aggressiveCutoff = int(float64(WEIGHT_AGGRESSIVE_CUTOFF) * LATE_GAME_TERRITORY)
		weights.defensiveSpacing = WEIGHT_DEFENSIVE_SPACING / 2
		weights.centerControl = WEIGHT_CENTER_CONTROL / 2
		weights.futureTerritory = WEIGHT_FUTURE_TERRITORY / 2
		weights.mobilityProjection = WEIGHT_MOBILITY_PROJECTION / 2
	}

	return weights
}

func buildGameSnapshot(myTrail, otherTrail [][]int, myBoosts, playerNumber, turnCount int) GameStateSnapshot {
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
		turnCount:  turnCount,
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

func calculateSecondOrderVoronoi(myAgent *Agent, otherAgent *Agent) int {
	if !myAgent.Alive || !otherAgent.Alive {
		return 0
	}

	_, _, control := calculateVoronoiControl(myAgent, otherAgent)

	mySecondOrder := 0
	oppSecondOrder := 0

	for y := 0; y < myAgent.Board.Height; y++ {
		for x := 0; x < myAgent.Board.Width; x++ {
			pos := Position{X: x, Y: y}

			if control[y][x] == 1 {
				for _, dir := range AllDirections {
					next := Position{X: pos.X + dir.DX, Y: pos.Y + dir.DY}
					next = myAgent.Board.TorusCheck(next)

					if control[next.Y][next.X] == 2 {
						myHead := myAgent.GetHead()
						oppHead := otherAgent.GetHead()

						myDist := torusDistance(myHead, next, myAgent.Board)
						oppDist := torusDistance(oppHead, next, otherAgent.Board)

						if myDist < oppDist-1 {
							mySecondOrder++
						}
					}
				}
			} else if control[y][x] == 2 {
				for _, dir := range AllDirections {
					next := Position{X: pos.X + dir.DX, Y: pos.Y + dir.DY}
					next = myAgent.Board.TorusCheck(next)

					if control[next.Y][next.X] == 1 {
						myHead := myAgent.GetHead()
						oppHead := otherAgent.GetHead()

						myDist := torusDistance(myHead, next, myAgent.Board)
						oppDist := torusDistance(oppHead, next, otherAgent.Board)

						if oppDist < myDist-1 {
							oppSecondOrder++
						}
					}
				}
			}
		}
	}

	return mySecondOrder - oppSecondOrder
}

func calculatePotentialMobility(myAgent *Agent, otherAgent *Agent) int {
	if !myAgent.Alive || !otherAgent.Alive {
		return 0
	}

	myMoves := myAgent.GetValidMoves()
	oppMoves := otherAgent.GetValidMoves()

	myPotential := 0
	for _, move := range myMoves {
		testBoard := myAgent.Board.Clone()
		testAgent := myAgent.Clone(testBoard)
		testOpp := otherAgent.Clone(testBoard)

		success := testAgent.Move(move, testOpp, false)
		if success && testAgent.Alive {
			myPotential += len(testAgent.GetValidMoves())
		}
	}

	oppPotential := 0
	for _, move := range oppMoves {
		testBoard := otherAgent.Board.Clone()
		testOpp := otherAgent.Clone(testBoard)
		testAgent := myAgent.Clone(testBoard)

		success := testOpp.Move(move, testAgent, false)
		if success && testOpp.Alive {
			oppPotential += len(testOpp.GetValidMoves())
		}
	}

	return myPotential - oppPotential
}

func calculateTrailThreat(myAgent *Agent, otherAgent *Agent) int {
	if !myAgent.Alive || !otherAgent.Alive {
		return 0
	}

	myHead := myAgent.GetHead()
	oppTrail := otherAgent.Trail

	threat := 0
	lookback := 15
	if len(oppTrail) < lookback {
		lookback = len(oppTrail)
	}

	for i := len(oppTrail) - 1; i >= len(oppTrail)-lookback && i >= 0; i-- {
		trailPos := oppTrail[i]
		dist := torusDistance(myHead, trailPos, myAgent.Board)

		if dist == 0 {
			continue
		}

		threat += 100 / (dist + 1)
	}

	return threat
}

func calculateInfluence(myAgent *Agent, otherAgent *Agent) int {
	if !myAgent.Alive || !otherAgent.Alive {
		return 0
	}

	centralityMap := make([][]int, myAgent.Board.Height)
	for y := 0; y < myAgent.Board.Height; y++ {
		centralityMap[y] = make([]int, myAgent.Board.Width)
		for x := 0; x < myAgent.Board.Width; x++ {
			centerX := myAgent.Board.Width / 2
			centerY := myAgent.Board.Height / 2

			dx := abs(x - centerX)
			dy := abs(y - centerY)

			if dx > myAgent.Board.Width/2 {
				dx = myAgent.Board.Width - dx
			}
			if dy > myAgent.Board.Height/2 {
				dy = myAgent.Board.Height - dy
			}

			centralityMap[y][x] = 100 - (dx+dy)*5
			if centralityMap[y][x] < 0 {
				centralityMap[y][x] = 0
			}
		}
	}

	myInfluence := 0
	myHead := myAgent.GetHead()
	visited := make(map[Position]bool)
	queue := []Position{myHead}
	visited[myHead] = true
	depth := 0
	maxDepth := 7

	for len(queue) > 0 && depth < maxDepth {
		levelSize := len(queue)
		for i := 0; i < levelSize; i++ {
			current := queue[0]
			queue = queue[1:]

			myInfluence += centralityMap[current.Y][current.X]

			for _, dir := range AllDirections {
				next := Position{X: current.X + dir.DX, Y: current.Y + dir.DY}
				next = myAgent.Board.TorusCheck(next)

				if !visited[next] && myAgent.Board.GetCellState(next) == EMPTY {
					visited[next] = true
					queue = append(queue, next)
				}
			}
		}
		depth++
	}

	oppInfluence := 0
	oppHead := otherAgent.GetHead()
	visited = make(map[Position]bool)
	queue = []Position{oppHead}
	visited[oppHead] = true
	depth = 0

	for len(queue) > 0 && depth < maxDepth {
		levelSize := len(queue)
		for i := 0; i < levelSize; i++ {
			current := queue[0]
			queue = queue[1:]

			oppInfluence += centralityMap[current.Y][current.X]

			for _, dir := range AllDirections {
				next := Position{X: current.X + dir.DX, Y: current.Y + dir.DY}
				next = otherAgent.Board.TorusCheck(next)

				if !visited[next] && otherAgent.Board.GetCellState(next) == EMPTY {
					visited[next] = true
					queue = append(queue, next)
				}
			}
		}
		depth++
	}

	return myInfluence - oppInfluence
}

func calculateWallPenalty(agent *Agent) int {
	if !agent.Alive {
		return 0
	}

	head := agent.GetHead()
	penalty := 0

	for _, dir := range AllDirections {
		next := Position{X: head.X + dir.DX, Y: head.Y + dir.DY}
		next = agent.Board.TorusCheck(next)

		if agent.Board.GetCellState(next) != EMPTY {
			penalty += 50
		}
	}

	return penalty
}

func calculateTerritoryDensity(agent *Agent, control [][]int, playerID int) int {
	if !agent.Alive {
		return 0
	}

	head := agent.GetHead()
	visited := make(map[Position]bool)
	queue := []Position{head}
	visited[head] = true

	controlledCells := 0
	totalCells := 0
	maxDepth := 10
	depth := 0

	for len(queue) > 0 && depth < maxDepth {
		levelSize := len(queue)
		for i := 0; i < levelSize; i++ {
			current := queue[0]
			queue = queue[1:]
			totalCells++

			if control[current.Y][current.X] == playerID {
				controlledCells++
			}

			for _, dir := range AllDirections {
				next := Position{X: current.X + dir.DX, Y: current.Y + dir.DY}
				next = agent.Board.TorusCheck(next)

				if !visited[next] && agent.Board.GetCellState(next) == EMPTY {
					visited[next] = true
					queue = append(queue, next)
				}
			}
		}
		depth++
	}

	if totalCells == 0 {
		return 0
	}

	return (controlledCells * 100) / totalCells
}

func calculateEscapeRoutes(agent *Agent) int {
	if !agent.Alive {
		return 0
	}

	escapeRoutes := 0

	for _, move := range agent.GetValidMoves() {
		testBoard := agent.Board.Clone()
		testAgent := agent.Clone(testBoard)

		success := testAgent.Move(move, testAgent, false)
		if !success || !testAgent.Alive {
			continue
		}

		visited := make(map[Position]bool)
		queue := []Position{testAgent.GetHead()}
		visited[queue[0]] = true
		reachable := 0
		maxReach := 20

		for len(queue) > 0 && reachable < maxReach {
			current := queue[0]
			queue = queue[1:]
			reachable++

			for _, dir := range AllDirections {
				next := Position{X: current.X + dir.DX, Y: current.Y + dir.DY}
				next = testAgent.Board.TorusCheck(next)

				if !visited[next] && testAgent.Board.GetCellState(next) == EMPTY {
					visited[next] = true
					queue = append(queue, next)
				}
			}
		}

		if reachable >= 15 {
			escapeRoutes++
		}
	}

	return escapeRoutes
}

func calculateOpponentMobilityRestriction(myAgent *Agent, otherAgent *Agent) int {
	if !myAgent.Alive || !otherAgent.Alive {
		return 0
	}

	oppHead := otherAgent.GetHead()
	restrictionScore := 0

	for _, dir := range AllDirections {
		next := Position{X: oppHead.X + dir.DX, Y: oppHead.Y + dir.DY}
		next = otherAgent.Board.TorusCheck(next)

		if myAgent.ContainsPosition(next) {
			restrictionScore += 30
		} else if otherAgent.Board.GetCellState(next) != EMPTY {
			restrictionScore += 10
		}
	}

	myHead := myAgent.GetHead()
	dist := torusDistance(myHead, oppHead, myAgent.Board)

	if dist <= 3 {
		restrictionScore += (4 - dist) * 20
	}

	return restrictionScore
}

func calculateLookaheadControl(myAgent *Agent, otherAgent *Agent) int {
	if !myAgent.Alive || !otherAgent.Alive {
		return 0
	}

	myMoves := myAgent.GetValidMoves()
	if len(myMoves) == 0 {
		return -1000
	}

	bestControl := -999999

	for _, move := range myMoves {
		testBoard := myAgent.Board.Clone()
		testMe := myAgent.Clone(testBoard)
		testOpp := otherAgent.Clone(testBoard)

		success := testMe.Move(move, testOpp, false)
		if !success || !testMe.Alive {
			continue
		}

		myT, oppT, _ := calculateVoronoiControl(testMe, testOpp)
		control := myT - oppT

		if control > bestControl {
			bestControl = control
		}
	}

	return bestControl
}

func calculateSpaceEfficiency(agent *Agent, otherAgent *Agent) int {
	if !agent.Alive {
		return 0
	}

	head := agent.GetHead()

	visited := make(map[Position]bool)
	queue := []Position{head}
	visited[head] = true

	emptySpaces := 0
	trailSpaces := 0
	maxDepth := 12
	depth := 0

	for len(queue) > 0 && depth < maxDepth {
		levelSize := len(queue)
		for i := 0; i < levelSize; i++ {
			current := queue[0]
			queue = queue[1:]

			for _, dir := range AllDirections {
				next := Position{X: current.X + dir.DX, Y: current.Y + dir.DY}
				next = agent.Board.TorusCheck(next)

				if !visited[next] {
					visited[next] = true

					if agent.Board.GetCellState(next) == EMPTY {
						emptySpaces++
						queue = append(queue, next)
					} else if agent.ContainsPosition(next) {
						trailSpaces++
					}
				}
			}
		}
		depth++
	}

	total := emptySpaces + trailSpaces
	if total == 0 {
		return 0
	}

	return (trailSpaces * 100) / total
}

func calculateAggressiveCutoff(myAgent *Agent, otherAgent *Agent) int {
	if !myAgent.Alive || !otherAgent.Alive {
		return 0
	}

	myHead := myAgent.GetHead()
	oppHead := otherAgent.GetHead()

	dist := torusDistance(myHead, oppHead, myAgent.Board)

	if dist > 8 {
		return 0
	}

	oppReachable := countReachableSpace(otherAgent, 20)

	cutoffScore := 0
	if oppReachable < 30 {
		cutoffScore += (30 - oppReachable) * 5
	}

	myReachable := countReachableSpace(myAgent, 20)
	if float64(myReachable) > float64(oppReachable)*1.3 {
		cutoffScore += 50
	}

	for _, dir := range AllDirections {
		next := Position{X: oppHead.X + dir.DX, Y: oppHead.Y + dir.DY}
		next = otherAgent.Board.TorusCheck(next)

		myDist := torusDistance(myHead, next, myAgent.Board)
		oppDist := 1

		if myDist < oppDist {
			cutoffScore += 20
		}
	}

	return cutoffScore
}

func calculateDefensiveSpacing(myAgent *Agent, otherAgent *Agent) int {
	if !myAgent.Alive || !otherAgent.Alive {
		return 0
	}

	myHead := myAgent.GetHead()
	oppHead := otherAgent.GetHead()

	dist := torusDistance(myHead, oppHead, myAgent.Board)

	optimalDist := 8
	spacing := -abs(dist-optimalDist) * 10

	if dist <= 2 {
		spacing -= 100
	} else if dist <= 4 {
		spacing -= 50
	}

	return spacing
}

func calculateCenterControl(agent *Agent) int {
	if !agent.Alive {
		return 0
	}

	head := agent.GetHead()
	centerX := agent.Board.Width / 2
	centerY := agent.Board.Height / 2

	dx := abs(head.X - centerX)
	dy := abs(head.Y - centerY)

	if dx > agent.Board.Width/2 {
		dx = agent.Board.Width - dx
	}
	if dy > agent.Board.Height/2 {
		dy = agent.Board.Height - dy
	}

	centerDist := dx + dy

	return 100 - centerDist*5
}

func calculateFutureTerritory(myAgent *Agent, otherAgent *Agent) int {
	if !myAgent.Alive || !otherAgent.Alive {
		return 0
	}

	myMoves := myAgent.GetValidMoves()
	if len(myMoves) == 0 {
		return -1000
	}

	totalFuture := 0
	count := 0

	for _, move := range myMoves {
		testBoard := myAgent.Board.Clone()
		testMe := myAgent.Clone(testBoard)
		testOpp := otherAgent.Clone(testBoard)

		success := testMe.Move(move, testOpp, false)
		if !success || !testMe.Alive {
			continue
		}

		myT, oppT := calculateVoronoiTerritory(testMe, testOpp)
		totalFuture += (myT - oppT)
		count++
	}

	if count == 0 {
		return -1000
	}

	return totalFuture / count
}

func calculateMobilityProjection(agent *Agent, opponent *Agent) int {
	if !agent.Alive {
		return 0
	}

	moves := agent.GetValidMoves()
	if len(moves) == 0 {
		return 0
	}

	totalMobility := 0
	futureDepth := 3

	for _, move := range moves {
		testBoard := agent.Board.Clone()
		testAgent := agent.Clone(testBoard)
		testOpp := opponent.Clone(testBoard)

		success := testAgent.Move(move, testOpp, false)
		if !success || !testAgent.Alive {
			continue
		}

		mobilitySum := len(testAgent.GetValidMoves())

		for d := 1; d < futureDepth; d++ {
			nextMoves := testAgent.GetValidMoves()
			if len(nextMoves) == 0 {
				break
			}

			testBoard2 := testAgent.Board.Clone()
			testAgent2 := testAgent.Clone(testBoard2)
			testOpp2 := testOpp.Clone(testBoard2)

			success := testAgent2.Move(nextMoves[0], testOpp2, false)
			if success && testAgent2.Alive {
				mobilitySum += len(testAgent2.GetValidMoves())
			}
		}

		totalMobility += mobilitySum
	}

	return totalMobility
}
