package main

import (
	"fmt"
	"math"
	"time"
)

// ============================================================================
// Search: Principal Variation Search (PVS) with optimizations
// ============================================================================

type SearchContext struct {
	startTime     time.Time
	deadline      time.Time
	moveOrdering  *MoveOrderingContext
	nodesSearched int
}

type TTEntry struct {
	score int
	depth int
	flag  int
}

var transpositionTable map[uint64]TTEntry

func logDebug(format string, args ...interface{}) {
	if debugMode {
		fmt.Printf("[DEBUG] "+format+"\n", args...)
	}
}

// Timer check
func (ctx *SearchContext) timeExpired() bool {
	return time.Now().After(ctx.deadline)
}

// Iterative deepening search
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
		ctx.moveOrdering.bestMoveCache[depth] = bestMove

		logDebug("Completed depth %d: best move %s (boost=%v) with score %d",
			depth, directionToString(bestMove.direction), bestMove.useBoost, bestMove.score)

		depth++

		if bestMove.score >= WIN_SCORE || bestMove.score <= LOSE_SCORE {
			logDebug("Found terminal score, stopping search")
			break
		}
	}

	logDebug("Search completed: reached depth %d, nodes: %d", depth-1, ctx.nodesSearched)
	return bestMove
}

// Root search with move ordering
func searchAtDepth(snapshot GameStateSnapshot, maxDepth int, ctx *SearchContext) Move {
	validMoves := snapshot.myAgent.GetValidMoves()

	if len(validMoves) == 0 {
		return Move{direction: RIGHT, useBoost: false, score: LOSE_SCORE}
	}

	bestMove := Move{direction: validMoves[0], useBoost: false, score: math.MinInt32}
	alpha := math.MinInt32
	beta := math.MaxInt32

	orderedMoves := ctx.moveOrdering.orderMovesAtRoot(validMoves, snapshot, maxDepth)

	firstMove := true
	moveIndex := 0

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

			var score int

			if firstMove {
				// Full window search for first move
				score = evaluateMoveAtDepthPVS(snapshot, dir, useBoost, maxDepth, alpha, beta, ctx, true)
				firstMove = false
			} else {
				// Null window search
				score = evaluateMoveAtDepthPVS(snapshot, dir, useBoost, maxDepth, alpha, alpha+1, ctx, false)

				// Re-search if it improves alpha
				if score > alpha && score < beta {
					score = evaluateMoveAtDepthPVS(snapshot, dir, useBoost, maxDepth, alpha, beta, ctx, true)
				}
			}

			// Boost aggressive bonus
			if shouldBoostAggressively(snapshot, dir, useBoost) {
				score += BOOST_AGGRESSIVE_BONUS
			}

			if score > bestMove.score {
				bestMove = Move{
					direction: dir,
					useBoost:  useBoost,
					score:     score,
				}
				alpha = score

				ctx.moveOrdering.updateHistory(dir, maxDepth, true)
			} else {
				ctx.moveOrdering.updateHistory(dir, maxDepth, false)
			}

			if alpha >= beta {
				ctx.moveOrdering.updateKillerMove(maxDepth, dir)
				return bestMove
			}

			moveIndex++
		}
	}

	return bestMove
}

// Evaluate move with PVS
func evaluateMoveAtDepthPVS(snapshot GameStateSnapshot, dir Direction, useBoost bool, maxDepth int, alpha int, beta int, ctx *SearchContext, fullWindow bool) int {
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
			score = pvs(snapshot.board, snapshot.myAgent, snapshot.otherAgent, maxDepth-1, alpha, beta, true, snapshot.amIRed, ctx)
			snapshot.otherAgent.UndoMove(oppState, snapshot.myAgent)
			snapshot.myAgent.UndoMove(myState, snapshot.otherAgent)
		} else {
			_, oppState := snapshot.otherAgent.UndoableMove(oppDir, snapshot.myAgent, false)
			_, myState := snapshot.myAgent.UndoableMove(dir, snapshot.otherAgent, useBoost)
			score = pvs(snapshot.board, snapshot.myAgent, snapshot.otherAgent, maxDepth-1, alpha, beta, true, snapshot.amIRed, ctx)
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

// Principal Variation Search (PVS)
func pvs(board *GameBoard, myAgent *Agent, otherAgent *Agent, depth int, alpha int, beta int, isMaximizing bool, amIRed bool, ctx *SearchContext) int {
	ctx.nodesSearched++

	if ctx.timeExpired() {
		return evaluatePositionWithBias(myAgent, otherAgent, amIRed)
	}

	if depth == 0 || !myAgent.Alive || !otherAgent.Alive {
		return evaluatePositionWithBias(myAgent, otherAgent, amIRed)
	}

	if isMaximizing {
		maxScore := math.MinInt32
		myMoves := myAgent.GetValidMoves()

		if len(myMoves) == 0 {
			return LOSE_SCORE
		}

		firstMove := true
		moveIndex := 0

		for _, myDir := range myMoves {
			if ctx.timeExpired() {
				return maxScore
			}

			boostOptions := []bool{false}
			if myAgent.BoostsRemaining > 0 && depth >= 3 {
				boostOptions = append(boostOptions, true)
			}

			for _, myBoost := range boostOptions {
				// Determine if LMR should apply
				reduction := 0
				if shouldApplyLMR(moveIndex, depth, myBoost) {
					reduction = getLMRReduction(moveIndex, depth)
				}

				minOpponentScore := math.MaxInt32
				oppMoves := otherAgent.GetValidMoves()

				for _, oppDir := range oppMoves {
					if ctx.timeExpired() {
						break
					}

					oppBoost := false
					var score int

					if amIRed {
						_, myState := myAgent.UndoableMove(myDir, otherAgent, myBoost)
						_, oppState := otherAgent.UndoableMove(oppDir, myAgent, oppBoost)

						if firstMove {
							// Full window
							score = pvs(board, myAgent, otherAgent, depth-1-reduction, alpha, beta, true, amIRed, ctx)
							firstMove = false
						} else {
							// Null window
							score = pvs(board, myAgent, otherAgent, depth-1-reduction, alpha, alpha+1, true, amIRed, ctx)

							// Re-search with full window if needed
							if score > alpha && score < beta {
								score = pvs(board, myAgent, otherAgent, depth-1, alpha, beta, true, amIRed, ctx)
							}
						}

						otherAgent.UndoMove(oppState, myAgent)
						myAgent.UndoMove(myState, otherAgent)
					} else {
						_, oppState := otherAgent.UndoableMove(oppDir, myAgent, oppBoost)
						_, myState := myAgent.UndoableMove(myDir, otherAgent, myBoost)

						if firstMove {
							score = pvs(board, myAgent, otherAgent, depth-1-reduction, alpha, beta, true, amIRed, ctx)
							firstMove = false
						} else {
							score = pvs(board, myAgent, otherAgent, depth-1-reduction, alpha, alpha+1, true, amIRed, ctx)
							if score > alpha && score < beta {
								score = pvs(board, myAgent, otherAgent, depth-1, alpha, beta, true, amIRed, ctx)
							}
						}

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

				moveIndex++
			}
		}

		return maxScore
	} else {
		return evaluatePositionWithBias(myAgent, otherAgent, amIRed)
	}
}

// Boost aggressive check
func shouldBoostAggressively(snapshot GameStateSnapshot, dir Direction, useBoost bool) bool {
	if !useBoost || snapshot.myAgent.BoostsRemaining == 0 {
		return false
	}

	myHead := snapshot.myAgent.GetHead()
	opponentHead := snapshot.otherAgent.GetHead()

	// Escape from tight spots
	myValidMoves := len(snapshot.myAgent.GetValidMoves())
	if myValidMoves <= BOOST_ESCAPE_FREEDOM {
		return true
	}

	mySpace := countAvailableSpace(snapshot.myAgent)
	opponentSpace := countAvailableSpace(snapshot.otherAgent)

	// Boost when behind in space
	if mySpace < opponentSpace-30 {
		centerX, centerY := BOARD_WIDTH/2, BOARD_HEIGHT/2
		distToCenter := manhattanDistanceRaw(myHead.X, myHead.Y, centerX, centerY)
		if distToCenter > 5 {
			return true
		}
	}

	dist := manhattanDistance(myHead, opponentHead)

	// Aggressive cutting when ahead
	if dist >= 3 && dist <= 6 && mySpace > opponentSpace+BOOST_CUTTING_ADVANTAGE {
		return true
	}

	// Information denial in late-midgame
	snapshot2 := GameStateSnapshot{
		myAgent:    snapshot.myAgent,
		otherAgent: snapshot.otherAgent,
		board:      snapshot.board,
		amIRed:     true,
	}
	phase := detectGamePhase(snapshot2)
	if phase == Midgame && mySpace > opponentSpace {
		turnEstimate := len(snapshot.myAgent.Trail)
		if turnEstimate >= 80 && turnEstimate <= 120 {
			return true
		}
	}

	return false
}

// Calculate boost reserve value
func boostReserveValue(boostsRemaining int, currentMove int) int {
	return (boostsRemaining * (200 - currentMove) * 5) / 2 // BOOST_RESERVE_MULTIPLIER = 2.5
}
