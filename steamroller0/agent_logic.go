package main

import (
	"fmt"
	"math"
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

	if len(myTrail) == 0 {
		return "RIGHT"
	}

	snapshot := buildGameSnapshot(myTrail, otherTrail, myBoosts, playerNumber)

	if !snapshot.myAgent.Alive {
		return "RIGHT"
	}

	ctx := SearchContext{
		startTime: time.Now(),
		deadline:  time.Now().Add(SEARCH_TIME_LIMIT),
	}

	bestMove := iterativeDeepeningSearch(snapshot, &ctx)

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

		var score int
		if snapshot.amIRed {
			_, myState := snapshot.myAgent.UndoableMove(dir, snapshot.otherAgent, useBoost)
			_, oppState := snapshot.otherAgent.UndoableMove(oppDir, snapshot.myAgent, false)
			score = alphabeta(snapshot.myAgent, snapshot.otherAgent, maxDepth-1, alpha, beta, true, ctx)
			snapshot.otherAgent.UndoMove(oppState, snapshot.myAgent)
			snapshot.myAgent.UndoMove(myState, snapshot.otherAgent)
		} else {
			_, oppState := snapshot.otherAgent.UndoableMove(oppDir, snapshot.myAgent, false)
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

	score := 0

	mySpace := countAvailableSpace(myAgent)
	opponentSpace := countAvailableSpace(otherAgent)
	spaceDiff := mySpace - opponentSpace
	score += spaceDiff * 20

	myFreedom := len(myValidMoves)
	opponentFreedom := len(opponentValidMoves)
	score += (myFreedom - opponentFreedom) * 50

	return score
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
