package main

import (
	"fmt"
	"math"
	"os"
)

var debugMode = os.Getenv("DEBUG") == "1"

const (
	MAX_DEPTH        = 3
	WIN_SCORE        = 10000
	LOSE_SCORE       = -10000
	DRAW_SCORE       = 0
	BOARD_HEIGHT     = 18
	BOARD_WIDTH      = 20
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
}

type Move struct {
	direction Direction
	useBoost  bool
	score     int
}

func DecideMove(myTrail, otherTrail [][]int, turnCount, myBoosts int) string {
	logDebug("Turn %d: Starting minimax with %d boosts", turnCount, myBoosts)
	
	if len(myTrail) == 0 {
		return "RIGHT"
	}
	
	snapshot := buildGameSnapshot(myTrail, otherTrail, myBoosts)
	
	bestMove := findBestMove(snapshot, turnCount)
	
	moveStr := directionToString(bestMove.direction)
	if bestMove.useBoost {
		moveStr += ":BOOST"
	}
	
	logDebug("Selected move: %s (score: %d)", moveStr, bestMove.score)
	return moveStr
}

func buildGameSnapshot(myTrail, otherTrail [][]int, myBoosts int) GameStateSnapshot {
	board := NewGameBoard(BOARD_HEIGHT, BOARD_WIDTH)
	
	for _, pos := range myTrail {
		board.SetCellState(Position{X: pos[0], Y: pos[1]}, AGENT)
	}
	for _, pos := range otherTrail {
		board.SetCellState(Position{X: pos[0], Y: pos[1]}, AGENT)
	}
	
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

func findBestMove(snapshot GameStateSnapshot, turnCount int) Move {
	validMoves := snapshot.myAgent.GetValidMoves()
	
	if len(validMoves) == 0 {
		return Move{direction: RIGHT, useBoost: false, score: LOSE_SCORE}
	}
	
	bestMove := Move{direction: validMoves[0], useBoost: false, score: math.MinInt32}
	
	for _, dir := range validMoves {
		for _, useBoost := range []bool{false, true} {
			if useBoost && snapshot.myAgent.BoostsRemaining <= 0 {
				continue
			}
			
			score := evaluateMove(snapshot, dir, useBoost)
			
			logDebug("Move %s (boost=%v): score=%d", directionToString(dir), useBoost, score)
			
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

func evaluateMove(snapshot GameStateSnapshot, dir Direction, useBoost bool) int {
	simBoard := snapshot.board.Clone()
	simMyAgent := snapshot.myAgent.Clone(simBoard)
	simOtherAgent := snapshot.otherAgent.Clone(simBoard)
	
	simMyAgent.Move(dir, simOtherAgent, useBoost)
	
	return minimax(simBoard, simMyAgent, simOtherAgent, MAX_DEPTH, true)
}

func minimax(board *GameBoard, myAgent *Agent, otherAgent *Agent, depth int, isMaximizing bool) int {
	if depth == 0 || !myAgent.Alive || !otherAgent.Alive {
		return evaluatePosition(myAgent, otherAgent)
	}
	
	if isMaximizing {
		maxScore := math.MinInt32
		
		for _, dir := range myAgent.GetValidMoves() {
			simBoard := board.Clone()
			simMyAgent := myAgent.Clone(simBoard)
			simOtherAgent := otherAgent.Clone(simBoard)
			
			simMyAgent.Move(dir, simOtherAgent, false)
			
			score := minimax(simBoard, simMyAgent, simOtherAgent, depth-1, false)
			
			if score > maxScore {
				maxScore = score
			}
		}
		
		return maxScore
	} else {
		minScore := math.MaxInt32
		
		for _, dir := range otherAgent.GetValidMoves() {
			simBoard := board.Clone()
			simMyAgent := myAgent.Clone(simBoard)
			simOtherAgent := otherAgent.Clone(simBoard)
			
			simOtherAgent.Move(dir, simMyAgent, false)
			
			score := minimax(simBoard, simMyAgent, simOtherAgent, depth-1, true)
			
			if score < minScore {
				minScore = score
			}
		}
		
		return minScore
	}
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
	
	score := 0
	
	score += myAgent.Length * 10
	score -= otherAgent.Length * 10
	
	mySpace := countAvailableSpace(myAgent)
	otherSpace := countAvailableSpace(otherAgent)
	score += (mySpace - otherSpace) * 5
	
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

func inferDirection(trail [][]int) Direction {
	if len(trail) < 2 {
		return RIGHT
	}
	
	head := trail[len(trail)-1]
	prev := trail[len(trail)-2]
	
	dx := head[0] - prev[0]
	dy := head[1] - prev[1]
	
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
