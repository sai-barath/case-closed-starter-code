package main

import (
	"math"
)

// ============================================================================
// Territory Evaluation: Voronoi, Influence Functions, Biconnected Components
// ============================================================================

type VoronoiResult struct {
	myTerritory       int
	opponentTerritory int
	contestedCells    int
}

type InfluenceResult struct {
	myInfluence       float64
	opponentInfluence float64
	controlledCells   int
}

// Adaptive Voronoi: depth scales with game phase
func voronoiTerritoryAdaptive(myAgent *Agent, opponentAgent *Agent, phase GamePhase) VoronoiResult {
	maxSteps := MIDGAME_VORONOI_DEPTH

	switch phase {
	case Opening:
		maxSteps = OPENING_VORONOI_DEPTH
	case Endgame:
		maxSteps = ENDGAME_VORONOI_DEPTH
	}

	return voronoiTerritory(myAgent, opponentAgent, maxSteps)
}

// Voronoi territory calculation with BFS
func voronoiTerritory(myAgent *Agent, opponentAgent *Agent, maxSteps int) VoronoiResult {
	myHead := myAgent.GetHead()
	opponentHead := opponentAgent.GetHead()

	myQueue := []Position{myHead}
	opponentQueue := []Position{opponentHead}

	myVisited := make(map[Position]int)
	opponentVisited := make(map[Position]int)

	myVisited[myHead] = 0
	opponentVisited[opponentHead] = 0

	for step := 0; step < maxSteps; step++ {
		// Expand my territory
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

		// Expand opponent territory
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

	// Count territory with depth-squared weighting
	myTerritory := 0
	opponentTerritory := 0
	contested := 0

	for pos, myDist := range myVisited {
		weight := (maxSteps - myDist) * (maxSteps - myDist) // Depth-squared weighting

		if opponentDist, opponentReached := opponentVisited[pos]; opponentReached {
			if myDist < opponentDist {
				myTerritory += weight
			} else if opponentDist < myDist {
				opponentWeight := (maxSteps - opponentDist) * (maxSteps - opponentDist)
				opponentTerritory += opponentWeight
			} else {
				contested++
			}
		} else {
			myTerritory += weight
		}
	}

	for pos, opponentDist := range opponentVisited {
		if _, myReached := myVisited[pos]; !myReached {
			weight := (maxSteps - opponentDist) * (maxSteps - opponentDist)
			opponentTerritory += weight
		}
	}

	return VoronoiResult{
		myTerritory:       myTerritory,
		opponentTerritory: opponentTerritory,
		contestedCells:    contested,
	}
}

// Influence function evaluation (exponential decay)
func influenceFunctionTerritory(myAgent *Agent, opponentAgent *Agent) InfluenceResult {
	myHead := myAgent.GetHead()
	opponentHead := opponentAgent.GetHead()

	myInfluence := 0.0
	opponentInfluence := 0.0
	controlledCells := 0

	// Calculate influence for each empty cell
	for y := 0; y < BOARD_HEIGHT; y++ {
		for x := 0; x < BOARD_WIDTH; x++ {
			pos := Position{X: x, Y: y}

			if myAgent.Board.GetCellState(pos) != EMPTY {
				continue
			}

			// Calculate distance with torus wrap
			myDist := float64(manhattanDistance(myHead, pos))
			oppDist := float64(manhattanDistance(opponentHead, pos))

			// Exponential decay influence
			myInf := INFLUENCE_BASE_STRENGTH * math.Exp(-myDist/INFLUENCE_DECAY_RATE)
			oppInf := INFLUENCE_BASE_STRENGTH * math.Exp(-oppDist/INFLUENCE_DECAY_RATE)

			// Attenuation by opponent proximity
			myInf *= (1.0 - math.Pow(oppInf/INFLUENCE_BASE_STRENGTH, INFLUENCE_OPPONENT_ATTENUATION))
			oppInf *= (1.0 - math.Pow(myInf/INFLUENCE_BASE_STRENGTH, INFLUENCE_OPPONENT_ATTENUATION))

			myInfluence += myInf
			opponentInfluence += oppInf

			if myInf > oppInf*1.2 {
				controlledCells++
			} else if oppInf > myInf*1.2 {
				controlledCells--
			}
		}
	}

	return InfluenceResult{
		myInfluence:       myInfluence,
		opponentInfluence: opponentInfluence,
		controlledCells:   controlledCells,
	}
}

// Detect barriers (3-sided boxes created by opponent)
func detectBarriers(agent *Agent, opponent *Agent) int {
	barriers := 0
	head := agent.GetHead()

	// Check 3x3 grid around head for enclosure patterns
	for _, dir := range AllDirections {
		next := Position{X: head.X + dir.DX, Y: head.Y + dir.DY}
		next = agent.Board.TorusCheck(next)

		if agent.Board.GetCellState(next) == EMPTY {
			// Check if this direction has limited escape routes
			blockedSides := 0
			for _, perpDir := range AllDirections {
				if perpDir.DX*dir.DX+perpDir.DY*dir.DY == 0 { // Perpendicular
					checkPos := Position{X: next.X + perpDir.DX, Y: next.Y + perpDir.DY}
					checkPos = agent.Board.TorusCheck(checkPos)
					if agent.Board.GetCellState(checkPos) != EMPTY {
						blockedSides++
					}
				}
			}

			if blockedSides >= 2 {
				barriers++
			}
		}
	}

	return barriers * 30 // Penalty per near-barrier
}

// Count available space with BFS (limited search)
func countAvailableSpace(agent *Agent) int {
	if !agent.Alive {
		return 0
	}

	head := agent.GetHead()
	visited := make(map[Position]bool)
	queue := []Position{head}
	visited[head] = true
	count := 0

	for len(queue) > 0 && count < SPACE_COUNT_LIMIT {
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

// Flood fill to find reachable cells (used in endgame)
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

// Articulation point detection (cut vertices)
type ArticulationContext struct {
	visited map[Position]bool
	disc    map[Position]int
	low     map[Position]int
	parent  map[Position]Position
	ap      map[Position]bool
	time    int
}

func findArticulationPoints(board *GameBoard, agent *Agent) []Position {
	ctx := ArticulationContext{
		visited: make(map[Position]bool),
		disc:    make(map[Position]int),
		low:     make(map[Position]int),
		parent:  make(map[Position]Position),
		ap:      make(map[Position]bool),
		time:    0,
	}

	head := agent.GetHead()
	articulationDFS(head, board, &ctx)

	points := make([]Position, 0)
	for pos, isAP := range ctx.ap {
		if isAP {
			points = append(points, pos)
		}
	}

	return points
}

func articulationDFS(pos Position, board *GameBoard, ctx *ArticulationContext) {
	children := 0
	ctx.visited[pos] = true
	ctx.time++
	ctx.disc[pos] = ctx.time
	ctx.low[pos] = ctx.time

	for _, dir := range AllDirections {
		next := Position{X: pos.X + dir.DX, Y: pos.Y + dir.DY}
		next = board.TorusCheck(next)

		if board.GetCellState(next) != EMPTY {
			continue
		}

		if !ctx.visited[next] {
			children++
			ctx.parent[next] = pos
			articulationDFS(next, board, ctx)

			ctx.low[pos] = min(ctx.low[pos], ctx.low[next])

			// Check AP conditions
			if parent, hasParent := ctx.parent[pos]; !hasParent {
				// Root is AP if it has 2+ children
				if children > 1 {
					ctx.ap[pos] = true
				}
			} else if ctx.low[next] >= ctx.disc[pos] {
				ctx.ap[pos] = true
				_ = parent // Use parent to avoid unused variable warning
			}
		} else if next != ctx.parent[pos] {
			ctx.low[pos] = min(ctx.low[pos], ctx.disc[next])
		}
	}
}

// Evaluate articulation point control
func evaluateArticulationControl(agent *Agent, opponent *Agent) int {
	myAPs := findArticulationPoints(agent.Board, agent)

	if len(myAPs) == 0 {
		return 0
	}

	myHead := agent.GetHead()
	score := 0

	for _, ap := range myAPs {
		// Check if we're closer to this articulation point
		myDist := manhattanDistance(myHead, ap)
		oppDist := manhattanDistance(opponent.GetHead(), ap)

		if myDist < oppDist {
			score += 150 // Strong advantage if we control AP
		} else if myDist == oppDist {
			score += 50 // Contested
		}
	}

	return score
}
