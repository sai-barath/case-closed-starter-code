package main

import (
	"math"
)

// ============================================================================
// Endgame: Hamiltonian path heuristics, space-filling curves
// ============================================================================

// Hilbert curve index calculation (approximation for space-filling)
func hilbertIndex(x, y, order int) int {
	if order == 0 {
		return 0
	}

	hpow := 1 << (order - 1)

	var rx, ry int
	if x >= hpow {
		rx = 1
		x -= hpow
	}
	if y >= hpow {
		ry = 1
		y -= hpow
	}

	idx := 3 * (hpow * hpow) * ((rx * 2) + ry)

	if ry == 0 {
		if rx == 1 {
			x = hpow - 1 - x
			y = hpow - 1 - y
		}
		x, y = y, x
	}

	return idx + hilbertIndex(x, y, order-1)
}

// Generate Hilbert curve path through available space
func generateHilbertPath(agent *Agent) []Position {
	head := agent.GetHead()

	// Get all reachable cells
	reachable := getReachableCells(agent)

	if len(reachable) == 0 {
		return []Position{}
	}

	// Calculate Hilbert index for each cell
	order := 5 // 2^5 = 32, enough for our board

	type cellWithIndex struct {
		pos   Position
		index int
	}

	cells := make([]cellWithIndex, 0, len(reachable))
	for pos := range reachable {
		idx := hilbertIndex(pos.X, pos.Y, order)
		cells = append(cells, cellWithIndex{pos: pos, index: idx})
	}

	// Sort by Hilbert index
	for i := 0; i < len(cells)-1; i++ {
		for j := i + 1; j < len(cells); j++ {
			if cells[j].index < cells[i].index {
				cells[i], cells[j] = cells[j], cells[i]
			}
		}
	}

	// Find starting point closest to head
	startIdx := 0
	minDist := math.MaxInt32
	for i, cell := range cells {
		dist := manhattanDistance(head, cell.pos)
		if dist < minDist {
			minDist = dist
			startIdx = i
		}
	}

	// Reorder to start from closest point
	path := make([]Position, len(cells))
	for i := range cells {
		path[i] = cells[(startIdx+i)%len(cells)].pos
	}

	return path
}

// Greedy furthest-point path (fast and effective)
func greedyFurthestPoint(agent *Agent) []Position {
	head := agent.GetHead()
	reachable := getReachableCells(agent)

	if len(reachable) == 0 {
		return []Position{}
	}

	path := []Position{head}
	visited := make(map[Position]bool)
	visited[head] = true
	current := head

	for len(visited) < len(reachable) {
		// Find furthest unvisited reachable cell
		furthest := Position{}
		maxDist := -1

		for pos := range reachable {
			if !visited[pos] {
				dist := manhattanDistance(current, pos)
				if dist > maxDist {
					maxDist = dist
					furthest = pos
				}
			}
		}

		if maxDist == -1 {
			break // No more reachable cells
		}

		path = append(path, furthest)
		visited[furthest] = true
		current = furthest
	}

	return path
}

// Get all cells reachable from agent's head
func getReachableCells(agent *Agent) map[Position]bool {
	head := agent.GetHead()
	reachable := make(map[Position]bool)
	queue := []Position{head}
	visited := make(map[Position]bool)
	visited[head] = true

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		if agent.Board.GetCellState(current) == EMPTY {
			reachable[current] = true
		}

		for _, dir := range AllDirections {
			next := Position{X: current.X + dir.DX, Y: current.Y + dir.DY}
			next = agent.Board.TorusCheck(next)

			if !visited[next] && agent.Board.GetCellState(next) == EMPTY {
				visited[next] = true
				reachable[next] = true
				queue = append(queue, next)
			}
		}
	}

	return reachable
}

// Calculate longest path score using greedy heuristic
func longestPathScore(agent *Agent) int {
	path := greedyFurthestPoint(agent)
	return len(path) * 10 // Weight path length
}

// Evaluate endgame position with space-filling consideration
func evaluateEndgame(myAgent *Agent, opponentAgent *Agent, mySpace int, opponentSpace int) int {
	spaceDiff := mySpace - opponentSpace

	// Strong space advantage
	if spaceDiff > ENDGAME_SPACE_DIFF {
		return WIN_SCORE / 2
	} else if spaceDiff < -ENDGAME_SPACE_DIFF {
		return LOSE_SCORE / 2
	}

	// Calculate actual reachable space
	myReachable := floodFillReachable(myAgent)
	oppReachable := floodFillReachable(opponentAgent)

	reachableDiff := myReachable - oppReachable

	if reachableDiff > 10 {
		return WIN_SCORE / 3
	} else if reachableDiff < -10 {
		return LOSE_SCORE / 3
	}

	// Use longest path heuristic
	myPathScore := longestPathScore(myAgent)
	oppPathScore := longestPathScore(opponentAgent)

	return (myPathScore-oppPathScore)*20 + reachableDiff*200
}

// Check if we should switch to endgame mode (early detection)
func shouldUseEndgameMode(myAgent *Agent, opponentAgent *Agent) bool {
	totalCells := BOARD_HEIGHT * BOARD_WIDTH
	occupiedCells := len(myAgent.Trail) + len(opponentAgent.Trail)
	freeSpace := float64(totalCells-occupiedCells) / float64(totalCells)

	// Trigger if < 40% free space
	if freeSpace < PARTITION_THRESHOLD {
		return true
	}

	// Or if we have significant space advantage
	mySpace := countAvailableSpace(myAgent)
	oppSpace := countAvailableSpace(opponentAgent)

	if mySpace*10 > oppSpace*13 { // mySpace > oppSpace*1.3
		return true
	}

	return false
}

// Articulation-aware pathing: prioritize moves through articulation points
func articulationAwarePath(agent *Agent, aps []Position) Direction {
	if len(aps) == 0 {
		return agent.Direction // Keep current direction
	}

	head := agent.GetHead()
	validMoves := agent.GetValidMoves()

	if len(validMoves) == 0 {
		return agent.Direction
	}

	// Find move that gets us closest to nearest AP
	bestMove := validMoves[0]
	minDist := math.MaxInt32

	for _, dir := range validMoves {
		nextPos := Position{X: head.X + dir.DX, Y: head.Y + dir.DY}
		nextPos = agent.Board.TorusCheck(nextPos)

		// Find closest AP from this position
		for _, ap := range aps {
			dist := manhattanDistance(nextPos, ap)
			if dist < minDist {
				minDist = dist
				bestMove = dir
			}
		}
	}

	return bestMove
}

// Space-filling curve direction suggestion
func spaceFillingSuggestion(agent *Agent) Direction {
	path := generateHilbertPath(agent)

	if len(path) < 2 {
		return agent.Direction
	}

	head := agent.GetHead()
	nextTarget := path[1] // First cell after head

	// Find direction toward next target
	dx := nextTarget.X - head.X
	dy := nextTarget.Y - head.Y

	// Handle torus wraparound
	if abs(dx) > BOARD_WIDTH/2 {
		if dx > 0 {
			dx = -1
		} else {
			dx = 1
		}
	}
	if abs(dy) > BOARD_HEIGHT/2 {
		if dy > 0 {
			dy = -1
		} else {
			dy = 1
		}
	}

	// Convert to direction
	if abs(dx) > abs(dy) {
		if dx > 0 {
			return RIGHT
		}
		return LEFT
	} else {
		if dy > 0 {
			return DOWN
		}
		return UP
	}
}
