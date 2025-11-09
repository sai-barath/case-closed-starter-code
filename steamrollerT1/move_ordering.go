package main

import (
	"math"
)

// ============================================================================
// Move Ordering: Hash moves, killer moves, history heuristic, LMR
// ============================================================================

type MoveOrderingContext struct {
	killerMoves   map[int][2]Direction
	historyTable  map[Direction]int
	bestMoveCache map[int]Move
}

func NewMoveOrderingContext() *MoveOrderingContext {
	return &MoveOrderingContext{
		killerMoves:   make(map[int][2]Direction),
		historyTable:  make(map[Direction]int),
		bestMoveCache: make(map[int]Move),
	}
}

// Update killer moves for a given depth
func (ctx *MoveOrderingContext) updateKillerMove(depth int, move Direction) {
	killers := ctx.killerMoves[depth]
	if killers[0] != move {
		killers[1] = killers[0]
		killers[0] = move
		ctx.killerMoves[depth] = killers
	}
}

// Update history heuristic with gravity to prevent inflation
func (ctx *MoveOrderingContext) updateHistory(move Direction, depth int, isCutoff bool) {
	bonus := depth * depth
	if !isCutoff {
		bonus = -bonus / 2
	}

	// History with gravity: prevents unbounded growth
	current := ctx.historyTable[move]
	ctx.historyTable[move] = bonus - current*abs(bonus)/MAX_HISTORY_SCORE
}

// Order moves at root with all available heuristics
func (ctx *MoveOrderingContext) orderMovesAtRoot(moves []Direction, snapshot GameStateSnapshot, depth int) []Direction {
	if len(moves) == 0 {
		return moves
	}

	scored := make([]scoredMove, 0, len(moves))

	for _, dir := range moves {
		moveScore := 0

		// 1. Hash move (from previous iteration)
		if prevBest, exists := ctx.bestMoveCache[depth-1]; exists && prevBest.direction == dir {
			moveScore += HASH_MOVE_BONUS
		}

		// 2. Killer moves
		killers := ctx.killerMoves[depth]
		if killers[0] == dir {
			moveScore += KILLER_MOVE_1_BONUS
		} else if killers[1] == dir {
			moveScore += KILLER_MOVE_2_BONUS
		}

		// 3. History heuristic
		moveScore += ctx.historyTable[dir]

		scored = append(scored, scoredMove{dir: dir, score: moveScore})
	}

	// 4. Static evaluation
	scored = orderMovesStatic(scored, snapshot)

	// Sort by combined score
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

// Order moves with static evaluation
func orderMovesStatic(scored []scoredMove, snapshot GameStateSnapshot) []scoredMove {
	myHead := snapshot.myAgent.GetHead()
	centerX, centerY := BOARD_WIDTH/2, BOARD_HEIGHT/2

	phase := detectGamePhase(snapshot)

	for i := range scored {
		dir := scored[i].dir
		nextPos := Position{X: myHead.X + dir.DX, Y: myHead.Y + dir.DY}
		nextPos = snapshot.myAgent.Board.TorusCheck(nextPos)

		// Center control (more important in opening)
		centerDist := manhattanDistanceRaw(nextPos.X, nextPos.Y, centerX, centerY)
		if phase == Opening {
			scored[i].score -= centerDist * 8
		} else {
			scored[i].score -= centerDist * 3
		}

		// Freedom (count free neighbors)
		freeNeighbors := 0
		for _, d := range AllDirections {
			neighbor := Position{X: nextPos.X + d.DX, Y: nextPos.Y + d.DY}
			neighbor = snapshot.myAgent.Board.TorusCheck(neighbor)
			if snapshot.myAgent.Board.GetCellState(neighbor) == EMPTY {
				freeNeighbors++
			}
		}
		scored[i].score += freeNeighbors * 40

		// Local space count
		localSpace := countLocalSpace(nextPos, snapshot.myAgent, 5)
		scored[i].score += localSpace * 2
	}

	return scored
}

// Simple move ordering (for non-root nodes)
func orderMoves(moves []Direction, snapshot GameStateSnapshot) []Direction {
	if len(moves) <= 1 {
		return moves
	}

	scored := make([]scoredMove, 0, len(moves))
	myHead := snapshot.myAgent.GetHead()
	centerX, centerY := BOARD_WIDTH/2, BOARD_HEIGHT/2

	for _, dir := range moves {
		nextPos := Position{X: myHead.X + dir.DX, Y: myHead.Y + dir.DY}
		nextPos = snapshot.myAgent.Board.TorusCheck(nextPos)

		moveScore := 0

		// Center control
		centerDist := manhattanDistanceRaw(nextPos.X, nextPos.Y, centerX, centerY)
		moveScore -= centerDist * 5

		// Free neighbors
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

	// Bubble sort (fine for 4 moves max)
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

// Count local space within radius (used for move ordering)
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

// Determine if Late Move Reduction should apply
func shouldApplyLMR(moveIndex int, depth int, isCapture bool) bool {
	// Don't reduce tactical moves or shallow searches
	if isCapture || depth < 3 || moveIndex < LMR_FULL_DEPTH_MOVES {
		return false
	}
	return true
}

// Calculate LMR depth reduction
func getLMRReduction(moveIndex int, depth int) int {
	if moveIndex < LMR_FULL_DEPTH_MOVES {
		return 0
	}

	// More aggressive reduction for later moves
	reduction := LMR_DEPTH_REDUCTION
	if moveIndex >= 6 {
		reduction = int(math.Min(float64(depth-1), float64(LMR_DEPTH_REDUCTION+1)))
	}

	return reduction
}
