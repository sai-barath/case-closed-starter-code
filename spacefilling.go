package main

import (
	"math"
)

func countEdges(board *GameBoard, pos Position) int {
	edges := 0
	for _, dir := range AllDirections {
		neighbor := Position{
			X: pos.X + dir.DX,
			Y: pos.Y + dir.DY,
		}
		neighbor = board.TorusCheck(neighbor)

		if board.GetCellState(neighbor) == EMPTY {
			edges++
		}
	}
	return edges
}

func countWalls(board *GameBoard, pos Position) int {
	return 4 - countEdges(board, pos)
}

type ConnectedComponents struct {
	Board      *GameBoard
	Components [][]int
	NextID     int
	parent     []int
}

func NewConnectedComponents(board *GameBoard) *ConnectedComponents {
	components := make([][]int, board.Height)
	for y := 0; y < board.Height; y++ {
		components[y] = make([]int, board.Width)
		for x := 0; x < board.Width; x++ {
			components[y][x] = -1
		}
	}

	return &ConnectedComponents{
		Board:      board,
		Components: components,
		NextID:     0,
		parent:     make([]int, 0),
	}
}

func (cc *ConnectedComponents) find(x int) int {
	if cc.parent[x] == x {
		return x
	}
	cc.parent[x] = cc.find(cc.parent[x])
	return cc.parent[x]
}

func (cc *ConnectedComponents) union(x, y int) {
	rootX := cc.find(x)
	rootY := cc.find(y)
	if rootX != rootY {
		if rootX < rootY {
			cc.parent[rootY] = rootX
		} else {
			cc.parent[rootX] = rootY
		}
	}
}

func (cc *ConnectedComponents) Calculate() {
	cc.NextID = 0
	cc.parent = make([]int, cc.Board.Height*cc.Board.Width)

	for y := 0; y < cc.Board.Height; y++ {
		for x := 0; x < cc.Board.Width; x++ {
			cc.Components[y][x] = -1
		}
	}

	for i := range cc.parent {
		cc.parent[i] = i
	}

	for y := 0; y < cc.Board.Height; y++ {
		for x := 0; x < cc.Board.Width; x++ {
			pos := Position{X: x, Y: y}
			if cc.Board.GetCellState(pos) != EMPTY {
				continue
			}

			currentID := y*cc.Board.Width + x

			up := Position{X: x, Y: y - 1}
			up = cc.Board.TorusCheck(up)
			if cc.Board.GetCellState(up) == EMPTY {
				upID := up.Y*cc.Board.Width + up.X
				cc.union(currentID, upID)
			}

			left := Position{X: x - 1, Y: y}
			left = cc.Board.TorusCheck(left)
			if cc.Board.GetCellState(left) == EMPTY {
				leftID := left.Y*cc.Board.Width + left.X
				cc.union(currentID, leftID)
			}
		}
	}

	componentMap := make(map[int]int)
	nextComponentID := 0

	for y := 0; y < cc.Board.Height; y++ {
		for x := 0; x < cc.Board.Width; x++ {
			pos := Position{X: x, Y: y}
			if cc.Board.GetCellState(pos) != EMPTY {
				cc.Components[y][x] = -1
				continue
			}

			idx := y*cc.Board.Width + x
			root := cc.find(idx)

			if compID, exists := componentMap[root]; exists {
				cc.Components[y][x] = compID
			} else {
				componentMap[root] = nextComponentID
				cc.Components[y][x] = nextComponentID
				nextComponentID++
			}
		}
	}

	cc.NextID = nextComponentID
}

func (cc *ConnectedComponents) GetComponentID(pos Position) int {
	normalized := cc.Board.TorusCheck(pos)
	return cc.Components[normalized.Y][normalized.X]
}

func (cc *ConnectedComponents) GetComponentSize(componentID int) int {
	if componentID < 0 {
		return 0
	}

	count := 0
	for y := 0; y < cc.Board.Height; y++ {
		for x := 0; x < cc.Board.Width; x++ {
			if cc.Components[y][x] == componentID {
				count++
			}
		}
	}
	return count
}

func (cc *ConnectedComponents) AreConnected(pos1, pos2 Position) bool {
	comp1 := cc.GetComponentID(pos1)
	comp2 := cc.GetComponentID(pos2)
	return comp1 >= 0 && comp1 == comp2
}

type ArticulationPointFinder struct {
	board   *GameBoard
	visited map[Position]bool
	disc    map[Position]int
	low     map[Position]int
	parent  map[Position]Position
	ap      map[Position]bool
	time    int
}

func NewArticulationPointFinder(board *GameBoard) *ArticulationPointFinder {
	return &ArticulationPointFinder{
		board:   board,
		visited: make(map[Position]bool),
		disc:    make(map[Position]int),
		low:     make(map[Position]int),
		parent:  make(map[Position]Position),
		ap:      make(map[Position]bool),
		time:    0,
	}
}

func (apf *ArticulationPointFinder) FindArticulationPoints() map[Position]bool {
	apf.visited = make(map[Position]bool)
	apf.disc = make(map[Position]int)
	apf.low = make(map[Position]int)
	apf.parent = make(map[Position]Position)
	apf.ap = make(map[Position]bool)
	apf.time = 0

	for y := 0; y < apf.board.Height; y++ {
		for x := 0; x < apf.board.Width; x++ {
			pos := Position{X: x, Y: y}
			if apf.board.GetCellState(pos) == EMPTY && !apf.visited[pos] {
				apf.dfs(pos, Position{X: -1, Y: -1})
			}
		}
	}

	return apf.ap
}

func (apf *ArticulationPointFinder) dfs(u Position, p Position) {
	children := 0
	apf.visited[u] = true
	apf.time++
	apf.disc[u] = apf.time
	apf.low[u] = apf.time

	for _, dir := range AllDirections {
		v := Position{
			X: u.X + dir.DX,
			Y: u.Y + dir.DY,
		}
		v = apf.board.TorusCheck(v)

		if apf.board.GetCellState(v) != EMPTY {
			continue
		}

		if !apf.visited[v] {
			children++
			apf.parent[v] = u
			apf.dfs(v, u)

			apf.low[u] = min(apf.low[u], apf.low[v])

			if p.X == -1 && p.Y == -1 && children > 1 {
				apf.ap[u] = true
			}

			if p.X != -1 || p.Y != -1 {
				if apf.low[v] >= apf.disc[u] {
					apf.ap[u] = true
				}
			}
		} else if v.X != p.X || v.Y != p.Y {
			apf.low[u] = min(apf.low[u], apf.disc[v])
		}
	}
}

func IsArticulationPoint(board *GameBoard, pos Position) bool {
	if board.GetCellState(pos) != EMPTY {
		return false
	}

	apf := NewArticulationPointFinder(board)
	aps := apf.FindArticulationPoints()
	return aps[pos]
}

func EvaluateSpaceFilling(board *GameBoard, pos Position, articulationPoints map[Position]bool) int {
	if board.GetCellState(pos) != EMPTY {
		return -10000
	}

	score := 0

	edgeCount := countEdges(board, pos)
	wallCount := countWalls(board, pos)

	score += wallCount * 100
	score -= edgeCount * 50

	if articulationPoints[pos] {
		score -= 500
	}

	return score
}

func GetBestSpaceFillingMove(agent *Agent, articulationPoints map[Position]bool, components *ConnectedComponents) Direction {
	validMoves := agent.GetValidMoves()
	if len(validMoves) == 0 {
		return RIGHT
	}

	bestMove := validMoves[0]
	bestScore := math.MinInt32

	head := agent.GetHead()

	for _, dir := range validMoves {
		nextPos := Position{
			X: head.X + dir.DX,
			Y: head.Y + dir.DY,
		}
		nextPos = agent.Board.TorusCheck(nextPos)

		score := EvaluateSpaceFilling(agent.Board, nextPos, articulationPoints)

		if articulationPoints[nextPos] && components != nil {
			for _, checkDir := range AllDirections {
				beyondPos := Position{
					X: nextPos.X + checkDir.DX,
					Y: nextPos.Y + checkDir.DY,
				}
				beyondPos = agent.Board.TorusCheck(beyondPos)

				if agent.Board.GetCellState(beyondPos) == EMPTY {
					compID := components.GetComponentID(beyondPos)
					compSize := components.GetComponentSize(compID)
					score += compSize * 10
				}
			}
		}

		if score > bestScore {
			bestScore = score
			bestMove = dir
		}
	}

	return bestMove
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

type ChamberTree struct {
	board              *GameBoard
	articulationPoints map[Position]bool
}

func NewChamberTree(board *GameBoard) *ChamberTree {
	apf := NewArticulationPointFinder(board)
	aps := apf.FindArticulationPoints()
	return &ChamberTree{
		board:              board,
		articulationPoints: aps,
	}
}

func (ct *ChamberTree) EvaluateChamberTree(myHead, oppHead Position) int {
	myScore := ct.evaluateFromPosition(myHead, oppHead, true)
	oppScore := ct.evaluateFromPosition(oppHead, myHead, false)
	return myScore - oppScore
}

func (ct *ChamberTree) evaluateFromPosition(head, opponentHead Position, isMe bool) int {
	visited := make(map[Position]bool)
	currentChamberSize := ct.exploreChamber(head, visited)

	battlefrontChambers := make(map[Position]int)
	ct.findBattlefrontChambers(head, opponentHead, visited, battlefrontChambers)

	bestBattlefrontValue := 0
	for cutVertex, stepsToEnter := range battlefrontChambers {
		battlefrontVisited := make(map[Position]bool)
		battlefrontSize := ct.exploreChamber(cutVertex, battlefrontVisited)

		value := battlefrontSize - stepsToEnter*2
		if value > bestBattlefrontValue {
			bestBattlefrontValue = value
		}
	}

	if currentChamberSize > bestBattlefrontValue {
		return currentChamberSize
	}
	return bestBattlefrontValue
}

func (ct *ChamberTree) exploreChamber(start Position, visited map[Position]bool) int {
	if ct.board.GetCellState(start) != EMPTY {
		return 0
	}

	// Only explore the space reachable without crossing articulation point boundaries
	// We count cells that are not articulation points
	stack := []Position{start}
	visited[start] = true
	count := 0

	// Count start position if it's not an articulation point
	if !ct.articulationPoints[start] {
		count++
	}

	for len(stack) > 0 {
		current := stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		for _, dir := range AllDirections {
			next := Position{
				X: current.X + dir.DX,
				Y: current.Y + dir.DY,
			}
			next = ct.board.TorusCheck(next)

			if visited[next] || ct.board.GetCellState(next) != EMPTY {
				continue
			}

			visited[next] = true

			// Count non-articulation points and continue exploring
			if !ct.articulationPoints[next] {
				count++
				stack = append(stack, next)
			}
			// Articulation points are marked visited but not explored or counted
		}
	}

	return count
}

func (ct *ChamberTree) findBattlefrontChambers(myHead, oppHead Position, myVisited map[Position]bool, battlefronts map[Position]int) {
	for cutVertex := range ct.articulationPoints {
		if myVisited[cutVertex] {
			continue
		}

		distance := ct.manhattanDistance(myHead, cutVertex)

		oppVisited := make(map[Position]bool)
		ct.exploreChamber(oppHead, oppVisited)

		if oppVisited[cutVertex] {
			battlefronts[cutVertex] = distance
		}
	}
}

func (ct *ChamberTree) manhattanDistance(a, b Position) int {
	dx := a.X - b.X
	dy := a.Y - b.Y

	if dx < 0 {
		dx = -dx
	}
	if dy < 0 {
		dy = -dy
	}

	if dx > ct.board.Width/2 {
		dx = ct.board.Width - dx
	}
	if dy > ct.board.Height/2 {
		dy = ct.board.Height - dy
	}

	return dx + dy
}
