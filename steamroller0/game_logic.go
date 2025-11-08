package main

const (
	EMPTY = 0
	AGENT = 1
)

type Position struct {
	X int `json:"x"`
	Y int `json:"y"`
}

type Direction struct {
	DX int
	DY int
}

var (
	UP    = Direction{DX: 0, DY: -1}
	DOWN  = Direction{DX: 0, DY: 1}
	RIGHT = Direction{DX: 1, DY: 0}
	LEFT  = Direction{DX: -1, DY: 0}
	
	AllDirections = []Direction{UP, DOWN, LEFT, RIGHT}
)

type GameBoard struct {
	Height int
	Width  int
	Grid   [][]int
}

func NewGameBoard(height, width int) *GameBoard {
	grid := make([][]int, height)
	for i := range grid {
		grid[i] = make([]int, width)
	}
	return &GameBoard{
		Height: height,
		Width:  width,
		Grid:   grid,
	}
}

func (gb *GameBoard) Clone() *GameBoard {
	newGrid := make([][]int, gb.Height)
	for i := range gb.Grid {
		newGrid[i] = make([]int, gb.Width)
		copy(newGrid[i], gb.Grid[i])
	}
	return &GameBoard{
		Height: gb.Height,
		Width:  gb.Width,
		Grid:   newGrid,
	}
}

func (gb *GameBoard) TorusCheck(pos Position) Position {
	x := pos.X % gb.Width
	y := pos.Y % gb.Height
	if x < 0 {
		x += gb.Width
	}
	if y < 0 {
		y += gb.Height
	}
	return Position{X: x, Y: y}
}

func (gb *GameBoard) GetCellState(pos Position) int {
	normalized := gb.TorusCheck(pos)
	return gb.Grid[normalized.Y][normalized.X]
}

func (gb *GameBoard) SetCellState(pos Position, state int) {
	normalized := gb.TorusCheck(pos)
	gb.Grid[normalized.Y][normalized.X] = state
}

type Agent struct {
	AgentID         int
	Trail           []Position
	TrailSet        map[Position]bool
	Direction       Direction
	Board           *GameBoard
	Alive           bool
	Length          int
	BoostsRemaining int
}

func NewAgent(agentID int, startPos Position, startDir Direction, board *GameBoard) *Agent {
	second := Position{
		X: startPos.X + startDir.DX,
		Y: startPos.Y + startDir.DY,
	}
	
	trail := []Position{startPos, second}
	trailSet := make(map[Position]bool)
	trailSet[startPos] = true
	trailSet[second] = true
	
	agent := &Agent{
		AgentID:         agentID,
		Trail:           trail,
		TrailSet:        trailSet,
		Direction:       startDir,
		Board:           board,
		Alive:           true,
		Length:          2,
		BoostsRemaining: 3,
	}
	
	board.SetCellState(startPos, AGENT)
	board.SetCellState(second, AGENT)
	
	return agent
}

func (a *Agent) Clone(newBoard *GameBoard) *Agent {
	newTrail := make([]Position, len(a.Trail))
	copy(newTrail, a.Trail)
	
	newTrailSet := make(map[Position]bool, len(a.TrailSet))
	for k, v := range a.TrailSet {
		newTrailSet[k] = v
	}
	
	return &Agent{
		AgentID:         a.AgentID,
		Trail:           newTrail,
		TrailSet:        newTrailSet,
		Direction:       a.Direction,
		Board:           newBoard,
		Alive:           a.Alive,
		Length:          a.Length,
		BoostsRemaining: a.BoostsRemaining,
	}
}

func (a *Agent) IsHead(pos Position) bool {
	if len(a.Trail) == 0 {
		return false
	}
	head := a.Trail[len(a.Trail)-1]
	return pos.X == head.X && pos.Y == head.Y
}

func (a *Agent) ContainsPosition(pos Position) bool {
	return a.TrailSet[pos]
}

func (a *Agent) GetHead() Position {
	if len(a.Trail) == 0 {
		return Position{X: -1, Y: -1}
	}
	return a.Trail[len(a.Trail)-1]
}

func (a *Agent) GetValidMoves() []Direction {
	if !a.Alive {
		return nil
	}
	
	valid := make([]Direction, 0, 4)
	for _, dir := range AllDirections {
		if dir.DX == -a.Direction.DX && dir.DY == -a.Direction.DY {
			continue
		}
		valid = append(valid, dir)
	}
	return valid
}

type MoveState struct {
	AddedPositions   []Position
	OldDirection     Direction
	NewDirection     Direction
	BoostUsed        bool
	MyAliveChanged   bool
	OtherAliveChanged bool
	OldMyAlive       bool
	OldOtherAlive    bool
	LengthAdded      int
}

func (a *Agent) Move(direction Direction, otherAgent *Agent, useBoost bool) bool {
	if !a.Alive {
		return false
	}

	if useBoost && a.BoostsRemaining <= 0 {
		useBoost = false
	}

	numMoves := 1
	if useBoost {
		numMoves = 2
		a.BoostsRemaining--
	}

	for moveNum := 0; moveNum < numMoves; moveNum++ {
		if direction.DX == -a.Direction.DX && direction.DY == -a.Direction.DY {
			continue
		}

		head := a.Trail[len(a.Trail)-1]
		newHead := Position{
			X: head.X + direction.DX,
			Y: head.Y + direction.DY,
		}

		newHead = a.Board.TorusCheck(newHead)
		cellState := a.Board.GetCellState(newHead)
		a.Direction = direction

		if cellState == AGENT {
			if a.ContainsPosition(newHead) {
				a.Alive = false
				return false
			}

			if otherAgent != nil && otherAgent.Alive && otherAgent.ContainsPosition(newHead) {
				if otherAgent.IsHead(newHead) {
					a.Alive = false
					otherAgent.Alive = false
					return false
				} else {
					a.Alive = false
					return false
				}
			}
		}

		a.Trail = append(a.Trail, newHead)
		a.TrailSet[newHead] = true
		a.Length++
		a.Board.SetCellState(newHead, AGENT)
	}

	return true
}

func (a *Agent) UndoableMove(direction Direction, otherAgent *Agent, useBoost bool) (bool, MoveState) {
	state := MoveState{
		OldDirection: a.Direction,
		OldMyAlive:   a.Alive,
		BoostUsed:    false,
		LengthAdded:  0,
	}
	
	if otherAgent != nil {
		state.OldOtherAlive = otherAgent.Alive
	}

	if !a.Alive {
		return false, state
	}

	if useBoost && a.BoostsRemaining <= 0 {
		useBoost = false
	}

	numMoves := 1
	if useBoost {
		numMoves = 2
		a.BoostsRemaining--
		state.BoostUsed = true
	}

	for moveNum := 0; moveNum < numMoves; moveNum++ {
		if direction.DX == -a.Direction.DX && direction.DY == -a.Direction.DY {
			continue
		}

		head := a.Trail[len(a.Trail)-1]
		newHead := Position{
			X: head.X + direction.DX,
			Y: head.Y + direction.DY,
		}

		newHead = a.Board.TorusCheck(newHead)
		cellState := a.Board.GetCellState(newHead)
		state.NewDirection = direction
		a.Direction = direction

		if cellState == AGENT {
			if a.ContainsPosition(newHead) {
				a.Alive = false
				state.MyAliveChanged = true
				return false, state
			}

			if otherAgent != nil && otherAgent.Alive && otherAgent.ContainsPosition(newHead) {
				if otherAgent.IsHead(newHead) {
					a.Alive = false
					otherAgent.Alive = false
					state.MyAliveChanged = true
					state.OtherAliveChanged = true
					return false, state
				} else {
					a.Alive = false
					state.MyAliveChanged = true
					return false, state
				}
			}
		}

		a.Trail = append(a.Trail, newHead)
		a.TrailSet[newHead] = true
		a.Length++
		state.LengthAdded++
		state.AddedPositions = append(state.AddedPositions, newHead)
		a.Board.SetCellState(newHead, AGENT)
	}

	return true, state
}

func (a *Agent) UndoMove(state MoveState, otherAgent *Agent) {
	for i := len(state.AddedPositions) - 1; i >= 0; i-- {
		pos := state.AddedPositions[i]
		if len(a.Trail) > 0 && a.Trail[len(a.Trail)-1].X == pos.X && a.Trail[len(a.Trail)-1].Y == pos.Y {
			a.Trail = a.Trail[:len(a.Trail)-1]
		}
		delete(a.TrailSet, pos)
		a.Board.SetCellState(pos, EMPTY)
	}
	
	a.Length -= state.LengthAdded
	a.Direction = state.OldDirection
	
	if state.BoostUsed {
		a.BoostsRemaining++
	}
	
	if state.MyAliveChanged {
		a.Alive = state.OldMyAlive
	}
	
	if state.OtherAliveChanged && otherAgent != nil {
		otherAgent.Alive = state.OldOtherAlive
	}
}
