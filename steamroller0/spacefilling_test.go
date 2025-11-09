package main

import (
	"testing"
)

func TestCountEdgesCenter(t *testing.T) {
	board := NewGameBoard(18, 20)
	board.Grid[10][9] = AGENT

	pos := Position{X: 9, Y: 10}
	edges := countEdges(board, pos)

	if edges != 3 {
		t.Errorf("Expected 3 edges, got %d", edges)
	}

	walls := countWalls(board, pos)
	if walls != 1 {
		t.Errorf("Expected 1 wall, got %d", walls)
	}
}

func TestCountEdgesCornerWithTorusWraparound(t *testing.T) {
	board := NewGameBoard(18, 20)

	board.Grid[0][1] = AGENT
	board.Grid[1][0] = AGENT

	pos := Position{X: 0, Y: 0}
	edges := countEdges(board, pos)

	if edges != 2 {
		t.Errorf("Expected 2 edges at corner (0,0) with torus wraparound, got %d", edges)
	}

	walls := countWalls(board, pos)
	if walls != 2 {
		t.Errorf("Expected 2 walls at corner (0,0), got %d", walls)
	}
}

func TestCountEdgesBottomRightCornerTorus(t *testing.T) {
	board := NewGameBoard(18, 20)

	board.Grid[board.Height-1][0] = AGENT
	board.Grid[0][board.Width-1] = AGENT

	pos := Position{X: board.Width - 1, Y: board.Height - 1}
	edges := countEdges(board, pos)

	if edges != 2 {
		t.Errorf("Expected 2 edges at bottom-right corner with torus wraparound, got %d", edges)
	}
}

func TestCountEdgesSurroundedByWalls(t *testing.T) {
	board := NewGameBoard(18, 20)

	center := Position{X: 10, Y: 10}
	for _, dir := range AllDirections {
		neighbor := Position{
			X: center.X + dir.DX,
			Y: center.Y + dir.DY,
		}
		neighbor = board.TorusCheck(neighbor)
		board.Grid[neighbor.Y][neighbor.X] = AGENT
	}

	edges := countEdges(board, center)
	if edges != 0 {
		t.Errorf("Expected 0 edges when surrounded by walls, got %d", edges)
	}

	walls := countWalls(board, center)
	if walls != 4 {
		t.Errorf("Expected 4 walls when surrounded, got %d", walls)
	}
}

func TestCountEdgesAllEmpty(t *testing.T) {
	board := NewGameBoard(18, 20)
	pos := Position{X: 10, Y: 10}

	edges := countEdges(board, pos)
	if edges != 4 {
		t.Errorf("Expected 4 edges on empty board, got %d", edges)
	}

	walls := countWalls(board, pos)
	if walls != 0 {
		t.Errorf("Expected 0 walls on empty board, got %d", walls)
	}
}

func TestConnectedComponentsEmptyBoard(t *testing.T) {
	board := NewGameBoard(18, 20)
	cc := NewConnectedComponents(board)
	cc.Calculate()

	if cc.NextID != 1 {
		t.Errorf("Expected 1 component on empty board, got %d", cc.NextID)
	}

	pos1 := Position{X: 0, Y: 0}
	pos2 := Position{X: board.Width - 1, Y: board.Height - 1}

	if !cc.AreConnected(pos1, pos2) {
		t.Errorf("Expected positions to be connected on empty board")
	}
}

func TestConnectedComponentsWithWall(t *testing.T) {
	board := NewGameBoard(18, 20)

	for x := 0; x < board.Width; x++ {
		board.Grid[board.Height/2][x] = AGENT
	}

	cc := NewConnectedComponents(board)
	cc.Calculate()

	if cc.NextID != 2 {
		t.Errorf("Expected 2 components separated by horizontal wall, got %d", cc.NextID)
	}

	topPos := Position{X: 5, Y: board.Height/2 - 1}
	bottomPos := Position{X: 5, Y: board.Height/2 + 1}

	if cc.AreConnected(topPos, bottomPos) {
		t.Errorf("Expected positions on opposite sides of wall to be disconnected")
	}
}

func TestConnectedComponentsTorusWraparound(t *testing.T) {
	board := NewGameBoard(18, 20)

	for x := 1; x < board.Width-1; x++ {
		board.Grid[board.Height/2][x] = AGENT
	}

	cc := NewConnectedComponents(board)
	cc.Calculate()

	if cc.NextID != 1 {
		t.Errorf("Expected 1 component (wall doesn't span edges, torus connects), got %d", cc.NextID)
	}

	topPos := Position{X: 0, Y: 0}
	bottomPos := Position{X: 0, Y: board.Height - 1}

	if !cc.AreConnected(topPos, bottomPos) {
		t.Errorf("Expected torus wraparound to connect top and bottom edges")
	}
}

func TestConnectedComponentsIsolatedRegion(t *testing.T) {
	board := NewGameBoard(18, 20)

	for x := 5; x <= 10; x++ {
		board.Grid[5][x] = AGENT
		board.Grid[10][x] = AGENT
	}
	for y := 5; y <= 10; y++ {
		board.Grid[y][5] = AGENT
		board.Grid[y][10] = AGENT
	}

	cc := NewConnectedComponents(board)
	cc.Calculate()

	insidePos := Position{X: 7, Y: 7}
	outsidePos := Position{X: 0, Y: 0}

	if cc.AreConnected(insidePos, outsidePos) {
		t.Errorf("Expected isolated region to be disconnected from outside")
	}

	insideComp := cc.GetComponentID(insidePos)
	compSize := cc.GetComponentSize(insideComp)
	expectedSize := 4 * 4
	if compSize != expectedSize {
		t.Errorf("Expected isolated region size %d, got %d", expectedSize, compSize)
	}
}

func TestConnectedComponentsMultipleSmallRegions(t *testing.T) {
	board := NewGameBoard(18, 20)

	for y := 0; y < board.Height; y++ {
		for x := 0; x < board.Width; x++ {
			if (x+y)%2 == 0 {
				board.Grid[y][x] = AGENT
			}
		}
	}

	cc := NewConnectedComponents(board)
	cc.Calculate()

	if cc.NextID < 2 {
		t.Errorf("Expected multiple components with checkerboard pattern, got %d", cc.NextID)
	}
}

func TestArticulationPointSimplePath(t *testing.T) {
	board := NewGameBoard(18, 20)

	for y := 0; y < board.Height; y++ {
		for x := 0; x < board.Width; x++ {
			board.Grid[y][x] = AGENT
		}
	}

	for x := 5; x <= 10; x++ {
		board.Grid[10][x] = EMPTY
	}

	apf := NewArticulationPointFinder(board)
	aps := apf.FindArticulationPoints()

	middlePos := Position{X: 7, Y: 10}
	if !aps[middlePos] {
		t.Errorf("Expected middle of narrow corridor to be articulation point")
	}

	endPos := Position{X: 5, Y: 10}
	if aps[endPos] {
		t.Errorf("Expected end of corridor not to be articulation point")
	}
}

func TestArticulationPointTIntersection(t *testing.T) {
	board := NewGameBoard(18, 20)

	for y := 0; y < board.Height; y++ {
		for x := 0; x < board.Width; x++ {
			board.Grid[y][x] = AGENT
		}
	}

	for x := 5; x <= 15; x++ {
		board.Grid[10][x] = EMPTY
	}
	for y := 5; y < 10; y++ {
		board.Grid[y][10] = EMPTY
	}

	apf := NewArticulationPointFinder(board)
	aps := apf.FindArticulationPoints()

	junctionPos := Position{X: 10, Y: 10}
	if !aps[junctionPos] {
		t.Errorf("Expected T-junction center to be articulation point")
	}
}

func TestArticulationPointNoAPInOpenSpace(t *testing.T) {
	board := NewGameBoard(18, 20)

	apf := NewArticulationPointFinder(board)
	aps := apf.FindArticulationPoints()

	hasAP := false
	for y := 5; y < 15; y++ {
		for x := 5; x < 15; x++ {
			pos := Position{X: x, Y: y}
			if aps[pos] {
				hasAP = true
				break
			}
		}
	}

	if hasAP {
		t.Errorf("Expected no articulation points in large open space due to torus wraparound")
	}
}

func TestArticulationPointTorusWraparound(t *testing.T) {
	board := NewGameBoard(18, 20)

	for y := 0; y < board.Height; y++ {
		for x := 0; x < board.Width; x++ {
			board.Grid[y][x] = AGENT
		}
	}

	board.Grid[0][0] = EMPTY
	board.Grid[0][1] = EMPTY
	board.Grid[0][board.Width-1] = EMPTY

	apf := NewArticulationPointFinder(board)
	aps := apf.FindArticulationPoints()

	middlePos := Position{X: 0, Y: 0}
	if !aps[middlePos] {
		t.Errorf("Expected middle cell connecting wraparound edges to be articulation point")
	}
}

func TestEvaluateSpaceFillingTightSpace(t *testing.T) {
	board := NewGameBoard(18, 20)

	pos := Position{X: 10, Y: 10}

	for _, dir := range []Direction{UP, LEFT, DOWN} {
		neighbor := Position{
			X: pos.X + dir.DX,
			Y: pos.Y + dir.DY,
		}
		neighbor = board.TorusCheck(neighbor)
		board.Grid[neighbor.Y][neighbor.X] = AGENT
	}

	aps := make(map[Position]bool)
	score := EvaluateSpaceFilling(board, pos, aps)

	if score <= 0 {
		t.Errorf("Expected positive score for tight space (3 walls), got %d", score)
	}

	wallCount := countWalls(board, pos)
	if wallCount != 3 {
		t.Errorf("Expected 3 walls, got %d", wallCount)
	}
}

func TestEvaluateSpaceFillingOpenSpace(t *testing.T) {
	board := NewGameBoard(18, 20)
	pos := Position{X: 10, Y: 10}

	aps := make(map[Position]bool)
	score := EvaluateSpaceFilling(board, pos, aps)

	if score >= 0 {
		t.Errorf("Expected negative score for open space (4 edges), got %d", score)
	}
}

func TestEvaluateSpaceFillingArticulationPointPenalty(t *testing.T) {
	board := NewGameBoard(18, 20)
	pos := Position{X: 10, Y: 10}

	aps := make(map[Position]bool)
	scoreWithoutAP := EvaluateSpaceFilling(board, pos, aps)

	aps[pos] = true
	scoreWithAP := EvaluateSpaceFilling(board, pos, aps)

	if scoreWithAP >= scoreWithoutAP {
		t.Errorf("Expected articulation point to decrease score, got %d (without) vs %d (with)", scoreWithoutAP, scoreWithAP)
	}

	penalty := scoreWithoutAP - scoreWithAP
	if penalty != 500 {
		t.Errorf("Expected articulation point penalty of 500, got %d", penalty)
	}
}

func TestEvaluateSpaceFillingOccupiedCell(t *testing.T) {
	board := NewGameBoard(18, 20)
	pos := Position{X: 10, Y: 10}
	board.Grid[pos.Y][pos.X] = AGENT

	aps := make(map[Position]bool)
	score := EvaluateSpaceFilling(board, pos, aps)

	if score != -10000 {
		t.Errorf("Expected -10000 for occupied cell, got %d", score)
	}
}

func TestGetBestSpaceFillingMovePrefersTightSpace(t *testing.T) {
	board := NewGameBoard(18, 20)

	agent1Trail := []Position{{X: 10, Y: 10}}
	agent := &Agent{
		AgentID:         1,
		Trail:           agent1Trail,
		TrailSet:        map[Position]bool{{X: 10, Y: 10}: true},
		Direction:       RIGHT,
		Board:           board,
		Alive:           true,
		Length:          1,
		BoostsRemaining: 3,
	}

	board.Grid[11][10] = AGENT
	board.Grid[11][11] = AGENT
	board.Grid[10][11] = AGENT

	aps := make(map[Position]bool)
	move := GetBestSpaceFillingMove(agent, aps, nil)

	if move != RIGHT {
		t.Errorf("Expected RIGHT (toward tight space), got %v", move)
	}
}
