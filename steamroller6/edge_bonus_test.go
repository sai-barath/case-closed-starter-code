package main

import (
	"testing"
)

func TestCalculateEdgeBonus_EmptyTerritory(t *testing.T) {
	board := NewGameBoard(18, 20)

	// Create a 3x3 territory for owner 1 in the center
	control := make([][]int, BOARD_HEIGHT)
	for y := 0; y < BOARD_HEIGHT; y++ {
		control[y] = make([]int, BOARD_WIDTH)
	}

	// Mark a 3x3 area as owned by player 1
	for y := 5; y < 8; y++ {
		for x := 5; x < 8; x++ {
			control[y][x] = 1
		}
	}

	bonus := calculateEdgeBonus(board, control, 1)

	// Each of the 9 cells has 4 empty neighbors (since board is all empty)
	// 9 cells * 4 neighbors = 36
	expected := 36

	if bonus != expected {
		t.Errorf("Expected bonus %d, got %d", expected, bonus)
	}
}

func TestCalculateEdgeBonus_WithTrails(t *testing.T) {
	board := NewGameBoard(18, 20)
	_ = NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)

	// Agent has trail at {5,5} and {6,5}
	// Create control map where player 1 owns a 3x3 area
	control := make([][]int, BOARD_HEIGHT)
	for y := 0; y < BOARD_HEIGHT; y++ {
		control[y] = make([]int, BOARD_WIDTH)
	}

	for y := 4; y < 7; y++ {
		for x := 4; x < 7; x++ {
			control[y][x] = 1
		}
	}

	bonus := calculateEdgeBonus(board, control, 1)

	// The 3x3 has 9 cells, but 2 are occupied by trail
	// Only empty cells in territory count
	// 7 empty cells, each can have 0-4 empty neighbors
	// This is complex to calculate exactly, but should be > 0

	if bonus <= 0 {
		t.Errorf("Expected positive bonus, got %d", bonus)
	}
}

func TestCalculateEdgeBonus_TorusWrapping(t *testing.T) {
	board := NewGameBoard(18, 20)

	control := make([][]int, BOARD_HEIGHT)
	for y := 0; y < BOARD_HEIGHT; y++ {
		control[y] = make([]int, BOARD_WIDTH)
	}

	// Mark corner cell at (0, 0) as owned by player 1
	control[0][0] = 1

	bonus := calculateEdgeBonus(board, control, 1)

	// Corner cell (0,0) has 4 neighbors due to torus:
	// UP: (0, 19), DOWN: (0, 1), LEFT: (17, 0), RIGHT: (1, 0)
	// All should be EMPTY, so bonus = 4
	expected := 4

	if bonus != expected {
		t.Errorf("Expected bonus %d (torus wrapping), got %d", expected, bonus)
	}
}

func TestCalculateEdgeBonus_WallHugging(t *testing.T) {
	board := NewGameBoard(18, 20)
	_ = NewAgent(1, Position{X: 5, Y: 5}, RIGHT, board)
	_ = NewAgent(2, Position{X: 10, Y: 5}, LEFT, board)

	// Create control map
	control := make([][]int, BOARD_HEIGHT)
	for y := 0; y < BOARD_HEIGHT; y++ {
		control[y] = make([]int, BOARD_WIDTH)
	}

	// Player 1 owns cells around their trail (but trail cells at x=5,6 are not empty)
	for x := 3; x < 8; x++ {
		control[5][x] = 1
	}

	// Player 2 owns cells around their trail (but trail cells at x=10,11 are not empty)
	for x := 8; x < 13; x++ {
		control[5][x] = 2
	}

	bonus1 := calculateEdgeBonus(board, control, 1)
	bonus2 := calculateEdgeBonus(board, control, 2)

	// Both should have positive bonuses (empty cells in their territory)
	if bonus1 <= 0 {
		t.Errorf("Player 1 should have positive bonus, got %d", bonus1)
	}

	if bonus2 <= 0 {
		t.Errorf("Player 2 should have positive bonus, got %d", bonus2)
	}

	// The exact values depend on trail positions
	t.Logf("Player 1 bonus: %d, Player 2 bonus: %d", bonus1, bonus2)
}

func TestCalculateEdgeBonus_NoTerritory(t *testing.T) {
	board := NewGameBoard(18, 20)

	control := make([][]int, BOARD_HEIGHT)
	for y := 0; y < BOARD_HEIGHT; y++ {
		control[y] = make([]int, BOARD_WIDTH)
	}

	// No territory owned by player 1
	bonus := calculateEdgeBonus(board, control, 1)

	if bonus != 0 {
		t.Errorf("Expected 0 bonus for no territory, got %d", bonus)
	}
}

func TestCalculateEdgeBonus_CompareOpenVsClosed(t *testing.T) {
	board := NewGameBoard(18, 20)

	control := make([][]int, BOARD_HEIGHT)
	for y := 0; y < BOARD_HEIGHT; y++ {
		control[y] = make([]int, BOARD_WIDTH)
	}

	// Scenario 1: Open 5x1 corridor
	for x := 5; x < 10; x++ {
		control[5][x] = 1
	}

	bonus1 := calculateEdgeBonus(board, control, 1)

	// Scenario 2: Open 3x3 square
	control2 := make([][]int, BOARD_HEIGHT)
	for y := 0; y < BOARD_HEIGHT; y++ {
		control2[y] = make([]int, BOARD_WIDTH)
	}

	for y := 5; y < 8; y++ {
		for x := 5; x < 8; x++ {
			control2[y][x] = 1
		}
	}

	bonus2 := calculateEdgeBonus(board, control2, 1)

	// 3x3 square should have higher bonus (more internal edges)
	// than a 5x1 corridor (mostly end-to-end)
	t.Logf("5x1 corridor bonus: %d, 3x3 square bonus: %d", bonus1, bonus2)

	if bonus2 <= bonus1 {
		t.Errorf("3x3 square should have more open edges than 5x1 corridor. Got corridor=%d, square=%d", bonus1, bonus2)
	}
}
