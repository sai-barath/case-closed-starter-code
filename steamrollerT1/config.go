package main

import (
	"os"
	"time"
)

// ============================================================================
// Configuration and constants
// ============================================================================

var debugMode = os.Getenv("DEBUG") == "1"

const (
	SEARCH_TIME_LIMIT = 100 * time.Millisecond
	WIN_SCORE         = 10000
	LOSE_SCORE        = -10000
	DRAW_SCORE        = 0
	BOARD_HEIGHT      = 18
	BOARD_WIDTH       = 20
)

// Game phases based on free space percentage
type GamePhase int

const (
	Opening GamePhase = iota
	Midgame
	Endgame
)

// Phase detection thresholds
const (
	OPENING_THRESHOLD = 0.70 // > 70% free space
	MIDGAME_THRESHOLD = 0.45 // 45-70% free space
	// < 45% is endgame
)

// Phase-dependent heuristic weights
type HeuristicWeights struct {
	Territory         int
	Space             int
	TrailLength       int
	CenterControl     int
	Freedom           int
	Tightness         int
	CornerPenalty     int
	Blocking          int
	ForcingMoves      int
	TerritoryCutting  int
	SynergyMultiplier float64
}

var (
	OpeningWeights = HeuristicWeights{
		Territory:         20, // Increased from 15
		Space:             25,
		TrailLength:       8,
		CenterControl:     5,  // Increased from 2
		Freedom:           25, // Reduced from 40
		Tightness:         20,
		CornerPenalty:     15,
		Blocking:          8,
		ForcingMoves:      30,
		TerritoryCutting:  25,
		SynergyMultiplier: 0.3,
	}

	MidgameWeights = HeuristicWeights{
		Territory:         15,
		Space:             25,
		TrailLength:       8,
		CenterControl:     2,
		Freedom:           40,
		Tightness:         20,
		CornerPenalty:     15,
		Blocking:          8,
		ForcingMoves:      30,
		TerritoryCutting:  25,
		SynergyMultiplier: 0.3,
	}

	EndgameWeights = HeuristicWeights{
		Territory:         8, // Reduced from 15
		Space:             25,
		TrailLength:       15, // Increased from 8
		CenterControl:     1,
		Freedom:           50, // Increased from 40
		Tightness:         20,
		CornerPenalty:     15,
		Blocking:          8,
		ForcingMoves:      30,
		TerritoryCutting:  25,
		SynergyMultiplier: 0.3,
	}
)

// Voronoi configuration
const (
	OPENING_VORONOI_DEPTH = 45 // Deep search in opening
	MIDGAME_VORONOI_DEPTH = 30 // Balanced
	ENDGAME_VORONOI_DEPTH = 20 // Shallow but precise
	SPACE_COUNT_LIMIT     = 150
	ENDGAME_SPACE_DIFF    = 15
	PARTITION_THRESHOLD   = 0.4 // Trigger partition analysis
)

// Boost configuration
const (
	BOOST_RESERVE_MULTIPLIER = 2.5
	BOOST_ESCAPE_FREEDOM     = 2
	BOOST_ESCAPE_DISTANCE    = 4
	BOOST_CUTTING_ADVANTAGE  = 40
	BOOST_CUTTING_DISTANCE   = 3
	BOOST_AGGRESSIVE_BONUS   = 500
)

// Move ordering configuration
const (
	HASH_MOVE_BONUS      = 100000
	KILLER_MOVE_1_BONUS  = 10000
	KILLER_MOVE_2_BONUS  = 5000
	MAX_HISTORY_SCORE    = 10000
	LMR_FULL_DEPTH_MOVES = 4 // Search first N moves at full depth
	LMR_DEPTH_REDUCTION  = 2 // Reduce depth by this for late moves
)

// Search configuration
const (
	TT_EXACT = 0
	TT_LOWER = 1
	TT_UPPER = 2
)

// Influence function parameters
const (
	INFLUENCE_BASE_STRENGTH        = 100.0
	INFLUENCE_DECAY_RATE           = 3.0 // Range: 2.5-4.0
	INFLUENCE_OPPONENT_ATTENUATION = 0.7
)
