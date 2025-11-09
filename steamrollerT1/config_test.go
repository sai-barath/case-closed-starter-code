package main

import (
	"testing"
)

// ============================================================================
// Config Tests
// ============================================================================

func TestGamePhaseThresholds(t *testing.T) {
	if OPENING_THRESHOLD <= MIDGAME_THRESHOLD {
		t.Errorf("Opening threshold (%f) should be > midgame threshold (%f)",
			OPENING_THRESHOLD, MIDGAME_THRESHOLD)
	}
}

func TestHeuristicWeightsConsistency(t *testing.T) {
	testCases := []struct {
		name    string
		weights HeuristicWeights
	}{
		{"Opening", OpeningWeights},
		{"Midgame", MidgameWeights},
		{"Endgame", EndgameWeights},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// All weights should be positive
			if tc.weights.Territory < 0 {
				t.Errorf("%s: Territory weight should be positive", tc.name)
			}
			if tc.weights.Space < 0 {
				t.Errorf("%s: Space weight should be positive", tc.name)
			}
			if tc.weights.Freedom < 0 {
				t.Errorf("%s: Freedom weight should be positive", tc.name)
			}
		})
	}
}

func TestEndgameWeightsHigherFreedom(t *testing.T) {
	if EndgameWeights.Freedom <= MidgameWeights.Freedom {
		t.Errorf("Endgame freedom (%d) should be > midgame freedom (%d)",
			EndgameWeights.Freedom, MidgameWeights.Freedom)
	}
}

func TestOpeningWeightsCenterControl(t *testing.T) {
	if OpeningWeights.CenterControl <= MidgameWeights.CenterControl {
		t.Errorf("Opening center control (%d) should be > midgame center control (%d)",
			OpeningWeights.CenterControl, MidgameWeights.CenterControl)
	}
}
