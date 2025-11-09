package main

import (
	"encoding/csv"
	"fmt"
	"os"
)

type HeuristicSnapshot struct {
	TurnNumber         int
	TerritoryDiff      int
	FreedomDiff        int
	ReachableDiff      int
	BoostDiff          int
	ChamberScore       int
	HeadDistance       int
	Separated          bool
	P1ComponentSize    int
	P2ComponentSize    int
	P1ValidMoves       int
	P2ValidMoves       int
	ActualOutcomeScore int // Set at end: final component size diff when separated
}

type GameRecorder struct {
	snapshots      []HeuristicSnapshot
	separationTurn int
	finalP1Space   int
	finalP2Space   int
	gameResult     *GameResult
}

func NewGameRecorder() *GameRecorder {
	return &GameRecorder{
		snapshots:      make([]HeuristicSnapshot, 0, 200),
		separationTurn: -1,
	}
}

func (gr *GameRecorder) RecordTurn(myAgent *Agent, otherAgent *Agent, turnCount int) {
	if !myAgent.Alive || !otherAgent.Alive {
		return
	}

	myHead := myAgent.GetHead()
	oppHead := otherAgent.GetHead()

	// Check if separated
	myHeadState := myAgent.Board.GetCellState(myHead)
	oppHeadState := otherAgent.Board.GetCellState(oppHead)
	myAgent.Board.SetCellState(myHead, EMPTY)
	otherAgent.Board.SetCellState(oppHead, EMPTY)

	cc := NewConnectedComponents(myAgent.Board)
	cc.Calculate()

	myAgent.Board.SetCellState(myHead, myHeadState)
	otherAgent.Board.SetCellState(oppHead, oppHeadState)

	separated := !cc.AreConnected(myHead, oppHead)
	p1CompSize := 0
	p2CompSize := 0

	if separated && gr.separationTurn == -1 {
		gr.separationTurn = turnCount
		myComponentID := cc.GetComponentID(myHead)
		oppComponentID := cc.GetComponentID(oppHead)
		gr.finalP1Space = cc.GetComponentSize(myComponentID)
		gr.finalP2Space = cc.GetComponentSize(oppComponentID)
	}

	if separated {
		myComponentID := cc.GetComponentID(myHead)
		oppComponentID := cc.GetComponentID(oppHead)
		p1CompSize = cc.GetComponentSize(myComponentID)
		p2CompSize = cc.GetComponentSize(oppComponentID)
	}

	// Calculate all heuristics
	myTerritory, oppTerritory, _ := calculateVoronoiControl(myAgent, otherAgent)
	territoryDiff := myTerritory - oppTerritory

	myValidMoves := myAgent.GetValidMoves()
	oppValidMoves := otherAgent.GetValidMoves()
	freedomDiff := len(myValidMoves) - len(oppValidMoves)

	myReachable := countReachableSpace(myAgent, 15)
	oppReachable := countReachableSpace(otherAgent, 15)
	reachableDiff := myReachable - oppReachable

	boostDiff := myAgent.BoostsRemaining - otherAgent.BoostsRemaining

	ct := NewChamberTree(myAgent.Board)
	chamberScore := ct.EvaluateChamberTree(myHead, oppHead)

	headDist := torusDistance(myHead, oppHead, myAgent.Board)

	snapshot := HeuristicSnapshot{
		TurnNumber:      turnCount,
		TerritoryDiff:   territoryDiff,
		FreedomDiff:     freedomDiff,
		ReachableDiff:   reachableDiff,
		BoostDiff:       boostDiff,
		ChamberScore:    chamberScore,
		HeadDistance:    headDist,
		Separated:       separated,
		P1ComponentSize: p1CompSize,
		P2ComponentSize: p2CompSize,
		P1ValidMoves:    len(myValidMoves),
		P2ValidMoves:    len(oppValidMoves),
	}

	gr.snapshots = append(gr.snapshots, snapshot)
}

func (gr *GameRecorder) SetGameResult(result *GameResult) {
	gr.gameResult = result

	// Only use data from before separation
	if gr.separationTurn != -1 {
		// Calculate outcome score (final component size difference)
		outcomeScore := gr.finalP1Space - gr.finalP2Space

		// Update all snapshots before separation with the outcome
		for i := range gr.snapshots {
			if gr.snapshots[i].TurnNumber < gr.separationTurn {
				gr.snapshots[i].ActualOutcomeScore = outcomeScore
			}
		}
	}
}

func (gr *GameRecorder) WriteToCSV(filename string, append bool) error {
	var file *os.File
	var err error

	if append {
		file, err = os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	} else {
		file, err = os.Create(filename)
	}

	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header if not appending
	if !append {
		header := []string{
			"turn",
			"territory_diff",
			"freedom_diff",
			"reachable_diff",
			"boost_diff",
			"chamber_score",
			"head_distance",
			"p1_valid_moves",
			"p2_valid_moves",
			"outcome_score",
		}
		if err := writer.Write(header); err != nil {
			return fmt.Errorf("failed to write header: %w", err)
		}
	}

	// Only write snapshots before separation with valid outcome
	for _, snap := range gr.snapshots {
		if gr.separationTurn != -1 && snap.TurnNumber < gr.separationTurn {
			record := []string{
				fmt.Sprintf("%d", snap.TurnNumber),
				fmt.Sprintf("%d", snap.TerritoryDiff),
				fmt.Sprintf("%d", snap.FreedomDiff),
				fmt.Sprintf("%d", snap.ReachableDiff),
				fmt.Sprintf("%d", snap.BoostDiff),
				fmt.Sprintf("%d", snap.ChamberScore),
				fmt.Sprintf("%d", snap.HeadDistance),
				fmt.Sprintf("%d", snap.P1ValidMoves),
				fmt.Sprintf("%d", snap.P2ValidMoves),
				fmt.Sprintf("%d", snap.ActualOutcomeScore),
			}
			if err := writer.Write(record); err != nil {
				return fmt.Errorf("failed to write record: %w", err)
			}
		}
	}

	return nil
}
