package main

import (
	"math"
)

// ============================================================================
// Heuristics: Position evaluation with phase-dependent weights
// ============================================================================

// Detect current game phase based on free space
func detectGamePhase(snapshot GameStateSnapshot) GamePhase {
	totalCells := BOARD_HEIGHT * BOARD_WIDTH
	occupiedCells := len(snapshot.myAgent.Trail) + len(snapshot.otherAgent.Trail)
	freeSpaceRatio := float64(totalCells-occupiedCells) / float64(totalCells)

	if freeSpaceRatio > OPENING_THRESHOLD {
		return Opening
	} else if freeSpaceRatio > MIDGAME_THRESHOLD {
		return Midgame
	}
	return Endgame
}

// Get weights for current game phase
func getPhaseWeights(phase GamePhase) HeuristicWeights {
	switch phase {
	case Opening:
		return OpeningWeights
	case Midgame:
		return MidgameWeights
	case Endgame:
		return EndgameWeights
	default:
		return MidgameWeights
	}
}

// Main evaluation function with phase-dependent weights
func evaluatePosition(myAgent *Agent, otherAgent *Agent) int {
	// Terminal conditions
	if !myAgent.Alive && !otherAgent.Alive {
		return DRAW_SCORE
	}
	if !myAgent.Alive {
		return LOSE_SCORE
	}
	if !otherAgent.Alive {
		return WIN_SCORE
	}

	myValidMoves := myAgent.GetValidMoves()
	opponentValidMoves := otherAgent.GetValidMoves()

	if len(myValidMoves) == 0 {
		return LOSE_SCORE
	}
	if len(opponentValidMoves) == 0 {
		return WIN_SCORE
	}

	// Detect phase and get appropriate weights
	snapshot := GameStateSnapshot{
		myAgent:    myAgent,
		otherAgent: otherAgent,
		board:      myAgent.Board,
		amIRed:     true,
	}
	phase := detectGamePhase(snapshot)
	weights := getPhaseWeights(phase)

	// Calculate space
	mySpace := countAvailableSpace(myAgent)
	opponentSpace := countAvailableSpace(otherAgent)

	// Endgame detection
	if phase == Endgame || shouldUseEndgameMode(myAgent, otherAgent) {
		endgameResult := evaluateEndgame(myAgent, otherAgent, mySpace, opponentSpace)
		if endgameResult != 0 && abs(endgameResult) > 1000 {
			return endgameResult
		}
	}

	score := 0

	// Strategic components
	strategicScore := 0

	// 1. Territory control (Voronoi with influence)
	territoryResult := voronoiTerritoryAdaptive(myAgent, otherAgent, phase)
	territoryDiff := territoryResult.myTerritory - territoryResult.opponentTerritory
	strategicScore += territoryDiff * weights.Territory / 100 // Normalize

	// Add influence function (more sophisticated)
	if phase == Midgame || phase == Endgame {
		influenceResult := influenceFunctionTerritory(myAgent, otherAgent)
		influenceDiff := int(influenceResult.myInfluence - influenceResult.opponentInfluence)
		strategicScore += influenceDiff * weights.Territory / 200
	}

	// 2. Space advantage
	spaceDiff := mySpace - opponentSpace
	if spaceDiff > 50 {
		strategicScore += 3000
		// Add territory cutting bonus when ahead
		strategicScore += evaluateTerritoryCutting(myAgent, otherAgent) * weights.TerritoryCutting
	} else if spaceDiff < -50 {
		strategicScore -= 3000
	} else {
		strategicScore += spaceDiff * weights.Space
	}

	// 3. Trail length
	strategicScore += myAgent.Length * weights.TrailLength
	strategicScore -= otherAgent.Length * weights.TrailLength

	// 4. Center control
	myHead := myAgent.GetHead()
	opponentHead := otherAgent.GetHead()
	centerX, centerY := BOARD_WIDTH/2, BOARD_HEIGHT/2
	myCenterDist := manhattanDistanceRaw(myHead.X, myHead.Y, centerX, centerY)
	opponentCenterDist := manhattanDistanceRaw(opponentHead.X, opponentHead.Y, centerX, centerY)
	strategicScore += (opponentCenterDist - myCenterDist) * weights.CenterControl

	// Tactical components
	tacticalScore := 0

	// 5. Freedom degree (immediate mobility)
	myFreedom := countFreedomDegree(myAgent)
	opponentFreedom := countFreedomDegree(otherAgent)
	tacticalScore += (myFreedom - opponentFreedom) * weights.Freedom

	// 6. Tightness (avoid cramped positions)
	myTightness := measureTightness(myAgent)
	opponentTightness := measureTightness(otherAgent)
	tacticalScore += (opponentTightness - myTightness) * weights.Tightness

	// 7. Corner penalty
	myCornerPenalty := cornerProximityPenalty(myHead)
	opponentCornerPenalty := cornerProximityPenalty(opponentHead)
	tacticalScore += (opponentCornerPenalty - myCornerPenalty) * weights.CornerPenalty

	// 8. Blocking position
	myBlockingScore := blockingPositionScore(myAgent, otherAgent)
	opponentBlockingScore := blockingPositionScore(otherAgent, myAgent)
	tacticalScore += (myBlockingScore - opponentBlockingScore) * weights.Blocking

	// 9. Forcing moves
	forcingBonus := evaluateForcingMoves(myAgent, otherAgent)
	tacticalScore += forcingBonus * weights.ForcingMoves

	// 10. Barrier detection
	myBarriers := detectBarriers(myAgent, otherAgent)
	opponentBarriers := detectBarriers(otherAgent, myAgent)
	tacticalScore -= myBarriers
	tacticalScore += opponentBarriers

	// 11. Articulation point control (advanced)
	if phase == Midgame || phase == Endgame {
		apScore := evaluateArticulationControl(myAgent, otherAgent)
		strategicScore += apScore
	}

	// Combine scores
	score = strategicScore + tacticalScore

	// Multiplicative synergy bonus (non-linear interaction)
	if strategicScore > 0 && tacticalScore > 0 {
		synergy := math.Sqrt(float64(strategicScore) * float64(tacticalScore))
		score += int(synergy * weights.SynergyMultiplier)
	}

	// Distance-based adjustments
	dist := manhattanDistance(myHead, opponentHead)
	if mySpace > opponentSpace {
		score += dist * 3 // Maintain distance when ahead
	} else if mySpace < opponentSpace {
		score -= dist * 5 // Close distance when behind
	}

	// Boost value
	if myAgent.BoostsRemaining > 0 && otherAgent.BoostsRemaining == 0 {
		score += 100
	}

	return score
}

// Evaluation with player bias (first-move advantage)
func evaluatePositionWithBias(myAgent *Agent, otherAgent *Agent, amIRed bool) int {
	baseScore := evaluatePosition(myAgent, otherAgent)

	if !myAgent.Alive || !otherAgent.Alive {
		return baseScore
	}

	myHead := myAgent.GetHead()
	opponentHead := otherAgent.GetHead()
	dist := manhattanDistance(myHead, opponentHead)

	// Close combat bias
	if dist <= 4 {
		if amIRed {
			baseScore += 150 // First player has advantage in close combat
		} else {
			baseScore -= 200 // Second player is disadvantaged
		}
	}

	// Escape bonus for second player
	if !amIRed && dist <= 3 {
		for _, dir := range myAgent.GetValidMoves() {
			nextPos := Position{X: myHead.X + dir.DX, Y: myHead.Y + dir.DY}
			nextPos = myAgent.Board.TorusCheck(nextPos)

			distToOpp := manhattanDistance(nextPos, opponentHead)
			if distToOpp > dist {
				baseScore += 80 // Reward moves that increase distance
				break
			}
		}
	}

	return baseScore
}

// Territory cutting evaluation
func evaluateTerritoryCutting(myAgent *Agent, opponentAgent *Agent) int {
	if !myAgent.Alive || !opponentAgent.Alive {
		return 0
	}

	myHead := myAgent.GetHead()
	opponentHead := opponentAgent.GetHead()
	dist := manhattanDistance(myHead, opponentHead)

	if dist > 8 {
		return 0
	}

	cuttingScore := 0

	// Check if we're between opponent and their escape routes
	midX := (myHead.X + opponentHead.X) / 2
	midY := (myHead.Y + opponentHead.Y) / 2

	myDistToMid := manhattanDistanceRaw(myHead.X, myHead.Y, midX, midY)
	oppDistToMid := manhattanDistanceRaw(opponentHead.X, opponentHead.Y, midX, midY)

	if myDistToMid < oppDistToMid {
		cuttingScore += 50
	}

	// Check opponent's freedom
	opponentValidMoves := len(opponentAgent.GetValidMoves())
	if opponentValidMoves <= 2 {
		cuttingScore += 100
	} else if opponentValidMoves == 3 {
		cuttingScore += 30
	}

	return cuttingScore
}

// Forcing moves evaluation
func evaluateForcingMoves(myAgent *Agent, opponentAgent *Agent) int {
	if !myAgent.Alive || !opponentAgent.Alive {
		return 0
	}

	opponentValidMoves := opponentAgent.GetValidMoves()

	if len(opponentValidMoves) == 1 {
		return 80 // Very strong forcing position
	} else if len(opponentValidMoves) == 2 {
		opponentHead := opponentAgent.GetHead()

		badMoves := 0
		for _, dir := range opponentValidMoves {
			nextPos := Position{X: opponentHead.X + dir.DX, Y: opponentHead.Y + dir.DY}
			nextPos = opponentAgent.Board.TorusCheck(nextPos)

			freeNeighbors := 0
			for _, d := range AllDirections {
				neighbor := Position{X: nextPos.X + d.DX, Y: nextPos.Y + d.DY}
				neighbor = opponentAgent.Board.TorusCheck(neighbor)
				if opponentAgent.Board.GetCellState(neighbor) == EMPTY {
					freeNeighbors++
				}
			}

			if freeNeighbors <= 2 {
				badMoves++
			}
		}

		if badMoves >= 1 {
			return 40
		}
		return 20
	}

	return 0
}

// Count freedom degree (immediate neighbors)
func countFreedomDegree(agent *Agent) int {
	if !agent.Alive {
		return 0
	}

	head := agent.GetHead()
	freeCount := 0

	for _, dir := range AllDirections {
		next := Position{X: head.X + dir.DX, Y: head.Y + dir.DY}
		next = agent.Board.TorusCheck(next)

		if agent.Board.GetCellState(next) == EMPTY {
			freeCount++
		}
	}

	return freeCount
}

// Measure tightness (nearby occupied cells)
func measureTightness(agent *Agent) int {
	if !agent.Alive {
		return 0
	}

	head := agent.GetHead()
	tightness := 0

	for dx := -2; dx <= 2; dx++ {
		for dy := -2; dy <= 2; dy++ {
			if dx == 0 && dy == 0 {
				continue
			}
			pos := Position{X: head.X + dx, Y: head.Y + dy}
			pos = agent.Board.TorusCheck(pos)

			if agent.Board.GetCellState(pos) == AGENT {
				tightness++
			}
		}
	}

	return tightness
}

// Corner proximity penalty
func cornerProximityPenalty(pos Position) int {
	corners := []Position{
		{X: 0, Y: 0},
		{X: BOARD_WIDTH - 1, Y: 0},
		{X: 0, Y: BOARD_HEIGHT - 1},
		{X: BOARD_WIDTH - 1, Y: BOARD_HEIGHT - 1},
	}

	minDist := math.MaxInt32
	for _, corner := range corners {
		dist := manhattanDistanceRaw(pos.X, pos.Y, corner.X, corner.Y)
		if dist < minDist {
			minDist = dist
		}
	}

	if minDist <= 2 {
		return 50 - minDist*10
	}

	return 0
}

// Blocking position score
func blockingPositionScore(agent *Agent, opponent *Agent) int {
	if !agent.Alive || !opponent.Alive {
		return 0
	}

	myHead := agent.GetHead()
	opponentHead := opponent.GetHead()
	dist := manhattanDistance(myHead, opponentHead)

	if dist <= 5 {
		opponentValidMoves := 0
		for _, dir := range opponent.GetValidMoves() {
			next := Position{X: opponentHead.X + dir.DX, Y: opponentHead.Y + dir.DY}
			next = opponent.Board.TorusCheck(next)

			if opponent.Board.GetCellState(next) == EMPTY {
				opponentValidMoves++
			}
		}

		if opponentValidMoves <= 1 {
			return 100
		} else if opponentValidMoves == 2 {
			return 30
		}
	}

	return 0
}

// Manhattan distance between positions
func manhattanDistance(p1, p2 Position) int {
	return manhattanDistanceRaw(p1.X, p1.Y, p2.X, p2.Y)
}

// Manhattan distance with torus wrap
func manhattanDistanceRaw(x1, y1, x2, y2 int) int {
	dx := abs(x1 - x2)
	dy := abs(y1 - y2)

	if dx > BOARD_WIDTH/2 {
		dx = BOARD_WIDTH - dx
	}
	if dy > BOARD_HEIGHT/2 {
		dy = BOARD_HEIGHT - dy
	}

	return dx + dy
}

// Absolute value
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// Min of two ints
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Max of two ints
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
