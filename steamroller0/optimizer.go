package main

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

type WeightConfig struct {
	WEIGHT_TERRITORY      int
	WEIGHT_FREEDOM        int
	WEIGHT_REACHABLE      int
	WEIGHT_BOOST          int
	WEIGHT_CHAMBER        int
	WEIGHT_EDGE           int
	WEIGHT_COMPACTNESS    int
	WEIGHT_CUTOFF         int
	WEIGHT_GROWTH         int
	PENALTY_CORRIDOR_BASE int
	PENALTY_HEAD_DISTANCE int
}

func (w WeightConfig) String() string {
	return fmt.Sprintf(`WEIGHT_TERRITORY      = %d
WEIGHT_FREEDOM        = %d
WEIGHT_REACHABLE      = %d
WEIGHT_BOOST          = %d
WEIGHT_CHAMBER        = %d
WEIGHT_EDGE           = %d
WEIGHT_COMPACTNESS    = %d
WEIGHT_CUTOFF         = %d
WEIGHT_GROWTH         = %d
PENALTY_CORRIDOR_BASE = %d
PENALTY_HEAD_DISTANCE = %d`,
		w.WEIGHT_TERRITORY,
		w.WEIGHT_FREEDOM,
		w.WEIGHT_REACHABLE,
		w.WEIGHT_BOOST,
		w.WEIGHT_CHAMBER,
		w.WEIGHT_EDGE,
		w.WEIGHT_COMPACTNESS,
		w.WEIGHT_CUTOFF,
		w.WEIGHT_GROWTH,
		w.PENALTY_CORRIDOR_BASE,
		w.PENALTY_HEAD_DISTANCE)
}

func randomWeights() WeightConfig {
	return WeightConfig{
		WEIGHT_TERRITORY:      rand.Intn(200),
		WEIGHT_FREEDOM:        rand.Intn(200),
		WEIGHT_REACHABLE:      rand.Intn(200),
		WEIGHT_BOOST:          rand.Intn(50),
		WEIGHT_CHAMBER:        rand.Intn(100),
		WEIGHT_EDGE:           rand.Intn(50) - 25,
		WEIGHT_COMPACTNESS:    rand.Intn(100) - 50,
		WEIGHT_CUTOFF:         rand.Intn(100),
		WEIGHT_GROWTH:         rand.Intn(100),
		PENALTY_CORRIDOR_BASE: rand.Intn(1000) + 100,
		PENALTY_HEAD_DISTANCE: rand.Intn(500) + 50,
	}
}

type threadSafeWeights struct {
	mu      sync.Mutex
	weights WeightConfig
}

var globalWeights = &threadSafeWeights{}

func setWeights(w WeightConfig) {
	globalWeights.mu.Lock()
	defer globalWeights.mu.Unlock()
	WEIGHT_TERRITORY = w.WEIGHT_TERRITORY
	WEIGHT_FREEDOM = w.WEIGHT_FREEDOM
	WEIGHT_REACHABLE = w.WEIGHT_REACHABLE
	WEIGHT_BOOST = w.WEIGHT_BOOST
	WEIGHT_CHAMBER = w.WEIGHT_CHAMBER
	WEIGHT_EDGE = w.WEIGHT_EDGE
	WEIGHT_COMPACTNESS = w.WEIGHT_COMPACTNESS
	WEIGHT_CUTOFF = w.WEIGHT_CUTOFF
	WEIGHT_GROWTH = w.WEIGHT_GROWTH
	PENALTY_CORRIDOR_BASE = w.PENALTY_CORRIDOR_BASE
	PENALTY_HEAD_DISTANCE = w.PENALTY_HEAD_DISTANCE
}

func runSingleGame(weights1, weights2 WeightConfig) GameResult {
	game := NewGame()

	for game.Turns < 500 {
		if !game.Agent1.Alive || !game.Agent2.Alive {
			break
		}

		setWeights(weights1)
		myTrail1 := make([][]int, len(game.Agent1.Trail))
		for i, pos := range game.Agent1.Trail {
			myTrail1[i] = []int{pos.X, pos.Y}
		}
		otherTrail1 := make([][]int, len(game.Agent2.Trail))
		for i, pos := range game.Agent2.Trail {
			otherTrail1[i] = []int{pos.X, pos.Y}
		}
		move1 := DecideMove(myTrail1, otherTrail1, game.Turns, game.Agent1.BoostsRemaining, 1)
		dir1, boost1 := parseMove(move1)

		setWeights(weights2)
		myTrail2 := make([][]int, len(game.Agent2.Trail))
		for i, pos := range game.Agent2.Trail {
			myTrail2[i] = []int{pos.X, pos.Y}
		}
		otherTrail2 := make([][]int, len(game.Agent1.Trail))
		for i, pos := range game.Agent1.Trail {
			otherTrail2[i] = []int{pos.X, pos.Y}
		}
		move2 := DecideMove(myTrail2, otherTrail2, game.Turns, game.Agent2.BoostsRemaining, 2)
		dir2, boost2 := parseMove(move2)

		result := game.Step(dir1, dir2, boost1, boost2)
		if result != nil {
			return *result
		}
	}

	if game.Agent1.Length > game.Agent2.Length {
		return Agent1Win
	} else if game.Agent2.Length > game.Agent1.Length {
		return Agent2Win
	}
	return Draw
}

func parseMove(move string) (Direction, bool) {
	boost := false
	dirStr := move
	if len(move) > 6 && move[len(move)-6:] == ":BOOST" {
		boost = true
		dirStr = move[:len(move)-6]
	}

	switch dirStr {
	case "UP":
		return UP, boost
	case "DOWN":
		return DOWN, boost
	case "LEFT":
		return LEFT, boost
	case "RIGHT":
		return RIGHT, boost
	}
	return RIGHT, boost
}

type Individual struct {
	Weights WeightConfig
	Wins    int
	Losses  int
	Draws   int
}

func (ind Individual) Score() float64 {
	total := ind.Wins + ind.Losses + ind.Draws
	if total == 0 {
		return 0
	}
	return float64(ind.Wins)*1.0 + float64(ind.Draws)*0.5
}

type GameTask struct {
	weights1 WeightConfig
	weights2 WeightConfig
	idx      int
	isAgent1 bool
}

type GameTaskResult struct {
	idx      int
	result   GameResult
	isAgent1 bool
}

func runGamesParallel(individuals []Individual, baseline WeightConfig, gamesPerIndividual int, workers int) {
	tasks := make(chan GameTask, len(individuals)*gamesPerIndividual)
	results := make(chan GameTaskResult, len(individuals)*gamesPerIndividual)

	totalGames := len(individuals) * gamesPerIndividual
	completedGames := 0

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for task := range tasks {
				result := runSingleGame(task.weights1, task.weights2)
				results <- GameTaskResult{idx: task.idx, result: result, isAgent1: task.isAgent1}
			}
		}(w)
	}

	for i := range individuals {
		for game := 0; game < gamesPerIndividual; game++ {
			if game%2 == 0 {
				tasks <- GameTask{weights1: individuals[i].Weights, weights2: baseline, idx: i, isAgent1: true}
			} else {
				tasks <- GameTask{weights1: baseline, weights2: individuals[i].Weights, idx: i, isAgent1: false}
			}
		}
	}
	close(tasks)

	go func() {
		wg.Wait()
		close(results)
	}()

	resultCounts := make([]struct{ wins, losses, draws int }, len(individuals))
	for res := range results {
		completedGames++
		if completedGames%20 == 0 || completedGames == totalGames {
			fmt.Printf("  Progress: %d/%d games completed\r", completedGames, totalGames)
		}

		if res.isAgent1 {
			if res.result == Agent1Win {
				resultCounts[res.idx].wins++
			} else if res.result == Agent2Win {
				resultCounts[res.idx].losses++
			} else {
				resultCounts[res.idx].draws++
			}
		} else {
			if res.result == Agent2Win {
				resultCounts[res.idx].wins++
			} else if res.result == Agent1Win {
				resultCounts[res.idx].losses++
			} else {
				resultCounts[res.idx].draws++
			}
		}
	}
	fmt.Println()

	for i := range individuals {
		individuals[i].Wins = resultCounts[i].wins
		individuals[i].Losses = resultCounts[i].losses
		individuals[i].Draws = resultCounts[i].draws
	}
}

func RunOptimizer(populationSize int, gamesPerIndividual int, workers int) {
	rand.Seed(time.Now().UnixNano())

	population := make([]Individual, populationSize)
	for i := range population {
		population[i].Weights = randomWeights()
	}

	baseline := WeightConfig{
		WEIGHT_TERRITORY:      26,
		WEIGHT_FREEDOM:        120,
		WEIGHT_REACHABLE:      136,
		WEIGHT_BOOST:          14,
		WEIGHT_CHAMBER:        30,
		WEIGHT_EDGE:           -10,
		WEIGHT_COMPACTNESS:    -15,
		WEIGHT_CUTOFF:         12,
		WEIGHT_GROWTH:         30,
		PENALTY_CORRIDOR_BASE: 500,
		PENALTY_HEAD_DISTANCE: 200,
	}

	gen := 0
	var previousBest *WeightConfig

	for {
		gen++
		fmt.Printf("\n=== Generation %d ===\n", gen)
		fmt.Printf("Running %d games (%d individuals × %d games each) with %d workers...\n",
			populationSize*gamesPerIndividual, populationSize, gamesPerIndividual, workers)
		startTime := time.Now()

		runGamesParallel(population, baseline, gamesPerIndividual, workers)

		bestIdx := 0
		for i := range population {
			if population[i].Score() > population[bestIdx].Score() {
				bestIdx = i
			}
		}

		elapsed := time.Since(startTime)
		fmt.Printf("Generation completed in %.2fs\n", elapsed.Seconds())
		fmt.Printf("\nBest vs Baseline: %d wins, %d losses, %d draws (Score: %.1f/%d)\n",
			population[bestIdx].Wins,
			population[bestIdx].Losses,
			population[bestIdx].Draws,
			population[bestIdx].Score(),
			gamesPerIndividual)

		if previousBest != nil {
			fmt.Printf("Testing best vs previous generation...\n")
			vsWins, vsLosses, vsDraws := 0, 0, 0
			for game := 0; game < gamesPerIndividual; game++ {
				var result GameResult
				if game%2 == 0 {
					result = runSingleGame(population[bestIdx].Weights, *previousBest)
				} else {
					result = runSingleGame(*previousBest, population[bestIdx].Weights)
					if result == Agent1Win {
						result = Agent2Win
					} else if result == Agent2Win {
						result = Agent1Win
					}
				}
				if result == Agent1Win {
					vsWins++
				} else if result == Agent2Win {
					vsLosses++
				} else {
					vsDraws++
				}
			}
			vsScore := float64(vsWins) + float64(vsDraws)*0.5
			fmt.Printf("Best vs Previous: %d wins, %d losses, %d draws (Score: %.1f/%d)\n",
				vsWins, vsLosses, vsDraws, vsScore, gamesPerIndividual)
		}

		fmt.Printf("\n%s\n", population[bestIdx].Weights.String())

		currentBest := population[bestIdx].Weights
		previousBest = &currentBest

		newPopulation := make([]Individual, populationSize)
		newPopulation[0].Weights = population[bestIdx].Weights

		sortedPop := make([]Individual, len(population))
		copy(sortedPop, population)
		for i := 0; i < len(sortedPop)-1; i++ {
			for j := i + 1; j < len(sortedPop); j++ {
				if sortedPop[j].Score() > sortedPop[i].Score() {
					sortedPop[i], sortedPop[j] = sortedPop[j], sortedPop[i]
				}
			}
		}

		eliteSize := populationSize / 5
		if eliteSize < 2 {
			eliteSize = 2
		}

		for i := 1; i < populationSize; i++ {
			parent1 := sortedPop[rand.Intn(eliteSize)]
			parent2 := sortedPop[rand.Intn(eliteSize)]

			child := WeightConfig{
				WEIGHT_TERRITORY:      mutate(crossover(parent1.Weights.WEIGHT_TERRITORY, parent2.Weights.WEIGHT_TERRITORY), 200),
				WEIGHT_FREEDOM:        mutate(crossover(parent1.Weights.WEIGHT_FREEDOM, parent2.Weights.WEIGHT_FREEDOM), 200),
				WEIGHT_REACHABLE:      mutate(crossover(parent1.Weights.WEIGHT_REACHABLE, parent2.Weights.WEIGHT_REACHABLE), 200),
				WEIGHT_BOOST:          mutate(crossover(parent1.Weights.WEIGHT_BOOST, parent2.Weights.WEIGHT_BOOST), 50),
				WEIGHT_CHAMBER:        mutate(crossover(parent1.Weights.WEIGHT_CHAMBER, parent2.Weights.WEIGHT_CHAMBER), 100),
				WEIGHT_EDGE:           mutateSigned(crossover(parent1.Weights.WEIGHT_EDGE, parent2.Weights.WEIGHT_EDGE), 50, -25),
				WEIGHT_COMPACTNESS:    mutateSigned(crossover(parent1.Weights.WEIGHT_COMPACTNESS, parent2.Weights.WEIGHT_COMPACTNESS), 100, -50),
				WEIGHT_CUTOFF:         mutate(crossover(parent1.Weights.WEIGHT_CUTOFF, parent2.Weights.WEIGHT_CUTOFF), 100),
				WEIGHT_GROWTH:         mutate(crossover(parent1.Weights.WEIGHT_GROWTH, parent2.Weights.WEIGHT_GROWTH), 100),
				PENALTY_CORRIDOR_BASE: mutatePositive(crossover(parent1.Weights.PENALTY_CORRIDOR_BASE, parent2.Weights.PENALTY_CORRIDOR_BASE), 1000, 100),
				PENALTY_HEAD_DISTANCE: mutatePositive(crossover(parent1.Weights.PENALTY_HEAD_DISTANCE, parent2.Weights.PENALTY_HEAD_DISTANCE), 500, 50),
			}
			newPopulation[i].Weights = child
		}

		population = newPopulation
	}
}

func crossover(a, b int) int {
	if rand.Float64() < 0.5 {
		return a
	}
	return b
}

func mutate(val int, maxVal int) int {
	if rand.Float64() < 0.2 {
		return rand.Intn(maxVal)
	}
	if rand.Float64() < 0.4 {
		delta := rand.Intn(maxVal/5) - maxVal/10
		result := val + delta
		if result < 0 {
			result = 0
		}
		if result > maxVal {
			result = maxVal
		}
		return result
	}
	return val
}

func mutateSigned(val int, maxVal int, minVal int) int {
	if rand.Float64() < 0.2 {
		return rand.Intn(maxVal-minVal) + minVal
	}
	if rand.Float64() < 0.4 {
		delta := rand.Intn((maxVal-minVal)/5) - (maxVal-minVal)/10
		result := val + delta
		if result < minVal {
			result = minVal
		}
		if result > maxVal {
			result = maxVal
		}
		return result
	}
	return val
}

func mutatePositive(val int, maxVal int, minVal int) int {
	if rand.Float64() < 0.2 {
		return rand.Intn(maxVal) + minVal
	}
	if rand.Float64() < 0.4 {
		delta := rand.Intn(maxVal/5) - maxVal/10
		result := val + delta
		if result < minVal {
			result = minVal
		}
		if result > maxVal+minVal {
			result = maxVal + minVal
		}
		return result
	}
	return val
}
