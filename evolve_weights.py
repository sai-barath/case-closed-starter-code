#!/usr/bin/env python3
"""
Evolutionary parameter tuning for steamroller bot.
Runs tournaments between different parameter configurations and evolves the best ones.
"""

import subprocess
import random
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple
import statistics

@dataclass
class BotParams:
    """Parameter configuration for a bot"""
    WEIGHT_TERRITORY: int = 50
    WEIGHT_FREEDOM: int = 150
    WEIGHT_REACHABLE: int = 100
    WEIGHT_BOOST: int = 20
    WEIGHT_CHAMBER: int = 30
    WEIGHT_EDGE: int = 15
    WEIGHT_COMPACTNESS: int = 25
    WEIGHT_CUTOFF: int = 40
    WEIGHT_GROWTH: int = 30
    PENALTY_CORRIDOR_BASE: int = 500
    PENALTY_HEAD_DISTANCE: int = 200
    
    wins: int = 0
    losses: int = 0
    draws: int = 0
    
    def mutate(self, mutation_rate: float = 0.3) -> 'BotParams':
        """Create a mutated copy of this bot's parameters"""
        params = BotParams(**{k: v for k, v in asdict(self).items() if k not in ['wins', 'losses', 'draws']})
        
        for attr in ['WEIGHT_TERRITORY', 'WEIGHT_FREEDOM', 'WEIGHT_REACHABLE', 'WEIGHT_BOOST',
                     'WEIGHT_CHAMBER', 'WEIGHT_EDGE', 'WEIGHT_COMPACTNESS', 'WEIGHT_CUTOFF', 'WEIGHT_GROWTH']:
            if random.random() < mutation_rate:
                current = getattr(params, attr)
                # Mutate by ±20% to ±50%
                change = random.randint(-50, 50)
                new_val = max(5, current + change)
                setattr(params, attr, new_val)
        
        # Penalties are different scale
        for attr in ['PENALTY_CORRIDOR_BASE', 'PENALTY_HEAD_DISTANCE']:
            if random.random() < mutation_rate:
                current = getattr(params, attr)
                change = random.randint(-100, 100)
                new_val = max(50, current + change)
                setattr(params, attr, new_val)
        
        return params
    
    def crossover(self, other: 'BotParams') -> 'BotParams':
        """Create offspring by combining parameters from two parents"""
        params = {}
        for attr in ['WEIGHT_TERRITORY', 'WEIGHT_FREEDOM', 'WEIGHT_REACHABLE', 'WEIGHT_BOOST',
                     'WEIGHT_CHAMBER', 'WEIGHT_EDGE', 'WEIGHT_COMPACTNESS', 'WEIGHT_CUTOFF', 'WEIGHT_GROWTH',
                     'PENALTY_CORRIDOR_BASE', 'PENALTY_HEAD_DISTANCE']:
            # Randomly pick from one parent or average them
            if random.random() < 0.5:
                params[attr] = getattr(self, attr)
            else:
                params[attr] = getattr(other, attr)
        
        return BotParams(**params)
    
    @property
    def fitness(self) -> float:
        """Calculate fitness score: wins - losses + draws*0.5"""
        total = self.wins + self.losses + self.draws
        if total == 0:
            return 0
        return (self.wins - self.losses + self.draws * 0.5) / total
    
    def __str__(self) -> str:
        return f"W{self.wins}-L{self.losses}-D{self.draws} ({self.fitness:.2f})"


def write_params_to_file(params: BotParams, bot_dir: str):
    """Write parameters to agent_logic.go by replacing var block"""
    logic_file = os.path.join(bot_dir, "agent_logic.go")
    
    with open(logic_file, 'r') as f:
        lines = f.readlines()
    
    # Find the var block and replace it
    new_lines = []
    in_var_block = False
    
    for line in lines:
        if line.strip().startswith('var (') and 'WEIGHT_TERRITORY' in ''.join(lines[lines.index(line):lines.index(line)+20]):
            in_var_block = True
            new_lines.append(line)
            # Write new parameters
            new_lines.append(f"\tWEIGHT_TERRITORY       = {params.WEIGHT_TERRITORY}\n")
            new_lines.append(f"\tWEIGHT_FREEDOM         = {params.WEIGHT_FREEDOM}\n")
            new_lines.append(f"\tWEIGHT_REACHABLE       = {params.WEIGHT_REACHABLE}\n")
            new_lines.append(f"\tWEIGHT_BOOST           = {params.WEIGHT_BOOST}\n")
            new_lines.append(f"\tWEIGHT_CHAMBER         = {params.WEIGHT_CHAMBER}\n")
            new_lines.append(f"\tWEIGHT_EDGE            = {params.WEIGHT_EDGE}\n")
            new_lines.append(f"\tWEIGHT_COMPACTNESS     = {params.WEIGHT_COMPACTNESS}\n")
            new_lines.append(f"\tWEIGHT_CUTOFF          = {params.WEIGHT_CUTOFF}\n")
            new_lines.append(f"\tWEIGHT_GROWTH          = {params.WEIGHT_GROWTH}\n")
            new_lines.append(f"\tPENALTY_CORRIDOR_BASE  = {params.PENALTY_CORRIDOR_BASE}\n")
            new_lines.append(f"\tPENALTY_HEAD_DISTANCE  = {params.PENALTY_HEAD_DISTANCE}\n")
            continue
        
        if in_var_block:
            if line.strip() == ')':
                in_var_block = False
                new_lines.append(line)
            continue
        
        new_lines.append(line)
    
    with open(logic_file, 'w') as f:
        f.writelines(new_lines)


def build_bot(bot_dir: str) -> bool:
    """Compile the bot. Returns True if successful."""
    try:
        result = subprocess.run(
            ['go', 'build', '-o', 'steamroller'],
            cwd=bot_dir,
            capture_output=True,
            timeout=30
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Build error: {e}")
        return False


def play_game(bot1_path: str, bot2_path: str) -> str:
    """Play a game between two bots. Returns 'bot1', 'bot2', or 'draw'"""
    try:
        result = subprocess.run(
            ['python3', 'local-tester.py', bot1_path, bot2_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        output = result.stdout + result.stderr
        
        # Parse result
        if 'Player 1 wins' in output or 'Winner: Player 1' in output:
            return 'bot1'
        elif 'Player 2 wins' in output or 'Winner: Player 2' in output:
            return 'bot2'
        elif 'Draw' in output or 'draw' in output.lower():
            return 'draw'
        else:
            # Try to parse from the output
            lines = output.split('\n')
            for line in lines:
                if 'winner' in line.lower():
                    if '1' in line:
                        return 'bot1'
                    elif '2' in line:
                        return 'bot2'
            return 'draw'
    except Exception as e:
        print(f"Game error: {e}")
        return 'draw'


def tournament(population: List[BotParams], games_per_matchup: int = 10) -> List[BotParams]:
    """Run a round-robin tournament between all bots in population"""
    print(f"\n=== Running tournament with {len(population)} bots, {games_per_matchup} games per matchup ===")
    
    # Reset scores
    for bot in population:
        bot.wins = 0
        bot.losses = 0
        bot.draws = 0
    
    # Prepare bot directories
    bot_dirs = []
    for i, params in enumerate(population):
        bot_dir = f"steamroller0_gen{i}"
        os.makedirs(bot_dir, exist_ok=True)
        
        # Copy files
        for file in ['agent.go', 'agent_logic.go', 'game_logic.go', 'spacefilling.go', 'go.mod']:
            subprocess.run(['cp', f'steamroller0/{file}', f'{bot_dir}/{file}'], check=True)
        
        # Write parameters
        write_params_to_file(params, bot_dir)
        
        # Build
        print(f"Building bot {i}...", end=' ')
        if build_bot(bot_dir):
            print("✓")
            bot_dirs.append(bot_dir)
        else:
            print("✗ (build failed)")
            params.losses += 100  # Penalize failed builds heavily
            bot_dirs.append(None)
    
    # Run games
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            if bot_dirs[i] is None or bot_dirs[j] is None:
                continue
            
            print(f"\nBot {i} vs Bot {j}:")
            
            for game_num in range(games_per_matchup):
                # Alternate who plays first
                if game_num % 2 == 0:
                    result = play_game(f'{bot_dirs[i]}/steamroller', f'{bot_dirs[j]}/steamroller')
                    if result == 'bot1':
                        population[i].wins += 1
                        population[j].losses += 1
                        print("1", end='')
                    elif result == 'bot2':
                        population[i].losses += 1
                        population[j].wins += 1
                        print("2", end='')
                    else:
                        population[i].draws += 1
                        population[j].draws += 1
                        print("=", end='')
                else:
                    result = play_game(f'{bot_dirs[j]}/steamroller', f'{bot_dirs[i]}/steamroller')
                    if result == 'bot1':
                        population[j].wins += 1
                        population[i].losses += 1
                        print("2", end='')
                    elif result == 'bot2':
                        population[j].losses += 1
                        population[i].wins += 1
                        print("1", end='')
                    else:
                        population[i].draws += 1
                        population[j].draws += 1
                        print("=", end='')
            print()
    
    # Cleanup temp directories
    for bot_dir in bot_dirs:
        if bot_dir:
            subprocess.run(['rm', '-rf', bot_dir], check=False)
    
    # Sort by fitness
    population.sort(key=lambda x: x.fitness, reverse=True)
    
    print("\n=== Tournament Results ===")
    for i, bot in enumerate(population):
        print(f"{i+1}. {bot}")
    
    return population


def evolve(population_size: int = 8, generations: int = 10, games_per_matchup: int = 4):
    """Run evolutionary algorithm"""
    print(f"Starting evolution: {population_size} bots, {generations} generations, {games_per_matchup} games/matchup")
    
    # Create initial population (baseline + random variations)
    population = [BotParams()]  # Baseline
    for _ in range(population_size - 1):
        population.append(population[0].mutate(mutation_rate=0.5))
    
    best_overall = None
    
    for gen in range(generations):
        print(f"\n{'='*60}")
        print(f"GENERATION {gen + 1}/{generations}")
        print(f"{'='*60}")
        
        # Tournament
        population = tournament(population, games_per_matchup)
        
        # Track best
        if best_overall is None or population[0].fitness > best_overall.fitness:
            best_params = {k: v for k, v in asdict(population[0]).items() if k not in ['wins', 'losses', 'draws']}
            best_overall = BotParams(**best_params)
            best_overall.wins = population[0].wins
            best_overall.losses = population[0].losses
            best_overall.draws = population[0].draws
            print(f"\n🏆 NEW BEST BOT: {best_overall}")
        
        # Evolve next generation
        # Keep top 25%, breed them to fill 50%, random mutations for 25%
        survivors = population[:population_size // 4]
        
        next_gen = survivors.copy()
        
        # Breed
        while len(next_gen) < population_size * 3 // 4:
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)
            child = parent1.crossover(parent2)
            if random.random() < 0.3:  # Mutate some children
                child = child.mutate(mutation_rate=0.2)
            next_gen.append(child)
        
        # Random mutations
        while len(next_gen) < population_size:
            parent = random.choice(survivors)
            next_gen.append(parent.mutate(mutation_rate=0.5))
        
        population = next_gen
    
    print(f"\n\n{'='*60}")
    print("EVOLUTION COMPLETE!")
    print(f"{'='*60}")
    
    # Ensure we have a valid best bot
    if best_overall is None:
        best_overall = population[0]
    
    print(f"\nBest bot found: {best_overall}")
    print("\nParameters:")
    best_params_dict = asdict(best_overall)
    for k, v in best_params_dict.items():
        if k not in ['wins', 'losses', 'draws']:
            print(f"  {k}: {v}")
    
    return best_overall


if __name__ == '__main__':
    import sys
    
    pop_size = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    generations = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    games = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    
    best = evolve(population_size=pop_size, generations=generations, games_per_matchup=games)
    
    # Write best parameters back to steamroller0
    print("\nWriting best parameters to steamroller0/agent_logic.go...")
    write_params_to_file(best, 'steamroller0')
    print("✓ Done! Rebuild steamroller0 to use the new parameters.")
