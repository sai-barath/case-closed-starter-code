"""
Comprehensive DQN vs Heuristic Evaluation Suite

This script provides detailed metrics for comparing DQN agents against
advanced heuristic agents, including:
- Win rates and game statistics
- Move quality analysis
- Spatial control metrics
- Performance timing
- Head-to-head matchups with multiple configurations
"""

import time
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import json
from datetime import datetime

from training_env import TrainingEnvironment, run_episode, evaluate_agents
from advanced_heuristic_agent import (
    aggressive_heuristic_agent,
    defensive_heuristic_agent,
    balanced_heuristic_agent,
    create_advanced_heuristic_policy
)


def evaluate_with_detailed_metrics(agent1_policy, agent2_policy,
                                   agent1_name: str, agent2_name: str,
                                   num_episodes: int = 100) -> Dict:
    """
    Evaluate two agents with comprehensive metrics.
    
    Args:
        agent1_policy: Policy function for agent 1
        agent2_policy: Policy function for agent 2
        agent1_name: Name of agent 1
        agent2_name: Name of agent 2
        num_episodes: Number of games to play
        
    Returns:
        Dictionary with detailed statistics
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {agent1_name} vs {agent2_name}")
    print(f"{'='*70}")
    
    # Track statistics
    stats = {
        'agent1_name': agent1_name,
        'agent2_name': agent2_name,
        'num_episodes': num_episodes,
        'agent1_wins': 0,
        'agent2_wins': 0,
        'draws': 0,
        'game_lengths': [],
        'agent1_trail_lengths': [],
        'agent2_trail_lengths': [],
        'agent1_decision_times': [],
        'agent2_decision_times': [],
        'agent1_boost_usage': [],
        'agent2_boost_usage': [],
        'spatial_control': {
            'agent1_avg': 0.0,
            'agent2_avg': 0.0,
        },
        'crash_types': defaultdict(int),
    }
    
    # Run episodes
    for episode in range(num_episodes):
        env = TrainingEnvironment()
        state1, state2 = env.reset()
        
        done = False
        steps = 0
        
        episode_a1_time = []
        episode_a2_time = []
        
        while not done:
            # Agent 1 move with timing
            start = time.time()
            move1 = agent1_policy(state1)
            episode_a1_time.append(time.time() - start)
            
            # Agent 2 move with timing
            start = time.time()
            move2 = agent2_policy(state2)
            episode_a2_time.append(time.time() - start)
            
            # Execute step
            state1, state2, reward1, reward2, done, result = env.step(move1, move2)
            steps += 1
        
        # Record game statistics
        stats['game_lengths'].append(steps)
        stats['agent1_trail_lengths'].append(len(state1['my_trail']))
        stats['agent2_trail_lengths'].append(len(state2['my_trail']))
        stats['agent1_decision_times'].extend(episode_a1_time)
        stats['agent2_decision_times'].extend(episode_a2_time)
        
        # Count boosts used
        initial_boosts = 3
        boosts_used_1 = initial_boosts - state1.get('my_boosts', 0)
        boosts_used_2 = initial_boosts - state2.get('opponent_boosts', 0)
        stats['agent1_boost_usage'].append(boosts_used_1)
        stats['agent2_boost_usage'].append(boosts_used_2)
        
        # Record result
        from case_closed_game import GameResult
        if result == GameResult.AGENT1_WIN:
            stats['agent1_wins'] += 1
            stats['crash_types']['agent2_crashed'] += 1
        elif result == GameResult.AGENT2_WIN:
            stats['agent2_wins'] += 1
            stats['crash_types']['agent1_crashed'] += 1
        else:
            stats['draws'] += 1
            stats['crash_types']['both_crashed'] += 1
        
        # Progress indicator
        if (episode + 1) % 10 == 0:
            print(f"  Progress: {episode + 1}/{num_episodes} episodes completed")
    
    # Calculate aggregate statistics
    stats['agent1_win_rate'] = stats['agent1_wins'] / num_episodes
    stats['agent2_win_rate'] = stats['agent2_wins'] / num_episodes
    stats['draw_rate'] = stats['draws'] / num_episodes
    
    stats['avg_game_length'] = np.mean(stats['game_lengths'])
    stats['std_game_length'] = np.std(stats['game_lengths'])
    stats['min_game_length'] = np.min(stats['game_lengths'])
    stats['max_game_length'] = np.max(stats['game_lengths'])
    
    stats['agent1_avg_trail_length'] = np.mean(stats['agent1_trail_lengths'])
    stats['agent2_avg_trail_length'] = np.mean(stats['agent2_trail_lengths'])
    
    stats['agent1_avg_decision_time_ms'] = np.mean(stats['agent1_decision_times']) * 1000
    stats['agent2_avg_decision_time_ms'] = np.mean(stats['agent2_decision_times']) * 1000
    stats['agent1_max_decision_time_ms'] = np.max(stats['agent1_decision_times']) * 1000
    stats['agent2_max_decision_time_ms'] = np.max(stats['agent2_decision_times']) * 1000
    
    stats['agent1_avg_boost_usage'] = np.mean(stats['agent1_boost_usage'])
    stats['agent2_avg_boost_usage'] = np.mean(stats['agent2_boost_usage'])
    
    # Calculate spatial control (trail length as proxy)
    total_trail = np.array(stats['agent1_trail_lengths']) + np.array(stats['agent2_trail_lengths'])
    agent1_control = np.array(stats['agent1_trail_lengths']) / total_trail
    agent2_control = np.array(stats['agent2_trail_lengths']) / total_trail
    
    stats['spatial_control']['agent1_avg'] = np.mean(agent1_control)
    stats['spatial_control']['agent2_avg'] = np.mean(agent2_control)
    
    return stats


def print_detailed_report(stats: Dict):
    """Print a comprehensive report of the evaluation."""
    print(f"\n{'='*70}")
    print(f"DETAILED EVALUATION REPORT")
    print(f"{'='*70}")
    print(f"Matchup: {stats['agent1_name']} vs {stats['agent2_name']}")
    print(f"Episodes: {stats['num_episodes']}")
    print(f"{'='*70}")
    
    print(f"\n📊 WIN RATES:")
    print(f"  {stats['agent1_name']:30s}: {stats['agent1_wins']:3d} wins ({stats['agent1_win_rate']:6.2%})")
    print(f"  {stats['agent2_name']:30s}: {stats['agent2_wins']:3d} wins ({stats['agent2_win_rate']:6.2%})")
    print(f"  {'Draws':30s}: {stats['draws']:3d}      ({stats['draw_rate']:6.2%})")
    
    print(f"\n📏 GAME LENGTH STATISTICS:")
    print(f"  Average game length:  {stats['avg_game_length']:6.1f} steps")
    print(f"  Std deviation:        {stats['std_game_length']:6.1f} steps")
    print(f"  Shortest game:        {stats['min_game_length']:6d} steps")
    print(f"  Longest game:         {stats['max_game_length']:6d} steps")
    
    print(f"\n🗺️  SPATIAL CONTROL (Trail Coverage):")
    print(f"  {stats['agent1_name']:30s}: {stats['agent1_avg_trail_length']:6.1f} avg cells")
    print(f"  {stats['agent2_name']:30s}: {stats['agent2_avg_trail_length']:6.1f} avg cells")
    print(f"  {stats['agent1_name']:30s}: {stats['spatial_control']['agent1_avg']:6.2%} territory")
    print(f"  {stats['agent2_name']:30s}: {stats['spatial_control']['agent2_avg']:6.2%} territory")
    
    print(f"\n⚡ BOOST USAGE:")
    print(f"  {stats['agent1_name']:30s}: {stats['agent1_avg_boost_usage']:4.2f} avg boosts/game")
    print(f"  {stats['agent2_name']:30s}: {stats['agent2_avg_boost_usage']:4.2f} avg boosts/game")
    
    print(f"\n⏱️  DECISION TIME (Performance):")
    print(f"  {stats['agent1_name']:30s}: {stats['agent1_avg_decision_time_ms']:7.3f} ms avg")
    print(f"  {stats['agent2_name']:30s}: {stats['agent2_avg_decision_time_ms']:7.3f} ms avg")
    print(f"  {stats['agent1_name']:30s}: {stats['agent1_max_decision_time_ms']:7.3f} ms max")
    print(f"  {stats['agent2_name']:30s}: {stats['agent2_max_decision_time_ms']:7.3f} ms max")
    
    print(f"\n💥 CRASH ANALYSIS:")
    for crash_type, count in stats['crash_types'].items():
        print(f"  {crash_type:30s}: {count:3d} times ({count/stats['num_episodes']:6.2%})")
    
    print(f"\n{'='*70}")
    
    # Determine winner
    if stats['agent1_win_rate'] > stats['agent2_win_rate']:
        margin = stats['agent1_win_rate'] - stats['agent2_win_rate']
        print(f"🏆 WINNER: {stats['agent1_name']} (by {margin:.1%})")
    elif stats['agent2_win_rate'] > stats['agent1_win_rate']:
        margin = stats['agent2_win_rate'] - stats['agent1_win_rate']
        print(f"🏆 WINNER: {stats['agent2_name']} (by {margin:.1%})")
    else:
        print(f"🤝 TIE: Both agents equally matched")
    
    print(f"{'='*70}\n")


def run_tournament(dqn_policy, heuristic_agents: Dict[str, callable],
                   num_episodes: int = 100, save_results: bool = True):
    """
    Run a tournament with DQN against multiple heuristic agents.
    
    Args:
        dqn_policy: DQN policy function
        heuristic_agents: Dict of {name: policy_function}
        num_episodes: Episodes per matchup
        save_results: Whether to save results to JSON
    """
    print(f"\n{'='*70}")
    print(f"🏆 DQN TOURNAMENT - {len(heuristic_agents)} OPPONENTS")
    print(f"{'='*70}")
    print(f"Episodes per matchup: {num_episodes}")
    print(f"Total games: {num_episodes * len(heuristic_agents)}")
    print(f"{'='*70}\n")
    
    tournament_results = []
    
    for idx, (name, policy) in enumerate(heuristic_agents.items(), 1):
        print(f"\n[Matchup {idx}/{len(heuristic_agents)}]")
        
        stats = evaluate_with_detailed_metrics(
            dqn_policy, policy,
            "DQN Agent", name,
            num_episodes=num_episodes
        )
        
        print_detailed_report(stats)
        tournament_results.append(stats)
    
    # Tournament summary
    print(f"\n{'='*70}")
    print(f"🏆 TOURNAMENT SUMMARY")
    print(f"{'='*70}\n")
    
    total_wins = sum(s['agent1_wins'] for s in tournament_results)
    total_losses = sum(s['agent2_wins'] for s in tournament_results)
    total_draws = sum(s['draws'] for s in tournament_results)
    total_games = sum(s['num_episodes'] for s in tournament_results)
    
    print(f"Overall DQN Performance:")
    print(f"  Total Wins:   {total_wins:4d} / {total_games} ({total_wins/total_games:6.2%})")
    print(f"  Total Losses: {total_losses:4d} / {total_games} ({total_losses/total_games:6.2%})")
    print(f"  Total Draws:  {total_draws:4d} / {total_games} ({total_draws/total_games:6.2%})")
    
    print(f"\nMatchup Breakdown:")
    for stats in tournament_results:
        print(f"  vs {stats['agent2_name']:25s}: {stats['agent1_win_rate']:6.2%} win rate")
    
    print(f"\n{'='*70}\n")
    
    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tournament_results_{timestamp}.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, defaultdict):
                return dict(obj)
            return obj
        
        results_json = {
            'tournament_summary': {
                'total_games': total_games,
                'dqn_wins': total_wins,
                'dqn_losses': total_losses,
                'draws': total_draws,
                'win_rate': total_wins / total_games,
            },
            'matchups': []
        }
        
        for stats in tournament_results:
            matchup_data = {}
            for key, value in stats.items():
                matchup_data[key] = convert_to_serializable(value)
            results_json['matchups'].append(matchup_data)
        
        with open(filename, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"📁 Results saved to: {filename}\n")
    
    return tournament_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate_dqn_vs_heuristic.py <model_path> [num_episodes]")
        print("Example: python evaluate_dqn_vs_heuristic.py models/dqn_agent_final.pt 100")
        sys.exit(1)
    
    model_path = sys.argv[1]
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    # Load DQN model
    print(f"\n{'='*70}")
    print(f"Loading DQN model from: {model_path}")
    print(f"{'='*70}\n")
    
    from agent2 import create_inference_agent
    dqn_policy = create_inference_agent(model_path)
    
    # Define heuristic opponents
    heuristic_agents = {
        'Advanced Heuristic (Balanced)': balanced_heuristic_agent,
        'Advanced Heuristic (Aggressive)': aggressive_heuristic_agent,
        'Advanced Heuristic (Defensive)': defensive_heuristic_agent,
    }
    
    # Run tournament
    results = run_tournament(dqn_policy, heuristic_agents, num_episodes=num_episodes)
    
    print(f"\n🎉 Tournament complete!")
