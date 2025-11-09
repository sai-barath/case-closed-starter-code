"""
Master Training and Evaluation Pipeline

This script orchestrates the complete workflow:
1. Train DQN for specified episodes (200-500)
2. Save trained model
3. Evaluate against advanced heuristic agents
4. Generate comprehensive metrics

Usage:
    python train_and_evaluate.py [num_episodes] [eval_episodes]
    
Example:
    python train_and_evaluate.py 500 100
"""

import sys
import os
import time
from datetime import datetime


def main():
    # Parse arguments
    num_train_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    num_eval_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    print(f"\n{'='*70}")
    print(f"🚀 CASE CLOSED - DQN TRAINING & EVALUATION PIPELINE")
    print(f"{'='*70}")
    print(f"Training episodes: {num_train_episodes}")
    print(f"Evaluation episodes per opponent: {num_eval_episodes}")
    print(f"{'='*70}\n")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # ========================================================================
    # PHASE 1: TRAIN DQN
    # ========================================================================
    
    print(f"\n{'='*70}")
    print(f"PHASE 1: DQN TRAINING")
    print(f"{'='*70}\n")
    
    from agent2 import train_dqn_agent
    
    start_time = time.time()
    
    # Train with frequent saves and evaluations
    save_freq = 100  # Save every 100 episodes
    eval_freq = 100  # Evaluate every 100 episodes
    
    train_dqn_agent(
        num_episodes=num_train_episodes,
        save_freq=save_freq,
        eval_freq=eval_freq,
        save_dir="models"
    )
    
    training_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Time taken: {training_time/60:.1f} minutes")
    print(f"Model saved to: models/dqn_agent_final.pt")
    print(f"{'='*70}\n")
    
    # ========================================================================
    # PHASE 2: TEST HEURISTIC AGENTS (Baseline Performance)
    # ========================================================================
    
    print(f"\n{'='*70}")
    print(f"PHASE 2: HEURISTIC AGENT BASELINE")
    print(f"{'='*70}\n")
    
    from advanced_heuristic_agent import (
        balanced_heuristic_agent,
        aggressive_heuristic_agent,
        defensive_heuristic_agent
    )
    from training_env import evaluate_agents, wall_avoider_agent, greedy_space_agent
    
    print("Testing heuristic agents to establish baseline performance...\n")
    
    print("[1/5] Balanced Heuristic vs Wall Avoider...")
    stats1 = evaluate_agents(balanced_heuristic_agent, wall_avoider_agent,
                            num_episodes=50, verbose=False)
    print(f"  Result: {stats1['agent1_win_rate']:.1%} win rate\n")
    
    print("[2/5] Balanced Heuristic vs Greedy Space...")
    stats2 = evaluate_agents(balanced_heuristic_agent, greedy_space_agent,
                            num_episodes=50, verbose=False)
    print(f"  Result: {stats2['agent1_win_rate']:.1%} win rate\n")
    
    print("[3/5] Aggressive Heuristic vs Balanced Heuristic...")
    stats3 = evaluate_agents(aggressive_heuristic_agent, balanced_heuristic_agent,
                            num_episodes=50, verbose=False)
    print(f"  Result: {stats3['agent1_win_rate']:.1%} win rate\n")
    
    print("[4/5] Defensive Heuristic vs Balanced Heuristic...")
    stats4 = evaluate_agents(defensive_heuristic_agent, balanced_heuristic_agent,
                            num_episodes=50, verbose=False)
    print(f"  Result: {stats4['agent1_win_rate']:.1%} win rate\n")
    
    print("[5/5] Aggressive vs Defensive...")
    stats5 = evaluate_agents(aggressive_heuristic_agent, defensive_heuristic_agent,
                            num_episodes=50, verbose=False)
    print(f"  Result: {stats5['agent1_win_rate']:.1%} win rate\n")
    
    print(f"{'='*70}")
    print(f"HEURISTIC BASELINE SUMMARY:")
    print(f"{'='*70}")
    print(f"Balanced vs Wall Avoider:   {stats1['agent1_win_rate']:6.1%}")
    print(f"Balanced vs Greedy Space:   {stats2['agent1_win_rate']:6.1%}")
    print(f"Aggressive vs Balanced:     {stats3['agent1_win_rate']:6.1%}")
    print(f"Defensive vs Balanced:      {stats4['agent1_win_rate']:6.1%}")
    print(f"Aggressive vs Defensive:    {stats5['agent1_win_rate']:6.1%}")
    print(f"{'='*70}\n")
    
    # ========================================================================
    # PHASE 3: COMPREHENSIVE DQN EVALUATION
    # ========================================================================
    
    print(f"\n{'='*70}")
    print(f"PHASE 3: COMPREHENSIVE DQN EVALUATION")
    print(f"{'='*70}\n")
    
    from agent2 import create_inference_agent
    from evaluate_dqn_vs_heuristic import run_tournament
    
    # Load trained DQN
    print("Loading trained DQN model...\n")
    dqn_policy = create_inference_agent("models/dqn_agent_final.pt")
    
    # Define tournament opponents
    heuristic_opponents = {
        'Advanced Heuristic (Balanced)': balanced_heuristic_agent,
        'Advanced Heuristic (Aggressive)': aggressive_heuristic_agent,
        'Advanced Heuristic (Defensive)': defensive_heuristic_agent,
        'Greedy Space Agent': greedy_space_agent,
        'Wall Avoider Agent': wall_avoider_agent,
    }
    
    # Run tournament
    tournament_results = run_tournament(
        dqn_policy,
        heuristic_opponents,
        num_episodes=num_eval_episodes,
        save_results=True
    )
    
    # ========================================================================
    # PHASE 4: FINAL SUMMARY
    # ========================================================================
    
    print(f"\n{'='*70}")
    print(f"🎉 PIPELINE COMPLETE - FINAL SUMMARY")
    print(f"{'='*70}\n")
    
    total_time = time.time() - start_time
    
    print(f"📊 Execution Summary:")
    print(f"  Training time:        {training_time/60:6.1f} minutes")
    print(f"  Evaluation time:      {(total_time-training_time)/60:6.1f} minutes")
    print(f"  Total time:           {total_time/60:6.1f} minutes")
    print(f"  Training episodes:    {num_train_episodes}")
    print(f"  Evaluation episodes:  {num_eval_episodes * len(heuristic_opponents)}")
    
    print(f"\n🏆 DQN Performance:")
    total_wins = sum(r['agent1_wins'] for r in tournament_results)
    total_games = sum(r['num_episodes'] for r in tournament_results)
    overall_win_rate = total_wins / total_games
    
    print(f"  Overall win rate: {overall_win_rate:.1%}")
    
    if overall_win_rate >= 0.75:
        print(f"  🌟 EXCELLENT - DQN is championship-level!")
    elif overall_win_rate >= 0.60:
        print(f"  ✅ GOOD - DQN is competitive!")
    elif overall_win_rate >= 0.50:
        print(f"  ⚠️  FAIR - DQN needs more training")
    else:
        print(f"  ❌ POOR - DQN needs significant improvement")
    
    print(f"\n📁 Files Generated:")
    print(f"  • models/dqn_agent_final.pt - Trained model")
    print(f"  • tournament_results_*.json - Detailed metrics")
    
    print(f"\n{'='*70}")
    print(f"✅ All phases complete! Check tournament_results_*.json for details.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
