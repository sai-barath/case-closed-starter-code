"""
Test script to verify DQN training and inference pipeline.

This script tests:
1. DQN model architecture
2. State tensor conversion
3. Training for a few episodes
4. Model saving and loading
5. Inference with loaded model
"""

import torch
import numpy as np
from agent2 import DQNAgent, train_dqn_agent
from dqn_inference import DQNInference
from training_env import run_episode, evaluate_agents, wall_avoider_agent
import os


def test_model_architecture():
    """Test that DQN model can be instantiated and run forward pass."""
    print("\n" + "="*70)
    print("TEST 1: Model Architecture")
    print("="*70)
    
    from agent2 import DQNetwork
    
    model = DQNetwork()
    
    # Create dummy input
    batch_size = 4
    spatial = torch.randn(batch_size, 7, 18, 20)
    extra = torch.randn(batch_size, 8)
    
    # Forward pass
    output = model(spatial, extra)
    
    print(f"✓ Model instantiated successfully")
    print(f"✓ Input shape: spatial={spatial.shape}, extra={extra.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Expected output shape: ({batch_size}, 4)")
    
    assert output.shape == (batch_size, 4), "Output shape mismatch!"
    print(f"✓ Forward pass successful!")
    
    return True


def test_state_conversion():
    """Test state dictionary to tensor conversion."""
    print("\n" + "="*70)
    print("TEST 2: State Conversion")
    print("="*70)
    
    from training_env import TrainingEnvironment
    
    env = TrainingEnvironment()
    state1, state2 = env.reset()
    
    agent = DQNAgent()
    spatial, extra = agent.state_to_tensor(state1)
    
    print(f"✓ State dict converted to tensors")
    print(f"✓ Spatial tensor shape: {spatial.shape}")
    print(f"✓ Extra features shape: {extra.shape}")
    print(f"✓ Expected: spatial=(1, 7, 18, 20), extra=(1, 8)")
    
    assert spatial.shape == (1, 7, 18, 20), "Spatial tensor shape mismatch!"
    assert extra.shape == (1, 8), "Extra features shape mismatch!"
    
    print(f"✓ State conversion successful!")
    
    return True


def test_action_selection():
    """Test that agent can select valid actions."""
    print("\n" + "="*70)
    print("TEST 3: Action Selection")
    print("="*70)
    
    from training_env import TrainingEnvironment
    
    env = TrainingEnvironment()
    state1, state2 = env.reset()
    
    agent = DQNAgent()
    
    # Test action selection
    action = agent.select_action(state1, training=True)
    print(f"✓ Selected action (training mode): {action}")
    
    action = agent.select_action(state1, training=False)
    print(f"✓ Selected action (inference mode): {action}")
    
    # Verify action is valid
    valid_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    assert action in valid_actions, f"Invalid action: {action}"
    
    print(f"✓ Action selection successful!")
    
    return True


def test_training_episode():
    """Test training for a few episodes."""
    print("\n" + "="*70)
    print("TEST 4: Training Episode")
    print("="*70)
    
    from training_env import TrainingEnvironment
    
    agent = DQNAgent()
    
    print("Training for 5 episodes...")
    
    for episode in range(5):
        env = TrainingEnvironment()
        state1, state2 = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 100:
            action1 = agent.select_action(state1, training=True)
            action2 = agent.select_action(state2, training=True)
            
            next_state1, next_state2, reward1, reward2, done, result = env.step(action1, action2)
            
            agent.replay_buffer.push(state1, action1, reward1, next_state1, done)
            
            if len(agent.replay_buffer) >= agent.batch_size:
                loss = agent.train_step()
            
            state1 = next_state1
            state2 = next_state2
            total_reward += reward1
            steps += 1
        
        print(f"  Episode {episode + 1}: Reward={total_reward:.1f}, Steps={steps}, Result={result}")
    
    print(f"✓ Training episodes successful!")
    print(f"✓ Replay buffer size: {len(agent.replay_buffer)}")
    
    return True


def test_model_save_load():
    """Test saving and loading model."""
    print("\n" + "="*70)
    print("TEST 5: Model Save/Load")
    print("="*70)
    
    # Create and train agent
    agent1 = DQNAgent()
    
    # Create test directory
    os.makedirs("test_models", exist_ok=True)
    
    # Save model
    save_path = "test_models/test_model.pt"
    agent1.save_model(save_path, metadata={'test': 'data'})
    print(f"✓ Model saved to {save_path}")
    
    # Load model
    agent2 = DQNAgent()
    agent2.load_model(save_path)
    print(f"✓ Model loaded from {save_path}")
    
    # Verify weights match
    for p1, p2 in zip(agent1.policy_net.parameters(), agent2.policy_net.parameters()):
        assert torch.allclose(p1, p2), "Loaded weights don't match!"
    
    print(f"✓ Weights match!")
    
    # Clean up
    os.remove(save_path)
    os.rmdir("test_models")
    
    print(f"✓ Model save/load successful!")
    
    return True


def test_inference_module():
    """Test inference module."""
    print("\n" + "="*70)
    print("TEST 6: Inference Module")
    print("="*70)
    
    # Create and save a model
    os.makedirs("test_models", exist_ok=True)
    agent = DQNAgent()
    save_path = "test_models/inference_test.pt"
    agent.save_model(save_path)
    
    # Load with inference module
    inference = DQNInference(save_path)
    print(f"✓ Inference module loaded model")
    
    # Test with training environment state
    from training_env import TrainingEnvironment
    env = TrainingEnvironment()
    state, _ = env.reset()
    
    # Get move
    move = inference.get_move(state)
    print(f"✓ Inference module returned move: {move}")
    
    # Verify move is valid
    valid_moves = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'UP:BOOST', 'DOWN:BOOST', 'LEFT:BOOST', 'RIGHT:BOOST']
    assert any(move == vm or move.startswith(vm) for vm in valid_moves[:4]), f"Invalid move: {move}"
    
    # Clean up
    os.remove(save_path)
    os.rmdir("test_models")
    
    print(f"✓ Inference module successful!")
    
    return True


def test_full_game():
    """Test playing a full game with DQN agent."""
    print("\n" + "="*70)
    print("TEST 7: Full Game Playthrough")
    print("="*70)
    
    # Create agent
    agent = DQNAgent()
    
    # Create policy function
    def dqn_policy(state):
        return agent.select_action(state, training=False)
    
    # Play game against wall avoider
    print("Playing DQN vs Wall Avoider...")
    result, history = run_episode(dqn_policy, wall_avoider_agent, render=False)
    
    print(f"✓ Game completed!")
    print(f"  Result: {result}")
    print(f"  Game length: {len(history)} steps")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("DQN SYSTEM TEST SUITE")
    print("="*70)
    
    tests = [
        ("Model Architecture", test_model_architecture),
        ("State Conversion", test_state_conversion),
        ("Action Selection", test_action_selection),
        ("Training Episode", test_training_episode),
        ("Model Save/Load", test_model_save_load),
        ("Inference Module", test_inference_module),
        ("Full Game", test_full_game),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 All tests passed! DQN system is ready for training.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review errors above.")
    
    print("="*70)


if __name__ == "__main__":
    run_all_tests()
