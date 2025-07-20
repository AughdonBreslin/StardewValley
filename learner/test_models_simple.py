import torch
import numpy as np
from rl_coach_agent import RLCoachEnvironment, RLCoachAgent

def test_model_with_architecture(model_path, hidden_sizes, num_episodes=3):
    """Test a specific model with known architecture"""
    print(f"\nTesting model: {model_path}")
    print(f"Using architecture: hidden_sizes = {hidden_sizes}")
    
    # Create environment
    env = RLCoachEnvironment(width=800, height=600, render_game=True)
    
    # Create agent with specified architecture
    config = {
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'epsilon': 0.01,  # Low epsilon for testing
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'memory_size': 10000,
        'batch_size': 32,
        'target_update_freq': 1000,
        'hidden_sizes': hidden_sizes,
        'learning_starts': 1000
    }
    
    agent = RLCoachAgent(state_size=4, action_size=2, config=config)
    
    try:
        # Load the model
        agent.load(model_path)
        print("Model loaded successfully!")
        
        # Test episodes
        total_rewards = []
        total_scores = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            steps = 0
            
            while steps < 1800:  # 30 seconds
                action = agent.act(state, training=False)
                next_state, reward, done, info = env.step(action)
                
                episode_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
            total_scores.append(info['score'])
            
            print(f"  Episode {episode + 1}: Reward={episode_reward:.2f}, Score={info['score']}, Steps={steps}")
        
        avg_reward = np.mean(total_rewards)
        avg_score = np.mean(total_scores)
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average Score: {avg_score:.2f}")
        
        return avg_reward, avg_score
        
    except Exception as e:
        print(f"Error testing model: {e}")
        return None, None
    finally:
        env.close()

def main():
    """Test models with known architectures"""
    import os
    
    # Define models and their architectures
    models_to_test = [
        ("demo_rl_coach_model.pth", [32, 32]),  # Demo model
        ("rl_coach_final_model.pth", [32, 32]),  # Final model (same as demo)
        ("rl_coach_interrupted_model.pth", [64, 64]),  # Interrupted model (larger)
        ("rl_coach_model_episode_50.pth", [32, 32]),  # Episode 50 model
    ]
    
    print("Testing models with known architectures...")
    
    results = []
    
    for model_file, hidden_sizes in models_to_test:
        if os.path.exists(model_file):
            avg_reward, avg_score = test_model_with_architecture(model_file, hidden_sizes)
            if avg_reward is not None:
                results.append((model_file, avg_reward, avg_score))
        else:
            print(f"\nModel {model_file} not found, skipping...")
    
    # Summary
    if results:
        print("\n" + "="*50)
        print("SUMMARY OF RESULTS:")
        print("="*50)
        for model_file, avg_reward, avg_score in results:
            print(f"{model_file}:")
            print(f"  Average Reward: {avg_reward:.2f}")
            print(f"  Average Score: {avg_score:.2f}")
        
        # Find best model
        best_model = max(results, key=lambda x: x[1])  # Best by reward
        print(f"\nBest performing model: {best_model[0]}")
        print(f"Best average reward: {best_model[1]:.2f}")

if __name__ == "__main__":
    main() 