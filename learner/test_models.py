import torch
import numpy as np
from rl_coach_agent import RLCoachEnvironment, RLCoachAgent

def detect_model_architecture(model_path):
    """Detect the architecture of a saved model"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        q_network_state = checkpoint['q_network_state_dict']
        
        # Extract layer sizes from the state dict
        layer_sizes = []
        for key, param in q_network_state.items():
            if 'weight' in key and 'network' in key:
                layer_sizes.append(param.shape[1])  # Input size of the layer
        
        # The first layer input size is the state size (4)
        # The hidden sizes are the output sizes of hidden layers
        hidden_sizes = layer_sizes[1:-1]  # Exclude first (input) and last (output) layers
        
        return hidden_sizes
    except Exception as e:
        print(f"Error detecting architecture: {e}")
        return None

def test_model(model_path, num_episodes=3):
    """Test a specific model"""
    print(f"\nTesting model: {model_path}")
    
    # Detect architecture
    hidden_sizes = detect_model_architecture(model_path)
    if hidden_sizes is None:
        print("Could not detect architecture, skipping...")
        return
    
    print(f"Detected architecture: hidden_sizes = {hidden_sizes}")
    
    # Create environment
    env = RLCoachEnvironment(width=800, height=600, render_game=True)
    
    # Create agent with detected architecture
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
            
            while steps < 600:  # 10 seconds
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
        
    except Exception as e:
        print(f"Error testing model: {e}")
    finally:
        env.close()

def main():
    """Test all available models"""
    import os
    
    # List of model files to test
    model_files = [
        "demo_rl_coach_model.pth",
        "rl_coach_final_model.pth", 
        "rl_coach_interrupted_model.pth",
        "rl_coach_model_episode_50.pth"
    ]
    
    print("Testing all available models...")
    
    for model_file in model_files:
        if os.path.exists(model_file):
            test_model(model_file)
        else:
            print(f"\nModel {model_file} not found, skipping...")

if __name__ == "__main__":
    main() 