import torch
import numpy as np
import matplotlib.pyplot as plt
from rl_coach_agent import RLCoachEnvironment, RLCoachAgent, RLCoachTrainer

def demo_training():
    """Demo with shorter training session"""
    print("Starting RL Coach Demo...")
    
    # Create environment
    env = RLCoachEnvironment(width=800, height=600, render_game=True)
    
    # Create agent with demo config
    demo_config = {
        'learning_rate': 0.001,
        'gamma': 0.95,
        'epsilon': 1.0,
        'epsilon_min': 0.05,
        'epsilon_decay': 0.99,
        'memory_size': 5000,
        'batch_size': 32,
        'target_update_freq': 500,
        'hidden_sizes': [32, 32],
        'learning_starts': 100
    }
    
    agent = RLCoachAgent(state_size=4, action_size=2, config=demo_config)
    
    # Create trainer with demo config
    trainer_config = {
        'max_episodes': 100,  # Shorter for demo
        'max_steps_per_episode': 1800,  # 30 seconds at 60 FPS
        'evaluation_freq': 20,
        'save_freq': 50,
        'log_freq': 5
    }
    
    trainer = RLCoachTrainer(env, agent, config=trainer_config)
    
    try:
        # Start training
        trainer.train()
        
        # Plot results
        plot_training_results(trainer)
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        agent.save("models/demo_rl_coach_model.pth")
    finally:
        env.close()

def plot_training_results(trainer):
    """Plot training results"""
    try:
        plt.figure(figsize=(15, 5))
        
        # Plot rewards
        plt.subplot(1, 3, 1)
        plt.plot(trainer.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Plot moving average
        if len(trainer.episode_rewards) >= 10:
            moving_avg = np.convolve(trainer.episode_rewards, np.ones(10)/10, mode='valid')
            plt.plot(range(9, len(trainer.episode_rewards)), moving_avg, 'r-', label='Moving Average')
            plt.legend()
        
        # Plot episode lengths
        plt.subplot(1, 3, 2)
        plt.plot(trainer.episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        
        # Plot training losses
        plt.subplot(1, 3, 3)
        if trainer.training_losses:
            plt.plot(trainer.training_losses)
            plt.title('Training Losses')
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('training_results.png')
        plt.show()
        
        print("Training results plotted and saved as 'training_results.png'")
        
    except ImportError:
        print("Matplotlib not available, skipping plots")

def test_trained_agent(model_path="models/demo_rl_coach_model.pth"):
    """Test a trained agent"""
    try:
        # Create environment
        env = RLCoachEnvironment(width=800, height=600, render_game=True)
        
        # Create agent
        agent = RLCoachAgent(state_size=4, action_size=2)
        
        # Load trained model
        agent.load(model_path)
        print(f"Loaded trained model from {model_path}")
        
        # Test for a few episodes
        for episode in range(3):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            while steps < 1800:  # 30 seconds
                action = agent.act(state, training=False)
                next_state, reward, done, info = env.step(action)
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            print(f"Test Episode {episode + 1}: Reward={total_reward:.2f}, Score={info['score']}, Steps={steps}")
        
        env.close()
        
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train the agent first.")
    except Exception as e:
        print(f"Error testing agent: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_trained_agent()
    else:
        demo_training() 