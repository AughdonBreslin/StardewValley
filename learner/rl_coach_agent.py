import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import json
import os
from typing import Dict, List, Tuple, Any

from game import GameState, pygame

class RLCoachEnvironment:
    """Environment wrapper following RL Coach patterns"""
    
    def __init__(self, width=800, height=600, render_game=True):
        self.game = GameState(width, height, render_game)
        self.action_space = 2  # 0: no action, 1: click
        self.observation_space = 4  # 4 state variables
        
    def reset(self):
        """Reset the environment and return initial state"""
        self.game.reset_game()
        return self.get_state()
    
    def step(self, action):
        """Execute action and return (state, reward, done, info)"""
        # Set the action
        self.game.clicking = bool(action)
        
        # Run one game step
        self.game.handle_events()
        self.game.update()
        if self.game.render_game:
            self.game.render()
        
        # Calculate reward
        reward = 1.0 if self.game.contained else -0.1
        
        # Get next state
        next_state = self.get_state()
        
        # Check if episode is done (30 seconds or game closed)
        done = not self.game.running or pygame.time.get_ticks() > self.game.thirty_second_timer
        
        info = {
            'score': self.game.score,
            'contained': self.game.contained
        }
        
        return next_state, reward, done, info
    
    def get_state(self):
        """Get normalized state representation"""
        white_y = self.game.white_box['rect'].centery / self.game.screen_height
        red_y = self.game.red_box['rect'].centery / self.game.screen_height
        white_vel = self.game.white_box['y_speed'] / 8.0
        red_vel = self.game.red_box['y_speed'] / 4.0
        
        return np.array([white_y, red_y, white_vel, red_vel], dtype=np.float32)
    
    def close(self):
        """Close the environment"""
        pygame.quit()

class RLCoachNetwork(nn.Module):
    """Neural network following RL Coach architecture patterns"""
    
    def __init__(self, input_size, hidden_sizes, output_size):
        super(RLCoachNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)  # Regularization like RL Coach
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier initialization (RL Coach style)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class RLCoachAgent:
    """RL Coach-style agent with experience replay and target network"""
    
    def __init__(self, state_size, action_size, config=None):
        self.state_size = state_size
        self.action_size = action_size
        
        # Default configuration following RL Coach patterns
        self.config = config or {
            'learning_rate': 0.0001,
            'gamma': 0.99,
            'epsilon': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,
            'memory_size': 10000,
            'batch_size': 32,
            'target_update_freq': 1000,
            'hidden_sizes': [64, 64],
            'learning_starts': 1000
        }
        
        # Networks
        self.q_network = RLCoachNetwork(state_size, self.config['hidden_sizes'], action_size)
        self.target_network = RLCoachNetwork(state_size, self.config['hidden_sizes'], action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.MSELoss()
        
        # Experience replay
        self.memory = deque(maxlen=self.config['memory_size'])
        
        # Training state
        self.step_count = 0
        self.epsilon = self.config['epsilon']
        
    def act(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        """Train the agent using experience replay"""
        if len(self.memory) < self.config['batch_size'] or self.step_count < self.config['learning_starts']:
            return 0.0
        
        # Sample batch
        batch = random.sample(self.memory, self.config['batch_size'])
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values (using target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.config['gamma'] * next_q_values * ~dones)
        
        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()
        
        # Update target network
        if self.step_count % self.config['target_update_freq'] == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.config['epsilon_min']:
            self.epsilon *= self.config['epsilon_decay']
        
        return loss.item()
    
    def save(self, path):
        """Save the agent"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, path)
    
    def load(self, path):
        """Load the agent"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']

class RLCoachTrainer:
    """Training manager following RL Coach patterns"""
    
    def __init__(self, environment, agent, config=None):
        self.env = environment
        self.agent = agent
        
        self.config = config or {
            'max_episodes': 10000,
            'max_steps_per_episode': 1800,  # 30 seconds at 60 FPS
            'evaluation_freq': 100,
            'save_freq': 500,
            'log_freq': 10
        }
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        
    def train(self):
        """Main training loop"""
        print("Starting RL Coach training...")
        
        for episode in range(self.config['max_episodes']):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(self.config['max_steps_per_episode']):
                # Select action
                action = self.agent.act(state, training=True)
                
                # Execute action
                next_state, reward, done, info = self.env.step(action)
                
                # Store transition
                self.agent.store_transition(state, action, reward, next_state, done)
                
                # Train agent
                loss = self.agent.train()
                if loss > 0:
                    self.training_losses.append(loss)
                
                episode_reward += reward
                episode_length += 1
                self.agent.step_count += 1
                
                state = next_state
                
                if done:
                    break
            
            # Record episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Logging
            if episode % self.config['log_freq'] == 0:
                avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
                avg_loss = np.mean(self.training_losses[-100:]) if self.training_losses else 0
                print(f"Episode {episode}: Reward={episode_reward:.2f}, Avg Reward={avg_reward:.2f}, "
                      f"Epsilon={self.agent.epsilon:.3f}, Loss={avg_loss:.4f}")
            
            # Evaluation
            if episode % self.config['evaluation_freq'] == 0:
                self.evaluate(episode)
            
            # Save model
            if episode % self.config['save_freq'] == 0:
                self.agent.save(f"models/rl_coach_model_episode_{episode}.pth")
        
        # Final save
        self.agent.save("rl_coach_final_model.pth")
        self.save_training_stats()
        print("Training completed!")
    
    def evaluate(self, episode, num_episodes=5):
        """Evaluate the agent"""
        print(f"\nEvaluating agent at episode {episode}...")
        
        eval_rewards = []
        eval_scores = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_score = 0
            
            for step in range(self.config['max_steps_per_episode']):
                action = self.agent.act(state, training=False)
                next_state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_score = info['score']
                state = next_state
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_scores.append(episode_score)
        
        avg_reward = np.mean(eval_rewards)
        avg_score = np.mean(eval_scores)
        print(f"Evaluation: Avg Reward={avg_reward:.2f}, Avg Score={avg_score:.2f}")
    
    def save_training_stats(self):
        """Save training statistics"""
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'training_losses': self.training_losses,
            'config': self.config
        }
        
        with open('training_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

def main():
    """Main function to run RL Coach training"""
    # Create environment
    env = RLCoachEnvironment(width=800, height=600, render_game=True)
    
    # Create agent
    agent = RLCoachAgent(state_size=4, action_size=2)

    # Try to load previous model if it exists
    model_path = "rl_coach_final_model.pth"
    if os.path.exists(model_path):
        try:
            agent.load(model_path)
            print(f"Resumed training from saved model: {model_path}")
        except Exception as e:
            print(f"Failed to load previous model: {e}. Starting from scratch.")
    else:
        print("No previous model found. Starting training from scratch.")
    
    # Create trainer
    trainer = RLCoachTrainer(env, agent)
    
    try:
        # Start training
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        agent.save("rl_coach_interrupted_model.pth")
    finally:
        env.close()

if __name__ == "__main__":
    main() 