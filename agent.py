import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from game import GameState, pygame

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size),
            nn.Sigmoid()
        )
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        return 1 if act_values.item() > 0.5 else 0
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([t[0] for t in minibatch]))
        actions = torch.FloatTensor(np.array([t[1] for t in minibatch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch]))
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch]))
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch]))
        
        # Current Q values (probabilities)
        current_q = self.model(states).squeeze()
        
        # Target Q values
        with torch.no_grad():
            next_q = self.model(next_states).squeeze()
        targets = rewards + (1 - dones) * self.gamma * next_q
        
        # Action masking
        action_mask = (actions == 1).float()  # 1 for click, 0 for no-click
        
        # Only update Q-values for taken actions
        # For click (1): optimize predicted Q toward target
        # For no-click (0): optimize (1 - predicted Q) toward target
        loss = torch.mean(
            action_mask * (current_q - targets)**2 + 
            (1 - action_mask) * ((1 - current_q) - targets)**2
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def load(self, name):
        return self.model.load_state_dict(torch.load(name))

class NeuralNetworkPlayer(GameState):
    def __init__(self, width=800, height=600, train=True, render_game=True):
        super().__init__(width, height, render_game)
        self.agent = DQNAgent(4, 1)  # 4 inputs, 1 output
        self.train = train
        self.batch_size = 32
        self.episodes = 0
        self.max_episodes = 50000
        
    def get_state(self):
        # Normalize all values between 0-1
        white_y = self.white_box['rect'].centery / self.screen_height
        red_y = self.red_box['rect'].centery / self.screen_height
        white_vel = self.white_box['y_speed'] / 8.0  # Assuming max vel is 8
        red_vel = self.red_box['y_speed'] / 4.0
        
        return [white_y, red_y, white_vel, red_vel]
    
    def run(self):
        while self.running and self.episodes < self.max_episodes:
            state = self.get_state()
            total_reward = 0
            done = False
            
            while not done:
                action = self.agent.act(state)
                self.clicking = bool(action)
                
                # Run one game step
                self.handle_events()
                self.update()
                if self.render_game or self.episodes % 10 == 0:  # Render every 10 episodes
                    self.render()
                
                # Calculate reward
                reward = 1 if self.contained else -0.1
                total_reward += reward
                
                next_state = self.get_state()
                # End after 10 seconds
                done = not self.running or pygame.time.get_ticks() > self.ten_second_timer
                
                if self.train:
                    self.agent.remember(state, action, reward, next_state, done)
                    self.agent.replay(self.batch_size)
                
                state = next_state
            
            self.episodes += 1
            print(f"Episode: {self.episodes}, Score: {self.score}, Epsilon: {self.agent.epsilon:.2f}")

            self.reset_game()
        
        pygame.quit()
        self.agent.save("dqn_model.pth")

        

if __name__ == "__main__":
    # Set train=False to see the trained agent play
    game = NeuralNetworkPlayer(train=True, render_game=True)
    game.run()