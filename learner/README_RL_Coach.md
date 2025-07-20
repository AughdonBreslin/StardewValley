# RL Coach Implementation for Box Containment Game

This implementation provides a reinforcement learning agent using RL Coach architecture patterns for the Box Containment Game. The agent learns to control the white container box to keep the red box inside it.

## Features

### RL Coach Architecture Components

1. **RLCoachEnvironment**: Environment wrapper that follows RL Coach patterns
   - Standard `reset()`, `step()`, `close()` interface
   - Normalized state representation
   - Reward shaping for containment behavior

2. **RLCoachNetwork**: Neural network with RL Coach-style architecture
   - Xavier weight initialization
   - Dropout regularization
   - Configurable hidden layers

3. **RLCoachAgent**: DQN agent with RL Coach features
   - Experience replay buffer
   - Target network for stable training
   - Epsilon-greedy exploration
   - Gradient clipping
   - Configurable hyperparameters

4. **RLCoachTrainer**: Training manager with RL Coach patterns
   - Episode-based training loop
   - Regular evaluation
   - Training statistics tracking
   - Model checkpointing

## Key RL Coach Features Implemented

- **Experience Replay**: Stores transitions in a replay buffer for stable learning
- **Target Network**: Separate network for computing target Q-values
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation
- **Gradient Clipping**: Prevents exploding gradients
- **Regularization**: Dropout layers for better generalization
- **Configurable Architecture**: Easy to modify network structure and hyperparameters

## Usage

### Quick Demo
```bash
python demo_rl_coach.py
```

### Full Training
```bash
python rl_coach_agent.py
```

### Test Trained Agent
```bash
python demo_rl_coach.py test
```

## Configuration

The agent can be configured through the `config` parameter:

```python
config = {
    'learning_rate': 0.0001,      # Learning rate for Adam optimizer
    'gamma': 0.99,                # Discount factor
    'epsilon': 1.0,               # Initial exploration rate
    'epsilon_min': 0.01,          # Minimum exploration rate
    'epsilon_decay': 0.995,       # Exploration decay rate
    'memory_size': 10000,         # Replay buffer size
    'batch_size': 32,             # Training batch size
    'target_update_freq': 1000,   # Target network update frequency
    'hidden_sizes': [64, 64],     # Hidden layer sizes
    'learning_starts': 1000       # Steps before learning begins
}
```

## State Representation

The agent receives a 4-dimensional state vector:
- `white_y`: Normalized Y position of white container (0-1)
- `red_y`: Normalized Y position of red box (0-1)
- `white_vel`: Normalized Y velocity of white container (-1 to 1)
- `red_vel`: Normalized Y velocity of red box (-1 to 1)

## Action Space

- `0`: No action (let gravity pull down)
- `1`: Click (apply upward force)

## Reward Function

- `+1.0`: Red box is contained within white box
- `-0.1`: Red box is not contained (small penalty for exploration)

## Training Process

1. **Exploration Phase**: Agent starts with high epsilon (1.0) for maximum exploration
2. **Learning Phase**: After `learning_starts` steps, agent begins training from replay buffer
3. **Exploitation Phase**: Epsilon decays over time, agent relies more on learned policy
4. **Evaluation**: Regular evaluation episodes to monitor performance
5. **Checkpointing**: Model saved periodically during training

## Output Files

- `rl_coach_final_model.pth`: Final trained model
- `training_stats.json`: Training statistics and metrics
- `training_results.png`: Training plots (if matplotlib available)

## Comparison with Original DQN

| Feature | Original DQN | RL Coach Implementation |
|---------|-------------|------------------------|
| Network Architecture | Simple 3-layer | Configurable with regularization |
| Experience Replay | Basic | Advanced with configurable size |
| Target Network | No | Yes, with periodic updates |
| Gradient Clipping | No | Yes |
| Weight Initialization | Default | Xavier initialization |
| Regularization | No | Dropout layers |
| Configuration | Hardcoded | Fully configurable |
| Training Monitoring | Basic | Comprehensive logging |
| Evaluation | No | Regular evaluation episodes |

## Performance Tips

1. **Start with demo**: Use `demo_rl_coach.py` for quick testing
2. **Adjust learning rate**: Lower learning rates (0.0001) for more stable training
3. **Increase replay buffer**: Larger buffers (10000+) for better sample diversity
4. **Monitor epsilon**: Ensure exploration decays appropriately
5. **Use target network**: Helps with training stability

## Troubleshooting

- **Slow learning**: Try increasing learning rate or batch size
- **Unstable training**: Reduce learning rate or increase target update frequency
- **Poor exploration**: Adjust epsilon decay rate
- **Memory issues**: Reduce replay buffer size or batch size 