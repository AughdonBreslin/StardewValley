# Models Directory

This directory contains trained reinforcement learning models and training artifacts for the Stardew Valley RL project.

## Model Files

### RL Coach Models
- `demo_rl_coach_model.pth` - Demo model trained with shorter training session
- `rl_coach_final_model.pth` - Final trained model from complete training run
- `rl_coach_model_episode_50.pth` - Model checkpoint at episode 50
- `rl_coach_model_episode_0.pth` - Model checkpoint at episode 0
- `rl_coach_interrupted_model.pth` - Model saved during interrupted training session

### DQN Models
- `dqn_model.pth` - Deep Q-Network model

## Training Artifacts

- `training_results.png` - Visualization of training progress (rewards, episode lengths, losses)
- `training_stats.json` - Detailed training statistics and metrics

## Usage

To load a model in your code, update the path to include the models directory:

```python
# Example: Loading the demo model
agent.load("models/demo_rl_coach_model.pth")

# Example: Loading the final model
agent.load("models/rl_coach_final_model.pth")
```

## Model Information

These models were trained using the RL Coach framework with the following characteristics:
- State size: 4
- Action size: 2
- Architecture: DQN with hidden layers [32, 32]
- Learning rate: 0.001
- Gamma: 0.95
- Epsilon decay: 0.99

## Training Configuration

The models were trained with the following demo configuration:
- Max episodes: 100 (for demo)
- Max steps per episode: 1800 (30 seconds at 60 FPS)
- Evaluation frequency: 20 episodes
- Save frequency: 50 episodes
- Log frequency: 5 episodes 