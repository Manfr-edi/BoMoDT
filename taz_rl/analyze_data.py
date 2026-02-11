import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

"""
Analysis script for training logs.
Run this to visualize your episode statistics.
"""

logs_dir = "./training_logs"
csv_filepath = os.path.join(logs_dir, "episode_stats.csv")

# Load the CSV
df = pd.read_csv(csv_filepath)

# Create plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Training Progress', fontsize=16)

# Plot 1: Total Reward per Episode
axes[0, 0].plot(df.index, df['total_reward'], marker='o', markersize=4)
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Total Reward')
axes[0, 0].set_title('Episode Total Reward')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Number of Steps per Episode
axes[0, 1].plot(df.index, df['num_steps'], marker='s', markersize=4, color='orange')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Num Steps')
axes[0, 1].set_title('Episode Length')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Loss Curves
#axes[0, 2].plot(df.index, df['actor_loss'], label='Actor Loss', marker='^', markersize=4)
axes[0, 2].plot(df.index, df['critic_loss'], label='Critic Loss', marker='v', markersize=4)
axes[0, 2].set_xlabel('Episode')
axes[0, 2].set_ylabel('Loss')
axes[0, 2].set_title('Loss Over Time')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Average Action
axes[1, 0].plot(df.index, df['avg_action'], marker='o', markersize=4, color='green')
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Avg Action Value')
axes[1, 0].set_title('Average Action')
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Action Std (Exploration)
axes[1, 1].plot(df.index, df['action_std'], marker='s', markersize=4, color='purple')
axes[1, 1].set_xlabel('Episode')
axes[1, 1].set_ylabel('Action Std Dev')
axes[1, 1].set_title('Policy Exploration (Std Dev)')
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Value Estimate vs Advantage
axes[1, 2].plot(df.index, df['avg_value_estimate'], label='Avg Value Estimate', marker='o', markersize=4)
axes[1, 2].plot(df.index, df['avg_advantage'], label='Avg Advantage', marker='s', markersize=4)
axes[1, 2].set_xlabel('Episode')
axes[1, 2].set_ylabel('Value')
axes[1, 2].set_title('Value Estimates')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(logs_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
print(f"Plot saved to {logs_dir}/training_curves.png")
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("TRAINING SUMMARY STATISTICS")
print("="*60)
print(f"\nTotal episodes: {len(df)}")
print(f"\nReward Statistics:")
print(f"  Mean: {df['total_reward'].mean():.2f}")
print(f"  Std:  {df['total_reward'].std():.2f}")
print(f"  Min:  {df['total_reward'].min():.2f}")
print(f"  Max:  {df['total_reward'].max():.2f}")

print(f"\nLoss Statistics (last 10 episodes):")
last_10 = df.tail(10)
print(f"  Avg Actor Loss:  {last_10['actor_loss'].mean():.4f}")
print(f"  Avg Critic Loss: {last_10['critic_loss'].mean():.4f}")

print(f"\nPolicy Statistics (last 10 episodes):")
print(f"  Avg Action Std: {last_10['action_std'].mean():.3f}")
print(f"  Avg Value Est:  {last_10['avg_value_estimate'].mean():.2f}")
print(f"  Avg Advantage:  {last_10['avg_advantage'].mean():.2f}")

# Show trends
reward_trend = df['total_reward'].iloc[-10:].mean() - df['total_reward'].iloc[:10].mean()
print(f"\nReward Trend (last 10 vs first 10 episodes): {reward_trend:+.2f}")