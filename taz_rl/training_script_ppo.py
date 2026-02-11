from torch.optim import Adam
import torch.nn as nn
import torch
from tensordict import TensorDict
import csv
import json
import os
from datetime import datetime, timedelta

from libraries.classes.Planner import Planner
from libraries.classes.SumoSimulator import Simulator
from libraries.constants import EDGE_DATA_FILE_PATH
from libraries.utils.preprocessingUtils import generateEdgeDataFile
from taz_rl.rlenv.local_taz_env import SumoTazEnv
from libraries import constants
from libraries.utils.generalUtils import *

taz_id = "H"
tls_number = count_entries_by_letter(letter=taz_id, json_path=constants.TAZ_FILE)


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for PPO training"""
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # ACTOR head (policy)
        self.actor_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(32, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # CRITIC head (value function)
        self.critic_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, obs):
        # Shared features
        shared_features = self.shared(obs)

        # Actor output
        actor_out = self.actor_head(shared_features)
        mean = 5.0 * torch.tanh(self.mean_head(actor_out))
        std = torch.exp(self.log_std).clamp(0.1, 2.0)

        # Critic output
        value = self.critic_head(shared_features).squeeze(-1)

        return mean, std, value

    def get_action_and_value(self, obs):
        """Get action, log_prob, and value for PPO"""
        mean, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        return action, log_prob, value

    def evaluate_actions(self, obs, actions):
        """Evaluate log_prob and value for given actions (used in PPO update)"""
        mean, std, value = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, value, entropy


# ==================== CONFIGURATION ====================
policy = ActorCriticNetwork((tls_number * 6) + 4, tls_number)

optim = Adam(policy.parameters(), lr=3e-4)
logFile = "../sumoenv/standalone/command_log.txt"
sumo = Simulator(configurationPath='../sumoenv/standalone', logFile=logFile, tazTlsMapFile=constants.TAZ_FILE)
twinPlanner = Planner(simulator=sumo)

sumo.changeDetectorPath(detectorPath=constants.SUMO_NETWORK_PATH)

env = SumoTazEnv(
    sumoSimulator=sumo,
    tazID=taz_id,
    stepSize=300
)

# PPO Hyperparameters
n_epochs = 10
n_episodes = 24
gamma = 0.99
gae_lambda = 0.95  # GAE (Generalized Advantage Estimation) parameter
ppo_clip_ratio = 0.2  # Clipping parameter for PPO
ppo_epochs = 3  # Number of PPO update epochs per episode batch
entropy_coef = 0.01  # Entropy coefficient for exploration

base_datetime = datetime(2024, 2, 1)


# ==================== LOGGING SETUP ====================
logs_dir = "./training_logs"
os.makedirs(logs_dir, exist_ok=True)

csv_filepath = os.path.join(logs_dir, "episode_stats.csv")
csv_columns = [
    "epoch", "episode", "date", "timeslot",
    "total_reward", "num_steps", "avg_action", "action_std",
    "actor_loss", "critic_loss", "total_loss", "ppo_loss",
    "avg_value_estimate", "avg_advantage",
    "returns_mean", "returns_std", "policy_clip_fraction"
]

if not os.path.exists(csv_filepath):
    with open(csv_filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()

json_filepath = os.path.join(logs_dir, "training_history.json")
training_history = []

# Model loading
loading_model = False
if loading_model:
    checkpoint = torch.load("checkpoint_ppo.pt", weights_only=True)
    policy.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0) + 1
    print(f"Model loaded from epoch {checkpoint.get('epoch', 0)}")
    print(f"Previous avg reward: {checkpoint.get('avg_reward', 0):.2f}")


# ==================== HELPER FUNCTIONS ====================
def compute_gae(rewards, values, gamma=0.99, lambda_=0.95):
    """
    Compute Generalized Advantage Estimation (GAE).
    Better than standard advantage for variance reduction.
    """
    advantages = []
    gae = 0
    
    # Iterate backwards through trajectory
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0  # Terminal state
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lambda_ * gae
        advantages.insert(0, gae)
    
    return torch.tensor(advantages, dtype=torch.float32)


def ppo_update(policy, optimizer, batch_obs, batch_actions, batch_log_probs_old, 
               batch_advantages, batch_returns, clip_ratio=0.2, ppo_epochs=3, entropy_coef=0.01):
    """
    Perform PPO update on collected batch data
    """
    total_ppo_loss = 0
    clip_fractions = []
    
    for ppo_epoch in range(ppo_epochs):
        # Evaluate current policy on batch
        log_probs, values, entropies = policy.evaluate_actions(batch_obs, batch_actions)
        
        # Ratio of new to old policy
        ratio = torch.exp(log_probs - batch_log_probs_old)
        
        # PPO clipped objective
        surr1 = ratio * batch_advantages
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * batch_advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Critic loss
        critic_loss = nn.functional.mse_loss(values.squeeze(), batch_returns)
        
        # Entropy bonus (encourages exploration)
        entropy_loss = -entropies.mean() * entropy_coef
        
        # Total loss
        total_loss = policy_loss + 0.5 * critic_loss + entropy_loss
        
        # Update
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        
        # Track clipping fraction (diagnostic)
        clipped = torch.abs(ratio - 1.0) > clip_ratio
        clip_fraction = clipped.float().mean()
        clip_fractions.append(clip_fraction.item())
        
        total_ppo_loss += total_loss.item()
    
    avg_ppo_loss = total_ppo_loss / ppo_epochs
    avg_clip_fraction = sum(clip_fractions) / len(clip_fractions)
    
    return avg_ppo_loss, avg_clip_fraction, policy_loss.item(), critic_loss.item()


# ==================== TRAINING LOOP ====================
for epoch in range(n_epochs):
    # Change date for each epoch
    current_date = base_datetime + timedelta(days=epoch)
    simulation_date = current_date.strftime('%Y-%m-%d')
    epoch_reward = 0
    epoch_episodes_data = []

    for episode in range(n_episodes):
        # Change timeslot for each episode
        hour = episode
        timeslot = f"{hour:02d}:00-{(hour+1):02d}:00"

        print(f"[EPOCH {epoch}] Date: {simulation_date}, Timeslot: {timeslot}")
        generateEdgeDataFile(
            PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH,
            date=simulation_date,
            time_slot=timeslot
        )

        timeslot_clean = timeslot.replace(':', '-')
        route_folder_path = os.path.join(SUMO_PATH, 'routes', timeslot_clean)
        os.makedirs(route_folder_path + '/output/', exist_ok=True)

        twinPlanner.scenarioGenerator.generateRoute(
            inputEdgePath=EDGE_DATA_FILE_PATH,
            timeSlot=timeslot_clean,
            totalCount=5000,
            custom=False
        )
        sumo.changeTypePath(typePath=route_folder_path)
        sumo.changeRouteFilePath(route_folder_path)

        td = env.reset()
        print(f"[EPISODE START] epoch={epoch} episode={episode}")

        # ==================== TRAJECTORY COLLECTION ====================
        # PPO collects full trajectories before updating
        trajectories = []
        log_probs_list = []
        rewards_list = []
        values_list = []
        actions_list = []

        while True:
            obs = td["observation"]

            # Get action, log_prob, and value
            with torch.no_grad():
                action, log_prob, value = policy.get_action_and_value(obs)

            actions_list.append(action)
            log_probs_list.append(log_prob)
            values_list.append(value)

            # Step environment
            td = env.step(
                TensorDict(
                    {"action": action},
                    batch_size=[]
                )
            )

            print(f"[OBS] obs norm={obs.norm().item():.3f}")

            reward = td["next", "reward"].detach()
            rewards_list.append(reward)

            if episode == 0 and epoch == 0:
                print(f"[DEBUG] reward={reward.item():.3f}")
                print(f"[DEBUG] action_mean={action.mean().item():.3f}")
                fixed_obs = obs.clone()

            if td["next", "terminated"].item():
                break

            td = td["next"]

        # ==================== ADVANTAGE & RETURN CALCULATION ====================
        # Clamp rewards for stability
        rewards_list = [r.clamp(-1, 1) for r in rewards_list]
        rewards_tensor = torch.stack(rewards_list)
        episode_reward = rewards_tensor.sum()
        epoch_reward += episode_reward

        # Stack values
        values_tensor = torch.stack(values_list)
        actions_tensor = torch.stack(actions_list)
        log_probs_tensor = torch.stack(log_probs_list)

        # Calculate returns (discounted cumulative reward)
        returns = []
        G = 0
        for r in reversed(rewards_list):
            r_val = r.item() if torch.is_tensor(r) else r
            G = r_val + gamma * G
            returns.insert(0, G)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

        # Compute GAE (better advantage estimation)
        advantages = compute_gae(rewards_list, values_tensor.detach(), gamma=gamma, lambda_=gae_lambda)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ==================== PPO UPDATE ====================
        ppo_loss, clip_fraction, actor_loss, critic_loss = ppo_update(
            policy,
            optim,
            obs.unsqueeze(0),  # Add batch dimension
            actions_tensor.unsqueeze(0),
            log_probs_tensor.unsqueeze(0),
            advantages.unsqueeze(0),
            returns_tensor.unsqueeze(0),
            clip_ratio=ppo_clip_ratio,
            ppo_epochs=ppo_epochs,
            entropy_coef=entropy_coef
        )

        total_loss = actor_loss + 0.5 * critic_loss

        # ==================== LOGGING ====================
        with torch.no_grad():
            mean, std, _ = policy.forward(fixed_obs)

        episode_log = {
            "epoch": epoch,
            "episode": episode,
            "date": simulation_date,
            "timeslot": timeslot,
            "total_reward": float(episode_reward.item()),
            "num_steps": len(rewards_list),
            "avg_action": float(actions_tensor.mean().item()),
            "action_std": float(std.mean().item()),
            "actor_loss": float(actor_loss),
            "critic_loss": float(critic_loss),
            "total_loss": float(total_loss),
            "ppo_loss": float(ppo_loss),
            "avg_value_estimate": float(values_tensor.mean().item()),
            "avg_advantage": float(advantages.mean().item()),
            "returns_mean": float(returns_tensor.mean().item()),
            "returns_std": float(returns_tensor.std().item()),
            "policy_clip_fraction": float(clip_fraction)
        }

        epoch_episodes_data.append(episode_log)

        # Write to CSV
        with open(csv_filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writerow(episode_log)

        print(f"Episode {episode}:")
        print(f"  Total reward: {episode_reward.item():.2f}")
        print(f"  Steps: {len(rewards_list)}")
        print(f"  Avg action: {actions_tensor.mean():.3f}")
        print(f"  Action std: {std.mean().item():.3f}")
        print(f"  Actor loss: {actor_loss:.3f}")
        print(f"  Critic loss: {critic_loss:.3f}")
        print(f"  PPO loss: {ppo_loss:.3f}")
        print(f"  Avg value estimate: {values_tensor.mean().item():.3f}")
        print(f"  Avg advantage: {advantages.mean().item():.3f}")
        print(f"  Policy clip fraction: {clip_fraction:.3f}")

        with torch.no_grad():
            out = policy(fixed_obs)
        print(f"[CHECK] policy(fixed obs) = {out}\n")

    # Epoch summary
    avg_epoch_reward = epoch_reward / n_episodes
    print(f"\n=== Epoch {epoch} | Avg Reward: {avg_epoch_reward.item():.2f} ===\n")

    # Save epoch data to JSON
    epoch_summary = {
        "epoch": epoch,
        "date": simulation_date,
        "avg_reward": float(avg_epoch_reward.item()),
        "episodes": epoch_episodes_data
    }
    training_history.append(epoch_summary)

    # Save JSON
    with open(json_filepath, 'w') as f:
        json.dump(training_history, f, indent=2)

    # Save checkpoint
    if (epoch + 1) % 10 == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "avg_reward": avg_epoch_reward,
        }, f"checkpoint_ppo_epoch{epoch}.pt")

# Final checkpoint
torch.save({
    "epoch": epoch,
    "model_state_dict": policy.state_dict(),
    "optimizer_state_dict": optim.state_dict(),
    "avg_reward": avg_epoch_reward,
}, "checkpoint_ppo_final.pt")

print(f"Training complete! Logs saved to {logs_dir}")
