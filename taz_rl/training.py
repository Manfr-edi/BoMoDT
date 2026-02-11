from torch.optim import Adam
import torch.nn as nn
import torch
from tensordict import TensorDict
from torch.utils.checkpoint import checkpoint
import csv
from libraries.classes.Planner import Planner
from libraries.classes.SumoSimulator import Simulator
from libraries.constants import EDGE_DATA_FILE_PATH
from libraries.utils.preprocessingUtils import generateEdgeDataFile
from taz_rl.rlenv.local_taz_env import SumoTazEnv
from libraries import constants
from libraries.utils.generalUtils import *
from datetime import datetime, timedelta

taz_id = "H"
tls_number = count_entries_by_letter(letter=taz_id,json_path=constants.TAZ_FILE)

# Policy MLP minimale
#policy = nn.Sequential(
#    nn.Linear((tls_number * 6) + 4, 32),
#    nn.ReLU(),
#    nn.Linear(32, tls_number)
#)

class ActorCriticNetwork(nn.Module):
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

        # CRITIC head (value function) - NUOVO!
        self.critic_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, obs):
        # Feature condivise
        shared_features = self.shared(obs)

        # Actor output
        actor_out = self.actor_head(shared_features)
        mean = 5.0 * torch.tanh(self.mean_head(actor_out))
        std = torch.exp(self.log_std).clamp(0.1, 2.0)

        # Critic output - NUOVO!
        value = self.critic_head(shared_features).squeeze(-1)

        return mean, std, value

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

n_epochs = 50
n_episodes = 24
gamma = 0.99
episode_reward = 0
epoch_reward = 0

base_datetime = datetime(2024,2,1)


# ==================== LOGGING SETUP ====================
logs_dir = "./training_logs"
os.makedirs(logs_dir, exist_ok=True)

# CSV file for structured logging
csv_filepath = os.path.join(logs_dir, "episode_stats.csv")
csv_columns = [
    "epoch", "episode", "date", "timeslot",
    "total_reward", "num_steps", "avg_action", "action_std",
    "actor_loss", "critic_loss", "total_loss",
    "avg_value_estimate", "avg_advantage",
    "returns_mean", "returns_std"
]

# Create CSV header if file doesn't exist
if not os.path.exists(csv_filepath):
    with open(csv_filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()

# JSON file for full episode details (can be more detailed)
json_filepath = os.path.join(logs_dir, "training_history.json")
training_history = []


loading_model = False
if loading_model:
    checkpoint = torch.load("checkpoint_a2c.pt", weights_only=True)
    policy.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0) + 1
    print(f"Model loaded from epoch {checkpoint.get('epoch', 0)}")
    print(f"Previous avg reward: {checkpoint.get('avg_reward', 0):.2f}")


#for step in range(1000):
for epoch in range(n_epochs):
    #Change date for each epoch
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
        # Update simulator paths
        sumo.changeTypePath(typePath=route_folder_path)
        sumo.changeRouteFilePath(route_folder_path)

        td = env.reset()
        print(f"[EPISODE START] epoch={epoch} episode={episode}")
        episode_reward = 0
        # Collect trajectory
        log_probs = []
        rewards = []
        values = []
        actions = []
        while True:
            obs = td["observation"]

            #action = policy(obs)
            mean, std, value = policy(obs)
            #dist = torch.distributions.Normal(mean, torch.ones_like(mean))
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            actions.append(action)

            log_prob = dist.log_prob(action).sum()
            log_probs.append(log_prob)
            values.append(value)


            td = env.step(
                TensorDict(
                    {"action": action},
                    batch_size=[]
                )
            )
            print(f"[OBS] obs norm={obs.norm().item():.3f}")

            reward = td["next", "reward"].detach()
            rewards.append(reward)
            #episode_reward += reward.float()
            loss = -(log_prob * reward)

            if td["next", "terminated"].item():
                break

            if episode == 0 and epoch == 0:
                print(f"[DEBUG] reward={reward.item():.3f}")
                print(f"[DEBUG] action_mean={action.mean().item():.3f}")
                fixed_obs = obs.clone()

            # Prepare TD for next step
            td = td["next"]

        # CALCULATE RETURNS (G_t = sum of future rewards)
        returns = []
        G = 0
        for r in reversed(rewards):
            r_val = r.item() if torch.is_tensor(r) else r  # if is a scalar, r is good to go
            G = r + 0.99 * G  # gamma=0.99
            returns.insert(0, G)
        returns = torch.tensor(returns)
        # Normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Convert values to tensor
        values = torch.stack(values)

        # Compute Advantages
        advantages = returns - values.detach()

        # Normalization
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Actor Loss (policy gradient with advantage instead of return)
        actor_loss = 0
        for log_p, adv in zip(log_probs, advantages):
            actor_loss += -log_p * adv

        # Critic loss (critic learns predicting the returns)
        critic_loss = nn.functional.mse_loss(values, returns)

        total_loss = actor_loss + 0.5 * critic_loss  # 0.5 is a typical weight

        optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optim.step()


        rewards = [r.clamp(-1, 1) for r in rewards]
        rewards_tensor = torch.stack(rewards)
        episode_reward = rewards_tensor.sum()
        epoch_reward += episode_reward

        # ==================== LOGGING ====================
        episode_log = {
            "epoch": epoch,
            "episode": episode,
            "date": simulation_date,
            "timeslot": timeslot,
            "total_reward": float(episode_reward.item()),
            "num_steps": len(rewards),
            "avg_action": float(torch.stack(actions).mean().item()),
            "action_std": float(std.mean().item()),
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "total_loss": float(total_loss.item()),
            "avg_value_estimate": float(values.mean().item()),
            "avg_advantage": float(advantages.mean().item()),
            "returns_mean": float(returns.mean().item()),
            "returns_std": float(returns.std().item()),
        }

        epoch_episodes_data.append(episode_log)

        # Write to CSV (append mode)
        with open(csv_filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writerow(episode_log)


        print(f"Episode {episode}:")
        print(f"  Total reward: {episode_reward.item():.2f}")
        print(f"  Steps: {len(rewards)}")
        print(f"  Avg action: {torch.stack(actions).mean():.3f}")
        print(f"  Action std: {std.mean().item():.3f}")
        #print(f"  Policy loss: {policy_loss.item():.3f}")
        print(f"  Actor loss: {actor_loss.item():.3f}")
        print(f"  Critic loss: {critic_loss.item():.3f}")  # NUOVO
        print(f"  Avg value estimate: {values.mean().item():.3f}")  # NUOVO
        print(f"  Avg advantage: {advantages.mean().item():.3f}")  # NUOVO

        with torch.no_grad():
            out = policy(fixed_obs)
        print(f"[CHECK] policy(fixed obs) = {out}\n")

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

    # Save JSON periodically
    with open(json_filepath, 'w') as f:
        json.dump(training_history, f, indent=2)

    if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "avg_reward": avg_epoch_reward,
            }, f"checkpoint_a2c_epoch{epoch}.pt")

torch.save({
    "epoch": epoch,
    "model_state_dict": policy.state_dict(),
    "optimizer_state_dict": optim.state_dict(),
    "loss": loss,
}, "checkpoint_taz.pt")