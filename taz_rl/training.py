from torch.optim import Adam
import torch.nn as nn
import torch
from tensordict import TensorDict
from torch.utils.checkpoint import checkpoint

from libraries.classes.Planner import Planner
from libraries.classes.SumoSimulator import Simulator
from libraries.constants import EDGE_DATA_FILE_PATH
from libraries.utils.preprocessingUtils import generateEdgeDataFile
from taz_rl.rlenv.local_taz_env import SumoTazEnv
from libraries import constants
from libraries.utils.generalUtils import *

taz_id = "H"
tls_number = count_entries_by_letter(letter=taz_id,json_path=constants.TAZ_FILE)

# Policy MLP minimale
policy = nn.Sequential(
    nn.Linear((tls_number * 6) + 4, 32),
    nn.ReLU(),
    nn.Linear(32, tls_number)
)

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

n_epochs = 24
n_episodes = 10
episode_reward = 0
epoch_reward = 0

base_datetime = datetime(2024,2,1)


loading_model = False

if loading_model:
    checkpoint = torch.load("checkpoint_taz.pt", weights_only=True)
    policy.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_reward = checkpoint['epoch_reward']
    loss = checkpoint['loss']

#for step in range(1000):
for epoch in range(n_epochs):
    #Change date for each epoch
    current_date = base_datetime + timedelta(days=epoch)
    simulation_date = current_date.strftime('%Y-%m-%d')

    epoch_reward = 0

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
        while True:
            obs = td["observation"]

            #action = policy(obs)
            mean = policy(obs)
            dist = torch.distributions.Normal(mean, torch.ones_like(mean))
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()



            td = env.step(
                TensorDict(
                    {"action": action},
                    batch_size=[]
                )
            )
            print(f"[OBS] obs norm={obs.norm().item():.3f}")

            if td["next", "terminated"].item():
                break


            # semplice gradient ascent sul reward (toy)
            reward = td["next", "reward"].detach()
            episode_reward += reward.float()
            loss = -(log_prob * reward)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if episode == 0 and epoch == 0:
                print(f"[DEBUG] reward={reward.item():.3f}")
                print(f"[DEBUG] action_mean={action.mean().item():.3f}")
                fixed_obs = obs.clone()

            # prepara il TD per il prossimo step
            td = td["next"]
        epoch_reward += episode_reward
        print(f"Epoch {epoch} | avg reward = {epoch_reward / n_episodes}")

        with torch.no_grad():
            out = policy(fixed_obs)
        print(f"[CHECK] policy(fixed obs) = {out}")

torch.save({
    "epoch": epoch,
    "model_state_dict": policy.state_dict(),
    "optimizer_state_dict": optim.state_dict(),
    "loss": loss,
}, "checkpoint_taz.pt")