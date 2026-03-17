import os
import libtraci
import torch
from tensordict.nn import TensorDictModule
from torchrl.envs import GymEnv

from libraries.classes.SumoSimulator import Simulator
from libraries.classes.Planner import Planner
from libraries.classes.DigitalTwinManager import DigitalTwinManager
from libraries.utils.preprocessingUtils import generateEdgeDataFile
from taz_rl.rlenv.aggregation import *
from libraries.constants import SUMO_PATH, TAZ_FILE, PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH, EDGE_DATA_FILE_PATH, \
    SUMO_NETWORK_PATH

import torch
from torchrl.envs.common import EnvBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, BoundedTensorSpec
from tensordict import TensorDict
import numpy as np


class SumoTazEnv(EnvBase):
    """
    Class representing SUMO environment for a single Traffic Analysis Zone (TAZ).

    Attributes
    - sumo: SumoWrapper instance to interact with SUMO environment.
    - tazID: TAZ id, usually labelled with a single upper-case letter, to identify a TAZ.
    - stepSize: simulated seconds for the RL model to take an action.
    - warmupSteps: number of warmup steps to take before start evaluating observation.
    - cooldownSteps: number of cooldown steps to take after which observations are not evaluated anymore.
    """

    def __init__(self, sumoSimulator, tazID, stepSize=300, warmupSteps=2, cooldownSteps=15, device="cpu"):
        super().__init__(device=device)
        self.sumo = sumoSimulator
        self.tazID = tazID
        self.stepSize = stepSize
        ### LOOK FOR WARMUP AND COOLDOWN STEPS IF YOU CHANGE stepSize
        self.warmupSteps = warmupSteps
        self.cooldownSteps = cooldownSteps
        self.current_step = 0

        # save all TLS files from TAZ
        self.tls_list = self.sumo.get_tls_for_taz(tazID)

        # example: each action changes the phase duration within a range [-5, 5]
        self.action_spec = BoundedTensorSpec(
            low=-5.0, high=5.0, shape=(len(self.tls_list),)
        )

        # status = aggregated TAZ values + individual TLS values
        # decide feature size state
        # For each TLS: phaseID, phaseDuration, nVeh, meanSpeed, maxJamLength, occupancy → 6 feature
        n_tls_features = len(self.tls_list) * 6
        n_taz_features = 4  # nVeh, meanSpeed, maxJamLength, occupancy TAZ

        state_dim = n_tls_features + n_taz_features

        self.observation_spec = UnboundedContinuousTensorSpec(shape=(state_dim,))
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))

        # FIXED: Initialize previous state with reasonable defaults instead of 0
        self.prev_nVeh = 0
        self.prev_mean_speed = 10.0  # Typical urban speed (km/h)
        self.prev_critical_ratio = 0.1  # Small but non-zero
        self.prev_max_jam_len = 0
        self.prev_occupancy = 0.1  # Small but non-zero

        # ADDED: Reward component tracking for debugging
        self.last_reward_components = {}

    def _get_state_vector(self):
        """
        Function to build the observation tensor. This tensor is composed of aggregated tls metrics linked with current
        phase id and duration, together with aggregated TAZ metrics which summarize the average status of the involved tls.
        """
        if not self.sumo.isRunning():
            return torch.zeros(
                self.observation_spec.shape,
                dtype=torch.float32,
                device=self.device
            )
        raw = self.sumo.get_taz_e2_metrics(self.tazID, interval="last", mode="dict")

        tls_metrics = []
        for tls in self.tls_list:
            current_phase = libtraci.trafficlight.getPhase(tls)
            phase_duration = libtraci.trafficlight.getPhaseDuration(tls)
            time_to_switch = libtraci.trafficlight.getNextSwitch(tls) - libtraci.simulation.getTime()

            elapsed = phase_duration - time_to_switch
            phase_progress = elapsed / phase_duration if phase_duration > 0 else 0.0  # FIXED: Added safety check
            dets = raw["tls_data"].get(tls, [])
            # The phase progress can be added if needed
            phase_data = {"phaseID": current_phase, "phase_duration": phase_duration}

            # Data from all the road intersections in this TAZ are evaluated and aggregated using the dedicated function.
            # The results are combined with the traffic light phase data.
            tls_metrics.append(phase_data | aggregate_tls_metrics(dets))

        taz_metrics = aggregate_taz_metrics(tls_metrics)

        obs = []
        for m in tls_metrics:
            obs.extend([
                float(m.get("phase_ID", 0)),
                float(m.get("phase_duration", 0)),
                float(m.get("nVeh", 0)),
                float(m.get("meanSpeed", 0)),
                float(m.get("maxJamLength", 0)),
                float(m.get("occupancy", 0)),
            ])
        taz_features = [
            taz_metrics.get("veh_total", 0),
            taz_metrics.get("mean_speed", 0),
            taz_metrics.get("critical_ratio", 0),
            taz_metrics.get("mean_occupancy", 0),
        ]
        obs.extend(taz_features)

        return torch.tensor(obs, dtype=torch.float32, device=self.device)

    def _apply_action_to_tls(self, tls_id, action_value):
        program = libtraci.trafficlight.getAllProgramLogics(tls_id)[0]
        phase_id = program.currentPhaseIndex
        phase = program.phases[phase_id]

        # non toccare gialli / rossi
        if "g" not in phase.state and "G" not in phase.state:
            return

        base = phase.duration
        # delta = int(action_value * 10)  # max ±10s
        new_dur = int(np.clip(base + action_value, 10, 60))
        self.sumo.set_tls_phase_duration(tls_id, phase_id, new_dur)

    def _set_seed(self, seed: int):
        """Setta il seed dell'environment."""
        self.rng.manual_seed(seed)
        return seed

    # -------------------------------------------------

    def _reset(self, tensordict=None, **kwargs):
        if self.sumo.isLoaded():
            self.sumo.end()
        self.sumo.start(activeGui=False, logFilePath=self.sumo.logFile, rl_mode=True)
        self.current_step = 0

        obs = self._get_state_vector()

        # FIXED: Initialize previous state with current observations or safe defaults
        self.prev_mean_speed = max(obs[-3].item(), 1.0)  # At least 1.0 to prevent division issues
        self.prev_critical_ratio = max(obs[-2].item(), 0.01)  # At least 0.01
        self.prev_occupancy = max(obs[-1].item(), 0.01)  # At least 0.01

        # ADDED: Reset reward tracking
        self.last_reward_components = {}

        return TensorDict({"observation": obs}, batch_size=[])

    def _step(self, tensordict):
        action = tensordict["action"]
        if action.requires_grad:
            action = action.detach()
        action = action.cpu().numpy()

        for tls, a in zip(self.tls_list, action):
            self._apply_action_to_tls(tls, a)

        # Continue with the simulation of rl_dt seconds
        self.sumo.step(quantity=self.stepSize)
        self.current_step += 1
        if not self.sumo.isRunning() or self.current_step >= self.cooldownSteps:
            if self.sumo.isLoaded():
                self.sumo.end()
            return TensorDict(
                {
                    "observation": torch.zeros(
                        self.observation_spec.shape,
                        device=self.device
                    ),
                    "reward": torch.tensor(0.0, dtype=torch.float32, device=self.device),
                    "terminated": torch.tensor(True, device=self.device),
                    "truncated": torch.tensor(False, device=self.device),
                },
                batch_size=[]
            )
        obs = self._get_state_vector()
        # reward TAZ-level
        total_vehicles = obs[-4].item()
        mean_speed = obs[-3].item()
        critical_ratio = obs[-2].item()
        mean_occupancy = obs[-1].item()

        if self.current_step <= self.warmupSteps:
            reward = 0.0
            self.last_reward_components = {
                "speed_improvement": 0.0,
                "occupancy_reduction": 0.0,
                "critical_reduction": 0.0,
                "total": 0.0,
                "warmup": True
            }
        else:
            if total_vehicles > 0:
                # FIXED: Use max() to prevent division by near-zero values
                # This prevents the reward explosion when prev values are very small
                speed_improvement = (mean_speed - self.prev_mean_speed) / max(self.prev_mean_speed, 1.0)
                occupancy_reduction = -(mean_occupancy - self.prev_occupancy) / max(self.prev_occupancy, 0.1)
                critical_reduction = -2.0 * (critical_ratio - self.prev_critical_ratio) / max(self.prev_critical_ratio,
                                                                                              0.05)

                # ADDED: Clip individual components to prevent any single one from dominating
                #speed_improvement = np.clip(speed_improvement, -10.0, 10.0)
                #occupancy_reduction = np.clip(occupancy_reduction, -10.0, 10.0)
                #critical_reduction = np.clip(critical_reduction, -20.0, 20.0)

                reward = speed_improvement + occupancy_reduction + critical_reduction

                # ADDED: Final safety clip (should rarely trigger with the fixes above)
                #reward = np.clip(reward, -50.0, 50.0)

                # ADDED: Store components for debugging
                self.last_reward_components = {
                    "speed_improvement": float(speed_improvement),
                    "occupancy_reduction": float(occupancy_reduction),
                    "critical_reduction": float(critical_reduction),
                    "total": float(reward),
                    "mean_speed": mean_speed,
                    "prev_mean_speed": self.prev_mean_speed,
                    "critical_ratio": critical_ratio,
                    "prev_critical_ratio": self.prev_critical_ratio,
                    "occupancy": mean_occupancy,
                    "prev_occupancy": self.prev_occupancy,
                    "warmup": False
                }
            else:
                reward = 0.0
                self.last_reward_components = {
                    "speed_improvement": 0.0,
                    "occupancy_reduction": 0.0,
                    "critical_reduction": 0.0,
                    "total": 0.0,
                    "no_vehicles": True
                }

        # FIXED: Update previous state with safe minimum values
        self.prev_mean_speed = max(mean_speed, 1.0)
        self.prev_critical_ratio = max(critical_ratio, 0.01)
        self.prev_occupancy = max(mean_occupancy, 0.01)

        return TensorDict(
            {
                "observation": obs,
                "reward": torch.tensor(reward, dtype=torch.float32, device=self.device),
                "terminated": torch.tensor(False, device=self.device),
                "truncated": torch.tensor(False, device=self.device)
            },
            batch_size=[]
        )


if __name__ == "__main__":

    simulationDate = '2024-02-01'
    timeslot = "00:00-01:00"
    generateEdgeDataFile(PROCESSED_TRAFFIC_FLOW_EDGE_FILE_PATH, date=simulationDate, time_slot=timeslot)

    configurationPath = SUMO_PATH + "/standalone"
    logFile = SUMO_PATH + "/standalone/command_log.txt"
    sumoSimulator = Simulator(configurationPath=configurationPath, logFile=logFile, tazTlsMapFile=TAZ_FILE)
    taz_id = "H"  # esempio
    sumoSimulator.changeDetectorPath(detectorPath=SUMO_NETWORK_PATH)

    twinPlanner = Planner(simulator=sumoSimulator)
    timeslot = timeslot.replace(':', '-')
    timeslotPath = SUMO_PATH + "/routes/" + timeslot
    twinPlanner.scenarioGenerator.generateRoute(inputEdgePath=EDGE_DATA_FILE_PATH, timeSlot=timeslot, totalCount=5000,
                                                custom=False)
    routefolder_name = os.path.join(SUMO_PATH, 'routes')
    route_folder_path = os.path.join(routefolder_name, timeslot)
    os.makedirs(route_folder_path + '/output/', exist_ok=True)
    sumoSimulator.changeTypePath(typePath=route_folder_path)
    print(route_folder_path)
    sumoSimulator.changeRouteFilePath(route_folder_path)

    env = SumoTazEnv(sumoSimulator, tazID=taz_id, device="cpu")

    td = env.reset()
    # env.sumo.step(100)

    # for _ in range(10):
    step_count = 0
    while True:
        # azione random: durations
        action = env.action_spec.rand()
        # td = env.step(action)
        td = env.step(
            TensorDict(
                {"action": action},
                batch_size=[]
            )
        )

        print(f"Step {step_count}: Reward:", td["next", "reward"].item())

        # ADDED: Print reward components for debugging
        if hasattr(env, 'last_reward_components'):
            components = env.last_reward_components
            if not components.get('warmup', False) and not components.get('no_vehicles', False):
                print(f"  Speed improvement: {components['speed_improvement']:.4f}")
                print(f"  Occupancy reduction: {components['occupancy_reduction']:.4f}")
                print(f"  Critical reduction: {components['critical_reduction']:.4f}")

        step_count += 1

        if td["next", "terminated"].item():
            break
