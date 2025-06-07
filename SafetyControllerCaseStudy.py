import logging
import os
import string
import sys

import gymnasium as gym
from highway_env.envs import HighwayEnvFast
from highway_env.envs.common.observation import KinematicObservation
from highway_env.envs.highway_env import HighwayEnv
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    # format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    # datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


class HighwayAgent:

    def __init__(self, save_base_path: string, model_id: string = None) -> None:
        self._test_seed = None
        self._rss_plus_enabled = False
        self._shield_enabled = False
        self._test_runs = 10
        self._save_base_path = save_base_path
        self._model_id = model_id if model_id else "new"
        self._aggressive_vehicles = False
        self._env_config = HighwayAgent.base_config()
        return

    def set_rss_plus_enabled(self, enabled):
        self._rss_plus_enabled = enabled
        return self

    def set_shield_enabled(self, enabled):
        self._shield_enabled = enabled
        return self

    def set_test_runs(self, nb_runs):
        self._test_runs = nb_runs
        return self

    def set_test_seed(self, seed):
        self._test_seed = seed
        return self

    def set_aggressive_vehicles(self, aggressive: bool):
        self._aggressive_vehicles = aggressive
        return self

    @staticmethod
    def find_vehicle_in_front(env: HighwayEnvFast):
        base_env = env.unwrapped
        # Get ego vehicle
        ego_vehicle = base_env.vehicle  # Ego vehicle object
        # Get all other vehicles
        other_vehicles = base_env.road.vehicles

        # vehicles in front and same lane
        front_vehicles = [v for v in other_vehicles
                          if ego_vehicle.front_distance_to(v) > 0 and ego_vehicle.lane_index == v.lane_index]

        if front_vehicles:
            # Find the closest vehicle in front
            return min(front_vehicles, key=lambda v: v.position[0])
        else:
            return None
        # Compute distances to vehicles ahead
        # distances = [ego_vehicle.front_distance_to(v) for v in other_vehicles
        #              if ego_vehicle.front_distance_to(v) > 0 and ego_vehicle.lane_index == v.lane_index]
        # # Find the nearest vehicle in front
        # if distances:
        #     return float(min(distances))
        #     # print("\rDistance to next vehicle in front:", front_distance)
        # else:
        #     print("\rNo vehicle in front")
        #     return None

    @staticmethod
    def compute_rss_old(rss_plus_enabled, env: HighwayEnv, state):
        # find index of action 'SLOWER'
        actions = env.action_type.actions
        keys = [key for key, val in actions.items() if val == 'SLOWER']
        safe_action_key = keys[0]

        obs_type: KinematicObservation = env.observation_type
        ego = env.vehicle

        # compute d_real: denormalize position
        max_x = obs_type.features_range.get("x")[1]
        d_real: float = state[1][1] * max_x

        # compute d_rss
        # denormalize speeds
        max_vx = obs_type.features_range.get("vx")[1]
        v_r = state[0][3] * max_vx
        v_f = v_r + state[1][3] * max_vx

        rho = 1.0
        a_ego = ego.action.get("acceleration")
        if rss_plus_enabled:  # compute with RSS+ formula
            logger.debug("USING RSS+: a_ego=%", a_ego)
            a_max = a_ego
        else:  # use standard RSS formula
            a_max = 5.0
        b_min = 3.0
        b_max = 5.0
        d_rss: float \
            = max(0,
                  v_r * rho + 1 / 2 * a_max * rho ** 2
                  + (v_r + a_max * rho) ** 2 / (2 * b_min) - v_f ** 2 / (2 * b_max))

        # compute d_min
        d_min: float = max(0, v_r ** 2 / (2 * b_min) - v_f ** 2 / (2 * b_max))

        if d_real < d_rss:
            safe_action = safe_action_key
        else:
            safe_action = None
        return safe_action, d_real, d_rss, d_min, v_r, v_f

    @staticmethod
    def compute_rss(rss_plus_enabled, env: HighwayEnv, state):
        base_env = env.unwrapped
        # find index of action 'SLOWER'
        actions = base_env.action_type.actions
        keys = [key for key, val in actions.items() if val == 'SLOWER']
        slower_action_key = keys[0]

        obs_type: KinematicObservation = base_env.observation_type
        ego = base_env.vehicle
        veh_in_front = HighwayAgent.find_vehicle_in_front(base_env)

        safe_action = None
        d_real = float("inf")
        d_rss = 0
        d_min = 0
        v_r = ego.speed
        v_f = 0

        if veh_in_front:
            v_f = veh_in_front.speed
            d_real = ego.front_distance_to(veh_in_front)

            # compute d_rss
            rho = 1.0
            a_ego = ego.action.get("acceleration")
            if rss_plus_enabled:  # compute with RSS+ formula
                logger.debug("USING RSS+: a_ego=%", a_ego)
                a_max = a_ego
            else:  # use standard RSS formula
                a_max = 5.0
            b_min = 3.0
            b_max = 5.0
            d_rss: float \
                = max(0,
                      v_r * rho + 1 / 2 * a_max * rho ** 2
                      + (v_r + a_max * rho) ** 2 / (2 * b_min) - v_f ** 2 / (2 * b_max))

            # compute d_min
            d_min: float = max(0, v_r ** 2 / (2 * b_min) - v_f ** 2 / (2 * b_max))

            if d_real < d_rss:
                safe_action = slower_action_key
            else:
                safe_action = None

        return safe_action, d_real, d_rss, d_min, v_r, v_f

    def train(self):
        save_path, model_path = self.get_paths()

        env = self.make_env()
        # Settings adapted from
        # https://github.com/Farama-Foundation/HighwayEnv/blob/master/scripts/sb3_highway_dqn.py
        model = DQN('MlpPolicy', env,
                    policy_kwargs=dict(net_arch=[256, 256]),
                    learning_rate=5e-4,
                    buffer_size=15000,
                    learning_starts=200,
                    batch_size=32,
                    gamma=0.9,  # Discount factor
                    exploration_fraction=0.3,
                    exploration_initial_eps=1.0,
                    exploration_final_eps=0.05,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=50,
                    verbose=1,
                    tensorboard_log=save_path)

        # Save a checkpoint every 1000 steps
        checkpoint_callback = CheckpointCallback(
            save_freq=1000,
            save_path=save_path,
            name_prefix="rl_model"
        )

        model.learn(int(20_000), callback=checkpoint_callback, tb_log_name="new_dqn", progress_bar=True)
        model.save(model_path)

        env.close()

    def make_env(self):
        """
        Creates and returns the configured Highway environment.
        """
        # Ensure that the environment is registered and can accept configurations
        env = gym.make(
            "highway-fast-v0",
            render_mode='human',
            config=self._env_config  # Pass the environment configuration directly here.
        )

        #env.reset()  # Reset the environment to ensure it's ready for training.
        return env

    # def make_env(self):
    #     env: HighwayEnvFast = gym.make("highway-fast-v0", render_mode='human')
    #     env.configure(self._env_config)
    #     env.reset()
    #     return env

    @staticmethod
    def base_config():
        return {
            "lanes_count": 3,  # (4)
            "collision_reward": -1,
            "right_lane_reward": 0.0,  # (0.1),
            "reward_speed_range": [0, 40],
            "high_speed_reward": 2,
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": [0, 5, 10, 15, 20, 25, 30, 35, 40]},
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle"
        }

    # an alternative way to compute inter-distance
    @staticmethod
    def compute_distance_to_front(env: HighwayEnvFast):
        # Get ego vehicle
        ego_vehicle = env.vehicle  # Ego vehicle object
        ego_x = ego_vehicle.position[0]  # Ego's x position
        # Get all other vehicles
        other_vehicles = env.road.vehicles
        # Compute distances to vehicles ahead
        distances = [v.position[0] - ego_x for v in other_vehicles if v.position[0] > ego_x]
        # Find the nearest vehicle in front
        if distances:
            return float(min(distances))
            # print("\rDistance to next vehicle in front:", front_distance)
        else:
            logger.warn("\rNo vehicle in front")
            return None

    def get_paths(self):
        save_path = os.path.join(self._save_base_path, self._model_id)
        model_path = os.path.join(save_path, "trained_model")

        return save_path, model_path

    def test(self):
        print("\r===== BEGIN TEST ->", f"Model: {self._save_base_path}/{self._model_id}, "
                                       f"seed = {self._test_seed}, runs = {self._test_runs}, "
                                       f"shield = {self._shield_enabled}")

        save_path, model_path = self.get_paths()

        model = DQN.load(model_path)

        cfg = dict(self._env_config)
        cfg.update({"simulation_frequency": 15})
        if self._aggressive_vehicles:
            cfg.update({"other_vehicles_type": "highway_env.vehicle.behavior.AggressiveVehicle"})
        env = gym.make(
            "highway-fast-v0",
            render_mode='human',
            config=cfg
        )
        env.reset(seed=self._test_seed)

        base_env = env.unwrapped
        print(f"config = {base_env.config}")
        print("\rPossible actions:", base_env.action_type.actions)

        model_counters = [0] * 5
        shield_counters = [0] * 5
        actual_counters = [0] * 5
        unsafe_steps = 0
        crashes = 0
        min_net_safe_d = 100000.0
        run_ix = 0
        total_distance = 0.0

        obs_type: KinematicObservation = base_env.observation_type

        total_steps = 0
        safe_steps = 0
        unsafe_initial_steps = 0
        for t in range(1, self._test_runs + 1):
            logger.info("\r==== RUN %d ====", t)
            done = False
            truncated = False
            safety_reached = False  # a run often starts in an unsafe situation
            state = env.reset()[0]
            while not done and not truncated:
                total_steps += 1
                action = model.predict(state, deterministic=True)[0]
                safe_action, d_real, d_rss, d_min, v_r, v_f \
                    = HighwayAgent.compute_rss(self._rss_plus_enabled, env, state)
                d_inv = d_real - d_min
                vehicle_in_front = HighwayAgent.find_vehicle_in_front(env)
                base_env = env.unwrapped
                d_real2 = base_env.vehicle.front_distance_to(vehicle_in_front)
                v_r2 = base_env.vehicle.speed
                v_f2 = vehicle_in_front.speed

                if d_inv >= 0:
                    safety_reached = True  # True when safe for the first time in this run
                    safe_steps += 1

                if not safety_reached:
                    unsafe_initial_steps += 1

                model_counters[action] += 1
                # unsafe situation detected!
                if safe_action is not None and safe_action != action:
                    shield_counters[action] += 1
                    # apply safety shield if enabled
                    if self._shield_enabled:
                        action = safe_action
                actual_counters[action] += 1

                if safe_action is not None and safe_action != action:
                    unsafe_steps += 1

                # apply action, reach next state
                next_state, reward, done, truncated, info = env.step(action)

                env.render()

                if info and info['crashed']:
                    crashes += 1

                logger.debug(
                    "\raction=%d%s\td_real=%.2f\td_real2=%.2f\td_min=%.2f\td_rss=%.2f\tv_r=%.2f\tv_r2=%.2f\tv_f=%.2f\tv_f2=%.2f",
                    action,
                    "\tSHIELD" if safe_action is not None else "\t\t",
                    d_real, d_real2, d_min, d_rss, v_r, v_r2, v_f, v_f2)

                # only save new lowest d_min if scenario reached a safe state
                if safety_reached and d_inv < min_net_safe_d:
                    min_net_safe_d, run_ix = (d_inv, t)
                    logger.debug("\rNEW INV MIN: min_net_safe_d=%.2f", min_net_safe_d)

                state = next_state

            total_distance += base_env.vehicle.position[0]
            logger.debug("\ractions: %s\tunsafe_model_actions=%s\ttotal_distance=%.2f",
                         model_counters, unsafe_steps, total_distance)
            logger.info("\rCrashes: %d / %d runs %s", crashes, self._test_runs,
                        f"({crashes / self._test_runs * 100:0.1f} %)")

        print(f"\rAction counters: model = {model_counters}\tshield = {shield_counters}\t actual = {actual_counters}")
        print("\rCrashes:", crashes, "/", self._test_runs, "runs", f"({crashes / self._test_runs * 100:0.1f} %)")
        print("\runsafe_steps:", unsafe_steps, "/", total_steps, "steps",
              f"({unsafe_steps / (total_steps) * 100:0.1f} %)")
        print("\rmin_net_safe_d =", min_net_safe_d, f"(@run #{run_ix})")
        print("\ravg_distance =", total_distance / self._test_runs)
        runs = float(self._test_runs)
        # print("\rSteps (avg.): unsafe_initial=", float(unsafe_initial_steps) / runs,
        #       "\tsafe=", float(safe_steps) / runs,
        #       "/", (steps - unsafe_initial_steps) / runs,
        #       f"({safe_steps / (steps - unsafe_initial_steps) * 100:0.1f} %)")

        env.close()
        print("END TEST =======")


def main():
    if len(sys.argv) < 2:
        display_script_help()
        sys.exit(1)

    if sys.argv[1] == 'train':
        train_1lane_base_agent()
        #train_all_single_lane_agents()

    elif sys.argv[1] == 'test':
        test_1lane_base_agent(100)
        #test_all_single_lane_agents(100)

    else:
        display_script_help()


def display_script_help():
    print("Usage: python3 SafetyControllerCaseStudy.py train")
    print("       python3 SafetyControllerCaseStudy.py test")
    print()


class HighwayAgent3lanes(HighwayAgent):
    def __init__(self, save_base_path: string = "base", model_id: string = None):
        super().__init__(save_base_path, model_id)


class HighwayAgent3lanesAdversarial(HighwayAgent):
    def __init__(self, save_base_path: string = "adversarial", model_id: string = None):
        super().__init__(save_base_path, model_id)
        self._env_config["collision_reward"] = 1


class LegacyHighwayAgent1laneBase(HighwayAgent):
    def __init__(self, save_base_path: string = "base", model_id: string = None):
        super().__init__(save_base_path, model_id)
        self._env_config["lanes_count"] = 1


class HighwayAgent1laneBase(HighwayAgent):
    def __init__(self, save_base_path: string = "1lane_base", model_id: string = None):
        super().__init__(save_base_path, model_id)
        self._test_seed = 5
        self._env_config["lanes_count"] = 1
        self._env_config.get("action")["lateral"] = False


class HighwayAgent1laneAdversarial(HighwayAgent1laneBase):
    def __init__(self, save_base_path: string = "1lane_adversarial", model_id: string = None):
        super().__init__(save_base_path, model_id)
        self._env_config["collision_reward"] = 1


# === 3 LANES ===
def test_3lanes_agent_no_shield(runs):
    HighwayAgent3lanes().set_test_runs(runs).test()


def test_3lanes_agent_with_shield(runs):
    HighwayAgent3lanes().set_shield_enabled(True).set_test_runs(runs).test()


def test_3lanes_agent_adversarial_no_shield(runs):
    HighwayAgent3lanesAdversarial().set_test_runs(runs).test()


def test_3lanes_agent_adversarial_with_shield(runs):
    HighwayAgent3lanesAdversarial().set_shield_enabled(True).set_test_runs(runs).test()


# === SINGLE LANE ===
# *** train ***
def train_1lane_legacy_base_agent():
    LegacyHighwayAgent1laneBase().train()


def train_1lane_base_agent():
    HighwayAgent1laneBase().train()


def train_1lane_adversarial_agent():
    HighwayAgent1laneAdversarial().train()


def train_all_single_lane_agents():
    train_1lane_base_agent()
    train_1lane_adversarial_agent()


# *** test ***
def test_legacy_1lane_agent(runs):
    LegacyHighwayAgent1laneBase().set_test_runs(runs).test()


def test_legacy_1lane_agent_with_shield(runs):
    LegacyHighwayAgent1laneBase().set_shield_enabled(True).set_test_runs(runs).test()


def test_1lane_base_agent(runs):
    HighwayAgent1laneBase().set_test_runs(runs).test()


def test_1lane_base_agent_with_shield(runs):
    HighwayAgent1laneBase().set_shield_enabled(True).set_test_runs(runs).test()


def test_1lane_adversarial_agent(runs):
    HighwayAgent1laneAdversarial().set_test_runs(runs).test()


def test_1lane_adversarial_agent_with_shield(runs):
    HighwayAgent1laneAdversarial().set_shield_enabled(True).set_test_runs(runs).test()


def test_1lane_base_agent_against_aggressive(runs):
    HighwayAgent1laneBase().set_aggressive_vehicles(True).set_test_runs(runs).test()


def test_1lane_base_agent_against_aggressive_with_shield(runs):
    HighwayAgent1laneBase().set_aggressive_vehicles(True).set_shield_enabled(True).set_test_runs(runs).test()


def test_all_single_lane_agents(runs):
    # test agents with or without shield
    test_1lane_base_agent(runs)
    test_1lane_base_agent_with_shield(runs)
    test_1lane_adversarial_agent(runs)
    test_1lane_adversarial_agent_with_shield(runs)
    # test base agent with aggressive vehicles
    test_1lane_base_agent_against_aggressive(runs)
    test_1lane_base_agent_against_aggressive_with_shield(runs)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.WARN)
    main()
