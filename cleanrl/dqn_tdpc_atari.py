# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
# from stable_baselines3.common.buffers import ReplayBuffer
from cleanrl_utils.buffers import TrajectoryBuffer
from torch.utils.tensorboard import SummaryWriter

from collections import deque

# TODO: check the how far we are predicting in the future. How long is 1 step in the future?
# TODO: If too short, either increase horizon or dilate trajectories
# TODO: ensure trajectories are not drawn across episodes


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""
    pc_gamma: float = 0.9
    """the discount factor gamma for pc"""
    horizon: int = 1
    """the horizon of the prediction"""
    dilation: int = 5
    """the space between steps in a trajectory"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.n_action_space = env.single_action_space.n
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 3136),
            nn.Unflatten(1, (64, 7, 7)),

            nn.ConvTranspose2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, 4, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(32, 32, 8, stride=4),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 4, 3, padding=1),
            nn.Sigmoid(),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(self.n_action_space, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
        )

        self.transition = nn.Sequential(
            nn.Linear(512+64, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

        self.network = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_action_space),
        )
    
    def encode(self, obs):
        if obs.dim() == 5:
            seq_len = obs.shape[0]
            obs = obs.view(-1, *obs.shape[2:])
            z = self.encoder(obs)
            z = z.view(seq_len, -1, *z.shape[1:])
        else:
            z = self.encoder(obs)
        return z

    def predict_next_latent(self, z, action):
        action = F.one_hot(action.long(), num_classes=self.n_action_space).float()#.squeeze(1).squeeze(2)
        if action.dim() == 3:
            action = action.squeeze(1)
        elif action.dim() == 4:
            action = action.squeeze(2)

        a = self.action_encoder(action.float())
        z = torch.cat([z, a], dim=-1)
        z = self.transition(z)

        return z
    
    def reconstruct(self, z):
        if z.dim() == 3:
            seq_len = z.shape[0]
            z = z.view(-1, *z.shape[2:])
            pred = self.decoder(z)
            pred = pred.view(seq_len, -1, *pred.shape[1:])
        else:
            pred = self.decoder(z)
        return pred

    def forward(self, obs):
        z = self.encode(obs)
        return self.network(z.detach())

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = TrajectoryBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)

    r_thresholds = [0, 2, 10, 50, 150]
    # [(obs, action, next_obs)]
    transitions = []
    
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                    # Collect raw observations for the state space analysis
                    if len(transitions) < 5 and info["episode"]["r"] >= r_thresholds[len(transitions)]:
                        transitions.append((obs, actions, next_obs))
                        writer.add_image(f"state prediction/{len(transitions)-1}", next_obs[0], global_step, dataformats="CHW")
                        

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size, args.horizon, args.dilation)

                # DQN
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations/255.0).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                enc = q_network.encoder(data.observations[-1] / 255.0)
                old_val = q_network.network(enc.detach()).gather(1, data.actions[-1]).squeeze()
                td_loss = F.mse_loss(td_target, old_val)

                # TDPC
                targets = data.observations[1:] / 255.0
                acts = data.actions[:-1]
                latent_state = q_network.encode(data.observations[:-1] / 255.0)
                # # AE loss
                # recon_loss = F.binary_cross_entropy(q_network.reconstruct(latent_state), data.observations / 255.0)
                # norm = 1.0
                recon_loss = 0.0
                norm = 0.0
                # PAE loss
                for i in range(args.horizon):
                    latent_state = q_network.predict_next_latent(latent_state, acts)
                    preds = q_network.reconstruct(latent_state)
                    recon_loss += F.binary_cross_entropy(preds, targets) * args.pc_gamma ** i
                    norm += args.pc_gamma ** i

                    targets = targets[1:]
                    acts = acts[1:]
                    latent_state = latent_state[:-1]
                recon_loss /= norm

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", td_loss, global_step)
                    writer.add_scalar("losses/reconstruction_loss", recon_loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                loss = td_loss + recon_loss

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )
        
        # Prediction eval
        if global_step % 100000 == 0:
            for idx, (obs, action, next_obs) in enumerate(transitions):
                z = q_network.encode(torch.Tensor(obs).to(device) / 255.0)
                z = q_network.predict_next_latent(z, torch.Tensor(action).to(device))
                pred_next_obs = q_network.reconstruct(z).detach().cpu().numpy()
                writer.add_image(f"state prediction/{idx}", (pred_next_obs[0] * 255.0).clip(0, 255).astype(np.uint8), global_step, dataformats="CHW")

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
