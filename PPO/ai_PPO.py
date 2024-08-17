import json
import os

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.optim.optimizer import Optimizer

from ai_base import RL, Action, DecayingFloat, State

class AdaptivePGD(Optimizer):
    def __init__(
            self,
            params,
            lr=0.15,
            sigma=1.0,
            beta1=0.9,
            beta2=0.999,
            damping=1e-8,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= damping:
            raise ValueError("Invalid damping value: {}".format(damping))
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta1))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(beta2))

        names, params = zip(*params)

        defaults = dict(
            lr=lr, beta1=beta1, beta2=beta2, damping=damping, sigma=sigma, names=names
        )

        super(AdaptivePGD, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = T.zeros_like(p.data)
                    # Exponential moving average of gradient^2 values
                    state["exp_avg_sq"] = T.zeros_like(p.data)
                    # noise
                    state["prev_noise"] = T.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                beta1, beta2 = group["beta1"], group["beta2"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(p.grad.data, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(
                    p.grad.data ** 2, alpha=1 - beta2
                )
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                denom = ((exp_avg_sq.sqrt()) / math.sqrt(bias_correction2)).add_(
                    group["damping"]
                )

                noise = T.randn_like(p.grad) * group["sigma"]
                perturbed_adaptive_grad = exp_avg / denom / bias_correction1 + noise
                p.data.add_(perturbed_adaptive_grad, alpha=-group["lr"])

def orthogonal_init(layer, gain=1.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, chkpt_dir='ppo'):
        super(ActorNetwork, self).__init__()
        self.logits = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim))
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')

    def forward(self, x: T.Tensor, action_mask: T.Tensor):
        logits = self.logits(x)
        action_mask = action_mask.to(dtype=T.bool)
        logits = T.where(action_mask, logits, logits - (1 << 16))
        return T.distributions.Categorical(logits=logits)

    def save_checkpoint(self):
        state_dict = self.state_dict()
        state_dict = {k: v.cpu().numpy().tolist() for k, v in state_dict.items()}
        if not os.path.exists(self.checkpoint_file):
            os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(state_dict, f)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_file):
            return -1
        with open(self.checkpoint_file, 'r') as f:
            state_dict = json.load(f)
        state_dict = {k: T.tensor(v) for k, v in state_dict.items()}
        self.load_state_dict(state_dict)


class CriticNetwork(nn.Module):
    def __init__(self, input_dim, chkpt_dir="ppo"):
        super(CriticNetwork, self).__init__()
        self.v = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1))
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')

    def forward(self, x: T.Tensor):
        v = self.v(x)
        return v

    def save_checkpoint(self):
        state_dict = self.state_dict()
        state_dict = {k: v.cpu().numpy().tolist() for k, v in state_dict.items()}
        if not os.path.exists(self.checkpoint_file):
            os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(state_dict, f)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_file):
            return -1
        with open(self.checkpoint_file, 'r') as f:
            state_dict = json.load(f)
        state_dict = {k: T.tensor(v) for k, v in state_dict.items()}
        self.load_state_dict(state_dict)


class ProgressiveColumn(nn.Module):
    def __init__(self, input_dims, output_dims, alpha, fc1_dims=128, fc2_dims=128, prev_columns=None):
        super(ProgressiveColumn, self).__init__()

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims + (fc1_dims * len(prev_columns) if prev_columns else 0), fc2_dims)
        self.fc3 = nn.Linear(fc2_dims + (fc2_dims * len(prev_columns) if prev_columns else 0), output_dims)

        self.prev_columns = prev_columns if prev_columns else []

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.apply(orthogonal_init)
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))

        lateral_connections1 = [F.relu(col.fc1(state)) for col in self.prev_columns]
        if lateral_connections1:
            x = T.cat([x] + lateral_connections1, dim=1)

        x = F.relu(self.fc2(x))

        lateral_connections2 = [F.relu(col.fc2(T.cat([F.relu(col.fc1(state))] +
                                                     [F.relu(prev_col.fc1(state)) for prev_col in col.prev_columns],
                                                     dim=1)))
                                for col in self.prev_columns]
        if lateral_connections2:
            x = T.cat([x] + lateral_connections2, dim=1)

        return self.fc3(x)


class ProgressiveActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='ppo'):
        super(ProgressiveActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'progressive_actor_torch_ppo')
        self.columns = nn.ModuleList([ProgressiveColumn(input_dims, n_actions, alpha, fc1_dims, fc2_dims)])

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action_mask=None):
        outputs = [col(state) for col in self.columns]
        combined_output = sum(outputs) / len(outputs)  # Averaging the outputs
        dist = F.softmax(combined_output, dim=-1)

        if action_mask is not None:
            dist = dist * action_mask
            dist = dist / dist.sum(dim=-1, keepdim=True)

        return T.distributions.Categorical(dist)

    def add_column(self, alpha):
        new_column = ProgressiveColumn(self.columns[0].fc1.in_features,
                                       self.columns[0].fc3.out_features,
                                       alpha,
                                       prev_columns=self.columns)
        self.columns.append(new_column)

    def save_checkpoint(self):
        state_dict = self.state_dict()
        state_dict = {k: v.cpu().numpy().tolist() for k, v in state_dict.items()}
        if not os.path.exists(self.checkpoint_file):
            os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(state_dict, f)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_file):
            return -1
        with open(self.checkpoint_file, 'r') as f:
            state_dict = json.load(f)
        state_dict = {k: T.tensor(v) for k, v in state_dict.items()}
        self.load_state_dict(state_dict)


class ProgressiveCriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='ppo'):
        super(ProgressiveCriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'progressive_critic_torch_ppo')
        self.columns = nn.ModuleList([ProgressiveColumn(input_dims, 1, alpha, fc1_dims, fc2_dims)])

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        outputs = [col(state) for col in self.columns]
        return sum(outputs) / len(outputs)  # Averaging the outputs

    def add_column(self, alpha):
        new_column = ProgressiveColumn(self.columns[0].fc1.in_features,
                                       self.columns[0].fc3.out_features,
                                       alpha,
                                       prev_columns=self.columns)
        self.columns.append(new_column)

    def save_checkpoint(self):
        state_dict = self.state_dict()
        state_dict = {k: v.cpu().numpy().tolist() for k, v in state_dict.items()}
        if not os.path.exists(self.checkpoint_file):
            os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(state_dict, f)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_file):
            return -1
        with open(self.checkpoint_file, 'r') as f:
            state_dict = json.load(f)
        state_dict = {k: T.tensor(v) for k, v in state_dict.items()}
        self.load_state_dict(state_dict)


class PPOMemory:
    def __init__(self, batch_size):
        self.clear_memory()
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), \
            np.array(self.actions), \
            np.array(self.probs), \
            np.array(self.vals), \
            np.array(self.rewards), \
            np.array(self.dones), \
            np.array(self.next_states), \
            np.array(self.actions_masks), \
            batches

    def store_memory(self, state, action, probs, vals, reward, done, next_state, actions_mask):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)
        self.actions_masks.append(actions_mask)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
        self.next_states = []
        self.actions_masks = []


class PPO(RL):
    def __init__(self, exploration):
        super().__init__("PPO")
        self.action_multiplier = 1
        self.state_dim = 81
        self.n_actions = Action.COUNT
        self.gamma = 0.99  # Increased from 0.9
        self.policy_clip = 0.2  # Increased from 0.1
        self.n_epochs = 5  # Increased from 2
        self.entropy_coefficient = 0.05
        self.gae_lambda = 0.95
        self.batch_size = 1024
        self.lr = 0.0003
        self.learn_after = 1024
        self.n_steps = 0
        self.progress_file = 'progressive_ppo_progress.json'
        self.epsilon = 0
        self.is_exploration = True

        #     print("- Load data requested")
        #     progress = ai.load_data()
        #     if progress != -1:
        #         STAT.round = progress + 1
        #     else:
        #         print("- Failed to load data")
        self.epsilon = 0
        self.is_exploration = True

        self.device = "cuda:0" if T.cuda.is_available() else "cpu"
        self.actor = ActorNetwork(self.state_dim, self.n_actions * self.action_multiplier)
        self.actor.to(self.device)
        self.actor_optimizer = T.optim.AdamW(self.actor.parameters(), self.lr)
        self.critic = CriticNetwork(self.state_dim)
        self.critic.to(self.device)
        self.critic_optimzer = T.optim.AdamW(self.critic.parameters(), self.lr)

        self.memory = PPOMemory(self.batch_size)
        self.reset_state()

    def reset_state(self):
        self.current_state = None
        self.current_action = None
        self.current_value = None
        self.current_log_prob = None
        self.current_actions_mask = None

    def add_column(self):
        self.actor.add_column(self.lr)
        self.critic.add_column(self.lr)

    def encode_state(self, state: State):
        encoded = np.zeros((self.state_dim,), dtype=np.float32)
        encoded[state.row] = 1
        encoded[15 + state.col] = 1
        encoded[30 + state.step] = 1
        return encoded

    def store_transition(self, state, action, probs, vals, reward, done, next_state, actions_mask):
        self.memory.store_memory(state, action, probs, vals, reward, done, next_state, actions_mask)

    def state_to_numpy(self, state: State):
        return np.array(self.encode_state(state), dtype=np.float32)

    def state_action_mask(self, state: State):
        mask = np.zeros((self.n_actions * self.action_multiplier,), dtype=np.float32)
        valid_actions = state.valid_actions()
        for i in range(self.action_multiplier):
            start = i * self.n_actions
            end = start + self.n_actions
            mask[start: end][valid_actions] = 1.0
        return mask

    def save_data(self, round_id):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.save_progress(round_id)
        print(f"- saved models at round {round_id}")

    def load_data(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.load_progress()
        return self.n_steps

    def save_progress(self, round_id):
        with open(self.progress_file, 'w') as f:
            json.dump({'round': round_id}, f)

    def load_progress(self):
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
                self.n_steps = progress.get('round', 0)
        else:
            self.n_steps = 0

    def scale_reward(self, reward, step):
        if step == 50:
            return reward / 10
        return reward / 100

    def execute(self, state, reward, is_terminal=False):
        if (state.col == 0 and state.row == 14 and state.step != 0) or state.step == 50:
            is_terminal = True
        if self.current_state is not None and self.is_exploration:
            reward = self.scale_reward(reward, state.step)
            self.store_transition(self.state_to_numpy(self.current_state), self.current_action, self.current_log_prob,
                                  self.current_value, reward, is_terminal, self.state_to_numpy(state),
                                  self.current_actions_mask)
            self.n_steps += 1
            if self.n_steps % self.learn_after == 0:
                self.learn()

        if is_terminal:
            self.reset_state()
            return -1

        state_np = self.state_to_numpy(state)
        state_tensor = T.tensor(state_np, dtype=T.float).to(self.device).unsqueeze(0)
        action_mask = T.tensor(self.state_action_mask(state), dtype=T.float).to(self.device).unsqueeze(0)

        distributions = self.actor(state_tensor, action_mask)
        dist: T.distributions.Categorical = distributions  # Select the appropriate head
        values = self.critic(state_tensor)
        value = values  # Select the appropriate head
        if self.is_exploration is False:
            action = T.argmax(dist.probs, dim=-1)
            self.reset_state()
        else:
            action = dist.sample()
            probs = T.squeeze(dist.log_prob(action)).item()
            action = T.squeeze(action).item()
            value = T.squeeze(value).item()
            self.current_state = state
            self.current_action = action
            self.current_log_prob = probs
            self.current_value = value
            self.current_actions_mask = action_mask.cpu().numpy()[0]
        # number of action is higher than the number of actions
        return action % self.n_actions

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, next_states_arr, actions_masks_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            last_state_np = next_states_arr[-1]
            last_state_tensor = T.tensor(last_state_np, dtype=T.float, device=self.device)
            with T.no_grad():
                last_value = self.critic(last_state_tensor).squeeze(-1).cpu().item()
            advantage, returns = self.calculate_advantages_and_returns(reward_arr, vals_arr, dones_arr, last_value)

            values = T.tensor(values).to(self.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float, device=self.device)
                old_probs = T.tensor(old_prob_arr[batch], dtype=T.float, device=self.device)
                actions = T.tensor(action_arr[batch], device=self.device)
                returns_batch = T.tensor(returns[batch], dtype=T.float, device=self.device)
                advantage_batch = T.tensor(advantage[batch], dtype=T.float, device=self.device)
                advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-8)
                advantage_batch = advantage_batch.clip(-10, 10)
                actions_masks_batch = T.tensor(actions_masks_arr[batch], dtype=T.float, device=self.device)
                dist = self.actor(states, actions_masks_batch)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage_batch * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage_batch
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # returns_batch = advantage[batch] + values[batch]
                criterion = T.nn.HuberLoss()
                critic_loss = criterion(critic_value, returns_batch)
                # critic_loss = (returns_batch - critic_value) ** 2
                # critic_loss = critic_loss.mean()

                entropy_loss = dist.entropy().mean()
                total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coefficient * entropy_loss

                self.actor_optimizer.zero_grad()
                self.critic_optimzer.zero_grad()
                total_loss.backward()

                from torch.nn.utils import clip_grad_norm_
                # clip_grad_norm_(self.actor.parameters(),1)
                # clip_grad_norm_(self.critic.parameters(),1)
                self.actor_optimizer.step()
                self.critic_optimzer.step()

        self.entropy_coefficient *= (1 - 1e-5) ** len(batches)
        self.entropy_coefficient = max(self.entropy_coefficient, 0.001)
        self.memory.clear_memory()

    def calculate_advantages_and_returns(self, rewards, values, dones, last_value):
        sample_size = len(rewards)
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)

        next_val = last_value
        next_adv = 0
        for t in reversed(range(sample_size)):
            current_terminal = dones[t]
            current_reward = rewards[t]
            current_value = values[t]

            if current_terminal:
                next_val = 0
                next_adv = 0

            delta = current_reward + self.gamma * next_val - current_value
            current_adv = delta + self.gamma * self.gae_lambda * next_adv

            next_adv = current_adv
            next_val = current_value

            advantages[t] = current_adv
            returns[t] = current_adv + current_value
        return advantages, returns



