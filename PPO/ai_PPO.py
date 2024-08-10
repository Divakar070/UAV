import json
import os

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

from ai_base import RL, Action, DecayingFloat, State


def orthogonal_init(layer, gain=1.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


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
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return self.states, \
            np.array(self.actions), \
            np.array(self.probs), \
            np.array(self.vals), \
            np.array(self.rewards), \
            np.array(self.dones), \
            batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


# class ActorNetwork(nn.Module):
#     def __init__(self, n_actions, input_dims, alpha,
#                  fc1_dims=128, fc2_dims=128, n_heads=1, chkpt_dir='ppo'):
#         super(ActorNetwork, self).__init__()

#         self.checkpoint_file = os.path.join(chkpt_dir, 'multihead_actor_torch_ppo')
#         self.shared_layers = nn.Sequential(
#             nn.Linear(input_dims, fc1_dims),
#             nn.ReLU(),
#             nn.Linear(fc1_dims, fc2_dims),
#             nn.ReLU()
#         )

#         self.heads = nn.ModuleList([nn.Sequential(
#             nn.Linear(fc2_dims, n_actions),
#             nn.Softmax(dim=-1)
#         ) for _ in range(n_heads)])

#         self.optimizer = optim.Adam(self.parameters(), lr=alpha)
#         self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
#         self.apply(orthogonal_init)
#         self.to(self.device)

#     def forward(self, state, action_mask=None):
#         shared_output = self.shared_layers(state)
#         head_outputs = [head(shared_output) for head in self.heads]

#         if action_mask is not None:
#             head_outputs = [dist * action_mask for dist in head_outputs]
#             head_outputs = [dist / dist.sum(dim=-1, keepdim=True) for dist in head_outputs]

#         distributions = [Categorical(dist) for dist in head_outputs]
#         return distributions
#     def save_checkpoint(self):
#         state_dict = self.state_dict()
#         # Convert tensors to lists for JSON serialization
#         state_dict = {k: v.cpu().numpy().tolist() for k, v in state_dict.items()}
#         if not os.path.exists(self.checkpoint_file):
#             os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)

#         with open(self.checkpoint_file, 'w') as f:
#             json.dump(state_dict, f)

#     def load_checkpoint(self):
#         if not os.path.exists(self.checkpoint_file):
#             return -1
#         with open(self.checkpoint_file, 'r') as f:
#             state_dict = json.load(f)
#         # Convert lists back to tensors
#         state_dict = {k: T.tensor(v) for k, v in state_dict.items()}
#         self.load_state_dict(state_dict)

# class CriticNetwork(nn.Module):
#     def __init__(self, input_dims, alpha, fc1_dims=128, fc2_dims=128,
#                  n_heads=1, chkpt_dir='ppo'):
#         super(CriticNetwork, self).__init__()

#         self.checkpoint_file = os.path.join(chkpt_dir, 'multihead_critic_torch_ppo')
#         self.shared_layers = nn.Sequential(
#             nn.Linear(input_dims, fc1_dims),
#             nn.ReLU(),
#             nn.Linear(fc1_dims, fc2_dims),
#             nn.ReLU()
#         )

#         self.heads = nn.ModuleList([nn.Linear(fc2_dims, 1) for _ in range(n_heads)])

#         self.optimizer = optim.Adam(self.parameters(), lr=alpha)
#         self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
#         self.apply(orthogonal_init)
#         self.to(self.device)

#     def forward(self, state):
#         shared_output = self.shared_layers(state)
#         values = [head(shared_output) for head in self.heads]
#         return values

#     def save_checkpoint(self):
#         state_dict = self.state_dict()
#         # Convert tensors to lists for JSON serialization
#         state_dict = {k: v.cpu().numpy().tolist() for k, v in state_dict.items()}
#         if not os.path.exists(self.checkpoint_file):
#             os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
#         with open(self.checkpoint_file, 'w') as f:
#             json.dump(state_dict, f)

#     def load_checkpoint(self):
#         if not os.path.exists(self.checkpoint_file):
#             return -1
#         with open(self.checkpoint_file, 'r') as f:
#             state_dict = json.load(f)
#         # Convert lists back to tensors
#         state_dict = {k: T.tensor(v) for k, v in state_dict.items()}
#         self.load_state_dict(state_dict)

class PPO(RL):
    def __init__(self, exploration):
        super().__init__("PPO")
        self.state_dim = 81
        self.n_actions = Action.COUNT
        self.gamma = 0.99  # Increased from 0.9
        self.policy_clip = 0.2  # Increased from 0.1
        self.n_epochs = 4  # Increased from 2
        self.entropy_coefficient = 0.03
        self.gae_lambda = 0.95
        self.batch_size = 64  # Decreased from 100
        self.lr = 0.0003
        self.learn_after = 128  # Decreased from 200
        self.n_steps = 0
        self.progress_file = 'progressive_ppo_progress.json'
        self.epsilon = 0
        self.is_exploration = False

        #     print("- Load data requested")
        #     progress = ai.load_data()
        #     if progress != -1:
        #         STAT.round = progress + 1
        #     else:
        #         print("- Failed to load data")
        self.epsilon = 0
        self.is_exploration = False

        # self.actor = ActorNetwork(self.n_actions, self.state_dim, lr)
        # self.critic = CriticNetwork(self.state_dim, lr)
        self.actor = ProgressiveActorNetwork(self.n_actions, self.state_dim, self.lr)
        self.critic = ProgressiveCriticNetwork(self.state_dim, self.lr)

        self.memory = PPOMemory(self.batch_size)

    def add_column(self):
        self.actor.add_column(self.lr)
        self.critic.add_column(self.lr)

    def encode_state(self, state: State):
        encoded = np.zeros((self.state_dim,), dtype=np.float32)
        encoded[state.row] = 1
        encoded[15 + state.col] = 1
        encoded[30 + state.step] = 1
        return encoded

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    # def state_to_tensor(self, state:State):
    #     return T.tensor(self.encode_state(state), dtype=T.float32)
    def state_to_numpy(self, state: State):
        return np.array(self.encode_state(state), dtype=np.float32)

    def state_action_mask(self, state: State):
        mask = np.zeros((self.n_actions,), dtype=np.float32)
        mask[state.valid_actions()] = 1.0
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
            return reward
        return reward / 10

    def execute(self, state, reward, is_terminal=False):
        reward = self.scale_reward(reward, state.step)
        if (state.col == 0 and state.row == 14) or state.step == 50:
            is_terminal = True
        state_np = self.state_to_numpy(state)

        state_tensor = T.tensor(state_np, dtype=T.float).to(self.actor.device)
        action_mask = T.tensor(self.state_action_mask(state), dtype=T.float).to(self.actor.device)

        distributions = self.actor(state_tensor, action_mask)
        dist = distributions  # Select the appropriate head
        values = self.critic(state_tensor)
        value = values  # Select the appropriate head

        action = dist.sample()
        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()
        self.n_steps += 1
        self.store_transition(state_np, action, probs, value, reward, is_terminal)

        if self.n_steps % self.learn_after == 0:
            self.learn()

        return action

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] *
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor([state_arr[i] for i in batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                entropy_loss = dist.entropy().mean()
                total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coefficient * entropy_loss
                self.entropy_coefficient *= (1 - 1e-5)
                self.entropy_coefficient = max(self.entropy_coefficient, 0.001)

                self.actor.columns[-1].optimizer.zero_grad()
                self.critic.columns[-1].optimizer.zero_grad()
                total_loss.backward()
                self.actor.columns[-1].optimizer.step()
                self.critic.columns[-1].optimizer.step()

        self.memory.clear_memory()
