
import json
import os
import random
from collections import deque, namedtuple
from time import localtime, strftime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Importing necessary classes from ai_base
from ai_base import RL, Action, DecayingFloat, State


def scale_reward(self,reward):
    return reward / 1000  # Adjust the divisor based on your reward scale
    # return reward   # Adjust the divisor based on your reward scale
class PrioritizedExperienceReplay:
    def __init__(self, capacity: int, alpha: float) -> None:
        self.alpha = alpha
        self.capacity = capacity
        self.current_capacity = 0
        self.data = np.zeros(capacity, dtype=object)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.data_pointer = 0

    def add(self, data: object, priority: float):
        self.data[self.data_pointer] = data
        self.priorities[self.data_pointer] = priority
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        if self.current_capacity < self.capacity:
            self.current_capacity += 1

    def sample(self, batch_size: int, beta: float):
        scaled_priorities = self.priorities[:self.current_capacity] ** self.alpha
        sampling_probabilities = scaled_priorities / np.sum(scaled_priorities)
        indices = np.random.choice(self.current_capacity, batch_size, p=sampling_probabilities, replace=False)
        sampled_data = [self.data[i] for i in indices]

        weights = (self.current_capacity * sampling_probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return sampled_data, indices, weights

    def __len__(self):
        return self.current_capacity

    def __iter__(self):
        return iter(self.data[:self.current_capacity])

class DQNNetwork(nn.Module):

    # def __init__(self, input_dim, output_dim):
    #     super(DQNNetwork, self).__init__()
    #     self.fc1 = nn.Linear(input_dim, 100)
    #     self.fc2 = nn.Linear(100, 100)
    #     self.dropout= nn.Dropout(0.2)
    #     self.fc3 = nn.Linear(100, output_dim)

    # def forward(self, x):
    #     x = torch.relu(self.fc1(x))
    #     # x = self.dropout(x)
    #     x = torch.relu(self.fc2(x))
    #     # x = self.dropout(x)
    #     x = self.fc3(x)
        # return x
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            # nn.Linear(256, 128),
            nn.Linear(128, output_dim),
        )
    
    def forward(self, x: torch.Tensor, action_mask: torch.Tensor):
        q_values = self.model(x)
        # print("before: ", q_values)
        q_values = torch.where(action_mask, q_values, q_values - (1 << 16))
        # print("after: ",q_values)
        return q_values


class viv_DeepQLearning(RL):
    def __init__(self, exploration=True):
        super().__init__("Deep-Q-Learning")
        self.is_exploration = exploration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = 81  
        self.action_dim = Action.COUNT
        self.q_network = DQNNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DQNNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=2.5e-4)
        self.criterion = nn.HuberLoss()

        self.memory_size = 100000
        self.memory = PrioritizedExperienceReplay(capacity=self.memory_size, alpha=0.9)
        self.batch_size = 32
        # self.gamma = 0.999
        self.gamma = 0.90
        self.train_after = 10000
        self.epsilon = DecayingFloat(value=1, factor=1 - 1e-5, minval=0.05)
        # self.epsilon = DecayingFloat(value=1.0, factor=1.0 - 1e-4, minval=0.005)
        # self.target_update_frequency = 2500
        self.target_update_frequency = 1000
        self.steps_done = 0
        self.n_updates = 0

        self.current_state = None
        self.current_action = None

    def store_transition(self, state, action, reward, next_state, is_terminal_next_state):
        self.memory.add((self.state_to_tensor(state), action, reward, self.state_action_mask(state),
                         self.state_to_tensor(next_state), self.state_action_mask(next_state), is_terminal_next_state),
                        priority=1.0)

    def sample_memory(self):
        beta = min(1.0, self.steps_done / 1000)  
        # print("beta",beta)
        sampled_data, indices, weights = self.memory.sample(self.batch_size, beta)
        return sampled_data, indices, weights
    def state_to_tensor(self, state:State):
        return torch.tensor(self.encode_state(state), dtype=torch.float32)
    def encode_state(self,state:State):
        encoded = np.zeros((self.state_dim,),dtype=np.float32)
        encoded[state.row] = 1
        encoded[15+state.col] = 1
        encoded[30+state.step] = 1
        return encoded
    def state_action_mask(self,state:State):
        mask = torch.zeros((self.action_dim,),dtype=torch.bool)
        # print(state.row, state.col)
        mask[state.valid_actions()] = 1 # or True
        return mask
    def choose_action(self, state:State):
        if self.is_exploration and random.uniform(0, 1) < float(self.epsilon):
            action = random.choice(state.valid_actions())
        else:
            state_tensor = self.state_to_tensor(state).unsqueeze(0).to(self.device)
            state_action_mask_tensor = self.state_action_mask(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor,state_action_mask_tensor)
                q_values = q_values.cpu().numpy()[0]

            best_action = np.argmax(q_values)
            valid_actions = state.valid_actions()
            action = valid_actions[np.argmax(q_values[valid_actions])]
            assert best_action == action
        return action

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    

    def optimize_model(self):
        if len(self.memory) < self.train_after or len(self.memory) < self.batch_size:
            return
        sampled_data, indices, weights = self.sample_memory()
        (sampled_states, sampled_actions, sampled_rewards, sampled_action_masks,
         sampled_next_states, sampled_next_action_masks, sampled_is_terminal_next_states) = zip(*sampled_data)

        batch_state = torch.stack([s for s in sampled_states]).to(self.device)
        batch_action_mask = torch.stack([mask for mask in sampled_action_masks]).to(self.device)
        batch_action = torch.tensor(sampled_actions, dtype=torch.int64, device=self.device)
        batch_reward = torch.tensor(sampled_rewards, dtype=torch.float32, device=self.device)
        batch_next_state = torch.stack([s for s in sampled_next_states]).to(self.device)
        batch_next_action_mask = torch.stack([mask for mask in sampled_next_action_masks]).to(self.device)
        batch_is_terminal_next_state = torch.tensor(sampled_is_terminal_next_states, device=self.device)
        batch_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # print("action mask: ",len(sampled_action_masks))

        current_q_values = self.q_network(batch_state, batch_action_mask).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(batch_next_state, batch_next_action_mask).max(1)[0].detach()
            next_q_values = torch.where(batch_is_terminal_next_state, torch.zeros_like(next_q_values), next_q_values)

        expected_q_values = batch_reward + (self.gamma * next_q_values)
        loss = (batch_weights * self.criterion(current_q_values, expected_q_values)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        new_priorities = torch.abs(current_q_values - expected_q_values).detach().cpu().numpy() + 1e-6
        for idx, priority in zip(indices, new_priorities):
            self.memory.priorities[idx] = priority

    def execute(self, state, reward, is_terminal=False) -> int:
        if state.col == 0 and state.row == 14:
            is_terminal = True
        if self.current_state is not None:
            reward = scale_reward(self, reward)
            self.store_transition(self.current_state, self.current_action, reward, state, is_terminal)
            self.optimize_model()
            self.n_updates += 1
            if self.n_updates == 1 or (self.n_updates % self.target_update_frequency == 0):
                self.update_target_network()

        self.current_action = self.choose_action(state)
        self.current_state = state
        self.steps_done += 1

        if isinstance(self.epsilon, DecayingFloat):
            self.epsilon.decay()

        return self.current_action

    def load_data(self) -> int:
        filename = f"{self.name}-load.json"
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                checkpoint = json.load(f)
            
            def convert_to_tensor(obj):
                if isinstance(obj, list):
                    try:
                        return torch.tensor(obj)
                    except (ValueError, RuntimeError):
                        return [convert_to_tensor(item) for item in obj]
                elif isinstance(obj, dict):
                    return {k: convert_to_tensor(v) for k, v in obj.items()}
                elif isinstance(obj, (int, float)):
                    return torch.tensor([obj])
                else:
                    return obj

            # Convert data structures back to tensors
            converted_checkpoint = {}
            for k, v in checkpoint.items():
                if k in ['q_network', 'target_network']:
                    converted_checkpoint[k] = {param_name: convert_to_tensor(param_data) for param_name, param_data in v.items()}
                elif k == 'optimizer':
                    # Handle optimizer state differently
                    converted_checkpoint[k] = v  # Keep as is for now
                else:
                    converted_checkpoint[k] = convert_to_tensor(v)

            self.q_network.load_state_dict(converted_checkpoint['q_network'])
            self.target_network.load_state_dict(converted_checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])  # Use original optimizer state
            self.steps_done = converted_checkpoint['steps_done']
            self.epsilon.value = float(converted_checkpoint['epsilon'])
            
            print(f"- loaded '{filename}' successfully")
            return converted_checkpoint['round'].item()
        else:
            print(f"- '{filename}' not found, no experience is used")
            return -1

    def save_data(self, round_id):
        filename = strftime(f"{self.name}-load.json", localtime())
        checkpoint = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': float(self.epsilon),
            'round': round_id
        }
        
        # Convert tensors to lists for JSON serialization
        def convert_tensor_to_list(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensor_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensor_to_list(v) for v in obj]
            else:
                return obj

        checkpoint = {k: convert_tensor_to_list(v) for k, v in checkpoint.items()}

        with open(filename, 'w') as f:
            json.dump(checkpoint, f)

        print(f"- saved models at round {round_id}")
        return True
