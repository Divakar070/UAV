
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json, os
from time import strftime, localtime

#Importing necessary classes from ai_base
from ai_base import State, Action, RL, DecayingFloat

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DeepQLearning(RL):
    def __init__(self, exploration=True):
        super().__init__("Deep-Q-Learning")
        self.is_exploration = exploration
        
        self.state_dim = 3  # (col, row, step)
        self.action_dim = Action.COUNT
        self.q_network = DQNNetwork(self.state_dim, self.action_dim)
        self.target_network = DQNNetwork(self.state_dim, self.action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        self.memory = []
        self.memory_size = 10000
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = DecayingFloat(value=0.9, factor=1.0-1e-6, minval=0.05)
        self.target_update_frequency = 1000
        self.steps_done = 0
        
        self.current_state = None
        self.current_action = None

    def store_transition(self, state, action, reward, next_state):
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state))

    def sample_memory(self):
        return random.sample(self.memory, self.batch_size)
    
    def state_to_tensor(self, state):
        return torch.tensor([state.col, state.row, state.step], dtype=torch.float32)

    def choose_action(self, state):
        if self.is_exploration and random.uniform(0, 1) < float(self.epsilon):
            action = random.choice(state.valid_actions())
        else:
            state_tensor = self.state_to_tensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            valid_actions = state.valid_actions()
            q_values = q_values.detach().numpy()[0]
            action = valid_actions[np.argmax(q_values[valid_actions])]
        return action

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.sample_memory()
        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)
        
        batch_state = torch.stack([self.state_to_tensor(s) for s in batch_state])
        batch_action = torch.tensor(batch_action)
        batch_reward = torch.tensor(batch_reward)
        batch_next_state = torch.stack([self.state_to_tensor(s) for s in batch_next_state])
        
        current_q_values = self.q_network(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(batch_next_state).max(1)[0].detach()
        expected_q_values = batch_reward + (self.gamma * next_q_values)
        
        loss = self.criterion(current_q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def execute(self, state, reward) -> int:
        if self.current_state is not None:
            self.store_transition(self.current_state, self.current_action, reward, state)
            self.optimize_model()
            if self.steps_done % self.target_update_frequency == 0:
                self.update_target_network()
        
        self.current_action = self.choose_action(state)
        self.current_state = state
        self.steps_done += 1
        
        if isinstance(self.epsilon, DecayingFloat):
            self.epsilon.decay()
        
        return self.current_action

    def load_data(self) -> int:
        filename = f"{self.name}-load.pth"
        if os.path.exists(filename):
            checkpoint = torch.load(filename)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.steps_done = checkpoint['steps_done']
            self.epsilon.value = checkpoint['epsilon']
            print(f"- loaded '{filename}' successfully")
            return checkpoint['round']
        else:
            print(f"- '{filename}' not found, no experience is used")
            return -1

    def save_data(self, round_id) -> bool:
        filename = strftime(f"{self.name}-[%Y-%m-%d][%Hh%Mm%Ss].pth", localtime())
        checkpoint = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': float(self.epsilon),
            'round': round_id
        }
        torch.save(checkpoint, filename)
        return True
