
import numpy as np
import random
import json, os
from time import strftime, localtime
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from ai_base import State, Action, RL, DecayingFloat,SystemState


class DeepQLearning(RL):
    def __init__(self, exploration=True):
        super().__init__("Deep-Q-Learning")
        self.is_exploration = exploration
        
        self.state_dim = 3  # Assuming (col, row, step) as state dimensions
        self.action_dim = Action.COUNT
        self.q_network = self.build_model()
        self.target_network = self.build_model()
        self.target_network.set_weights(self.q_network.get_weights())
        self.optimizer = Adam(learning_rate=0.001)
        
        self.memory = []
        self.memory_size = 10000
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = DecayingFloat(value=0.9, factor=1.0-1e-6, minval=0.05)
        self.target_update_frequency = 1000
        self.steps_done = 0
        
        self.current_state = None
        self.current_action = None

    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_dim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def store_transition(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def sample_memory(self):
        return random.sample(self.memory, self.batch_size)
    
    def state_to_tensor(self, state):
        return np.array([state.col, state.row, state.step], dtype=np.float32).reshape(1, -1)

    def choose_action(self, state):
        if self.is_exploration and random.uniform(0, 1) < float(self.epsilon):
            action = random.choice(state.valid_actions())
        else:
            state_tensor = self.state_to_tensor(state)
            q_values = self.q_network.predict(state_tensor)
            valid_actions = state.valid_actions()
            action = valid_actions[np.argmax(q_values[0][valid_actions])]
        return action

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.sample_memory()
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
        
        batch_state = np.vstack([self.state_to_tensor(s) for s in batch_state])
        batch_action = np.array(batch_action)
        batch_reward = np.array(batch_reward)
        batch_next_state = np.vstack([self.state_to_tensor(s) for s in batch_next_state])
        batch_done = np.array(batch_done)
        
        current_q_values = self.q_network.predict(batch_state)
        next_q_values = self.target_network.predict(batch_next_state)
        target_q_values = current_q_values.copy()
        
        for i in range(self.batch_size):
            target = batch_reward[i]
            if not batch_done[i]:
                target += self.gamma * np.amax(next_q_values[i])
            target_q_values[i][batch_action[i]] = target
        
        self.q_network.fit(batch_state, target_q_values, epochs=1, verbose=0)

    def execute(self, state, reward) -> int:
        if self.current_state is not None:
            done = reward <= -1000  # Define a condition for terminal state
            self.store_transition(self.current_state, self.current_action, reward, state, done)
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
        filename = f"{self.name}-load.h5"
        if os.path.exists(filename):
            self.q_network.load_weights(filename)
            self.target_network.load_weights(filename)
            print(f"- loaded '{filename}' successfully")
            return 0  # Update as necessary
        else:
            print(f"- '{filename}' not found, no experience is used")
            return -1

    def save_data(self, round_id) -> bool:
        filename = strftime(f"{self.name}-[%Y-%m-%d][%Hh%Mm%Ss].h5", localtime())
        self.q_network.save_weights(filename)
        self.target_network.save_weights(filename)
        return True
