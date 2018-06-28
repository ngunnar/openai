
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
from keras import backend as K


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self._update_target_model()

    def fit(self, env, episodes, save):
        done = False
        batch_size = 32
        
        for e in range(episodes):
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            steps = 500
            for time in range(steps):
                # env.render()
                action = self._act(state)
                next_state, reward, done, _ = env.step(action)
                reward = reward if not done else -steps/time
                next_state = np.reshape(next_state, [1, state_size])
                self._remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    self._update_target_model()
                    print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, episodes, time, agent.epsilon))
                    break                
            if len(self.memory) > batch_size:
                self._replay(batch_size)       
            if e % 10 == 0 & save:
                self._save("./save/cartpole-model_test.h5")
        
        return self.model
        
    def test(self, env):
        observation = env.reset()
        obs = observation
        state = obs
        done = False
        tot_reward = 0.0
        
        while not done:
            env.render()
            #state = np.squeeze(state).reshape(1,4)
            state = np.reshape(state, [1, state_size])
            action = self._run(state)
            observation, reward, done, info = env.step(action)
            obs = observation
            state = obs    
            tot_reward += reward
        env.close()
        print('Game ended! Total reward: {}'.format(tot_reward))
        return tot_reward

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
        
    def _update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def _remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def _act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return self._run(state)  # returns action
    
    def _run(self, state):
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def _replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _load(self, name):
        self.model.load_weights(name)

    def _save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    #env = gym.make("MountainCar-v0")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(state_size, action_size)
    model = agent.fit(env, 1000, True)
    agent.test(env)