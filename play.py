from train import MyModel
import gym
import numpy as np

def play(agent):
    env = gym.make('CartPole-v1') 
    observation = env.reset()
    obs = observation
    state = obs
    done = False
    tot_reward = 0.0
        
    while not done:
        env.render()
        state = np.squeeze(state).reshape(1,4)
        action = agent.run(state)
        observation, reward, done, info = env.step(action)
        obs = observation
        state = obs    
        tot_reward += reward
    env.close()
    print('Game ended! Total reward: {}'.format(tot_reward))
    return tot_reward


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = MyModel(state_size, action_size)
    agent.load("./save/cartpole-model.h5")
    play(agent)