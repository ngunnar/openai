import gym
import numpy as np

#Mountain car playground
env  = gym.make("MountainCar-v0")
env.reset()

env.render()
print(str(env.action_space.n))
action = np.random.random_integers(env.action_space.n) - 1
print(action)
state, reward, done, info = env.step(action)

done = False
while done != True:
    action = 1
    env.render()
    state, reward, done, info = env.step(action)
    print(state)
    print(reward)