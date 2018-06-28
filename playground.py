import gym
import numpy as np

#Mountain car playground
env  = gym.make("MountainCar-v0")
env = gym.make("CartPole-v1")
env = gym.make("Taxi-v2")
env.reset()

possibleStates = env.observation_space.n
availableActions = env.action_space.n

Q = np.zeros([possibleStates, availableActions])

G = 0 # Total Accumulated reward for each episode
alpha = 0.618 # Learning rate

for episode in range(1, 1001):
    done = False
    G, reward = 0,0
    state = env.reset()
    while done != True :
        #print(state)
        #1 Agent start by choosing an action with the highest Q value for the current state.
        action = np.argmax(Q[state]) 
        #2 Take the action and store the future state as state2
        state2, reward, done, info = env.step(action) 
        # 3 Update state action pair for Q using the reward and the max Q value for state2. Update is done using the action value formula (based on Bellman equation) and allows state-action pairs to be updated in a recursive fashion (based on future values)
        Q[state, action] += alpha * (reward + np.max(Q[state2]) - Q[state, action]) 
        G += reward
        state = state2
    if episode % 50 == 0:
        print('Episode {} Total Reward: {}'.format(episode, G))

# Run model

done = False
state = env.reset()
while done != True:
    env.render()
    action = np.argmax(Q[state])
    state, reward, done, info = env.step(action)
print(reward)