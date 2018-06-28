import gym

# Load Enviroment
#env = gym.make("Taxi-v2")
#env = gym.make("CartPole-v1")


#env = gym.make("MsPacman-v0")
env = gym.make("MountainCarContinuous-v0")

#Start to reset Enviroment
initialState = env.reset()

#Number of possible states
#possibleStates = env.observation_space.n
possibleStates = env.observation_space.shape[0]
print("Possible states: " + str(possibleStates))

#Render Enviroment
env.render()

#Numer of available actions
#availableActions = env.action_space.n
availableActions = env.action_space.shape[0]
print("Available actions: " + str(availableActions))
# 0 = down
# 1 = up
# 2 = right
# 3 = left
# 4 = pickup
# 5 = drop-off

#Get random actions
randomAction = env.action_space.sample()

#Override initial states
initialState = 114
print("Initial state " + str(initialState))
env.env.s = initialState
env.render()

#Move
#print("Go up")
#state, reward, done, info = env.step(1)
#print("State: " + str(state))
#print("Reward: " + str(reward))
#print("Done: " + str(done))
#print("Info: " + str(info))
#env.render()


#Random walk
#print("Random walk")
#state = env.reset()
#counter = 0
#reward = None
#while reward != 20:
#    state, reward, done, info = env.step(env.action_space.sample())
#    counter += 1

#print(counter)

#Q Action value table
print("Q Action value table")
import numpy as np
Q = np.zeros([possibleStates, availableActions])

G = 0 # Total Accumulated reward for each episode
alpha = 0.618 # Learning rate

for episode in range(1, 1001):
    done = False
    G, reward = 0,0
    state = env.reset()
    while done != True :
        print(state)
        action = np.argmax(Q[state]) #1 Agent start by choosing an action with the highest Q value for the current state.
        state2, reward, done, info = env.step(action) #2 Take the action and store the future state as state2
        Q[state, action] += alpha * (reward + np.max(Q[state2]) - Q[state, action]) # 3 Update state action pair for Q using the reward and the max Q value for state2. Update is done using the action value formula (based on Bellman equation) and allows state-action pairs to be updated in a recursive fashion (based on future values)
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