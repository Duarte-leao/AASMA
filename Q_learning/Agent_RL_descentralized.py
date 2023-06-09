from Environment_clean_QL import CleaningEnv
from Results_analysis import compare_results_learning

import argparse
import numpy as np
from gym import Env
import time
import random 
import math
from scipy.spatial.distance import cityblock
from collections import defaultdict
import itertools    
import pickle
import joblib
import dill



def train_eval_loop_single( team, agents, n_evaluations, n_training_episodes, n_eval_episodes):

    print(f"Train-Eval Loop for {team}\n")
    results = np.zeros((n_evaluations, n_eval_episodes))

    for evaluation in range(n_evaluations):

        print(f"\tIteration {evaluation+1}/{n_evaluations}")
        # Train
        print(f"\t\tTraining {team} for {n_training_episodes} episodes.")

        for agent in agents: agent.train()   # Enables training mode
        # Run train iteration
        run_agent( agents, n_training_episodes)

        # Eval
        print(f"\t\tEvaluating {team} for {n_eval_episodes} episodes.")
        for agent in agents: agent.eval()  

        # Run eval iteration
        results[evaluation] = run_agent( agents ,n_eval_episodes)
        print(f"\t\tAverage Steps To Capture: {round(results[evaluation].mean(), 2)}")
   

    return results


def run_agent( agents,  n_episodes: int) -> np.ndarray:

    results = np.zeros(n_episodes)
    

    for episode in range(n_episodes):
        environment = CleaningEnv(5, 2, 400)
        terminals = [False for _ in range(len(agents))]
        observations = environment.reset()
        observations = see(observations, environment.num_dirt1, environment.num_robots)
        done = False
        while not done:
            
            # environment.render()
            if (environment.num_dirt1 + environment.num_dirt2) == 0:
                 break
            
            # if agents[0].training == False:
            #     time.sleep(0.5)
        

            actions = [agent.action(observations, environment.num_dirt1) for agent in agents]
            next_observations, rewards, done, info = environment.step(actions)
            next_observations = see(next_observations, environment.num_dirt1, environment.num_robots)

            for a, agent in enumerate(agents):
                if agent.training:
                    agent.update(observations, actions[a], next_observations, rewards)
        
            observations = next_observations
        
        results[episode] = environment.current_step 
        environment.close()

    return results

def see(observation, n_dirt1, n_agents):

    dirt_1 = np.zeros(n_dirt1, dtype=int)
    dirt_2 = np.ones(len(observation)-n_agents-n_dirt1, dtype=int)
    dirt_levels = np.append(dirt_1, dirt_2)
    obs1 = tuple(map(tuple, np.append(observation[n_agents:], dirt_levels[:, np.newaxis], axis=1)))
    obs2 = tuple(map(tuple, observation[:n_agents]))
    observations = obs2 + obs1 
    observations = tuple(itertools.chain.from_iterable(observations))
    # print(observations)
    return observations


class QLearningAgent():

    def __init__(self, agent_id, n_robots, n_actions, learning_rate=0.18, discount_factor=0.95, exploration_rate=0.15, initial_q_values = 0.0):
        super(QLearningAgent, self).__init__
        self._Q = defaultdict(lambda: np.ones(n_actions) * initial_q_values)
        self.n_agents = n_robots
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._exploration_rate = exploration_rate
        self.training = True
        self._n_actions = n_actions
        
    def action (self, x, n_dirt1 ):


        q_values = self._Q[x]

        # if self.training == True:
        #     time.sleep(0.5)
        #     print(q_values)
        if not self.training or (self.training and np.random.uniform(0, 1) > self._exploration_rate):
            action = np.argwhere(q_values == np.max(q_values)).reshape(-1)

        else:
            action = range(self._n_actions)

        return np.random.choice(action)
    
    def update(self, observation, action, next_observation, reward ):
        
        observation = tuple(observation)
        next_observation = tuple(next_observation)

        x , a, r, y = observation, action, reward, next_observation
        # print(x, a, r)
        alpha, gamma = self._learning_rate, self._discount_factor
        Qxa, Qy = self._Q[x][a] , self._Q[y]
        maxQy = max(Qy)
        self._Q[x][a] = (1 - alpha) * Qxa + alpha * (r + gamma * maxQy)


    def train(self):
        self.training = True

    def eval(self):
        self.training = False
    

if __name__ == '__main__':


    n_agents_testing = 1
    agent_id = np.arange(n_agents_testing)
    grid_size = 5
    n_robots1 = 1
    max_steps = 300


    # env_train = CleaningEnv(grid_size, n_robots1, max_steps)
    # env_eval = CleaningEnv(grid_size, n_robots1, max_steps)
    # with open('\Q_learning_hard_2.pkl', 'rb') as file:
    #     agent = dill.load(file)
    # 2 - Setup agent
    agents = [
         QLearningAgent(agent_id=0, n_robots=n_robots1, n_actions=5),
         QLearningAgent(agent_id=1, n_robots=n_robots1, n_actions=5)
    ]

    # 3 - Evaluate agent
    results = {}
        
    result = train_eval_loop_single( 'QLearning Team' ,agent, 2, 200, 20)
    results[agent[0]] = result


    # with open(F'\Q_learning_hard_2.pkl', 'wb') as file:
    #     dill.dump(agents, file)
    # joblib.dump(agents, 'Q_learning_easy_1.pkl')
compare_results_learning(results, title="Agents on 'Cleaning' Environment", colors=["orange"])