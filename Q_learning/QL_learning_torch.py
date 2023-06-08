from Env_torch import CleaningEnv
from Agents.Results_analysis import compare_results_learning

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
import torch

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



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

    results = torch.zeros(n_episodes)
    

    for episode in range(n_episodes):
        environment = CleaningEnv(5, 2, 600)
        environment.to(device)
        terminals = [False for _ in range(len(agents))]
        observations = environment.reset()

        observations = see(observations, environment.num_dirt1, environment.num_robots)
        
        done = False
        while not done:
            
            environment.render()
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

    dirt_1 = torch.zeros(n_dirt1, dtype=int).to(device)
    dirt_2 = torch.ones(len(observation)-n_agents-n_dirt1, dtype=int).to(device)
    dirt_levels = torch.cat(dirt_1, dirt_2)
    obs1 = tuple(map(tuple, torch.cat(observation[n_agents:], dirt_levels[:, np.newaxis], dim=1)))
    obs2 = tuple(map(tuple, observation[:n_agents]))
    observations = obs2 + obs1 
    observations = tuple(itertools.chain.from_iterable(observations))
    # print(observations)
    return observations


class QLearningAgent():

    def __init__(self, agent_id, n_robots, n_actions, learning_rate=0.18, discount_factor=0.95, exploration_rate=0.15, initial_q_values = 0.0):
        super(QLearningAgent, self).__init__
        self._Q = defaultdict(lambda: torch.ones(n_actions) * initial_q_values)
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
            action = np.argwhere(q_values.detach().cpu().numpy() == torch.max(q_values).detach().cpu().numpy()).reshape(-1)

        else:
            action = range(self._n_actions)

        return np.choice(action.detach().cpu().numpy())
    
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
    grid_size = 5
    n_robots1 = 1
    max_steps = 300


    # 2 - Setup agent

    Q1 = QLearningAgent(agent_id=0, n_robots=n_robots1, n_actions=5)
    Q2 = QLearningAgent(agent_id=0, n_robots=n_robots1, n_actions=5)
    Q1.to(device); Q2.to(device);  



    agents = [
         Q1,
         Q2
    ]


    # 3 - Evaluate agent
    results = {}
    
    result = train_eval_loop_single( 'QLearning Team' ,agents, 4000, 100, 20)
    results[agents[0]] = result


    with open(F'\Q_learning_hard_1.pkl', 'wb') as file:
        dill.dump(agents, file)
    # joblib.dump(agents, 'Q_learning_easy_1.pkl')
compare_results_learning(results, title="Agents on 'Cleaning' Environment", colors=["orange", "green"])