from Agents_reactive_0 import Reactive_0
from Environment_clean_QL import CleaningEnvQL
from Environment_clean_Reactive import CleaningEnv
from Agents_reactive_0 import Reactive_0
from Agents_reactive_1 import Reactive_1
from Agents_reactive_2 import Reactive_2
from Agents_Role import Agent_role
from Results_analysis import compare_results
import itertools
import dill
import cloudpickle

import numpy as np
import time
from scipy.spatial.distance import cityblock

def run_agent( agent,  n_episodes: int,  grid_size, n_robots1, max_steps)  -> np.ndarray:

    results_t = np.zeros(n_episodes)
    results_b = np.zeros(n_episodes)

    for episode in range(n_episodes):

        environment = CleaningEnv(grid_size, n_robots1, max_steps)
        observation = environment.reset()
        done = False
        while not done:
    
            if (environment.num_dirt1 + environment.num_dirt2) == 0:
                break

            # environment.render()
            actions = agent.action(observation, environment.num_dirt1)
            observation, reward, done, info = environment.step(actions)
        environment.close()
        results_t[episode] = environment.current_step
        results_b[episode] = environment.battery

    return results_t, results_b

def run_agent_Ql( agents, n_episodes: int,  grid_size, n_robots1, max_steps) -> np.ndarray:

    results_t= np.zeros(n_episodes)
    results_b = np.zeros(n_episodes)

    for episode in range(n_episodes):

        agents[0].eval()
        agents[1].eval()
        environment = CleaningEnvQL(grid_size, n_robots1, max_steps)
        terminals = [False for _ in range(len(agents))]
        observations = environment.reset()
        observations = see(observations, environment.num_dirt1, environment.num_robots)
        done = False
        while not done:
            
            # environment.render()
            if (environment.num_dirt1 + environment.num_dirt2) == 0:
                 break
        
            actions = [agent.action(observations, environment.num_dirt1) for agent in agents]
            next_observations, rewards, done, info = environment.step(actions)
            next_observations = see(next_observations, environment.num_dirt1, environment.num_robots)

            for a, agent in enumerate(agents):
                if agent.training:
                    agent.update(observations, actions[a], next_observations, rewards)
        
            observations = next_observations
        
        results_t[episode] = environment.current_step 
        results_b[episode] = environment.battery
        environment.close()

    return results_t, results_b


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


if __name__ == '__main__':

    grid_size = 5
    n_robots = 2
    max_steps = 500
    n_episodes = 20

    with open('Q_learning_hard_3.pkl', 'rb') as file:
        Q_learning = dill.load(file)

    # 2 - Setup agents
    agents = [
        Reactive_0(n_robots, n_actions=5),
        Reactive_1(n_robots, n_actions=5),
        Reactive_2(n_robots, n_actions=5),
        Agent_role(n_robots, n_actions=5),
        Q_learning
    ]

    results_time = {}
    results_battery = {}

    for i, agent in enumerate (agents):
        
        if i !=4:
            result_t, result_b = run_agent( agent, n_episodes, grid_size, n_robots, max_steps)
            results_time[agent.name] = result_t
            results_battery[agent.name] = result_b
        else:
            result_t, result_b = run_agent_Ql(agent, n_episodes, grid_size, n_robots, max_steps)
            results_time['Q_learning Agent'] = result_t
            results_battery['Q_learning Agent'] =  result_b
        # results_battery[agent] = result_b

    compare_results(results_time, title="Agents on 'Cleaning' Environment - Time spent", colors=["orange", "green", 'red', 'blue', 'yellow'])
    compare_results(results_battery, title="Agents on 'Cleaning' Environment - Battery spent", colors=["orange", "green", 'red', 'blue', 'yellow'])

