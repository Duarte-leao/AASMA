from Environment_clean import CleaningEnv
from Results_analysis import compare_results
from Agents import Agent

import numpy as np
import time
import random 
import math
from scipy.spatial.distance import cityblock


def run_agent( agent, n_episodes: int,  grid_size, n_robots1, max_steps)  -> np.ndarray:

    results_t = np.zeros(n_episodes)
    results_b = np.zeros(n_episodes)

    for episode in range(n_episodes):

        environment = CleaningEnv(grid_size, n_robots1, max_steps)
        observation = environment.reset()
        done = False
        while not done:
            time.sleep(0.3)
            environment.render()
            
            actions = agent.action(observation, environment.num_dirt1)
            observation, reward, done, info = environment.step(actions)

        environment.close()
        results_t[episode] = environment.current_step
        results_b[episode] = environment.battery

    return results_t, results_b

class Agent_role(Agent):

    def __init__(self, n_robots, n_actions):
        
        super(Agent_role, self).__init__(f"Role Agents")
        self.n_agents = n_robots
        self.n_actions = n_actions
        self.observation = None 


    def action(self, observation, num_dir1) -> int:

        self.observation = observation
        agents_positions = self.observation[:self.n_agents, :]
        dirt1_position = self.observation[self.n_agents:num_dir1+self.n_agents, :]
        dirt2_position = self.observation[num_dir1+self.n_agents:, :]

        dirt1 = math.inf
        dirt2 = math.inf
        actions = []

        if len(dirt1_position) > 0:
            distance1 = self.distances_to_dirt(agents_positions[0], dirt1_position)
            dirt1 = np.min(distance1)
        if len(dirt2_position) > 0:
            distance2 = self.distances_to_dirt(agents_positions[0], dirt2_position)
            dirt2 = np.min(distance2)
    
        role = 0
        if dirt1 > dirt2:

            role = 2 
            if np.any(np.all(np.equal(agents_positions[0], dirt2_position[np.argmin(distance2)]))) :
                
                same_position_robots = np.where(np.all(agents_positions == agents_positions[0], axis=1))[0]
                previous_agent_block_idx = np.argmin(distance2)
                if len(same_position_robots) > 1:
                    actions.append(5) 
                else:
                    actions.append(0) 
            else:
                actions.append(self.direction_to_go(agents_positions[0], dirt2_position[np.argmin(distance2)]))
                previous_agent_block_idx = np.argmin(distance2)

        else:
            role = 1 

            if np.all(np.equal(agents_positions[0], dirt1_position[np.argmin(distance1)])):
                previous_agent_block_idx = np.argmin(distance1)
                actions.append(5)
            else:
                actions.append(self.direction_to_go(agents_positions[0],dirt1_position[np.argmin(distance1)] ))
                previous_agent_block_idx = np.argmin(distance1)
    
        if role == 1:   

            dirt1_position = np.delete(dirt1_position,previous_agent_block_idx)
            distance11 = self.distances_to_dirt(agents_positions[1], dirt1_position)

            if np.all(np.equal(agents_positions[1], dirt1_position[np.argmin(distance11)])):
                actions.append(5)
            else:
                actions.append(self.direction_to_go(agents_positions[1],dirt1_position[np.argmin(distance11)] ))

        elif role == 2: 

            if np.any(np.all(np.equal(agents_positions[1], dirt2_position[previous_agent_block_idx]))) :
                
                same_position_robots = np.where(np.all(agents_positions == agents_positions[1], axis=1))[0]
                if len(same_position_robots) > 1:
                    actions.append(5)
                else:
                    actions.append(0) 
            else:
                actions.append(self.direction_to_go(agents_positions[1], dirt2_position[previous_agent_block_idx]))
        return actions
        

    def direction_to_go(self, agent_position, dirt_position):

        distances = np.array(dirt_position) - np.array(agent_position)
        abs_distances = np.absolute(distances)

        if abs_distances[0] < abs_distances[1]:
            return self._close_horizontally(distances)
        elif abs_distances[0] > abs_distances[1]:
            return self._close_vertically(distances)
        else:
            roll = random.uniform(0, 1)

        return self._close_horizontally(distances) if roll > 0.5 else self._close_vertically(distances)


    def distances_to_dirt(self, agent_position, dirt_positions):

        distances = np.zeros(len(dirt_positions))
        for p in range (len(dirt_positions)):

            distances[p]=(cityblock(agent_position, dirt_positions[p]))
        
        return distances


    def _close_horizontally(self, distances):

        if distances[1] > 0:
            return 4
        elif distances[1] < 0:
            return 3


    def _close_vertically(self, distances):
        if distances[0] > 0:
            return 2
        elif distances[0] < 0:
            return 1


if __name__ == '__main__':

    grid_size = 7
    n_robots = 2
    max_steps = 300
    n_actions = 2
    n_episodes = 20

    agent = Agent_role( n_robots, n_actions)

    results_time = {}
    results_battery = {}

    result_t, result_b = run_agent( agent, n_episodes, grid_size, n_robots, max_steps)
    results_time[agent.name] = result_t
    results_battery[agent.name] = result_b

    compare_results(results_time, title="Agents on 'Cleaning' Environment - Time spent", colors=["yellow"])
    compare_results(results_battery, title="Agents on 'Cleaning' Environment - Battery spent", colors=["yellow"])









