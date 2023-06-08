from Agents.Environment_clean import CleaningEnv
from Agents.Results_analysis import compare_results
from Agents.Agents import Agent

import argparse
import numpy as np
from gym import Env
import time
import random 
import math
from scipy.spatial.distance import cityblock


def run_agent( agent, n_episodes: int,  grid_size, n_robots1, max_steps) -> np.ndarray:

    results = np.zeros(n_episodes)

    for episode in range(n_episodes):
        
        print('Episode Number', episode)   
        environment = CleaningEnv(grid_size, n_robots1, max_steps)
        observations = environment.reset()
        done = False

        while not done:
            
            environment.render()
            if (environment.num_dirt1 + environment.num_dirt2) == 0:
                 break
            time.sleep(0.5)
            actions = agent.action(observations, environment.num_dirt1)
            observations, reward, done, info = environment.step(actions)

            print(f"Timestep {environment.current_step}")
            # print(f"\tObservation: {observations}")
            print(f"\tAction: {actions}\n")
            
        environment.close()

        results[episode] = environment.current_step
    return results

class Agents_Social_convention(Agent):

    def __init__(self, agent_id, n_robots, n_actions):
        
        super(Agents_Social_convention, self).__init__(f"Coordination Agents")
        self.agent_id = agent_id
        self.n_agents = n_robots
        self.n_actions = n_actions
        self.observation = None 
        self.previous_agent_block_idx = 100
        self.role_of_previous_agent = 0


    def action(self, observation, num_dir1) -> int:

        self.observation = observation
        agents_positions = self.observation[:self.n_agents, :]
        dirt1_position = self.observation[self.n_agents:num_dir1+self.n_agents, :]
        dirt2_position = self.observation[num_dir1+self.n_agents:, :]

        dirt1 = math.inf
        dirt2 = math.inf
        actions = np.zeros(self.n_agents)

        for i in range(self.n_agents):

            print('AGENT', i, 'position', agents_positions[i])
            actions[i] = self.function( agents_positions, i,dirt1_position, dirt2_position)


        return actions

    def function(self, agent_position, agent_id, dirt1, dirt2):
        # print('dirt level 2', dirt2)
        print('role', self.role_of_previous_agent)


        min1 = math.inf
        min2 = math.inf
        if len(dirt1) > 0:
            dist1 = self.distances_to_dirt(agent_position[agent_id], dirt1)
            min1 = min(dist1)
            min_dist_1 = np.argmin(dist1)
            
        if len(dirt2) > 0: 
            dist2 = self.distances_to_dirt(agent_position[agent_id], dirt2)
            min2 = min(dist2)
            min_dist_2 = np.argmin(dist2)

        if self.role_of_previous_agent == 0:

            if self.previous_agent_block_idx != 100 and len(dirt1) > 1 :

                print('dist1', dist1, 'index', self.previous_agent_block_idx, 'dirt1', dirt1)
                dist1 = np.delete(dist1, self.previous_agent_block_idx, axis=0)
                min_dist_1 = np.argmin(dist1)

            if min1 < min2:

                action = self.direction_to_go(agent_position[agent_id], dirt1[min_dist_1])  
                self.role_of_previous_agent = 0 
                self.previous_agent_block_idx = min_dist_1
                
                if np.all(np.equal(agent_position[agent_id], dirt1[min_dist_1])):
                     action = 5
            else:
                print('min2', min_dist_2, 'dist1',dirt2 )

                self.role_of_previous_agent = 1 # leader
                self.previous_agent_block_idx = min_dist_2
                action = self.direction_to_go(agent_position[agent_id], dirt2[min_dist_2]) 

                if np.any(np.all(np.equal(agent_position[agent_id], dirt2[self.previous_agent_block_idx]))):
            
                    same_position_robots = np.where(np.all(agent_position == agent_position[agent_id], axis=1))[0]
                    if len(same_position_robots) > 1:
                        action = 5
                        self.role_of_previous_agent = 0
                    else:
                        action = 0

        else:
            
            print('index', self.previous_agent_block_idx)
            print('dirt', dirt2)
            action = self.direction_to_go(agent_position[agent_id], dirt2[self.previous_agent_block_idx])

            if np.any(np.all(np.equal(agent_position[agent_id], dirt2[self.previous_agent_block_idx]))):
                        
                same_position_robots = np.where(np.all(agent_position == agent_position[agent_id], axis=1))[0]
                if len(same_position_robots) > 1:
                    action = 5
                else:
                    action = 0
                self.role_of_previous_agent = 0
                self.previous_agent_block_idx = 100
        
        return action
    

    def direction_to_go(self, agent_position, dirt_position):
        print('dirt',dirt_position )
        distances = np.array(dirt_position) - np.array(agent_position)
        abs_distances = np.absolute(distances)

        if abs_distances[0] < abs_distances[1]:
            return self._close_horizontally(distances)
        elif abs_distances[0] > abs_distances[1]:
            return self._close_vertically(distances)
        else:
            roll = random.uniform(0, 1)

        return self._close_horizontally(distances) if roll > 0.5 else self._close_vertically(distances)

    # def clean_verification(self, agent_pos, id, dirt):
        
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


    n_agents_testing = 1
    agent_id = np.arange(n_agents_testing)
    grid_size = 5
    n_robots1 = 2
    max_steps = 300
    

    # 2 - Setup agents
    agents = [
        Agents_Social_convention(0, n_robots=n_robots1, n_actions=5)
    ]

    # 3 - Evaluate agents
    results = {}
    for agent in agents:
        
        
        result = run_agent( agent, 20, grid_size, n_robots1, max_steps)
        results[agent] = result

    compare_results(results, title="Agents on 'Cleaning' Environment", colors=["orange", "green"])






