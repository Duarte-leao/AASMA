from Environment_clean import CleaningEnv
from Agents_random import RandomAgent
from Results_analysis import compare_results

import argparse
import numpy as np
from gym import Env
import time
import random 
import math
from scipy.spatial.distance import cityblock



def run_agent(environment: Env, agent, n_episodes: int) -> np.ndarray:

    results = np.zeros(n_episodes)

    for episode in range(n_episodes):


        observations = environment.reset()
        done = False
        while not done:

            time.sleep(1)
            env.render()


            actions = agent.action(observations)
        

            # Take a step in the environment

            observations, reward, done, info = env.step(actions)

            # print(f"Timestep {env.current_step}")
            # print(f"\tObservation: {observations}")
            # print(f"\tAction: {actions}\n")
            
            if (env.num_dirt1 + env.num_dirt2)== 0:
                done = True


        env.close()

        results[episode] = env.current_step
    return results

class GreedyAgent():

    def __init__(self, agent_id, n_robots, n_actions):
        
        super(GreedyAgent, self).__init__
        self.agent_id = agent_id
        self.n_agents = n_robots
        self.n_actions = n_actions
        self.observation = None 

    def action(self, observation) -> int:

        self.observation = observation
        agents_positions = self.observation[:self.n_agents, :]

        dirt_position1 = self.observation[self.n_agents:env.num_dirt1+self.n_agents, :]
        dirt_position2 = self.observation[env.num_dirt1+self.n_agents:env.num_dirt1+self.n_agents+env.num_dirt2, :]

        actions = []
        for i in range(self.n_agents):

            if len(dirt_position1) == 0:

                if np.any(np.all(np.equal(agents_positions[i], dirt_position2), axis=1)) :
                    same_position_robots = np.where(np.all(agents_positions == agents_positions[i], axis=1))[0]
                
                    if len(same_position_robots) > 1:
                        action = 5
                        actions.append(action)
                        break
                
                print('Dirt 2: ', dirt_position2)
                near_dirt = self.closest_dirt2( agents_positions, dirt_position2, agents_positions[i])
                action = self.direction_to_go1(agents_positions[i], near_dirt)
             
                actions.append(action)
            else:
                print('Number of the agent: ' , i)
                near_dirt = self.closest_dirt1(agents_positions[i], dirt_position1)
                action = self.direction_to_go1(agents_positions[i], near_dirt)
                print('Action:', action)
                actions.append(action)
            
        return actions
    

    def direction_to_go1(self, agent_position, dirt_position):

        distances = np.array(dirt_position) - np.array(agent_position)
        abs_distances = np.absolute(distances)

        if abs_distances[0]== 0 and  abs_distances[1]== 0 :
            return 5
        elif abs_distances[0] < abs_distances[1]:
            return self._close_horizontally(distances)
        elif abs_distances[0] > abs_distances[1]:
            return self._close_vertically(distances)
        else:
            roll = random.uniform(0, 1)

        return self._close_horizontally(distances) if roll > 0.5 else self._close_vertically(distances)

    

    def closest_dirt2(self, agent_positions, dirt_postions, atual_agent):

        distance_dirt = np.zeros(env.num_dirt2)
        distance_agents = np.zeros(self.n_agents)
        distance = np.zeros((len(distance_agents), len(distance_dirt)))
        min = math.inf
        n_dirt = int(len(dirt_postions) / 2)
        
        other_agents = np.delete(agent_positions, np.where(np.all(agent_positions == atual_agent, axis=1)), axis=0)
        print('All agents postions:', agent_positions)
        print('Other agens: ', other_agents )
        print('Agent: ', atual_agent)
        for p in range(n_dirt):

            distance_dirt[p] = cityblock(atual_agent, dirt_postions[p])

        for i, agent in enumerate (other_agents):

            distance_agents[i] = cityblock(atual_agent, other_agents[i])

        distance = np.outer(distance_agents, distance_dirt)
        max_index = np.argmax(distance)
        max_position = np.unravel_index(max_index, distance.shape)

        return dirt_postions[max_position[1],:]


    def check_same_position_partner(self, agent_positions, actual_agent):

        other_agents = np.delete(agent_positions, np.where(np.all(agent_positions == actual_agent, axis=1)), axis=0)

        for position in other_agents:
     
            if position == actual_agent:
                return True
        return False


    def closest_dirt1(self, agent_position, dirt_positions):

        min = math.inf
        closest_dirt_position = None
        
        for p in (dirt_positions):

            distance = cityblock(agent_position, p)
            if distance < min:
                min = distance
                closest_dirt_position = p
            
        return closest_dirt_position


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


    # def direction_to_go(self, pos_agents, pos_dirt1, pos_dirt2):

    #     actions = np.zeros(self.n_agents, dtype=int)
    #     for i in range(self.n_agents):
            
    #         if np.any(np.all(np.equal(pos_agents[i], pos_dirt1), axis=1)):      
    #             actions[i] = 5   
    #         elif np.any(np.all(np.equal(pos_agents[i], pos_dirt2), axis=1)) and np.all(pos_agents == pos_agents[i]):
    #             actions[i] = 5 
    #         else:     
    #             actions[i] = np.random.randint(env.n_actions, size=1)

    #     return actions

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--episodes", type=int, default=30)
    # opt = parser.parse_args()

    n_agents_testing = 1
    agent_id = np.arange(n_agents_testing)
    grid_size = 5
    n_robots = 2
    max_steps = 300
    

    # 1 - Setup environment
    env = CleaningEnv(grid_size, n_robots, max_steps)

    # 2 - Setup agents
    agents = [
        GreedyAgent(agent_id=0, n_robots=2, n_actions=env.n_actions),
        RandomAgent(agent_id=1, n_robots=2, n_actions=env.n_actions)
    ]

    # 3 - Evaluate agents
    results = {}
    for agent in agents:
        result = run_agent(env, agent, 1)
        results[agent.name] = result

    compare_results(results, title="Agents on 'Predator Prey' Environment", colors=["orange", "green"])







