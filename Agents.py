from Environment_clean import CleaningEnv
import argparse
import numpy as np
from gym import Env
import time

def run_agent(environment: Env, agent, n_episodes: int) -> np.ndarray:

    results = np.zeros(n_episodes)

    for episode in range(n_episodes):


        observation = environment.reset()
        done = False
        while not done:

            time.sleep(0.5)
            env.render()

            # Change action taking into account the agent we will use ( right now is random)
            actions = agent.action(observation)
            # actions = np.random.randint(env.n_actions, size=2)

            # Take a step in the environment

            observation, reward, done, info = env.step(actions)

            print(f"Timestep {env.current_step}")
            print(f"\tObservation: {observation}")
            print(f"\tAction: {actions}\n")
            
            if len(env.dirt_positions1)+ len(env.dirt_positions2) == 0:
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
        dirt_position1 = self.observation[self.n_agents:env.dirt_positions1.shape[0], :]
        dirt_position2 = self.observation[env.dirt_positions1.shape[0]:env.dirt_positions2.shape[0], :]
        actions = self.direction_to_go(agents_positions, dirt_position1, dirt_position2)

        return actions
    
    def direction_to_go(self, pos_agents, pos_dirt1, pos_dirt2):

        actions = np.zeros(self.n_agents, dtype=int)
        for i in range(self.n_agents):
            # If the robot is on top of dirt 
            if np.any(np.all(np.equal(pos_agents[i], pos_dirt1), axis=1)):       
                actions[i] = 5   #Clean
            else:
                actions[i] = np.random.randint(env.n_actions, size=1)

        return actions

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--episodes", type=int, default=30)
    # opt = parser.parse_args()

    n_agents_testing = 1
    agent_id = np.arange(n_agents_testing)
    grid_size= 5
    n_robots=2
    max_steps=300

    # 1 - Setup environment
    env = CleaningEnv(grid_size, n_robots, max_steps)

    # 2 - Setup agents
    agents = [
        GreedyAgent(agent_id, n_robots, env.n_actions)
    ]

    # 3 - Evaluate agents
    results = {}
    for agent in agents:
        result = run_agent(env, agent, 1)
        # results[agent.name] = result






