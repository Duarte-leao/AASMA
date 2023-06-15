from Environment_clean_QL import CleaningEnvQL
from Results_analysis import compare_results_learning
from Agents import Agent
import argparse
import numpy as np
from gym import Env
from scipy.spatial.distance import cityblock
from collections import defaultdict
import itertools    
import cloudpickle
import dill



def train_eval_loop_single( team, agents, n_evaluations, n_training_episodes, n_eval_episodes, grid_size, n_robots, max_steps):

    print(f"Train-Eval Loop for {team}\n")
    results = np.zeros((n_evaluations, n_eval_episodes))

    for evaluation in range(n_evaluations):

        # Train

        for agent in agents: agent.train()   # Enables training mode
        # Run train iteration
        run_agent( agents, n_training_episodes, grid_size, n_robots, max_steps)

        # Eval
        for agent in agents: agent.eval()  

        # Run eval iteration
        results[evaluation] = run_agent( agents ,n_eval_episodes, grid_size, n_robots, max_steps)
   
    return results


def run_agent( agents,  n_episodes: int, grid_size, n_robots, max_steps) -> np.ndarray:

    results = np.zeros(n_episodes)
    

    for episode in range(n_episodes):
        environment = CleaningEnvQL(grid_size, n_robots, max_steps)
        terminals = [False for _ in range(len(agents))]
        observations = environment.reset()
        observations = tuple(itertools.chain.from_iterable(observations))
        done = False
        while not done:
            
            # environment.render()
            if (environment.num_dirt1 + environment.num_dirt2) == 0:
                 break
            
            actions = [agent.action(observations, environment.num_dirt1) for agent in agents]
            next_observations, rewards, done, info = environment.step(actions)
            next_observations = tuple(itertools.chain.from_iterable(next_observations))

            for a, agent in enumerate(agents):
                if agent.training:
                    agent.update(observations, actions[a], next_observations, rewards)
        
            observations = next_observations
        
        results[episode] = environment.current_step 
        environment.close()

    return results



class QLearningAgent(Agent):

    def __init__(self, n_robots, n_actions, learning_rate=0.18, discount_factor=0.95, exploration_rate=0.15, initial_q_values = 0.0):
        super(QLearningAgent, self).__init__(f"Q Learning Agent")
        self._Q = defaultdict(lambda: np.ones(n_actions) * initial_q_values)
        self.n_agents = n_robots
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._exploration_rate = exploration_rate
        self.training = True
        self._n_actions = n_actions
        
    def action (self, x, n_dirt1 ):

        q_values = self._Q[x]
        if not self.training or (self.training and np.random.uniform(0, 1) > self._exploration_rate):
            action = np.argwhere(q_values == np.max(q_values)).reshape(-1)

        else:
            action = range(self._n_actions)

        return np.random.choice(action)
    
    def update(self, observation, action, next_observation, reward ):
        
        observation = tuple(observation)
        next_observation = tuple(next_observation)

        x , a, r, y = observation, action, reward, next_observation
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
    n_robots = 2
    max_steps = 300
    n_evaluations = 3000
    n_epsidoe_evaluations = 10
    n_train = 100

    agents = [
         QLearningAgent(n_robots, n_actions=5),
         QLearningAgent(n_robots, n_actions=5)
    ]

    results = {}
        
    result = train_eval_loop_single( 'QLearning Team' ,agents, n_evaluations, n_train, n_epsidoe_evaluations, grid_size, n_robots, max_steps)
    results[agents[0].name] = result

    with open(F'\Q_learning_model_4.pkl', 'wb') as file:
        cloudpickle.dump(agents, file)

    with open(F'results_4.pkl', 'wb') as file:
        cloudpickle.dump(results[agents[0].name], file)

compare_results_learning(results, title="Agents on 'Cleaning' Environment", colors=["orange"])