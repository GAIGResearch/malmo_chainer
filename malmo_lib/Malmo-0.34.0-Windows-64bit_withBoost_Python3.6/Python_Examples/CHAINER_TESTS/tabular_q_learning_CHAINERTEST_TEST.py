from __future__ import print_function

import sys
sys.path.append("../")
# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# The "Cliff Walking" example using Q-learning.
# From pages 148-150 of:
# Richard S. Sutton and Andrews G. Barto
# Reinforcement Learning, An Introduction
# MIT Press, 1998


#-----------------------------------------CHAINER STUFF-------------------------------------------------
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np

class QFunction(chainer.Chain):

    def __init__(self, obs_size, n_actions, n_hidden_channels=50):
        super().__init__()
        with self.init_scope():
            self.l0 = L.Linear(obs_size, n_hidden_channels)
            self.l1 = L.Linear(n_hidden_channels, n_hidden_channels)
            self.l2 = L.Linear(n_hidden_channels, n_actions)

    def __call__(self, x, test=False):
        """
        Args:
            x (ndarray or chainer.Variable): An observation
            test (bool): a flag indicating whether it is in test mode
        """
        print("QFunction call x = "+str(x))
        #raise RuntimeError
        h = F.tanh(self.l0(x))
        h = F.tanh(self.l1(h))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))

#-----------------------------------------CHAINER STUFF-------------------------------------------------


from future import standard_library
standard_library.install_aliases()
from builtins import input
from builtins import range
from builtins import object
import MalmoPython
import json
import logging
import math
import os
import random
from random import randint
import time
import malmoutils
from mission_CHAINERTEST import MissionEnvironment
from qAgent_CHAINERTEST import TabQAgent


def uniformRandom():
    act = randint(0,len(actionSet)-1)
    print("uniformRandom act="+str(act)+", len(actionSet)="+str(len(actionSet)))
    return act


# -- MALMO AGENT -- #
# Actually initialize the agent
agent_host = MalmoPython.AgentHost()

# Find the default mission file by looking next to the schemas folder:
schema_dir = None
try:
    schema_dir = os.environ['MALMO_XSD_PATH']
except KeyError:
    print("MALMO_XSD_PATH not set? Check environment.")
    exit(1)
mission_file = os.path.abspath(os.path.join(schema_dir, '..', 'sample_missions', 'cliff_walking_1.xml'))

# add some args
agent_host.addOptionalStringArgument('mission_file',
    'Path/to/file from which to load the mission.', mission_file)
agent_host.addOptionalFloatArgument('alpha',
    'Learning rate of the Q-learning agent.', 0.1)
agent_host.addOptionalFloatArgument('epsilon',
    'Exploration rate of the Q-learning agent.', 0.01)
agent_host.addOptionalFloatArgument('gamma', 'Discount factor.', 1.0)
agent_host.addOptionalFlag('load_model', 'Load initial model from model_file.')
agent_host.addOptionalStringArgument('model_file', 'Path to the initial model file', '')
agent_host.addOptionalFlag('debug', 'Turn on debugging.')

malmoutils.parse_command_line(agent_host)

# -- set up the agent -- #
#actionSet = list()
#actionSet.append(agent_host.sendCommand("movenorth 1"))
#actionSet.append(agent_host.sendCommand("movesouth 1"))
#actionSet.append(agent_host.sendCommand("movewest 1"))
#actionSet.append(agent_host.sendCommand("moveeast 1"))
#actionSet = [lambda: agent_host.sendCommand("movenorth 1"), lambda: agent_host.sendCommand("movesouth 1"), lambda: agent_host.sendCommand("movewest 1"), lambda: agent_host.sendCommand("moveeast 1")]
actionSet = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
#actionSet = ["stuff", "staff", lambda: print("stiff!")]
#actionSet = [print("act1"), print("act2"), print("act3"), print("act4")]

my_clients = MalmoPython.ClientPool()
my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available

# CREATE and START the mission using the MALMO agent
my_mission = MissionEnvironment(agent_host.getStringArgument('mission_file'), agent_host, my_clients, actionSet)

print("Starting mission now")
my_mission.startMission()

print("Waiting for the mission to start", end=' ')
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)
print()

world_state = agent_host.getWorldState()

while world_state.number_of_observations_since_last_state <= 0:
    world_state = agent_host.getWorldState()

# -- Chainer RL implementation here -- #
print("DEBUG ---- observations = "+str(world_state.observations))
msg = world_state.observations[-1].text
observations = json.loads(msg)
obs_size = len(observations)
print("obs_size = "+str(obs_size))
n_actions = np.array(actionSet).shape[0]
print("n_actions = "+str(n_actions))

q_func = QFunction(obs_size, n_actions)
_q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(
    obs_size, n_actions,
    n_hidden_layers=2, n_hidden_channels=50)

# Use Adam to optimize q_func. eps=1e-2 is for stability.
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)

# Set the discount factor that discounts future rewards.
gamma = 0.95

# Use epsilon-greedy for exploration
print("ACTIONSET[0] = "+str(actionSet))
explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=uniformRandom)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

# Since observations from CartPole-v0 is numpy.float64 while
# Chainer only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
phip = lambda x: (lambda y: print("PHIP FUNCTION: x[y]="+x[y]))
phi = lambda x: (lambda y: x[y].astype(np.float32, copy=False))

# Now create an agent that will interact with the environment.
agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500, update_interval=1,
    target_update_interval=100)

# Assign this as the acting agent for the mission
my_mission.assignAgent(agent)

# -- RUN CHAINER -- #
chainerrl.experiments.train_agent_with_evaluation(
    agent, my_mission,
    steps=2000,           # Train the agent for 2000 steps
    eval_n_runs=10,       # 10 episodes are sampled for each evaluation
    max_episode_len=200,  # Maximum length of each episodes
    eval_interval=1000,   # Evaluate the agent after every 1000 steps
    outdir='result'       # Save everything to 'result' directory
)

while world_state.is_mission_running:
    continue

print("Done.")

print()
print("Cumulative rewards for all runs:")
