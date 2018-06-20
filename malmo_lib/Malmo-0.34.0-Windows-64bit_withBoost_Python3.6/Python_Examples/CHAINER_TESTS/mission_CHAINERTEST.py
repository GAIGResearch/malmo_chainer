from __future__ import print_function

import sys
sys.path.append('../')

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
import numpy as np
import time
import malmoutils

class MissionEnvironment:
    def __init__(self, mission_file, ah, clients, _actions):
        self.agent_host = ah
        self.acting_agent = 0
        self.my_clients = clients
        self.agentID = 0
        self.actions = _actions
        
        with open(mission_file, 'r') as f:
            print("Loading mission from %s" % mission_file)
            mission_xml = f.read()
            self.my_mission = MalmoPython.MissionSpec(mission_xml, True)
        self.my_mission.removeAllCommandHandlers()
        self.my_mission.allowAllDiscreteMovementCommands()
        self.my_mission.requestVideo( 320, 240 )
        self.my_mission.setViewpoint( 1 )

        # add holes for interest
        for z in range(2,12,2):
            x = random.randint(1,3)
            self.my_mission.drawBlock( x,45,z,"lava")

        self.my_mission_record = malmoutils.get_default_recording_object(self.agent_host, "./RESULTS/save_test")

    def observe(self):
        #return json.loads(self.agent_host.getWorldState().observations[-1].text)
        return np.full((20,), 5, dtype=np.float32)
    
    def reset(self):
        #return json.loads(self.agent_host.getWorldState().observations[-1].text)
        self.agent_host.startMission(self.my_mission, self.my_mission_record)
        return np.full((20,), 5, dtype=np.float32)

    def startMission(self):
        self.agent_host.startMission(self.my_mission, self.my_mission_record)
        
    def step(self, action):
        print("stepping agent, acting_agent="+str(self.acting_agent))
        world_state = self.agent_host.getWorldState()
        current_r = sum(r.getValue() for r in world_state.rewards)
        print("STEP action="+str(action))
        self.agent_host.sendCommand(self.actions[action])
        #action = self.acting_agent.act_and_train(np.full((20,), 5, dtype=np.float32), current_r)
        return np.full((20,), 5, dtype=np.float32), current_r, False, {}

    def assignAgent(self, agent):
        self.acting_agent = agent
        
        
