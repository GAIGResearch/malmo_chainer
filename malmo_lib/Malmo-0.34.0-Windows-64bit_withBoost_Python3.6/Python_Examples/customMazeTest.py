from __future__ import print_function
from __future__ import division
from builtins import range
from past.utils import old_div
import MalmoPython
import os
import random
import sys
import time
import json
import malmoutils

malmoutils.fix_print()
MalmoPython.setLogging("", MalmoPython.LoggingSeverityLevel.LOG_OFF)

agent_host = MalmoPython.AgentHost()
agent_host.addOptionalIntArgument( "speed,s", "Length of tick, in ms.", 50)
malmoutils.parse_command_line(agent_host)

def processFrame( frame ):
    '''Track through the middle line of the depth data and find the max discontinuities'''
    global current_yaw_delta_from_image

    y = int(old_div(video_height, 2))
    rowstart = y * video_width
    
    v = 0
    v_max = 0
    v_max_pos = 0
    v_min = 0
    v_min_pos = 0
    
    dv = 0
    dv_max = 0
    dv_max_pos = 0
    dv_max_sign = 0
    
    d2v = 0
    d2v_max = 0
    d2v_max_pos = 0
    d2v_max_sign = 0
    
    for x in range(0, video_width):
        nv = frame[(rowstart + x) * 4 + 3]
        ndv = nv - v
        nd2v = ndv - dv

        if nv > v_max or x == 0:
            v_max = nv
            v_max_pos = x
            
        if nv < v_min or x == 0:
            v_min = nv
            v_min_pos = x

        if abs(ndv) > dv_max or x == 1:
            dv_max = abs(ndv)
            dv_max_pos = x
            dv_max_sign = ndv > 0
            
        if abs(nd2v) > d2v_max or x == 2:
            d2v_max = abs(nd2v)
            d2v_max_pos = x
            d2v_max_sign = nd2v > 0
            
        d2v = nd2v
        dv = ndv
        v = nv
    
    print("d2v, dv, v: " + str(d2v) + ", " + str(dv) + ", " + str(v))

    # We want to steer towards the greatest d2v (ie the biggest discontinuity in the gradient of the depth map).
    # If it's a possitive value, then it represents a rapid change from close to far - eg the left-hand edge of a gap.
    # Aiming to put this point in the leftmost quarter of the screen will cause us to aim for the gap.
    # If it's a negative value, it represents a rapid change from far to close - eg the right-hand edge of a gap.
    # Aiming to put this point in the rightmost quarter of the screen will cause us to aim for the gap.
    if dv_max_sign:
        edge = old_div(video_width, 4)
    else:
        edge = 3 * video_width / 4

    # Now, if there is something noteworthy in d2v, steer according to the above comment:
    if d2v_max > 8:
        current_yaw_delta_from_depth = (old_div(float(d2v_max_pos - edge), video_width))
    else:
        # Nothing obvious to aim for, so aim for the farthest point:
        if v_max < 255:
            current_yaw_delta_from_depth = (old_div(float(v_max_pos), video_width)) - 0.5
        else:
            # No real data to be had in d2v or v, so just go by the direction we were already travelling in:
            if current_yaw_delta_from_depth < 0:
                current_yaw_delta_from_depth = -1
            else:
                current_yaw_delta_from_depth = 1
    
#----------------------------------------------------------------------------------------------------------------------------------

current_yaw_delta_from_image = 0

video_width = 800
video_height = 600

maze = '''
      <MazeDecorator>
        <Seed>random</Seed>
        <SizeAndPosition width="10" length="10" height="10" xOrigin="-32" yOrigin="69" zOrigin="-5"/>
        <StartBlock type="emerald_block" fixedToEdge="true"/>
        <EndBlock type="redstone_block" fixedToEdge="true"/>
        <PathBlock type="diamond_block"/>
        <FloorBlock type="air"/>
        <GapBlock type="air"/>
        <GapProbability>'''+str(float(old_div(4,10.0)))+'''</GapProbability>
        <AllowDiagonalMovement>false</AllowDiagonalMovement>
      </MazeDecorator>
'''

def GetMissionXML( agent_host ):
    return '''<?xml version="1.0" encoding="UTF-8" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <About>
            <Summary>Run the maze!</Summary>
        </About>
        
        <ModSettings>
            <MsPerTick>''' + str(TICK_LENGTH) + '''</MsPerTick>
        </ModSettings>

        <ServerSection>
            <ServerInitialConditions>
                <Time>
                    <StartTime>0</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <Weather>clear</Weather>
            </ServerInitialConditions>
            <ServerHandlers>
                <FlatWorldGenerator generatorString="random" />
                ''' + maze + '''
                <ServerQuitFromTimeUp timeLimitMs="30000"/>
                <ServerQuitWhenAnyAgentFinishes />
            </ServerHandlers>
        </ServerSection>

        <AgentSection mode="Survival">
            <Name>marLo</Name>
            <AgentStart>
                <Placement x="-204" y="81" z="217"/>
            </AgentStart>
            <AgentHandlers>
                <ContinuousMovementCommands turnSpeedDegs="840">
                    <ModifierList type="deny-list"> <!-- Example deny-list: prevent agent from strafing -->
                        <command>strafe</command>
                    </ModifierList>
                </ContinuousMovementCommands>''' + malmoutils.get_video_xml(agent_host) + '''
                <AgentQuitFromTouchingBlockType>
                    <Block type="redstone_block"/>
                </AgentQuitFromTouchingBlockType>
                <VideoProducer>
                    <Width>''' + str(video_width) + '''</Width>
                    <Height>''' + str(video_height) + '''</Height>
                </VideoProducer>
            </AgentHandlers>
        </AgentSection>

    </Mission>'''


validate = True
agent_host.setObservationsPolicy(MalmoPython.ObservationsPolicy.LATEST_OBSERVATION_ONLY)

if agent_host.receivedArgument("test"):
    num_reps = 10
else:
    num_reps = 30000

TICK_LENGTH = agent_host.getIntArgument("speed")

for iRepeat in range(num_reps):
    my_mission_record = malmoutils.get_default_recording_object(agent_host, "Mission_{}".format(iRepeat + 1))
    my_mission = MalmoPython.MissionSpec(GetMissionXML(agent_host),validate)

    max_retries = 3
    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2)

    print("Waiting for the mission to start", end=' ')
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        if len(world_state.errors):
            print()
            for error in world_state.errors:
                print("Error:",error.text)
                exit()
    print()

    agent_host.sendCommand("pitch 0.2")
    time.sleep(1)
    agent_host.sendCommand("pitch 0")
    agent_host.sendCommand("move 0.2")
    
    # main loop:
    while world_state.is_mission_running:
        world_state = agent_host.getWorldState()
        while world_state.number_of_video_frames_since_last_state < 1 and world_state.is_mission_running:
            print("Waiting for frames...")
            time.sleep(0.05)
            world_state = agent_host.getWorldState()

        print("Got frame!")
        
        if world_state.is_mission_running:
            processFrame(world_state.video_frames[0].pixels)
            
            agent_host.sendCommand( "turn " + str(current_yaw_delta_from_image) )
                
    print("Mission has stopped.")
    time.sleep(0.5) # Give mod a little time to get back to dormant state.
