# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 23:00:12 2018

@author: Oliver Gent

Master Script For UAS iMechE challenge
"""

print('initialise GENT SYSTEMS')
# perform health checks

# connect to MAVlink - lED once complete

# wait for arm command from PWM input or from MAVlink Command
# mission selection pre determined

# payload mission
#   arm
#   take off
#   loiter
#   auto mode
#   when landed - disArm


# reconasance mission
#   arm
#   takeoff
#   loiter
#   auto-mode
#   start camera worker
#   start image recognition
#       these processes need to be in parallel.. 
#       need to get gps co-ords of where photos are taken
#   communicate letters found - perhaps at time found.. 
#   once auto complete and landed
#   disArm