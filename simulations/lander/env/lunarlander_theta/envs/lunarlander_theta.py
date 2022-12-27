"""
Rocket trajectory optimization is a classic topic in Optimal Control.

According to Pontryagin's maximum principle it's optimal to fire engine full throttle or
turn it off. That's the reason this environment is OK to have discreet actions (engine on or off).

The landing pad is always at coordinates (0,0). The coordinates are the first two numbers in the state vector.
Reward for moving from the top of the screen to the landing pad and zero speed is about 100..140 points.
If the lander moves away from the landing pad it loses reward. The episode finishes if the lander crashes or
comes to rest, receiving an additional -100 or +100 points. Each leg with ground contact is +10 points.
Firing the main engine is -0.3 points each frame. Firing the side engine is -0.03 points each frame.
Solved is 200 points.

Landing outside the landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land
on its first attempt. Please see the source code for details.

To see a heuristic landing, run:

python gym/envs/box2d/lunar_lander.py

To play yourself, run:

python examples/agents/keyboard_agent.py LunarLander-v2

Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
"""


"""
s (list): The state. Attributes:
          s[0] is the horizontal coordinate
          s[1] is the vertical coordinate
          s[2] is the horizontal speed
          s[3] is the vertical speed
          s[4] is the angle
          s[5] is the angular speed
          s[6] 1 if first leg has contact, else 0
          s[7] 1 if second leg has contact, else 0
"""


import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle
import random

FPS = 50
SCALE = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

INITIAL_RANDOM = 0.0   # Set 1500 to make game harder

LANDER_POLY =[
    (-14, +17), (-17, 0), (-17 ,-10),
    (+17, -10), (+17, 0), (+14, +17)
    ]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY = 12.0

VIEWPORT_W = 600
VIEWPORT_H = 400

HYPOTHESES = {'center': [0, 0, -100, 0, 0, -100, 0, -100, 0, 10, 10], 'anywhere': [0, -100, 0, 0, 0, -100, 0, -100, 0, 10, 10], 'crash': [0, 0, -100, 0, 0, -100, 0, 0, 0, 0, 0], 'hypo0': [0, 0, -101, 0, 0, -104, 0, 0, 0, 0, 0], 'hypo1': [0, 0, -135, 0, 0, -66, 0, -138, 0, 6, 14], 'hypo2': [0, -119, 0, 0, 0, -64, 0, -131, 0, 2, 4], 'hypo3': [0, 0, -116, 0, 0, -95, 0, -58, 0, 18, 12], 'hypo4': [0, 0, -74, 0, 0, -67, 0, -100, 0, 7, 18], 'hypo5': [0, 0, -85, 0, 0, -70, 0, 0, 0, 0, 0], 'hypo6': [0, 0, -130, 0, 0, -112, 0, -68, 0, 4, 2], 'hypo7': [0, 0, -81, 0, 0, -61, 0, 0, 0, 0, 0], 'hypo8': [0, -125, 0, 0, 0, -52, 0, -106, 0, 13, 10], 'hypo9': [0, -101, 0, 0, 0, -113, 0, -138, 0, 4, 14], 'hypo10': [0, 0, -70, 0, 0, -112, 0, 0, 0, 0, 0], 'hypo11': [0, -139, 0, 0, 0, -127, 0, -147, 0, 15, 17], 'hypo12': [0, -95, 0, 0, 0, -134, 0, -146, 0, 19, 3], 'hypo13': [0, 0, -64, 0, 0, -60, 0, -123, 0, 19, 8], 'hypo14': [0, -140, 0, 0, 0, -147, 0, -54, 0, 6, 6], 'hypo15': [0, -78, 0, 0, 0, -139, 0, -62, 0, 4, 1], 'hypo16': [0, -121, 0, 0, 0, -147, 0, -144, 0, 10, 10], 'hypo17': [0, 0, -52, 0, 0, -123, 0, -88, 0, 19, 11], 'hypo18': [0, 0, -57, 0, 0, -138, 0, 0, 0, 0, 0], 'hypo19': [0, -69, 0, 0, 0, -90, 0, -77, 0, 7, 10], 'hypo20': [0, -114, 0, 0, 0, -118, 0, -132, 0, 18, 13], 'hypo21': [0, -119, 0, 0, 0, -134, 0, -77, 0, 16, 7], 'hypo22': [0, 0, -92, 0, 0, -75, 0, 0, 0, 0, 0], 'hypo23': [0, 0, -121, 0, 0, -57, 0, 0, 0, 0, 0], 'hypo24': [0, -106, 0, 0, 0, -108, 0, -98, 0, 1, 8], 'hypo25': [0, 0, -128, 0, 0, -70, 0, -78, 0, 15, 16], 'hypo26': [0, 0, -134, 0, 0, -138, 0, 0, 0, 0, 0], 'hypo27': [0, 0, -140, 0, 0, -122, 0, 0, 0, 0, 0], 'hypo28': [0, 0, -55, 0, 0, -95, 0, -93, 0, 19, 16], 'hypo29': [0, -109, 0, 0, 0, -91, 0, -79, 0, 12, 15], 'hypo30': [0, 0, -109, 0, 0, -111, 0, 0, 0, 0, 0], 'hypo31': [0, 0, -93, 0, 0, -54, 0, 0, 0, 0, 0], 'hypo32': [0, -82, 0, 0, 0, -126, 0, -138, 0, 18, 2], 'hypo33': [0, 0, -64, 0, 0, -116, 0, -141, 0, 2, 7], 'hypo34': [0, 0, -112, 0, 0, -118, 0, 0, 0, 0, 0], 'hypo35': [0, 0, -150, 0, 0, -129, 0, 0, 0, 0, 0], 'hypo36': [0, 0, -51, 0, 0, -105, 0, -131, 0, 17, 2], 'hypo37': [0, 0, -136, 0, 0, -68, 0, -122, 0, 17, 9], 'hypo38': [0, 0, -89, 0, 0, -110, 0, 0, 0, 0, 0], 'hypo39': [0, -97, 0, 0, 0, -65, 0, -113, 0, 7, 13], 'hypo40': [0, -92, 0, 0, 0, -57, 0, -109, 0, 3, 3], 'hypo41': [0, -140, 0, 0, 0, -121, 0, -144, 0, 1, 3], 'hypo42': [0, 0, -142, 0, 0, -143, 0, 0, 0, 0, 0], 'hypo43': [0, 0, -70, 0, 0, -118, 0, -135, 0, 18, 11], 'hypo44': [0, -103, 0, 0, 0, -66, 0, -86, 0, 8, 7], 'hypo45': [0, 0, -79, 0, 0, -54, 0, 0, 0, 0, 0], 'hypo46': [0, 0, -67, 0, 0, -148, 0, -76, 0, 13, 14], 'hypo47': [0, 0, -61, 0, 0, -118, 0, 0, 0, 0, 0], 'hypo48': [0, 0, -121, 0, 0, -150, 0, 0, 0, 0, 0], 'hypo49': [0, -150, 0, 0, 0, -85, 0, -97, 0, 15, 3], 'hypo50': [0, 0, -78, 0, 0, -58, 0, -125, 0, 3, 7], 'hypo51': [0, -67, 0, 0, 0, -91, 0, -118, 0, 1, 13], 'hypo52': [0, 0, -136, 0, 0, -88, 0, -141, 0, 1, 15], 'hypo53': [0, 0, -140, 0, 0, -119, 0, 0, 0, 0, 0], 'hypo54': [0, 0, -51, 0, 0, -106, 0, -118, 0, 10, 1], 'hypo55': [0, 0, -111, 0, 0, -121, 0, -66, 0, 11, 11], 'hypo56': [0, -59, 0, 0, 0, -142, 0, -111, 0, 12, 10], 'hypo57': [0, 0, -108, 0, 0, -68, 0, 0, 0, 0, 0], 'hypo58': [0, 0, -150, 0, 0, -128, 0, 0, 0, 0, 0], 'hypo59': [0, 0, -114, 0, 0, -113, 0, -140, 0, 6, 8], 'hypo60': [0, 0, -113, 0, 0, -92, 0, -91, 0, 4, 12], 'hypo61': [0, 0, -76, 0, 0, -149, 0, 0, 0, 0, 0], 'hypo62': [0, -141, 0, 0, 0, -78, 0, -72, 0, 1, 11], 'hypo63': [0, 0, -91, 0, 0, -82, 0, -129, 0, 7, 6], 'hypo64': [0, 0, -110, 0, 0, -96, 0, -89, 0, 4, 7], 'hypo65': [0, 0, -67, 0, 0, -72, 0, -131, 0, 8, 10], 'hypo66': [0, -128, 0, 0, 0, -109, 0, -118, 0, 3, 1], 'hypo67': [0, 0, -54, 0, 0, -131, 0, -127, 0, 17, 11], 'hypo68': [0, -80, 0, 0, 0, -62, 0, -113, 0, 17, 17], 'hypo69': [0, 0, -76, 0, 0, -78, 0, -101, 0, 17, 10], 'hypo70': [0, -107, 0, 0, 0, -58, 0, -123, 0, 10, 16], 'hypo71': [0, 0, -136, 0, 0, -106, 0, 0, 0, 0, 0], 'hypo72': [0, -52, 0, 0, 0, -128, 0, -95, 0, 4, 5], 'hypo73': [0, 0, -97, 0, 0, -94, 0, -68, 0, 11, 8], 'hypo74': [0, 0, -96, 0, 0, -61, 0, 0, 0, 0, 0], 'hypo75': [0, -130, 0, 0, 0, -138, 0, -116, 0, 7, 8], 'hypo76': [0, 0, -128, 0, 0, -54, 0, -55, 0, 2, 15], 'hypo77': [0, 0, -60, 0, 0, -141, 0, 0, 0, 0, 0], 'hypo78': [0, -52, 0, 0, 0, -95, 0, -53, 0, 3, 8], 'hypo79': [0, 0, -87, 0, 0, -95, 0, -73, 0, 19, 5], 'hypo80': [0, -126, 0, 0, 0, -135, 0, -90, 0, 16, 3], 'hypo81': [0, 0, -59, 0, 0, -94, 0, -105, 0, 6, 10], 'hypo82': [0, 0, -131, 0, 0, -135, 0, 0, 0, 0, 0], 'hypo83': [0, -100, 0, 0, 0, -113, 0, -94, 0, 19, 8], 'hypo84': [0, 0, -72, 0, 0, -109, 0, -110, 0, 13, 13], 'hypo85': [0, 0, -147, 0, 0, -104, 0, -125, 0, 8, 15], 'hypo86': [0, 0, -144, 0, 0, -64, 0, 0, 0, 0, 0], 'hypo87': [0, 0, -125, 0, 0, -127, 0, -123, 0, 5, 18], 'hypo88': [0, 0, -76, 0, 0, -135, 0, 0, 0, 0, 0], 'hypo89': [0, 0, -99, 0, 0, -69, 0, 0, 0, 0, 0], 'hypo90': [0, 0, -112, 0, 0, -132, 0, -60, 0, 15, 14], 'hypo91': [0, 0, -82, 0, 0, -104, 0, -120, 0, 3, 5], 'hypo92': [0, 0, -133, 0, 0, -53, 0, -73, 0, 15, 17], 'hypo93': [0, 0, -124, 0, 0, -71, 0, 0, 0, 0, 0], 'hypo94': [0, 0, -90, 0, 0, -67, 0, -147, 0, 7, 13], 'hypo95': [0, 0, -115, 0, 0, -65, 0, -78, 0, 5, 6], 'hypo96': [0, 0, -141, 0, 0, -125, 0, 0, 0, 0, 0]}
HYPOTHESES.update({'hypo97': [0, 0, -128, 0, 0, -98, 0, -139, 0, 17, 2], 'hypo98': [0, 0, -302, 0, 0, 38, 0, -295, 0, 8, 4], 'hypo99': [0, 0, -210, 0, 0, -72, 0, -165, 0, 6, 2], 'hypo100': [0, 0, -97, 0, 0, -124, 0, -100, 0, 12, 8], 'hypo101': [0, 0, -110, 0, 0, -74, 0, -143, 0, 19, 19], 'hypo102': [0, 0, -140, 0, 0, -88, 0, -149, 0, 8, 2], 'hypo103': [0, 0, -120, 0, 0, -62, 0, -130, 0, 5, 1], 'hypo104': [0, 0, -154, 0, 0, -43, 0, -204, 0, 3, 4], 'hypo105': [0, 0, -162, 0, 0, -5, 0, -152, 0, 2, 3], 'hypo106': [0, 0, -149, 0, 0, -113, 0, -81, 0, 11, 10], 'hypo107': [0, 0, -195, 0, 0, -15, 0, -186, 0, 6, 3], 'hypo108': [0, 0, -241, 0, 0, -32, 0, -189, 0, 6, 1], 'hypo109': [0, 0, -213, 0, 0, -2, 0, -219, 0, 8, 7], 'hypo110': [0, 0, -54, 0, 0, -69, 0, -142, 0, 9, 18], 'hypo111': [0, 0, -140, 0, 0, -118, 0, -140, 0, 9, 15], 'hypo112': [0, 0, -205, 0, 0, -37, 0, -240, 0, 8, 3], 'hypo113': [0, 0, -117, 0, 0, -131, 0, -90, 0, 1, 13], 'hypo114': [0, 0, -196, 0, 0, -24, 0, -230, 0, 1, 4], 'hypo115': [0, 0, -228, 0, 0, -63, 0, -187, 0, 5, 3], 'hypo116': [0, 0, -320, 0, 0, -19, 0, -269, 0, 1, 2], 'hypo117': [0, 0, -166, 0, 0, -35, 0, -247, 0, 4, 7], 'hypo118': [0, 0, -156, 0, 0, -94, 0, -180, 0, 9, 8], 'hypo119': [0, 0, -56, 0, 0, -108, 0, -146, 0, 2, 15], 'hypo120': [0, 0, -97, 0, 0, -150, 0, -138, 0, 9, 6], 'hypo121': [0, 0, -268, 0, 0, -4, 0, -319, 0, 3, 12], 'hypo122': [0, 0, -144, 0, 0, -139, 0, -132, 0, 17, 7], 'hypo123': [0, 0, -308, 0, 0, -14, 0, -309, 0, 2, 6], 'hypo124': [0, 0, -198, 0, 0, -86, 0, -152, 0, 7, 9], 'hypo125': [0, 0, -350, 0, 0, -1, 0, -257, 0, 9, 4], 'hypo126': [0, 0, -78, 0, 0, -119, 0, -141, 0, 9, 17], 'hypo127': [0, 0, -205, 0, 0, -87, 0, -217, 0, 1, 6], 'hypo128': [0, 0, -335, 0, 0, 32, 0, -279, 0, 7, 6], 'hypo129': [0, 0, -52, 0, 0, -144, 0, -124, 0, 2, 11], 'hypo130': [0, 0, -255, 0, 0, 16, 0, -263, 0, 5, 13], 'hypo131': [0, 0, -114, 0, 0, -78, 0, -107, 0, 14, 9], 'hypo132': [0, 0, -184, 0, 0, -32, 0, -170, 0, 6, 5], 'hypo133': [0, 0, -118, 0, 0, -137, 0, -106, 0, 1, 2], 'hypo134': [0, 0, -259, 0, 0, -13, 0, -295, 0, 2, 9], 'hypo135': [0, 0, -301, 0, 0, -37, 0, -254, 0, 9, 11], 'hypo136': [0, 0, -99, 0, 0, -68, 0, -94, 0, 14, 14], 'hypo137': [0, 0, -320, 0, 0, -19, 0, -287, 0, 9, 8], 'hypo138': [0, 0, -226, 0, 0, -57, 0, -233, 0, 7, 4], 'hypo139': [0, 0, -341, 0, 0, 25, 0, -321, 0, 1, 5]})
HYPOTHESES.update({'hypo140': [0, 0, -219, 0, 0, -1, 0, -195, 0, 6, 4], 'hypo141': [0, 0, -126, 0, 0, -73, 0, -57, 0, 11, 6], 'hypo142': [0, 0, -296, 0, 0, 35, 0, -272, 0, 8, 6], 'hypo143': [0, 0, -343, 0, 0, -14, 0, -267, 0, 4, 9], 'hypo144': [0, 0, -60, 0, 0, -55, 0, -141, 0, 1, 19], 'hypo145': [0, 0, -117, 0, 0, -81, 0, -61, 0, 5, 4], 'hypo146': [0, 0, -176, 0, 0, -75, 0, -192, 0, 1, 6], 'hypo147': [0, 0, -124, 0, 0, -65, 0, -108, 0, 3, 9], 'hypo148': [0, 0, -285, 0, 0, -43, 0, -297, 0, 12, 2], 'hypo149': [0, 0, -289, 0, 0, -24, 0, -272, 0, 8, 4], 'hypo150': [0, 0, -238, 0, 0, -57, 0, -186, 0, 3, 3], 'hypo151': [0, 0, -119, 0, 0, -68, 0, -102, 0, 16, 4], 'hypo152': [0, 0, -98, 0, 0, -60, 0, -69, 0, 17, 12], 'hypo153': [0, 0, -62, 0, 0, -132, 0, -111, 0, 13, 1], 'hypo154': [0, 0, -129, 0, 0, -141, 0, -146, 0, 14, 14], 'hypo155': [0, 0, -254, 0, 0, -52, 0, -341, 0, 3, 8], 'hypo156': [0, 0, -115, 0, 0, -54, 0, -54, 0, 9, 11], 'hypo157': [0, 0, -220, 0, 0, -40, 0, -179, 0, 3, 3], 'hypo158': [0, 0, -236, 0, 0, -8, 0, -177, 0, 3, 6], 'hypo159': [0, 0, -339, 0, 0, -1, 0, -310, 0, 2, 2], 'hypo160': [0, 0, -157, 0, 0, -76, 0, -211, 0, 9, 4], 'hypo161': [0, 0, -159, 0, 0, -89, 0, -151, 0, 7, 6], 'hypo162': [0, 0, -259, 0, 0, -5, 0, -291, 0, 8, 9], 'hypo163': [0, 0, -305, 0, 0, -11, 0, -333, 0, 9, 7], 'hypo164': [0, 0, -103, 0, 0, -115, 0, -78, 0, 16, 2], 'hypo165': [0, 0, -174, 0, 0, -87, 0, -245, 0, 7, 4], 'hypo166': [0, 0, -311, 0, 0, -29, 0, -281, 0, 6, 12], 'hypo167': [0, 0, -308, 0, 0, -51, 0, -317, 0, 4, 3], 'hypo168': [0, 0, -135, 0, 0, -67, 0, -96, 0, 7, 18], 'hypo169': [0, 0, -204, 0, 0, -74, 0, -173, 0, 4, 6], 'hypo170': [0, 0, -82, 0, 0, -104, 0, -71, 0, 13, 15], 'hypo171': [0, 0, -251, 0, 0, -33, 0, -327, 0, 3, 9], 'hypo172': [0, 0, -323, 0, 0, 35, 0, -254, 0, 3, 6], 'hypo173': [0, 0, -305, 0, 0, 14, 0, -257, 0, 6, 2], 'hypo174': [0, 0, -61, 0, 0, -54, 0, -132, 0, 1, 2], 'hypo175': [0, 0, -120, 0, 0, -86, 0, -58, 0, 14, 18], 'hypo176': [0, 0, -101, 0, 0, -57, 0, -90, 0, 11, 12], 'hypo177': [0, 0, -224, 0, 0, -75, 0, -246, 0, 1, 3], 'hypo178': [0, 0, -58, 0, 0, -95, 0, -63, 0, 14, 19], 'hypo179': [0, 0, -317, 0, 0, -23, 0, -347, 0, 5, 2], 'hypo180': [0, 0, -315, 0, 0, 17, 0, -268, 0, 13, 11], 'hypo181': [0, 0, -221, 0, 0, -54, 0, -250, 0, 9, 4], 'hypo182': [0, 0, -260, 0, 0, -29, 0, -289, 0, 2, 12], 'hypo183': [0, 0, -231, 0, 0, -15, 0, -200, 0, 1, 8], 'hypo184': [0, 0, -291, 0, 0, -59, 0, -305, 0, 2, 13], 'hypo185': [0, 0, -337, 0, 0, 13, 0, -323, 0, 10, 14], 'hypo186': [0, 0, -349, 0, 0, -40, 0, -275, 0, 11, 13], 'hypo187': [0, 0, -183, 0, 0, -55, 0, -248, 0, 5, 4], 'hypo188': [0, 0, -202, 0, 0, -29, 0, -173, 0, 1, 4], 'hypo189': [0, 0, -122, 0, 0, -142, 0, -145, 0, 3, 17], 'hypo190': [0, 0, -216, 0, 0, -67, 0, -202, 0, 2, 5], 'hypo191': [0, 0, -189, 0, 0, -13, 0, -242, 0, 8, 2], 'hypo192': [0, 0, -350, 0, 0, -13, 0, -348, 0, 1, 7], 'hypo193': [0, 0, -227, 0, 0, -64, 0, -153, 0, 1, 4], 'hypo194': [0, 0, -325, 0, 0, -53, 0, -328, 0, 14, 2], 'hypo195': [0, 0, -231, 0, 0, -16, 0, -177, 0, 1, 6], 'hypo196': [0, 0, -177, 0, 0, -54, 0, -229, 0, 3, 2], 'hypo197': [0, 0, -105, 0, 0, -137, 0, -128, 0, 11, 11], 'hypo198': [0, 0, -283, 0, 0, 20, 0, -309, 0, 1, 12], 'hypo199': [0, 0, -154, 0, 0, -1, 0, -234, 0, 1, 3]})

class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.lander == contact.fixtureA.body or self.env.lander == contact.fixtureB.body:
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class LunarLanderTheta(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    continuous = False

    def __init__(self):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.moon = None
        self.lander = None
        self.particles = []

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)

        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.moon: return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def reset(self, theta):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shapings = {hypo: 0 for hypo in HYPOTHESES}
        self.prev_shaping = None
        self.theta = theta
        self.initial_x = None

        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE

        # terrain
        CHUNKS = 15

        chunk_x = [W/(CHUNKS-1)*i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS//2-1]
        self.helipad_x2 = chunk_x[CHUNKS//2+1]
        self.helipad_y = H/4

        # Use to create flat ground
        # height = [H/4] * (CHUNKS+1)
        # Use to create mountains around helipad
        height = self.np_random.uniform(0, 0, size=(CHUNKS+1,))
        height[CHUNKS//2-5] = self.helipad_y
        height[CHUNKS//2-4] = self.helipad_y
        height[CHUNKS//2-2] = self.helipad_y
        height[CHUNKS//2-1] = self.helipad_y
        height[CHUNKS//2+0] = self.helipad_y
        height[CHUNKS//2+1] = self.helipad_y
        height[CHUNKS//2+2] = self.helipad_y
        height[CHUNKS//2+4] = self.helipad_y
        height[CHUNKS//2+5] = self.helipad_y
        smooth_y = [3.3 for i in range(CHUNKS)]

        self.moon = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        self.sky_polys = []
        for i in range(CHUNKS-1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i+1], smooth_y[i+1])
            self.moon.CreateEdgeFixture(
                vertices=[p1,p2],
                density=0,
                friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        initial_y = VIEWPORT_H/SCALE
        x1 = np.random.uniform(CHUNKS//2-1,CHUNKS//2+1)
        x2 = np.random.uniform(CHUNKS//2+5,CHUNKS//2+7)
        initial_x = random.choice([x1, x2])
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=polygonShape(vertices=[(x/SCALE, y/SCALE) for x, y in LANDER_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,   # collide only with ground
                restitution=0.0)  # 0.99 bouncy
                )
        self.lander.color1 = (0.5, 0.4, 0.9)
        self.lander.color2 = (0.3, 0.3, 0.5)
        self.lander.ApplyForceToCenter( (
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
            ), True)

        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i*LEG_AWAY/SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W/SCALE, LEG_H/SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
                )
            leg.ground_contact = False
            leg.color1 = (0.5, 0.4, 0.9)
            leg.color2 = (0.3, 0.3, 0.5)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY/SCALE, LEG_DOWN/SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i  # low enough not to jump back into the sky
                )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5  # The most esoteric numbers here, angled legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs
        self.reset_state = True

        return self.step(np.array([0, 0]) if self.continuous else 0)[0]

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position = (x, y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=circleShape(radius=2/SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3)
                )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def step(self, action):
        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        # Engines
        tip  = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action == 2):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0,1.0) + 1.0)*0.5   # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0
            ox = (tip[0] * (4/SCALE + 2 * dispersion[0]) +
                  side[0] * dispersion[1])  # 4 is move a bit downwards, +-2 for randomness
            oy = -tip[1] * (4/SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            p = self._create_particle(3.5,  # 3.5 is here to make particle speed adequate
                                      impulse_pos[0],
                                      impulse_pos[1],
                                      m_power)  # particles are just a decoration
            p.ApplyLinearImpulse((ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power),
                                 impulse_pos,
                                 True)
            self.lander.ApplyLinearImpulse((-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                                           impulse_pos,
                                           True)

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (not self.continuous and action in [1, 3]):
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                direction = action-2
                s_power = 1.0
            ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY/SCALE)
            oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY/SCALE)
            impulse_pos = (self.lander.position[0] + ox - tip[0] * 17/SCALE,
                           self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT/SCALE)
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse((ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power),
                                 impulse_pos
                                 , True)
            self.lander.ApplyLinearImpulse((-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                                           impulse_pos,
                                           True)

        self.world.Step(1.0/FPS, 6*30, 2*30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            (pos.y - (self.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2),
            vel.x*(VIEWPORT_W/SCALE/2)/FPS,
            vel.y*(VIEWPORT_H/SCALE/2)/FPS,
            self.lander.angle,
            20.0*self.lander.angularVelocity/FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0
            ]
        assert len(state) == 8

        if self.reset_state:
            self.reset_state = False
            self.initial_x = state[0]

        # Get rewards for R1, R2 and R3
        rewards, done, lander_state = self.shape_reward(state, m_power, s_power)
        # Return reward for the selected reward type
        reward = rewards[self.theta]

        return np.array(state, dtype=np.float32), reward,\
                 done, {'reward':reward, 'rewards':rewards,\
                  'awake':lander_state, 'theta':self.theta}

    def shape_reward(self, state, m_power, s_power):
        rewards = {hypo: 0 for hypo in HYPOTHESES}
        shapings = {hypo: 0 for hypo in HYPOTHESES}

        for shaping in shapings:
            theta_vec = np.array(HYPOTHESES[shaping])
            features = np.array([state[1], state[0], np.sqrt(state[0]**2 + state[1]**2), state[2], state[3], np.sqrt(state[2]**2 + state[3]**2), state[4], np.abs(state[4]), state[5], state[6], state[7]])
            shapings[shaping] = np.dot(theta_vec, features)

        for reward in rewards:
            if self.prev_shapings[reward] is not None:
                rewards[reward] = shapings[reward] - self.prev_shapings[reward]
            self.prev_shapings[reward] = shapings[reward]

            # rewards[i] -= m_power*0.30  # less fuel spent is better, about -30 for heuristic landing
            # rewards[i] -= s_power*0.03

        done = False
        lander_state = True

        if self.game_over or abs(state[0]) >= 1.0:
            done = True
            # rewards[0] = -100
            # rewards[1] = -100
            # rewards[2] = +100.0
            # # if np.linalg.norm(state[0]) < 0.1:
            # #     rewards[2] = +100
            lander_state = False

        if not self.lander.awake:
            done = True
        #     rewards[0] = 0.0
        #     rewards[1] = 0.0
        #     rewards[2] = -100.0
        #     if np.linalg.norm(state[0]) < 0.1:
        #         rewards[0] = +100
        #     if np.linalg.norm(state[0] - self.initial_x) < 0.1:
        #         rewards[1] = +100
        #     # if np.linalg.norm(state[0]) < 0.1:
        #     #     rewards[2] = +100
            lander_state = True

        return rewards, done, lander_state


    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (max(0.2, 0.2+obj.ttl), max(0.2, 0.5*obj.ttl), max(0.2, 0.5*obj.ttl))
            obj.color2 = (max(0.2, 0.2+obj.ttl), max(0.2, 0.5*obj.ttl), max(0.2, 0.5*obj.ttl))

        self._clean_particles(False)

        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=(0, 0, 0))

        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        for x in [self.helipad_x1, self.helipad_x2]:
            flagy1 = self.helipad_y
            flagy2 = flagy1 + 50/SCALE
            self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(1, 1, 1))
            self.viewer.draw_polygon([(x, flagy2), (x, flagy2-10/SCALE), (x + 25/SCALE, flagy2 - 5/SCALE)],
                                     color=(0.8, 0.8, 0))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
