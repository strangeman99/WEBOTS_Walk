"""Hipy controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
from abc import ABC
import numpy as np
import gym
from gym import spaces
import time


# class CustomEnv(gym.env,ABC):

def __init__(self):

    devices_name = ["PelvYL", "PelvYR", "PelvL", "PelvR", "LegUpperL", "LegUpperR", "LegLowerL", "LegLowerL", "AnkleL",
                    "AnkleR", "FootL", "FootR"]
    devices = []
    for name in devices_name:
        devices.append(robot.getDevice(name))

    for device in devices:
        device.setVelocity(0.0)

    Cam = robot.getCamera("Camera")
