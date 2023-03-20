"""Control controller."""
from abc import ABC
import gym
from webots.controller import Robot, Camera, Motor
from gym.core import ActType, ObsType
from gym.spaces import Tuple


# This is the main class that inherits from environments
class CustomEnv(gym.Env, ABC):
    # Constructor setting up sensors and motors
    def __init__(self):
        # Init accel, gyro, global pos, target pos, camera, and dist
        robot = Robot()

        # Setting the position to the middle
        robot_node = robot.getFromDef("ROBOTIS OP3")
        self.position = [1, 0, 1]  # TODO Check the proper starting position
        robot_node.setPosition(self.position)

        # get the time step of the current world.
        timestep = int(robot.getBasicTimeStep())

        # Setting the devices
        # TODO Check the oder of these joints
        motor_devices_name = ["PelvYL", "PelvYR", "PelvL", "PelvR", "LegUpperL", "LegUpperR", "LegLowerL", "LegLowerR",
                              "AnkleL", "AnkleR", "FootL", "FootR"]
        motor_sensors_name = ["PelvYLS", "PelvYRS", "PelvLS", "PelvRS", "LegUpperLS", "LegUpperRS", "LegLowerLS",
                              "LegLowerRS", "AnkleLS", "AnkleRS", "FootLS", "FootRS"]
        self.motor_devices = []
        self.motor_sensors = []

        # initialize devices
        for i, name in enumerate(motor_devices_name):
            self.motor_devices.append(robot.getDevice(name))
            self.motor_devices[i].setPosition(2.82743)

        # initialize sensors
        for name in motor_sensors_name:
            self.motor_sensors.append(robot.getDevice(name))

        # enable sensors
        for sen in self.motor_sensors:
            sen.enable(timestep)

        # Setting the camera
        self.cam = robot.getDevice("Camera")
        self.cam.enable(timestep)

        # Setting the gyro and accel
        self.gyro = robot.getDevice("Gyro")
        self.accel = robot.getDevice("Accelerometer")

    # This resets the scene. Returns the starting position of everything
    def reset(self):
        pass

    # Executed at each time step
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        pass

    # This determines if the simulation needs to be reset
    def isDone(self):
        pass

    # This calculates the reward from the action
    def rewardCalc(self):
        pass

    # This function executes the desired action. Sets motor positions.
    def takeAction(self):
        pass

