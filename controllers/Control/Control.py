"""Control controller."""
import math
import random
from abc import ABC
from typing import Optional
import numpy as np
import collections as col

import gym
import cv2
from gym.core import ActType, ObsType
from gym import spaces
from controller import Supervisor, Robot


# This checks for overlap of the new point on any point already created
def anyOverlap(prev_points, cur_point, max_size) -> bool:
    for point in prev_points:
        dist = math.sqrt((point[0] - cur_point[0]) ** 2 + (point[1] - cur_point[1]) ** 2)
        if dist <= max_size:
            return True

    return False


# This is the main class that inherits from environments
class CustomEnv(gym.Env, ABC):
    # Constructor setting up sensors and motors
    def __init__(self):
        # Starting position of the robot
        self.start_pos = [0.0, 0.0, 0.285]

        # Motor constraints (Maybe change)
        self.begin_motor_pos = 0.0
        self.motor_max_pos = np.pi
        self.motor_min_pos = -np.pi
        self.motor_num = 12
        self.obs_num = 4

        # The space containing a single motor. 4 positions total
        self.mot_space = spaces.Box(
            low=self.motor_min_pos,
            high=self.motor_max_pos,
            shape=(self.obs_num,),
            dtype=np.float32
        )

        # Camera constraints
        self.cam_fidelity = 512

        # Space containing the camera
        self.cam_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.cam_fidelity, self.cam_fidelity, self.obs_num),
            dtype=np.uint8
        )

        # Gyro constraints
        self.gyro_max = 32767
        self.gyro_min = -32767

        # Space containing the gyro
        self.gyro_space = spaces.Box(
            low=self.gyro_min,
            high=self.gyro_max,
            shape=(3, self.obs_num),
            dtype=np.float32
        )

        # Accel constraints
        self.accel_min = np.array([-10.0, -10.0, -10.0])
        self.accel_max = np.array([10.0, 10.0, 10.0])

        # Space containing the accel
        self.accel_space = spaces.Box(
            low=self.accel_min,
            high=self.accel_max,
            shape=(3, self.obs_num),
            dtype=np.float32
        )

        # The full observation space containing all sensors
        self.observation_space = spaces.Tuple((
            self.cam_space,
            spaces.Tuple([self.mot_space for _ in range(self.motor_num)]),
            self.gyro_space,
            self.accel_space
        ))

        # Setting a buffer to hold old states
        self.buffer_size = 3
        self.buffer = col.deque([], maxlen=self.buffer_size)

        # The action space for output. 12 motors
        self.action_space = spaces.Tuple([
            spaces.Box(
                low=self.motor_min_pos,
                high=self.motor_max_pos,
                shape=(1,),
                dtype=np.float32
            )
            for _ in range(self.motor_num)
        ])

        # Init accel, gyro, global pos, target pos, camera, and dist
        self.robot = Robot()
        self.world = Supervisor()

        # Setting the position to the middle
        self.robot_node = self.world.getFromDef("WEBOT")
        self.position = self.start_pos
        position_field = self.robot_node.getField("translation")
        position_field.setSFVec3f(self.position)

        # get the time step of the current world.
        self.timestep = int(self.robot.getBasicTimeStep())

        # Setting the devices
        # TODO Check the oder of these joints
        motor_devices_name = ["PelvYL", "PelvYR", "PelvL", "PelvR", "LegUpperL", "LegUpperR", "LegLowerL", "LegLowerR",
                              "AnkleL", "AnkleR", "FootL", "FootR"]
        self.motor_devices = []
        self.motor_positions = []

        # initialize devices
        for i, name in enumerate(motor_devices_name):
            self.motor_devices.append(self.robot.getDevice(name))
            self.motor_devices[i].setPosition(self.begin_motor_pos)
            self.motor_positions.append(self.begin_motor_pos)

        # Setting the camera
        self.cam = self.robot.getDevice("Camera")
        self.cam.enable(self.timestep)

        # Setting the gyro and accel
        self.gyro = self.robot.getDevice("Gyro")
        self.gyro.enable(self.timestep)
        self.accel = self.robot.getDevice("Accelerometer")
        self.accel.enable(self.timestep)

        # TODO figure out what radius and size to put
        # Setting the random objects to avoid
        self.play_radius = 100
        self.max_obj_size = 3
        self.num_objects = 20
        self.objects = self.placeObjects(self.num_objects, self.play_radius, self.max_obj_size)

        # Target that the robot walks to
        self.target = (random.uniform(-1, 1) * self.play_radius, random.uniform(-1, 1) * self.play_radius)

    # This resets the scene. Returns the starting position of everything
    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> spaces.Tuple[ObsType, dict]:

        # Setting the position to the middle
        self.position = self.start_pos
        position_field = self.robot_node.getField("translation")
        position_field.setSFVec3f(self.position)

        # Resetting motors
        for i in range(len(self.motor_devices)):
            self.motor_devices[i].setPosition(self.begin_motor_pos)
            self.motor_positions[i] = self.begin_motor_pos

        # Resetting the camera
        self.cam.disable()
        self.cam.enable(self.timestep)

        # Resetting sensors
        self.gyro.disable()
        self.gyro.enable()
        self.accel.disable()
        self.accel.disable()

        # Resetting map and target
        # TODO not sure how to delete it yet
        for obj in self.objects:
            pass

        self.objects = self.placeObjects(self.num_objects, self.play_radius, self.max_obj_size)
        self.target = (random.uniform(-1, 1) * self.play_radius, random.uniform(-1, 1) * self.play_radius)

        # Take an observation and return it
        obs = self.observe()
        for i in range(self.buffer_size):
            self.buffer.append(obs)

        return obs

    # This randomly places objects
    def placeObjects(self, num_objects, max_radius, max_size):
        objects_pos = set()
        objects_pos.add((0, 0))  # To protect robot at start
        objects = []

        for obj in range(num_objects):
            # Random position generated
            pos_dif = False
            pos_x = 0.0
            pos_y = 0.0

            while not pos_dif:
                pos_x = random.uniform(-1, 1) * max_radius
                pos_y = random.uniform(-1, 1) * max_radius

                # Ensuring no repeat positions
                if not anyOverlap(objects_pos, (pos_x, pos_y), max_size):
                    pos_dif = True
                    objects_pos.add((pos_x, pos_y))

            # TODO this has to be changed to the new object I created in webots
            cur_obj = self.world.getFromDef("BOX")
            cur_obj.getField("translation").setSFVec3f([pos_x, pos_y, 0])

            # Random size generated
            size_x = random.uniform(0, 1) * max_size
            size_y = random.uniform(0, 1) * max_size
            size_z = random.uniform(0, 1) * max_size
            cur_obj.getField("size").setSFVec3f([size_x, size_y, size_z])

            objects.append(cur_obj)

        return objects

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

    # This gets an image from the camera
    def takeImage(self):
        camera_data = self.cam.getImage()

        # Convert to grayscale and resize to 512x512
        # TODO check if there is an actual problem with this
        camera_data = cv2.cvtColor(camera_data, cv2.COLOR_RGB2GRAY)
        camera_data = cv2.resize(camera_data, (512, 512))

        # Rescale pixel values to be between 0 and 1
        camera_data = camera_data.astype(np.float32) / 255.0

        # Add a channel dimension
        camera_data = np.expand_dims(camera_data, axis=2)

        return camera_data

    # This takes an observation of the environment
    def observe(self):
        # TODO finish the observation space
        observation = (self.takeImage(),
                       tuple([self.motor_devices[i].getTargetPosition() for i in range(self.motor_num)]),
                       )
