from controller import Supervisor, Node

import random


def spawn_cube(supervisor: Supervisor):
    # Load the cube prototype
    cube_proto = supervisor.getFromDef("CUBE")

    # Create a new instance of the cube
    cube = supervisor.cloneProto(cube_proto)#????

    # Randomly position the cube within the simulation world
    x = random.uniform(-5, 5)
    y = random.uniform(0.1, 2)
    z = random.uniform(-5, 5)
    cube.getField("translation").setSFVec3f([x, y, z])


if __name__ == "__main__":
    # Create the Webots supervisor
    supervisor = Supervisor()

    # Enable the simulation
    supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_REAL_TIME)

    # Spawn 10 cubes
    for _ in range(10):
        spawn_cube(supervisor)

    # Run the simulation
    while supervisor.step(Supervisor.TIME_STEP) != -1:
        pass
