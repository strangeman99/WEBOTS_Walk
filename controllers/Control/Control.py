"""Control controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
from controller import Camera
from controller import Motor

class CustomEnv()
    def __init__(self):
        #INIt accel
        #init gyro
        #init global position
        #target pos, distance
        # get the time step of the current world.
        timestep = int(robot.getBasicTimeStep())
        # create the Robot instance.
        robot = Robot()
        #variable to see the sensor name in the output
        n = 0
        devices_name = ["PelvYL", "PelvYR", "PelvL", "PelvR", "LegUpperL", "LegUpperR", "LegLowerL", "LegLowerR", "AnkleL", "AnkleR", "FootL", "FootR"]
        sensors_name = ["PelvYLS", "PelvYRS", "PelvLS", "PelvRS", "LegUpperLS", "LegUpperRS", "LegLowerLS", "LegLowerRS", "AnkleLS", "AnkleRS", "FootLS", "FootRS"]
        devices = []
        sensors = []
        #initialize devices
        for name in devices_name:
            devices.append(robot.getDevice(name))
        robot.getDevice("Gyro")
        robot.getDevice("Accelerometer")
            #initialize sensors
        for name in sensors_name:
            sensors.append(robot.getDevice(name))
            #enable sensors
        for sens in sensors:
            sens.enable(timestep)

        #set the position for all the devices ( Position is the maximum angle of rotation)

        devices[0].setPosition(2.82743)
        Cam = robot.getDevice("Camera")
        Cam.enable(timestep)

    def reset(self):
            
# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read the sensors:
    image = Cam.getImage()
    for sens in sensors:
        print(sens.getValue(), " - " + sensors_name[n])
        n+=1
        if sens == sensors[0]:
            n = 0
        
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    pass

# Enter here exit cleanup code.
