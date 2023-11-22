#different ways of reading keys

1. 
#if (keyboard.is_pressed):
   # print(keyboard.read_event().name)

2.
#print(keyboard.read_key())

3.
#keyboard.add_hotkey('space', lambda: print('space was pressed!'))
#keyboard.wait()

import can
import keyboard

# need to replace with our things
channel = 'can0'
bus = can.interface.Bus(channel=channel, bustype='socketcan')

import can


def send_can(data):
    # Create a CAN message

    msg = can.Message(arbitration_id=0x123, data=data)

    try:
        bus = can.interface.Bus(channel='can0', bustype='socketcan')
        bus.send(msg)
        print("Message sent on {}: {}".format(bus.channel_info, msg))
    except can.CanError:
        print("Failed to send message")

while True:
    event = keyboard.read_event()
    if event.event_type == keyboard.KEY_DOWN and event.name == 'a':
        print("key pressed: " + event.name)
        send_can("Hello world")

    elif event.event_type == keyboard.KEY_DOWN and event.name == 'b':
        print("key pressed: " + event.name)
        send_can("Hello world!!")
    elif event.event_type == keyboard.KEY_DOWN and event.name == 'd':
        print("key pressed: " + event.name)
        send_can("Hello world!!??")
    
