# Xbee send pulses to the MSP430 

# Imports
import xbee
import machine
from machine import Pin
import time

# Pins setup
usr_led = Pin(machine.Pin.board.D4, Pin.OUT, value=0)  # Enable pin on XBEE3 set to low initially
asc_led = Pin(machine.Pin.board.D5, Pin.OUT, value=0)  # Enable pin on XBEE3 set to low initially
spkr_pin = Pin(machine.Pin.board.P0, Pin.OUT, value=0)  # Shush chatty monkey
msp_pin = Pin(machine.Pin.board.P7, Pin.OUT)  # Communication pin to MSP430
heater_pin = Pin(machine.Pin.board.D18, Pin.OUT, value=1)
Hall_pin = Pin(machine.Pin.board.D3, Pin.IN, pull=None)

print("Vent Control Initialized")

# Function to receive commands
# def GetCommand():
#     packet = None  # Set packet to None
#     while packet is None:  # Wait for a packet to arrive
#         packet = xbee.receive()  # Try to receive a packet
#     [xbee.receive() for i in range(100)]  # Clear the buffer
#     return packet.get('payload').decode('utf-8')[:3]  # Return the first three characters decoded
# hard_coding for testing
def GetCommand ():
    # Valid commands: 'JKL'; 'MNO'; 'PQR'
    return 'JKL'

# Process the received command and map it to '1', '2', or '3'
def ProcessCommand(command):
    print(f"Received Command: {command}")
    if command == 'JKL':
        SendToMSP('1')  # Send '1' for 25% open time
    elif command == 'MNO':
        SendToMSP('2')  # Send '2' for 50% open time
    elif command == 'PQR':
        SendToMSP('3')  # Send '3' for 100% open time

# Send the specified signal to the MSP430 via msp_pin
def SendToMSP(signal):
    for char in signal:
        # Sending each character as a pulse
        if char == '1':
            msp_pin.value(1)
            time.sleep_ms(100)  # Pulse duration
            msp_pin.value(0)
        elif char == '2':
            msp_pin.value(1)
            time.sleep_ms(200)  # Pulse duration for '2'
            msp_pin.value(0)
        elif char == '3':
            msp_pin.value(1)
            time.sleep_ms(300)  # Pulse duration for '3'
            msp_pin.value(0)
        time.sleep(1)  # Pause between signals to ensure separation

# Main loop
def main():
    while True:
        time.sleep(3)  # Pause to sync, otherwise transmission may be disrupted
        command = GetCommand()
        ProcessCommand(command)
        # TURN OF HARD-CODING TESTING
        # # Transmit hall effect status (VCR or VOA) to the coordinator
        # hall_effect = "VCR" if Hall_pin.value() == 1 else "VOA"
        # try:
        #     xbee.transmit(xbee.ADDR_COORDINATOR, hall_effect)
        # except Exception as e:
        #     print(f"Transmission error: {e}")

# Start the main function
main()
