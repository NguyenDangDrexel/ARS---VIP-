"""
Title: Iridium Dispatcher
Author: Lance Nichols
Created: 3/22/2022
Revised: 4/12/2022
Target Device: OCCAMS2-V2 XBEE3 SMT
"""

# Imports
import xbee
import machine
from machine import Pin
import time

commands = ["ABC", "DEF", "GHI", "JKL", "MNO", "PQR", "STU", "VWX"]

# Pin Definitions
usr_led = Pin(machine.Pin.board.D4, Pin.OUT, value=0)
asc_led = Pin(machine.Pin.board.D5, Pin.OUT, value=0)
spk = Pin(machine.Pin.board.D10, Pin.OUT, value=0)

signal0 = Pin(machine.Pin.board.D0, Pin.IN)
signal1 = Pin(machine.Pin.board.D2, Pin.IN)
signal2 = Pin(machine.Pin.board.D3, Pin.IN)

out0 = Pin(machine.Pin.board.D15, Pin.OUT, value=0) # Reserved for signaling cutdown states
out1 = Pin(machine.Pin.board.D16, Pin.OUT, value=0) # Reserved for signaling cutdown states
out2 = Pin(machine.Pin.board.D17, Pin.OUT, value=0)
out3 = Pin(machine.Pin.board.D18, Pin.OUT, value=0)


cutDownStatus = 0
auxStatus = 0
mspSpike = False

def resolveSettingCutdownSignal():
    #Set the output pins based on the status
    out0.value(cutDownStatus % 2)
    out1.value(int(cutDownStatus/2) % 2)
    out2.value(auxStatus % 2)
    out3.value(int(auxStatus / 2) % 2)

def checkAck():
    global cutDownStatus
    global auxStatus
    global mspSpike
    try:
        packet = xbee.receive()                             #Try to recieve packet
        [xbee.receive() for i in range(100)]                #Clear the buffer
        reply = packet.get('payload').decode('utf-8')[:3]
        if reply is not None:
            print("Reply:"+reply)
            if reply == "CAK" and cutDownStatus == 1:  # If cutdown was sent
                cutDownStatus = 2  # Set to ack received status
            if reply == "TXB":  # If the XBee temp has spiked
                cutDownStatus = 3  # Set to ack received status
            if reply == "TMS":  # If the MSP temp has spiked
                cutDownStatus = 3  # Set to ack received status
                mspSpike = True
            if reply == "VOA":
                auxStatus = 1
            if reply == "VCR":
                auxStatus = 2
    except:
        pass


def resolveDispatch():
    currentCommand = commands[signal0.value() + signal1.value() * 2 + signal2.value() * 4]
    global cutDownStatus
    global auxStatus
    global mspSpike
    # Check current state
    if currentCommand == "ABC":  # Idle is received from Iridium Modem and MSP not active
        auxStatus = 0
        if not mspSpike:
            cutDownStatus = 0

    elif currentCommand == "DEF":  # Cutdown is received from Iridium Modem
        if cutDownStatus == 0 or (mspSpike and cutDownStatus == 3):  # If was idle or ready to exit MSP temp
            cutDownStatus = 1  # Set to received status
            mspSpike = False
    # Broadcast current state

    try:
        xbee.transmit(xbee.ADDR_BROADCAST, currentCommand)
        print("Transmitted: " + currentCommand)
    except:
        print("No Endpoint")

def beep():
    spk.value(1)
    time.sleep_ms(10)
    spk.value(0)
# def blinkasc():
#     asc_led.value(1)
#     time.sleep_ms(5)
#     asc_led.value(0)
# Main Loop
while True:
    resolveDispatch()
    checkAck()
    resolveSettingCutdownSignal()


    # Blink and wait
    beep()
    usr_led.value(1)
    time.sleep_ms(1000)
    usr_led.value(0)
    time.sleep_ms(1000)

