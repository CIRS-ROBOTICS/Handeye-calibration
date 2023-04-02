from pymodbus.client.sync import ModbusSerialClient
from math import ceil
import time
import numpy as np
from multiprocessing import Process


class _Communication(object):
    def __init__(self):
        self.client = None

    def connect_to_device(self, device):
        self.client = ModbusSerialClient(method='rtu', port=device, stopbits=1,
                                         bytesize=8, baudrate=115200, timeout=0.2)
        if not self.client.connect():
            print("Failed to connect to %s", device)
            return False
        return True

    def disconnect(self):
        self.client.close()

    def send_command(self, data):
        if (len(data) % 2 == 1):
            data.append(0)

            # Initiate message as an empty list
        message = []

        # Fill message by combining two bytes in one register
        for i in range(0, len(data)//2):
            message.append((data[2 * i] << 8) + data[2 * i + 1])

        # To do!: Implement try/except
        self.client.write_registers(0x03E8, message, unit=0x0009)

    def get_status(self, numBytes):
        """Sends a request to read, wait for the response and returns the Gripper status. The method gets the number of bytes to read as an argument"""
        numRegs = int(ceil(numBytes / 2.0))

        # To do!: Implement try/except
        # Get status from the device
        response = self.client.read_holding_registers(0x07D0, numRegs, unit=0x0009)

        # Instantiate output as an empty list
        output = []

        # Fill the output with the bytes in the appropriate order
        for i in range(0, numRegs):
            output.append((response.getRegister(i) & 0xFF00) >> 8)
            output.append(response.getRegister(i) & 0x00FF)

        # Output the result
        return output


class _InputCMD(object):
    def __init__(self):
        self.gACT = 0
        self.gGTO = 0
        self.gSTA = 0
        self.gOBJ = 0
        self.gFLT = 0
        self.gPR = 0
        self.gPO = 0
        self.gCU = 0


class _OutputCMD(object):
    def __init__(self):
        self.rACT = 0
        self.rGTO = 0
        self.rATR = 0
        self.rPR = 0
        self.rSP = 0
        self.rFR = 0


class RobotiqBaseRobotiq2FGripper(object):
    def __init__(self, device):
        self.message = []
        self.client = _Communication()
        self.client.connect_to_device(device)

    def verify_command(self, command):
        command.rACT = max(0, command.rACT)
        command.rACT = min(1, command.rACT)

        command.rGTO = max(0, command.rGTO)
        command.rGTO = min(1, command.rGTO)

        command.rATR = max(0, command.rATR)
        command.rATR = min(1, command.rATR)

        command.rPR = max(0, command.rPR)
        command.rPR = min(255, command.rPR)

        command.rSP = max(0, command.rSP)
        command.rSP = min(255, command.rSP)

        command.rFR = max(0, command.rFR)
        command.rFR = min(255, command.rFR)

        return command

    def refresh_command(self, command):
        command = self.verify_command(command)
        self.message = []
        self.message.append(command.rACT + (command.rGTO << 3) + (command.rATR << 4))
        self.message.append(0)
        self.message.append(0)
        self.message.append(command.rPR)
        self.message.append(command.rSP)
        self.message.append(command.rFR)

    def send_command(self):
        self.client.send_command(self.message)

    def get_status(self):
        status = self.client.get_status(6)

        # Message to output
        message = _InputCMD()

        # Assign the values to their respective variables
        message.gACT = (status[0] >> 0) & 0x01;
        message.gGTO = (status[0] >> 3) & 0x01;
        message.gSTA = (status[0] >> 4) & 0x03;
        message.gOBJ = (status[0] >> 6) & 0x03;
        message.gFLT = status[2]
        message.gPR = status[3]
        message.gPO = status[4]
        message.gCU = status[5]

        return message


class RobotiqUSBCtrlGripper(object):
    def __init__(self, device):
        self.cur_status = None
        self.base_gripper = RobotiqBaseRobotiq2FGripper(device)
        self.reset()

    def wait_for_connection(self, timeout=-1):
        time.sleep(0.1)
        start_time = time.time()
        while True:
            cur_status = self.base_gripper.get_status()
            if (timeout >= 0. and time.time() - start_time > timeout):
                return False
            if cur_status is not None:
                return True
            time.sleep(0.1)

    def is_ready(self):
        cur_status = self.base_gripper.get_status()
        
        return cur_status.gSTA == 3 and cur_status.gACT == 1

    def is_reset(self):
        cur_status = self.base_gripper.get_status()
        return cur_status.gSTA == 0 and cur_status.gACT == 0

    def is_moving(self):
        cur_status = self.base_gripper.get_status()
        return cur_status.gGTO == 1 and cur_status.gOBJ == 0

    def is_stopped(self):
        cur_status = self.base_gripper.get_status()
        return cur_status.gOBJ != 0

    def object_detected(self):
        cur_status = self.base_gripper.get_status()
        return cur_status.gOBJ == 1 or cur_status.gOBJ == 2

    def get_fault_status(self):
        cur_status = self.base_gripper.get_status()
        return cur_status.gFLT

    def get_pos(self):
        cur_status = self.base_gripper.get_status()
        po = cur_status.gPO
        return np.clip(0.087 / (13. - 230.) * (po - 230.), 0, 0.087)

    def get_req_pos(self):
        cur_status = self.base_gripper.get_status()
        pr = cur_status.gPR
        return np.clip(0.087 / (13. - 230.) * (pr - 230.), 0, 0.087)

    def is_closed(self):
        cur_status = self.base_gripper.get_status()
        return cur_status.gPO >= 230

    def is_opened(self):
        cur_status = self.base_gripper.get_status()
        return cur_status.gPO <= 13

    # in mA
    def get_current(self):
        cur_status = self.base_gripper.get_status()
        return cur_status.gCU * 0.1

    # if timeout is negative, wait forever
    def wait_until_stopped(self, timeout=-1):
        start_time = time.time()
        while True:
            if (timeout >= 0. and time.time() - start_time > timeout) or self.is_reset():
                return False
            if self.is_stopped():
                return True
            time.sleep(0.1)

    def wait_until_moving(self, timeout=-1):
        start_time = time.time()
        while True:
            if (timeout >= 0. and time.time() - start_time > timeout) or self.is_reset():
                return False
            if not self.is_stopped():
                return True
            time.sleep(0.1)

    def reset(self):
        cmd = _OutputCMD()
        cmd.rACT = 0
        self.base_gripper.refresh_command(cmd)

    def activate(self, timeout=-1):
        cmd = _OutputCMD()
        cmd.rACT = 1
        cmd.rGTO = 1
        cmd.rPR = 0
        cmd.rSP = 255
        cmd.rFR = 150
        self.base_gripper.refresh_command(cmd)
        start_time = time.time()
        while True:
            if timeout >= 0. and time.time() - start_time > timeout:
                return False
            if self.is_ready():
                return True
            time.sleep(0.1)

    def auto_release(self):
        cmd = _OutputCMD()
        cmd.rACT = 1
        cmd.rATR = 1
        self.base_gripper.refresh_command(cmd)

    ##
    # Goto position with desired force and velocity
    # @param pos Gripper width in meters. [0, 0.087]
    # @param vel Gripper speed in m/s. [0.013, 0.100]
    # @param force Gripper force in N. [30, 100] (not precise)
    def goto(self, pos, vel, force, block=False, timeout=-1):
        cmd = _OutputCMD()
        cmd.rACT = 1
        cmd.rGTO = 1
        cmd.rPR = int(np.clip((13. - 230.) / 0.087 * pos + 230., 0, 255))
        cmd.rSP = int(np.clip(255. / (0.1 - 0.013) * (vel - 0.013), 0, 255))
        cmd.rFR = int(np.clip(255. / (100. - 30.) * (force - 30.), 0, 255))
        self.base_gripper.refresh_command(cmd)
        time.sleep(0.1)
        if block:
            if not self.wait_until_moving(timeout):
                return False
            return self.wait_until_stopped(timeout)
        return True

    def stop(self, block=False, timeout=-1):
        cmd = _OutputCMD()
        cmd.rACT = 1
        cmd.rGTO = 0
        self.base_gripper.refresh_command(cmd)
        time.sleep(0.1)
        if block:
            return self.wait_until_stopped(timeout)
        return True

    def open(self, vel=0.1, force=100, block=False, timeout=-1):
        if self.is_opened():
            return True
        return self.goto(1.0, vel, force, block=block, timeout=timeout)

    def close(self, vel=0.1, force=100, block=False, timeout=-1):
        if self.is_closed():
            return True
        return self.goto(-1.0, vel, force, block=block, timeout=timeout)

    def send_commond(self):
        self.base_gripper.send_command()

    def open_gripper(self):
        self.open()
        self.send_commond()

    def close_gripper(self):
        self.close()
        self.send_commond()


if __name__ == "__main__":
    gripper = RobotiqUSBCtrlGripper('/dev/ttyUSB0')
    gripper.activate()
    gripper.send_commond()
    time.sleep(0.5)
    gripper.close()
    gripper.send_commond()
    time.sleep(1)
