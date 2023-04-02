# from NetFT import *
import numpy as np
import socket
from multiprocessing import Process

from ..perception import Perception


class FT300Sensor(Perception):
    def __init__(self, ip='192.168.1.102', port=63351):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((ip, port))
        self.start_threading()  # use to clear the buffer to make the force data is latest
        self.biasValue = None
        self.bias_sensor()

    def _read_value(self, n=10):
        ret = []
        for _ in range(n):
            data = self._receive()
            # print(len(data))
            data = data.replace("b\'(", "")
            data = data.replace(")\'", "")
            data = data.replace(",", " ")
            data = data.split()
            try:
                data = list(map(float, data))
                ret.append(data)
            except:
                print('receiving wrong data')

        ret = np.array(ret)
        ave_ret = np.mean(ret, axis=0)

        return ave_ret

    def bias_sensor(self):
        self.biasValue = self.get_data(raw=True)

    def get_data(self, n=10, raw=False):
        if raw:
            return np.around(np.array(self._read_value(n)))
        else:
            return np.around(np.array(self._read_value(n))) - self.biasValue

    def _receive(self):
        return str(self.s.recv(1024))

    def _callback(self):
        """
        To record the data by a single thread
        :return:
        """
        while self.stream:
            self._receive()

    def start_threading(self):
        """

        :param data:
        :return:
        """
        self.stream = True
        self.thread = Process(target=self._callback, args=())
        self.thread.daemon = True
        # self._send()
        self.thread.start()

    def stop_threading(self):
        self.stream = False
        # self._send(self.STOP_COMMAND)  # Stop receiving data
        self.thread.join()

    def close(self):
        pass

if __name__ == '__main__':
    fts = FT300Sensor()

    while True:
        r = fts.get_data()
        print(r)
