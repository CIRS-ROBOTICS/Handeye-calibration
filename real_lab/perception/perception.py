class Perception(object):
    def get_data(self):
        """

        :return: Color Image with BGR style, Depth Image with Meters
        """
        raise NotImplementedError

    def close(self):
        """

        :return:
        """
        raise NotImplementedError
