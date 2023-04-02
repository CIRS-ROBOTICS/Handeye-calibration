class Robot(object):
    def reset(self):
        raise NotImplementedError

    def apply_action(self, *motor_commands):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    def get_robot_params(self):
        raise NotImplementedError
