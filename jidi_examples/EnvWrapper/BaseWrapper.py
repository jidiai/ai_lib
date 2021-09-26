import abc

class BaseWrapper(metaclass=abc.ABCMeta):
    def __init__(self, env=None):
        self.env = env

    def set_env(self, env):
        self.env = env

    @abc.abstractmethod
    def step(self, action):
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        '''
        reset返回的态势与step的情况一致，要求reset时一定返回重置时环境的状态情况
        '''
        raise NotImplementedError

    def seed(self):
        pass

    def close(self):
        pass

    @abc.abstractmethod
    def get_actionspace(self, character=None):
        raise NotImplementedError

    @abc.abstractmethod
    def get_observationspace(self, character=None):
        raise NotImplementedError
