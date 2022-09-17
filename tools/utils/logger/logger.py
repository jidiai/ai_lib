import logging
from pickle import LONG_BINGET
import settings
from colorlog import ColoredFormatter
from logging import LogRecord, Logger

class MyLoggerAdapter(logging.LoggerAdapter):
    def __init__(self, logger, extra):
         super().__init__(logger, extra)
         
    def process(self, msg, kwargs):
        
        if "tag" in self.extra:
            msg="{}\ntags: {}".format(msg,self.extra["tags"])
        
        return msg,kwargs
        
class LoggerFactory:    
    
    @staticmethod
    def build_logger(name="MALib"):
        Logger = logging.getLogger(name)
        
        Logger.setLevel(settings.LOG_LEVEL)
        Logger.handlers = []  # No duplicated handlers
        Logger.propagate = False  # workaround for duplicated logs in ipython

        formatter = ColoredFormatter(
        "%(log_color)s[%(asctime)s][%(levelname)s] %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "white,bold",
            "INFOV": "cyan,bold",
            "WARNING": "yellow",
            "ERROR": "red,bold",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
        style="%",
    )


        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(settings.LOG_LEVEL)
        stream_handler.setFormatter(formatter)
        Logger.addHandler(stream_handler)
        return Logger

    @staticmethod
    def add_file_handler(Logger,filepath):
        file_handler = logging.FileHandler(filepath,mode='a')
        
        formatter = ColoredFormatter(
        "%(log_color)s[%(asctime)s][%(levelname)s] %(message)s <pid: %(process)d, tid: %(thread)d, module: %(module)s, func: %(funcName)s>",
        datefmt=None,
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "white,bold",
            "INFOV": "cyan,bold",
            "WARNING": "yellow",
            "ERROR": "red,bold",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
        style="%",
    )
        file_handler.setLevel(settings.LOG_LEVEL)
        file_handler.setFormatter(formatter)
        Logger.addHandler(file_handler)
        return Logger
        
    @staticmethod
    def get_logger(name="MALib",extra=None):
        logger=logging.getLogger(name)
        if extra is not None:
            logger=MyLoggerAdapter(logger,extra)
        return logger