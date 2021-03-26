# -*- coding:utf-8  -*-
import json
import env
import os


def make(env_type, conf=None):
    file_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if not conf:
        with open(file_path) as f:
            conf = json.load(f)[env_type]
    class_literal = conf['class_literal']

    return getattr(env, class_literal)(conf)


if __name__ == "__main__":
    make("classic_MountainCar-v0")


