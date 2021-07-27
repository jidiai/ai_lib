# -*- coding:utf-8  -*-
# 作者：zruizhi   
# 创建时间： 2020/9/11 11:17 上午   
# 描述：选择运行环境，需要维护env/__ini__.py && config.json（存储环境默认参数）

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
