import time
import ray
from tools.utils.logger import Logger


def get_actor(obj_name, actor_name, max_retries=10):
    actor = None
    ctr = 0
    while ctr < max_retries:
        try:
            if actor is None:
                actor = ray.get_actor(actor_name)
            break
        except Exception as e:
            time.sleep(1)
            continue
    if actor is None:
        # TODO: how to fail the whole cluster in the error case?
        Logger.error("{} failed to get actor {}".format(obj_name, actor_name))
    else:
        Logger.warning("{} succeed to get actor {}".format(obj_name, actor_name))
    return actor
