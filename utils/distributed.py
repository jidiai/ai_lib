import time
import ray
from light_malib.utils.logger import Logger

def get_actor(obj_name,actor_name,max_retries=10):
    actor=None
    ctr=0
    while ctr<=max_retries:
        ctr+=1
        try:
            if actor is None:
                actor = ray.get_actor(actor_name)
            break
        except Exception as e:
            time.sleep(1)
            Logger.warning("{} retried to get actor {} (ctr: {})".format(obj_name,actor_name,ctr))
            continue
    if actor is None:
        # TODO (jh): how to fail the whole cluster in the error case?
        Logger.error("{} failed to get actor {}".format(obj_name,actor_name))
    else:
        Logger.debug("{} succeeded to get actor {}".format(obj_name,actor_name))
    return actor

def get_resources(resources):
    assert resources is not None
    if "resources" in resources:
        resources["resources"]=dict(resources["resources"])
    return resources