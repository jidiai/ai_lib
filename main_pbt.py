from utils.logger import Logger
import ray
import argparse
from utils.cfg import load_cfg, convert_to_easydict
from utils.random import set_random_seed
from framework.pbt_runner import PBTRunner
import time
import os
import yaml
from omegaconf import OmegaConf

import pathlib

BASE_DIR = str(pathlib.Path(__file__).resolve().parent)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    return args


def get_local_ip_address():
    import socket
    ip_address = socket.gethostbyname(socket.gethostname())
    return ip_address


def start_cluster():
    try:
        cluster_start_info = ray.init(address="auto")
    except ConnectionError:
        Logger.warning(
            "No active cluster detected, will create local ray instance."
        )
        cluster_start_info = ray.init(resources={})

    Logger.warning(
        "============== Cluster Info ==============\n{}".format(cluster_start_info)
    )
    Logger.warning("* cluster resources:\n{}".format(ray.cluster_resources()))
    Logger.warning("this worker ip: {}".format(ray.get_runtime_context().worker.node_ip_address))
    return cluster_start_info


def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    set_random_seed(cfg.seed)
    cluster_start_info = start_cluster()
    assert cfg.distributed.nodes.master.ip is not None
    if cfg.distributed.nodes.master.ip == "auto":
        ip = ray.get_runtime_context().worker.node_ip_address
        cfg.distributed.nodes.master.ip = ip
        Logger.warning("Automatically set master ip to local ip address: {}".format(ip))
    # cluster_start_info = start_cluster()

    # check cfg
    # check gpu number here
    assert cfg.training_manager.num_trainers <= ray.cluster_resources()[
        "GPU"], "#trainers({}) should be <= #gpus({})".format(cfg.training_manager.num_trainers,
                                                              ray.cluster_resources()["GPU"])
    # check batch size here
    assert cfg.training_manager.batch_size <= cfg.data_server.table_cfg.capacity, "batch_size({}) should be <= capacity({})".format(
        cfg.training_manager.batch_size, cfg.data_server.table_cfg.capacity)

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    cfg.expr_log_dir = os.path.join(cfg.log_dir, cfg.expr_group, cfg.expr_name, f'{cfg.log_name}-{timestamp}')
    cfg.expr_log_dir = os.path.join(BASE_DIR, cfg.expr_log_dir)
    os.makedirs(cfg.expr_log_dir, exist_ok=True)

    # copy config file
    yaml_path = os.path.join(cfg.expr_log_dir, 'config.yaml')
    with open(yaml_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
        # yaml.dump(OmegaConf.to_yaml(cfg), f, sort_keys=False)

    cfg = convert_to_easydict(cfg)

    from monitor.monitor import Monitor
    from utils.distributed import get_resources
    Monitor = ray.remote(**get_resources(cfg.monitor.distributed.resources))(Monitor)
    monitor = Monitor.options(name="Monitor", max_concurrency=100).remote(cfg)

    runner = PBTRunner(cfg)

    try:
        runner.run()
    except KeyboardInterrupt as e:
        Logger.warning(
            "Detected KeyboardInterrupt event, start background resources recycling threads ..."
        )
    finally:
        runner.close()
        ray.get(monitor.close.remote())
        ray.shutdown()


if __name__ == "__main__":
    main()