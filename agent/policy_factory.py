import copy
from utils.logger import Logger
import pickle as pkl
import os
import ray
from registry import registry


class PolicyFactory:
    """
    test whether equals to True

    "condition"="==0",
    "condition"=">1",
    "condition"="%2",
    "condition"="default",

    strategy: inherit_last,inherit_best,random,pretrained

    init_cfg:
        "condition":
            strategy: pretrained
            pid: ...
            policy_dir: ...
        "condition":
            strategy:

    """

    def __init__(self, agent_id, population_id, algorithm_cfg, policy_server):
        self.agent_id = agent_id
        self.population_id = population_id
        self.id = "PolicyFactory_{}_{}".format(self.agent_id, self.population_id)
        self.algorithm_cfg = algorithm_cfg
        self.policy_server = policy_server
        # breakpoint()
        self.policy_init_cfg = self.algorithm_cfg.policy_init_cfg[
            self.agent_id
        ].init_cfg
        self.new_policy_ctr = -1
        if "new_policy_ctr_start" in self.policy_init_cfg:
            self.new_policy_ctr = self.policy_init_cfg.new_policy_ctr_start
        Logger.info(
            "{} new policy ctr starts at {}".format(self.id, self.new_policy_ctr)
        )

        self.strategies = []
        self.parse()

    @staticmethod
    def default_policy_id(agent_id, population_id, idx):
        policy_id = "{}_{}_{}".format(agent_id, population_id, idx)
        return policy_id

    def gen_new_policy(self):
        """
        return pid,policy
        """
        self.new_policy_ctr += 1
        policy_id, policy = self.init(self.new_policy_ctr)
        return policy_id, policy

    def parse(self):
        for cfg in self.policy_init_cfg:
            condition = cfg["condition"]
            # sanity check
            strategy = cfg["strategy"]
            if strategy == "inherit_last":
                pass
            elif strategy == "inherit_last_best":
                pass
            elif strategy == "pretrained":
                assert "policy_id" in cfg
                assert "policy_dir" in cfg
            elif strategy == "random":
                pass
            else:
                raise NotImplementedError(
                    "policy initialization strategy {} is not implemented".format(
                        strategy
                    )
                )
            if condition == "default":
                predicate = lambda x: True
            else:
                predicate = eval("lambda x: x{}".format(condition))
            self.strategies.append((predicate, cfg))

    def init(self, new_policy_ctr):
        found = False
        for predicate, cfg in self.strategies:
            if predicate(new_policy_ctr):
                strategy = cfg["strategy"]
                # Logger.error("{} {}".format(strategy,new_policy_ctr))
                found = True
                # Logger.error("strategy: {}".format(strategy))
                if strategy == "inherit_last":
                    pid, policy = self.init_from_last_policy(
                        self.policy_server, new_policy_ctr
                    )
                elif strategy == "pretrained":
                    pid, policy = self.init_from_pretrained_model(
                        cfg["policy_id"], cfg["policy_dir"], new_policy_ctr
                    )
                elif strategy == "random":
                    pid, policy = self.init_from_random(
                        self.algorithm_cfg, new_policy_ctr
                    )
                elif strategy == "inherit_last_best":
                    pid, policy = self.init_from_last_best(
                        self.policy_server, new_policy_ctr
                    )
                else:
                    raise NotImplementedError(
                        "policy initialization strategy {} is not implemented".format(
                            strategy
                        )
                    )

                reset_layer = cfg.get("reset_layer", None)
                if reset_layer is not None:
                    policy.reset_layers(reset_layer)
                noise_layer = cfg.get("noise_layer", None)
                if noise_layer is not None:
                    policy.noise_layers(noise_layer)

                break
        if not found:
            if new_policy_ctr == 0:
                pid, policy = self.init_from_random(self.algorithm_cfg, new_policy_ctr)
            else:
                pid, policy = self.init_from_last_policy(
                    self.policy_server, new_policy_ctr
                )
        Logger.warning(
            "policy {} uses custom_config: {}".format(pid, policy.custom_config)
        )
        return pid, policy

    def init_from_last_policy(self, policy_server, new_policy_ctr):
        assert new_policy_ctr > 0, "policy_ctr: {}".format(new_policy_ctr)
        last_policy_ctr = new_policy_ctr - 1
        last_policy_id = self.default_policy_id(
            self.agent_id, self.population_id, last_policy_ctr
        )
        last_policy_desc = ray.get(
            policy_server.pull.remote(self.id, self.agent_id, last_policy_id)
        )
        if last_policy_desc is None:
            raise Exception(
                "last policy {} {} not found".format(self.agent_id, last_policy_id)
            )
        new_policy_id = self.default_policy_id(
            self.agent_id, self.population_id, new_policy_ctr
        )
        policy = copy.deepcopy(last_policy_desc.policy)
        Logger.warning(
            f"{self.agent_id}: {new_policy_id} is initialized from last policy {last_policy_id}"
        )
        return new_policy_id, policy

    def init_from_last_best(self, policy_server, new_policy_ctr):
        assert new_policy_ctr > 0, "policy_ctr: {}".format(new_policy_ctr)
        last_policy_ctr = new_policy_ctr - 1
        last_policy_id = self.default_policy_id(
            self.agent_id, self.population_id, f"{last_policy_ctr}.best"
        )
        last_policy_desc = ray.get(
            policy_server.pull.remote(self.id, self.agent_id, last_policy_id)
        )
        if last_policy_desc is None:
            last_policy_id = self.default_policy_id(
                self.agent_id, self.population_id, f"{last_policy_ctr}"
            )
            last_policy_desc = ray.get(
                policy_server.pull.remote(self.id, self.agent_id, last_policy_id)
            )
            if last_policy_desc is None:
                raise Exception(
                    "last policy {} {} not found".format(self.agent_id, last_policy_id)
                )
        new_policy_id = self.default_policy_id(
            self.agent_id, self.population_id, new_policy_ctr
        )
        policy = copy.deepcopy(last_policy_desc.policy)
        Logger.warning(
            f"{self.agent_id}: {new_policy_id} is initialized from last best policy {last_policy_id}"
        )
        return new_policy_id, policy

    def init_from_pretrained_model(self, policy_id, policy_dir, new_policy_ctr):
        policy_cfg_pickle = pkl.load(open(os.path.join(policy_dir, "desc.pkl"), "rb"))
        policy_cls = registry.get(registry.POLICY, policy_cfg_pickle["registered_name"])
        new_policy_id = self.default_policy_id(
            self.agent_id, self.population_id, new_policy_ctr
        )
        policy = policy_cls.load(policy_dir, env_agent_id=self.agent_id)
        policy.custom_config = self.algorithm_cfg["custom_config"]
        Logger.warning(
            f"{self.agent_id}: {new_policy_id} is initialized from a pretrained model {policy_id}:{policy_dir}, cls = {policy_cls}"
        )
        return new_policy_id, policy

    def init_from_random(self, algorithm_conf, new_policy_ctr):
        algorithm_conf = self.algorithm_cfg
        env_agent_id = self.agent_id
        policy_cls = registry.get(registry.POLICY, algorithm_conf["name"])
        policy = policy_cls(
            registered_name=algorithm_conf["name"],
            observation_space=None,  # jh: legacy
            action_space=None,  # jh: legacy
            model_config=algorithm_conf.get("model_config", {}),
            custom_config=algorithm_conf.get("custom_config", {}),
            env_agent_id=env_agent_id,
        )
        new_policy_id = self.default_policy_id(
            self.agent_id, self.population_id, new_policy_ctr
        )
        Logger.warning(f"{env_agent_id}: {new_policy_id} is initialized from random")
        return new_policy_id, policy
