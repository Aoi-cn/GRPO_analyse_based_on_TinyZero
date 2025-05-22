# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math, multiply, countdown
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _select_rm_score_fn(data_source):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math.compute_score
    elif "multiply" in data_source or "arithmetic" in data_source:
        return multiply.compute_score
    elif "countdown" in data_source:
        return countdown.compute_score
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        breakpoint()
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    # 从 verl.utils.fs 模块导入 copy_local_path_from_hdfs 函数，用于从 HDFS 复制文件到本地
    from verl.utils.fs import copy_local_path_from_hdfs
    # 从 transformers 库导入 AutoTokenizer，用于自动加载预训练模型的分词器
    from transformers import AutoTokenizer

    # 打印初始配置信息
    from pprint import pprint # 导入 pprint 模块，用于更美观地打印 Python 对象
    from omegaconf import OmegaConf # 导入 OmegaConf 库，用于处理配置文件
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True 会解析配置中的符号值（例如，${xxx}）
    OmegaConf.resolve(config) # 再次确保所有配置值都已解析
    breakpoint() # 设置一个断点，方便调试时检查程序状态

    # 从 HDFS 下载检查点文件
    # config.actor_rollout_ref.model.path 指定了模型在 HDFS 上的路径
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # 实例化分词器
    from verl.utils import hf_tokenizer # 从 verl.utils 模块导入 hf_tokenizer 函数
    # 使用下载到本地的模型路径来初始化分词器
    tokenizer = hf_tokenizer(local_path)

    # 定义 worker 类
    # 根据配置中 actor_rollout_ref.actor.strategy 的值来选择不同的 worker 实现
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        # 如果策略是 'fsdp' (Fully Sharded Data Parallel)
        # 断言 actor 和 critic 的策略必须相同
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        # 从 verl.workers.fsdp_workers 模块导入 FSDP 版本的 ActorRolloutRefWorker 和 CriticWorker
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        # 从 verl.single_controller.ray 模块导入 RayWorkerGroup，用于管理 Ray worker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup # 将 RayWorkerGroup 赋值给 ray_worker_group_cls

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        # 如果策略是 'megatron' (一种大规模模型训练框架)
        # 断言 actor 和 critic 的策略必须相同
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        # 从 verl.workers.megatron_workers 模块导入 Megatron 版本的 ActorRolloutRefWorker 和 CriticWorker
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        # 从 verl.single_controller.ray.megatron 模块导入 NVMegatronRayWorkerGroup
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup # 将 NVMegatronRayWorkerGroup 赋值给 ray_worker_group_cls

    else:
        # 如果策略不是 'fsdp' 或 'megatron'，则抛出 NotImplementedError
        raise NotImplementedError

    # 从 verl.trainer.ppo.ray_trainer 模块导入 ResourcePoolManager 和 Role
    # ResourcePoolManager 用于管理资源池，Role 用于定义不同 worker 的角色
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    # 定义角色到 worker 类的映射
    # Role.ActorRollout: 对应 ActorRolloutRefWorker，用于生成经验数据
    # Role.Critic: 对应 CriticWorker，用于评估状态价值
    # Role.RefPolicy: 对应 ActorRolloutRefWorker，作为参考策略（通常是初始模型或SFT模型）
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker), # 将 ActorRolloutRefWorker 声明为 Ray远程 actor
        Role.Critic: ray.remote(CriticWorker), # 将 CriticWorker 声明为 Ray远程 actor
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker) # 将 ActorRolloutRefWorker 声明为 Ray远程 actor
    }

    global_pool_id = 'global_pool' # 定义全局资源池的 ID
    # 定义资源池的规格
    # global_pool_id 对应一个列表，列表中的每个元素代表一个节点的 GPU 数量
    # config.trainer.n_gpus_per_node 是每个节点的 GPU 数量
    # config.trainer.nnodes 是节点的数量
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    # 定义角色到资源池的映射
    # 所有角色都使用 'global_pool' 资源池
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # 这里将采用多源奖励函数：
    # - 对于基于规则的奖励模型 (RM)，直接调用奖励分数函数
    # - 对于基于模型的 RM，调用一个模型进行评估
    # - 对于代码相关的提示，如果存在测试用例，则发送到沙箱执行
    # - 最后，将所有奖励组合起来
    # - 奖励类型取决于数据的标签

    # 如果启用了奖励模型 (config.reward_model.enable 为 True)
    if config.reward_model.enable:
        # 根据奖励模型的策略选择不同的 RewardModelWorker 实现
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        # 将 RewardModelWorker 添加到角色 worker 映射中
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        # 将 RewardModel 角色也映射到全局资源池
        mapping[Role.RewardModel] = global_pool_id

    # 实例化奖励管理器，用于训练过程中的奖励计算
    # tokenizer: 之前实例化的分词器
    # num_examine: 打印到控制台的已解码响应的批次数，这里设置为 0，表示不打印
    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)

    # 注意：验证过程始终使用基于函数的奖励模型 (RM)
    # 实例化用于验证的奖励管理器
    # num_examine: 设置为 1，表示在验证时会打印一个批次的解码响应
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    # 实例化资源池管理器
    # resource_pool_spec: 定义的资源池规格
    # mapping: 定义的角色到资源池的映射
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    # 实例化 RayPPOTrainer，这是 PPO 算法的训练器
    trainer = RayPPOTrainer(config=config, # 传入配置对象
                            tokenizer=tokenizer, # 传入分词器
                            role_worker_mapping=role_worker_mapping, # 传入角色到 worker 的映射
                            resource_pool_manager=resource_pool_manager, # 传入资源池管理器
                            ray_worker_group_cls=ray_worker_group_cls, # 传入 Ray worker 组的类
                            reward_fn=reward_fn, # 传入训练奖励函数
                            val_reward_fn=val_reward_fn) # 传入验证奖励函数
    breakpoint() # 设置另一个断点，方便调试
    trainer.init_workers() # 初始化所有 worker
    trainer.fit() # 开始训练过程


if __name__ == '__main__':
    main()
