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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=True,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        # breakpoint()
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # test_batch = test_batch.to('cuda')

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            reward_tensor = self.val_reward_fn(test_batch)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        return metric_dict

    def init_workers(self):
        """初始化资源池和工作组 (Worker Group)"""
        # 1. 创建实际的 Ray 资源池
        # self.resource_pool_manager 包含了资源池的规格和角色到池的映射
        # create_resource_pool() 方法会根据规格实际创建这些 RayResourcePool 对象
        self.resource_pool_manager.create_resource_pool()

        # 2. 初始化一个字典，用于存储每个资源池 (RayResourcePool 对象) 将要运行的 Worker 类及其配置
        # 键是 RayResourcePool 对象，值是一个字典 (角色名 -> RayClassWithInitArgs)
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # 3. 配置 Actor 和 Rollout Worker
        # self.hybrid_engine 通常表示 Actor 和 Rollout 功能合并在一个 Worker 中
        if self.hybrid_engine:
            # 获取 ActorRollout 角色应该使用的资源池
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            # 创建 RayClassWithInitArgs 对象，封装了 Worker 类、配置和角色名
            # cls: 从 role_worker_mapping 中获取 ActorRollout 对应的 Worker 类
            # config: ActorRollout 相关的配置 (self.config.actor_rollout_ref)
            # role: 指定角色名为 'actor_rollout'，用于后续 WorkerGroup 的命名和管理
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            # 将 actor_rollout_cls 添加到其对应的资源池的映射中
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            # 如果不是混合引擎模式，当前实现不支持，抛出错误
            raise NotImplementedError

        # 4. 配置 Critic Worker (如果使用 GAE 优势估计)
        if self.config.algorithm.adv_estimator == 'gae': # Generalized Advantage Estimation
            # 获取 Critic 角色应该使用的资源池
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            # 创建 Critic Worker 的 RayClassWithInitArgs 对象
            # cls: 从 role_worker_mapping 中获取 Critic 对应的 Worker 类
            # config: Critic 相关的配置 (self.config.critic)
            # 角色名默认为 'critic' (由 RayClassWithInitArgs 内部逻辑或后续 spawn 决定)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            # 将 critic_cls 添加到其对应的资源池的映射中
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            # 设置标志位，表示将使用 Critic
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo': # Group-wise Reward Policy Optimization
            # GRPO 可能不直接使用独立的 Critic 网络进行价值估计
            self.use_critic = False
        else:
            # 如果优势估计算法不是 'gae' 或 'grpo'，抛出错误
            raise NotImplementedError

        # 5. 配置 Reference Policy Worker (如果需要)
        # self.use_reference_policy 标志在 __init__ 中根据 role_worker_mapping 是否包含 RefPolicy 设置
        if self.use_reference_policy:
            # 获取 RefPolicy 角色应该使用的资源池
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            # 创建 Reference Policy Worker 的 RayClassWithInitArgs 对象
            # cls: 从 role_worker_mapping 中获取 RefPolicy 对应的 Worker 类
            # config: 参考策略通常使用与 ActorRollout 相同的模型结构和配置 (self.config.actor_rollout_ref)
            # role: 指定角色名为 'ref'
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            # 将 ref_policy_cls 添加到其对应的资源池的映射中
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # 6. 配置 Reward Model (RM) Worker (如果需要)
        # self.use_rm 标志在 __init__ 中根据 role_worker_mapping 是否包含 RewardModel 设置
        if self.use_rm:
            # 如果使用基于模型的奖励，则创建 RM Worker
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            # 创建 Reward Model Worker 的 RayClassWithInitArgs 对象
            # cls: 从 role_worker_mapping 中获取 RewardModel 对应的 Worker 类
            # config: Reward Model 相关的配置 (self.config.reward_model)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            # 将 rm_cls 添加到其对应的资源池的映射中
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # 7. 初始化 WorkerGroup
        # NOTE: 如果希望每个角色使用不同的资源池（可以支持不同的并行大小），
        # 则不应使用 `create_colocated_worker_cls`。而是直接将不同的资源池传递给不同的工作组。
        # 更多信息请参见 https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb。

        all_wg = {} # 用于存储所有创建的 Worker Group 实例，键为角色名
        self.wg_dicts = [] # 用于存储 WorkerDict 的引用，以支持 ray >= 2.31
        # 遍历之前构建的 self.resource_pool_to_cls 字典
        # resource_pool 是 RayResourcePool 对象
        # class_dict 是一个字典，例如 {'actor_rollout': actor_rollout_cls_obj, 'critic': critic_cls_obj}
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            # 如果一个资源池中有多个角色（例如，Actor 和 Critic 在同一个资源池），
            # create_colocated_worker_cls 会将它们组合成一个单一的 Ray Actor 类 (WorkerDict)，
            # 这个 Actor 类内部会管理这些不同的 Worker 实例。
            # class_dict: {'role_name_A': RayClassWithInitArgs_A, 'role_name_B': RayClassWithInitArgs_B}
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)

            # 使用指定的 RayWorkerGroup 类 (例如 RayWorkerGroup 或 NVMegatronRayWorkerGroup)
            # 为当前资源池和组合后的 worker_dict_cls 创建一个 WorkerGroup 管理对象 (wg_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)

            # 调用 wg_dict.spawn() 来实际创建 Ray Actor 实例 (Worker)
            # prefix_set=class_dict.keys() 告诉 spawn 方法要为 class_dict 中的哪些角色创建 Worker 实例
            # spawn_wg 返回一个字典，键是角色名，值是对应的 WorkerGroup 代理对象 (Ray Actor Handle)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            # 将新创建的 Worker Group 代理对象合并到 all_wg 字典中
            all_wg.update(spawn_wg)
            # 保留 WorkerDict 的引用，以支持 ray >= 2.31 (参考 Ray PR #45699)
            self.wg_dicts.append(wg_dict)

        # 8. 将创建的 Worker Group 代理对象赋值给相应的实例变量，并初始化模型
        if self.use_critic:
            self.critic_wg = all_wg['critic'] # 获取 Critic Worker Group
            self.critic_wg.init_model()       # 调用其 init_model 方法进行初始化

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref'] # 获取 Reference Policy Worker Group
            self.ref_policy_wg.init_model()    # 初始化模型

        if self.use_rm:
            self.rm_wg = all_wg['rm']          # 获取 Reward Model Worker Group
            self.rm_wg.init_model()            # 初始化模型

        # ActorRollout Worker Group 最后创建和初始化
        # 这对于使用 vLLM 等推理引擎的场景可能比较重要，因为它可以在其他模型加载后更好地估计 KV 缓存内存
        self.actor_rollout_wg = all_wg['actor_rollout'] # 获取 ActorRollout Worker Group
        self.actor_rollout_wg.init_model()              # 初始化模型

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        PPO 的训练循环。
        驱动程序进程只需要通过 RPC 调用工作组的计算函数来构建 PPO 数据流。
        轻量级的优势计算在驱动程序进程上完成。
        """
        # 从 verl.utils.tracking 模块导入 Tracking 类，用于实验跟踪和日志记录
        from verl.utils.tracking import Tracking
        # 从 omegaconf 模块导入 OmegaConf，用于处理配置文件
        from omegaconf import OmegaConf

        # 初始化 Tracking 对象，用于记录实验的指标和配置
        # project_name: 项目名称，从配置中获取
        # experiment_name: 实验名称，从配置中获取
        # default_backend: 日志记录的后端，从配置中获取 (例如，wandb, tensorboard)
        # config: 将 OmegaConf 配置对象转换为字典，并解析所有变量
        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        # 初始化全局训练步数
        self.global_steps = 0

        # 在训练开始前执行验证
        # 目前，我们只支持使用 reward_function 进行验证。
        # 如果配置了验证奖励函数 (self.val_reward_fn) 并且配置允许在训练前验证
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            # 调用 _validate 方法执行验证
            val_metrics = self._validate()
            # 打印初始验证指标
            pprint(f'Initial validation metrics: {val_metrics}')
            # 使用 logger 记录验证指标
            logger.log(data=val_metrics, step=self.global_steps)
            # 如果配置了 'val_only' 为 True，则只进行验证，不进行训练，直接返回
            if self.config.trainer.get('val_only', False):
                return

        # 训练从第 1 步开始
        self.global_steps += 1

        # 外层循环：遍历总的训练轮数 (epochs)
        for epoch in range(self.config.trainer.total_epochs):
            # 内层循环：遍历训练数据加载器中的每个批次 (batch)
            for batch_dict in self.train_dataloader:
                # 打印当前的 epoch 和全局步数
                print(f'epoch {epoch}, step {self.global_steps}')
                # 初始化用于存储当前批次指标的字典
                metrics = {}
                # 初始化用于存储当前批次各阶段耗时的字典
                timing_raw = {}

                # 将从 dataloader 获取的字典转换为 DataProto 对象，这是一种自定义的数据结构
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # 从批次数据中弹出用于序列生成的键 ('input_ids', 'attention_mask', 'position_ids')
                # 这些键对应的数据将用于 actor 模型生成响应序列
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                # 使用 _timer 上下文管理器记录整个训练步骤 (step) 的耗时
                with _timer('step', timing_raw):
                    # 1. 生成序列 (Rollout 阶段)
                    # 使用 _timer 记录序列生成 (gen) 的耗时
                    with _timer('gen', timing_raw):
                        # 调用 actor_rollout_wg (Actor-Rollout Worker Group) 的 generate_sequences 方法生成响应序列
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    # 为批次中的每个样本生成一个唯一的 ID (uid)
                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # 根据配置中的 rollout.n (每个 prompt 生成的响应数量) 重复批次数据，以与 rollout 过程中生成的多个响应对齐
                    # interleave=True 表示交错重复
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # 将生成的序列数据 (gen_batch_output) 合并回原始批次数据 (batch)
                    batch = batch.union(gen_batch_output)

                    # 2. 平衡每个数据并行 (DP) rank 上的有效 token 数量
                    # 注意：这会打乱批次内数据的顺序。
                    # 如果实现基于组的优势计算（如 GRPO 和 RLOO），需要特别注意。
                    self._balance_batch(batch, metrics=metrics)

                    # 计算全局有效 token 数量，并存储在批次的 meta_info 中
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # 3. 如果使用参考策略 (Reference Policy)
                    if self.use_reference_policy:
                        # 计算参考策略的 log_prob
                        with _timer('ref', timing_raw):
                            # 调用 ref_policy_wg (Reference Policy Worker Group) 计算参考 log_prob
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            # 将计算得到的 ref_log_prob 合并到批次数据中
                            batch = batch.union(ref_log_prob)

                    # 4. 如果使用 Critic 网络
                    if self.use_critic:
                        # 计算价值 (values)
                        with _timer('values', timing_raw):
                            # 调用 critic_wg (Critic Worker Group) 计算状态价值
                            values = self.critic_wg.compute_values(batch)
                            # 将计算得到的 values 合并到批次数据中
                            batch = batch.union(values)

                    # 5. 计算优势 (Advantage) 和奖励 (Reward)
                    with _timer('adv', timing_raw):
                        # 计算得分 (scores)。支持基于模型和基于函数的奖励。
                        # 首先使用奖励模型 (Reward Model, RM) 计算得分，然后调用 reward_fn 结合奖励模型的结果和基于规则的结果。
                        if self.use_rm: # 如果使用奖励模型
                            # 首先计算奖励模型的得分
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            # 将奖励模型的得分合并到批次数据中
                            batch = batch.union(reward_tensor)

                        # 结合基于规则的奖励模型 (rule-based RM)
                        # 调用 self.reward_fn (通常是一个 RewardManager 实例) 计算最终的 token 级别得分
                        reward_tensor = self.reward_fn(batch)
                        # 将最终的 token 级别得分存储在批次数据中
                        batch.batch['token_level_scores'] = reward_tensor

                        # 计算奖励 (rewards)。如果可用，则应用 KL 惩罚。
                        # 如果配置中 actor 不使用 KL 损失 (use_kl_loss 为 False)
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            # 应用 KL 惩罚，调整 token_level_scores 得到 token_level_rewards
                            # kl_ctrl: KL 控制器 (AdaptiveKLController 或 FixedKLController)
                            # kl_penalty: KL 惩罚的类型
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            # 更新指标字典
                            metrics.update(kl_metrics)
                        else:
                            # 如果 actor 使用 KL 损失，则 token_level_rewards 直接等于 token_level_scores
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # 计算优势 (advantages)，在驱动程序进程上执行
                        # adv_estimator: 优势估计算法 (例如 'gae', 'grpo')
                        # gamma: 折扣因子
                        # lam: GAE 的 lambda 参数
                        # num_repeat: rollout 的重复次数
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # 6. 更新 Critic 网络
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            # 调用 critic_wg 更新 Critic 网络
                            critic_output = self.critic_wg.update_critic(batch)
                        # 从 Critic 更新的输出中提取指标，并进行归约 (例如，计算均值)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        # 更新指标字典
                        metrics.update(critic_output_metrics)

                    # 7. 实现 Critic 预热 (warmup)
                    # 如果当前全局步数大于等于 Critic 预热步数
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # 更新 Actor 网络
                        with _timer('update_actor', timing_raw):
                            # 调用 actor_rollout_wg 更新 Actor 网络
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        # 从 Actor 更新的输出中提取指标，并进行归约
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        # 更新指标字典
                        metrics.update(actor_output_metrics)

                    # 8. 执行验证 (Validation)
                    # 如果配置了验证奖励函数，并且验证频率大于 0，并且当前全局步数是验证频率的倍数
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            # 调用 _validate 方法执行验证
                            val_metrics: dict = self._validate()
                        # 更新指标字典
                        metrics.update(val_metrics)

                    # 9. 保存检查点 (Checkpoint)
                    # 如果保存频率大于 0，并且当前全局步数是保存频率的倍数
                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            # 调用 _save_checkpoint 方法保存模型检查点
                            self._save_checkpoint()

                # 10. 收集和记录指标
                # 计算与数据相关的指标 (例如，奖励、优势、价值的均值/最大值/最小值等)
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # 计算与时间相关的指标 (例如，各阶段耗时，每 token 耗时)
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: 创建一个支持多种后端的规范化 logger
                # 使用 logger 记录当前步骤的所有指标
                logger.log(data=metrics, step=self.global_steps)

                # 全局步数加 1
                self.global_steps += 1

                # 如果当前全局步数达到总训练步数
                if self.global_steps >= self.total_training_steps:
                    # 在训练结束后执行最终验证
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    # 结束训练
                    return
