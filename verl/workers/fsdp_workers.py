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
The main entry point to run the PPO algorithm
"""

import logging
import os
import warnings

import torch
import torch.distributed
import verl.utils.hdfs_io as hdfs_io
import verl.utils.torch_functional as verl_F
from omegaconf import DictConfig, open_dict
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, offload_fsdp_grad, init_fn, get_init_weight_context_manager
from verl.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_param_and_grad, load_fsdp_optimizer, \
    load_fsdp_param_and_grad
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.flops_counter import FlopsCounter
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from codetiming import Timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


class ActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.role = role
        assert self.role in ['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']

        self._is_actor = self.role in ['actor', 'actor_rollout', 'actor_rollout_ref']
        self._is_rollout = self.role in ['rollout', 'actor_rollout', 'actor_rollout_ref']
        self._is_ref = self.role in ['ref', 'actor_rollout_ref']

        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get('param_offload', False)
            self._is_offload_grad = self.config.actor.fsdp_config.get('grad_offload', False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get('optimizer_offload', False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get('param_offload', False)

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size //= (self.device_mesh.shape[0] // self.ulysses_sequence_parallel_size)
            self.config.actor.ppo_micro_batch_size //= (self.device_mesh.shape[0] //
                                                        self.ulysses_sequence_parallel_size)
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_micro_batch_size *= self.config.rollout.n
        if self._is_rollout:
            self.config.rollout.log_prob_micro_batch_size //= (self.device_mesh.shape[0] //
                                                               self.ulysses_sequence_parallel_size)
            self.config.rollout.log_prob_micro_batch_size *= self.config.rollout.n
        if self._is_ref:
            self.config.ref.log_prob_micro_batch_size //= (self.device_mesh.shape[0] //
                                                           self.ulysses_sequence_parallel_size)
            self.config.ref.log_prob_micro_batch_size *= self.config.rollout.n

    def _build_model_optimizer(self,
                               model_path,
                               fsdp_config,
                               optim_config,
                               override_model_config,
                               use_remove_padding=False,
                               enable_gradient_checkpointing=False,
                               trust_remote_code=False):
        from verl.utils.model import print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType
        from transformers import AutoModelForCausalLM, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
        from torch import optim

        log_gpu_memory_usage('Before init from HF AutoModel', logger=logger)
        local_path = copy_local_path_from_hdfs(model_path)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = fsdp_config.get('model_dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)

        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(actor_model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(actor_model_config, verbose=True)

        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f'Model config after override: {actor_model_config}')

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actor_module = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                torch_dtype=torch_dtype,
                                                                config=actor_model_config,
                                                                attn_implementation='flash_attention_2',
                                                                trust_remote_code=trust_remote_code)
            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage('After init from HF AutoModel', logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        if self._is_ref:
            mixed_precision = None

        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get('wrap_policy', None))

        if self._is_rollout and self.config.rollout.name == 'hf':
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        print(f'wrap_policy: {auto_wrap_policy}')

        # TODO(sgm): support hybrid
        if auto_wrap_policy is None:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD

        # TODO: add transformer policy
        actor_module_fsdp = FSDP(
            actor_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            forward_prefetch=False)

        log_gpu_memory_usage('After Actor FSDP init', logger=logger)

        # TODO: add more optimizer args into config
        if self._is_actor:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup
            actor_optimizer = optim.AdamW(actor_module_fsdp.parameters(),
                                          lr=optim_config.lr,
                                          betas=optim_config.get('betas', (0.9, 0.999)),
                                          weight_decay=optim_config.get('weight_decay', 1e-2))

            total_steps = optim_config.get('total_training_steps', 0)
            num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

            actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer,
                                                                   num_warmup_steps=num_warmup_steps)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        log_gpu_memory_usage('After actor optimizer init', logger=logger)

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self):
        from torch.distributed.device_mesh import init_device_mesh
        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f'rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}'
        rollout_device_mesh = init_device_mesh('cuda', mesh_shape=(dp, infer_tp), mesh_dim_names=['dp', 'infer_tp'])

        if self.config.rollout.name == 'hf':
            from verl.workers.rollout import HFRollout
            from verl.workers.sharding_manager import BaseShardingManager
            rollout = HFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
            rollout_sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?
        elif self.config.rollout.name == 'vllm':
            from verl.workers.rollout.vllm_rollout import vLLMRollout
            from verl.workers.sharding_manager import FSDPVLLMShardingManager
            log_gpu_memory_usage('Before building vllm rollout', logger=None)
            rollout = vLLMRollout(actor_module=self.actor_module_fsdp,
                                  config=self.config.rollout,
                                  tokenizer=self.tokenizer,
                                  model_hf_config=self.actor_model_config)
            log_gpu_memory_usage('After building vllm rollout', logger=None)
            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = 'dummy_hf'
            rollout_sharding_manager = FSDPVLLMShardingManager(module=self.actor_module_fsdp,
                                                               inference_engine=rollout.inference_engine,
                                                               model_config=self.actor_model_config,
                                                               full_params='hf' in self.config.rollout.load_format,
                                                               device_mesh=rollout_device_mesh)
            log_gpu_memory_usage('After building sharding manager', logger=None)

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from omegaconf import OmegaConf
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))

        use_remove_padding = self.config.model.get('use_remove_padding', False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False),
                trust_remote_code=self.config.model.get('trust_remote_code', False))

            # get the original unwrapped module
            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                # param is require during state_dict in sharding manager
                offload_fsdp_grad(module=self.actor_module_fsdp)
                log_gpu_memory_usage('After offload actor grad during init', logger=logger)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage('After offload actor optimizer during init', logger=logger)
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
            self.actor = DataParallelPPOActor(config=self.config.actor,
                                              actor_module=self.actor_module_fsdp,
                                              actor_optimizer=self.actor_optimizer)

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout()

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(model_path=self.config.model.path,
                                                               fsdp_config=self.config.ref.fsdp_config,
                                                               optim_config=None,
                                                               override_model_config=override_model_config,
                                                               use_remove_padding=use_remove_padding,
                                                               trust_remote_code=self.config.model.get(
                                                                   'trust_remote_code', False))[0]
            if self._is_offload_param:
                offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=self._is_offload_grad)

            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        # 将输入数据（包含批次数据和元信息）整体移动到 CUDA 设备。
        # 这通常意味着 DataProto 内部的 TensorDict 中的张量会被移到 GPU。
        data = data.to('cuda')
        breakpoint()
        # breakpoint() # 调试断点，通常在开发和调试时使用。

        # 断言检查，确保当前 worker 实例确实扮演 Actor 的角色。
        # self._is_actor 是在 __init__ 中根据传入的 role 设置的布尔标志。
        assert self._is_actor

        # 如果配置了参数卸载 (offload_param)，则在更新前将 FSDP 包装的 Actor 模型的参数和梯度加载回 GPU。
        # self._is_offload_param 和 self._is_offload_grad 是根据配置设置的标志。
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp, # FSDP 包装的 Actor 模型
                                     device_id=torch.cuda.current_device(), # 当前 CUDA 设备 ID
                                     load_grad=self._is_offload_grad) # 是否也加载梯度

        # 如果配置了优化器状态卸载 (offload_optimizer)，则在更新前将优化器状态加载回 GPU。
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, # Actor 的优化器
                                device_id=torch.cuda.current_device()) # 当前 CUDA 设备 ID

        # 再次确保批次数据在 CUDA 设备上。
        # 尽管 data.to('cuda') 已经执行，这可以视为一个双重检查或针对特定情况的处理。
        data.batch = data.batch.cuda()

        # 记录更新策略前的 GPU 显存使用情况，用于调试和性能分析。
        log_gpu_memory_usage('Before update policy', logger=logger)

        # 使用 Ulysses Sharding Manager 的上下文管理器。
        # 这个管理器负责处理序列并行 (Sequence Parallelism) 相关的数据切分和收集。
        with self.ulysses_sharding_manager:
            # 对输入数据进行预处理，以适应序列并行的需求。
            # 例如，如果启用了序列并行，数据可能会在序列维度上被切分并分发到不同的 GPU。
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            # 执行实际的训练步骤（策略更新）
            # 使用 Timer 来记录 update_policy 方法的执行时间。
            with Timer(name='update_policy', logger=None) as timer:
                # 调用 self.actor (通常是 DataParallelPPOActor 实例) 的 update_policy 方法。
                # 这个方法会执行 PPO 算法的核心更新逻辑，包括计算损失、反向传播和参数更新。
                # data 对象包含了训练所需的所有信息，如 input_ids, attention_mask, old_log_probs, advantages, returns 等。
                metrics = self.actor.update_policy(data=data)
            # 获取 update_policy 的执行时间
            delta_time = timer.last
            # 从元信息中获取全局处理的 token 数量
            global_num_tokens = data.meta_info['global_token_num']
            # 使用 FlopsCounter 估算本次更新的 FLOPs (浮点运算次数) 和 MFU (模型浮点运算利用率)。
            # promised_flops 是模型的理论峰值 FLOPs。
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            # 计算 MFU (Model FLOPs Utilization) 并存入 metrics 字典。
            # ppo_epochs 是 PPO 算法在一个批次数据上迭代的次数。
            # world_size 是分布式训练中的总进程数 (GPU 数量)。
            metrics['mfu/actor'] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size

            # 更新学习率调度器 (Learning Rate Scheduler)
            self.actor_lr_scheduler.step()
            # 获取当前的学习率
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            # 将当前学习率存入 metrics 字典
            metrics['actor/lr'] = lr

            # 记录更新策略后的 GPU 显存使用情况。
            log_gpu_memory_usage('After update policy', logger=logger)

            # TODO: here, we should return all metrics
            # 创建一个新的 DataProto 对象，用于存储返回的 metrics。
            # 注意：此时的 output 只包含 meta_info，不包含批次数据 (batch=None 默认)。
            output = DataProto(meta_info={'metrics': metrics})

            # 对输出数据 (仅含 metrics) 进行后处理，以适应序列并行的需求。
            # 如果有序列并行，可能需要从不同设备收集或同步 metrics。
            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            # 将最终的输出数据 (metrics) 移动到 CPU。
            output = output.to('cpu')

        # 如果配置了参数卸载，则在更新后将 FSDP 包装的 Actor 模型的参数和梯度卸载回 CPU (或指定的存储)。
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        # 如果配置了优化器状态卸载，则在更新后将优化器状态卸载回 CPU。
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)

        # 清空 PyTorch 的 CUDA 缓存，尝试释放未被引用的 GPU 显存。
        torch.cuda.empty_cache()
        # 返回包含训练指标 (metrics) 的 DataProto 对象。
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        prompts = prompts.to('cuda')
        # set to False if it is validation
        recompute_log_prob = prompts.meta_info.get('recompute_log_prob', True)

        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        prompts.batch = prompts.batch.cuda()
        meta_info = {'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id}
        prompts.meta_info.update(meta_info)
        with self.rollout_sharding_manager:
            log_gpu_memory_usage('After entering rollout sharding manager', logger=logger)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage('After rollout generation', logger=logger)

            output = self.rollout_sharding_manager.postprocess_data(output)

        if self._is_actor and recompute_log_prob:
            # we should always recompute old_log_probs when it is HybridEngine
            output.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size
            output.meta_info['max_token_len'] = self.config.rollout.log_prob_max_token_len_per_gpu
            output.meta_info['use_dynamic_bsz'] = self.config.rollout.log_prob_use_dynamic_bsz
            output.meta_info['temperature'] = self.config.rollout.temperature
            # perform recompute log_prob
            with self.ulysses_sharding_manager:
                output = self.ulysses_sharding_manager.preprocess_data(output)
                old_log_probs = self.actor.compute_log_prob(data=output)
                output.batch['old_log_probs'] = old_log_probs
                output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        if self._is_offload_param:
            # NOTE(sgm): the grad is already in CPU, only offload param here
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)
        # clear kv cache
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After recompute log prob', logger=logger)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref

        data = data.to('cuda')

        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.ref_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        micro_batch_size = self.config.ref.log_prob_micro_batch_size
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.temperature
        data.meta_info['max_token_len'] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.ref.log_prob_use_dynamic_bsz
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={'ref_log_prob': output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.ref_module_fsdp, offload_grad=self._is_offload_grad)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        assert self._is_actor
        import torch
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.actor_module_fsdp,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        # TODO: support DCP and save sharded checkpoints
        import torch.distributed
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.actor.actor_module, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.actor.actor_module.state_dict()
        if self.rank == 0:
            print(f'Saving actor checkpoint to {local_path}')
            os.makedirs(local_path, exist_ok=True)
            self.actor_module.save_pretrained(local_path, state_dict=state_dict)
            self.tokenizer.save_pretrained(local_path)
            if hdfs_path is not None:
                print(f'Uploading actor checkpoint to {hdfs_path}')
                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                hdfs_io.copy(src=local_path, dst=hdfs_path)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.actor_module_fsdp, offload_grad=self._is_offload_grad)


class CriticWorker(Worker):

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_grad = self.config.model.fsdp_config.grad_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size //= (torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size)
        self.config.ppo_micro_batch_size //= (torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size)
        self.config.forward_micro_batch_size //= (torch.distributed.get_world_size() //
                                                  self.ulysses_sequence_parallel_size)

    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from verl.utils.model import LambdaLayer, print_model_size, squeeze
        from verl.utils.torch_dtypes import PrecisionType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
        from torch import optim

        local_path = copy_local_path_from_hdfs(config.model.path)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        tokenizer_path = copy_local_path_from_hdfs(config.model.tokenizer_path)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get('trust_remote_code', False))

        from omegaconf import OmegaConf
        override_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))
        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f'Critic overriding config {override_config_kwargs}')

        torch_dtype = self.config.model.fsdp_config.get('model_dtype', 'fp32')
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from transformers import AutoConfig, AutoModelForTokenClassification
        from torch import nn

        trust_remote_code = False
        critic_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        critic_model_config.num_labels = 1

        use_remove_padding = config.model.get('use_remove_padding', False)
        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(critic_model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(critic_model_config, verbose=True)

        init_context = get_init_weight_context_manager()
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setattr(critic_model_config, 'classifier_dropout', 0.)
            setattr(critic_model_config, 'hidden_dropout', '0')
            critic_module = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                            torch_dtype=torch_dtype,
                                                                            config=critic_model_config,
                                                                            attn_implementation='flash_attention_2',
                                                                            trust_remote_code=trust_remote_code)

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get('enable_gradient_checkpointing', False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=critic_module, config=self.config.model.fsdp_config.wrap_policy)

        log_gpu_memory_usage('Before critic FSDP', logger=None)

        critic_module = FSDP(critic_module,
                             param_init_fn=init_fn,
                             use_orig_params=False,
                             auto_wrap_policy=auto_wrap_policy,
                             device_id=torch.cuda.current_device(),
                             sharding_strategy=ShardingStrategy.FULL_SHARD,
                             mixed_precision=mixed_precision,
                             sync_module_states=True,
                             forward_prefetch=False)

        log_gpu_memory_usage('After critic FSDP', logger=None)

        critic_optimizer = optim.AdamW(critic_module.parameters(),
                                       lr=config.optim.lr,
                                       betas=config.optim.get('betas', (0.9, 0.999)),
                                       weight_decay=config.optim.get('weight_decay', 1e-2))

        total_steps = config.optim.get('total_training_steps', 0)
        num_warmup_steps_ratio = config.optim.get('lr_warmup_steps_ratio', 0.)
        num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

        from verl.utils.torch_functional import get_constant_schedule_with_warmup
        critic_lr_scheduler = get_constant_schedule_with_warmup(optimizer=critic_optimizer,
                                                                num_warmup_steps=num_warmup_steps)

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from verl.workers.critic import DataParallelPPOCritic
        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config)

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        self.critic = DataParallelPPOCritic(config=self.config,
                                            critic_module=self.critic_module,
                                            critic_optimizer=self.critic_optimizer)

        self.flops_counter = FlopsCounter(self.critic_model_config)

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        data = data.to('cuda')

        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        micro_batch_size = self.config.forward_micro_batch_size
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['max_token_len'] = self.config.forward_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.use_dynamic_bsz
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={'values': values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to('cpu')
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        data = data.to('cuda')
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=torch.cuda.current_device())

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            with Timer(name='update_critic', logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info['global_token_num']
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics['mfu/critic'] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            self.critic_lr_scheduler.step()
            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics['critic/lr'] = lr

            output = DataProto(batch=None, meta_info={'metrics': metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
        torch.cuda.empty_cache()
        output = output.to('cpu')
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None):
        import torch
        if self._is_offload_param:
            load_fsdp_param_and_grad(module=self.critic_module,
                                     device_id=torch.cuda.current_device(),
                                     load_grad=self._is_offload_grad)

        # TODO: support DCP and save sharded checkpoints
        import torch.distributed
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.critic_module, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.critic_module.state_dict()
        if self.rank == 0:
            print(f'Saving critic checkpoint to {local_path}')
            os.makedirs(local_path, exist_ok=True)
            self.critic_module._fsdp_wrapped_module.save_pretrained(local_path, state_dict=state_dict)
            self.tokenizer.save_pretrained(local_path)
            if hdfs_path is not None:
                print(f'Uploading critic checkpoint to {hdfs_path}')
                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                hdfs_io.copy(src=local_path, dst=hdfs_path)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_param_and_grad(module=self.critic_module, offload_grad=self._is_offload_grad)


# TODO(sgm): we may need to extract it to dp_reward_model.py
class RewardModelWorker(Worker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.use_remove_padding = self.config.model.get('use_remove_padding', False)
        self.config.micro_batch_size //= torch.distributed.get_world_size()

    def _build_model(self, config):
        # the following line is necessary
        from transformers import AutoModelForTokenClassification, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, CPUOffload

        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(config.model.path)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_local_path_from_hdfs(config.model.input_tokenizer)
            self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path,
                                                trust_remote_code=config.model.get('trust_remote_code', False))
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get('trust_remote_code', False))

        trust_remote_code = config.model.get('trust_remote_code', False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        model_config.num_labels = 1

        use_remove_padding = config.model.get('use_remove_padding', False)
        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(model_config, verbose=True)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setattr(model_config, 'classifier_dropout', 0.)
            reward_module = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                            config=model_config,
                                                                            torch_dtype=torch.bfloat16,
                                                                            attn_implementation='flash_attention_2',
                                                                            trust_remote_code=trust_remote_code)
            reward_module.to(torch.bfloat16)
        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        reward_module = FSDP(
            reward_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # zero3
            sync_module_states=True,
            cpu_offload=CPUOffload(offload_params=self.config.model.fsdp_config.param_offload),
            forward_prefetch=False)

        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))
        self.reward_module = self._build_model(config=self.config)
        torch.cuda.empty_cache()

    def _forward_micro_batch(self, micro_batch):
        from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis, rearrange
        from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.reward_module(input_ids=input_ids_rmpad,
                                            attention_mask=None,
                                            position_ids=position_ids_rmpad,
                                            use_cache=False)  # prevent model thinks we are generating
                reward_rmpad = output.logits
                reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    reward_rmpad = gather_outpus_and_unpad(reward_rmpad,
                                                           gather_dim=0,
                                                           unpad_dim=0,
                                                           padding_size=pad_size)

                # pad it back
                rm_score = pad_input(reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            else:
                output = self.reward_module(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids)
                rm_score = output.logits  # (batch_size, seq_len, 1)
                rm_score = rm_score.squeeze(-1)

            # extract the result of the last valid token
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
            rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch['attention_mask']
        position_ids = data.batch['position_ids']
        response_length = data.batch['responses'].shape[-1]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch['attention_mask'].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            chat: list = data.non_tensor_batch['raw_prompt'][i].tolist()

            # extract response
            response_ids = data.batch['responses'][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch['attention_mask'][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, '')

            chat.append({'role': 'assistant', 'content': response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(chat,
                                                                             add_generation_prompt=False,
                                                                             tokenize=False)
            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f'Switch template. chat: {prompt_with_chat_template}')

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get('max_length', src_max_length)
            if max_length is None:
                max_length = src_max_length
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=target_tokenizer,
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get('truncation', 'right'))  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {'input_ids': rm_input_ids, 'attention_mask': rm_attention_mask, 'position_ids': rm_position_ids}

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        """
        计算给定数据批次的奖励模型 (Reward Model, RM) 得分。

        Args:
            data (DataProto): 包含输入数据的数据对象。
                              期望包含 'input_ids', 'attention_mask', 'position_ids', 'responses'。
                              如果配置了 _do_switch_chat_template，还需要 'raw_prompt'。

        Returns:
            DataProto: 包含计算得到的 token 级别奖励模型得分 ('rm_scores') 的数据对象。
        """
        import itertools  # 用于处理可迭代对象，如此处的 indices
        from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx # 用于动态批次大小处理

        # 将输入数据移动到 CUDA 设备
        data = data.to('cuda')

        # 如果配置了需要切换聊天模板 (例如，RM 使用的 tokenizer 或模板与 Actor/Rollout 不同)
        if self._do_switch_chat_template:
            # 调用 _switch_chat_template 方法对输入数据进行预处理，
            # 将原始的 prompt 和 response 转换为 RM 期望的格式和 tokenizer。
            rm_data = self._switch_chat_template(data)
        else:
            # 如果不需要切换模板，直接使用原始数据作为 RM 的输入。
            # 注意：这里应该确保 rm_data 被正确赋值，即使不切换模板。
            # 通常情况下，如果 _do_switch_chat_template 为 False，rm_data 应该就是 data。
            # 为了代码的健壮性，显式赋值。
            rm_data = data
        # breakpoint() # 调试断点，通常在开发和调试时使用。

        # 将（可能经过模板切换的）RM 输入数据中的批次数据移动到 CUDA 设备。
        # 确保 rm_data.batch 存在，如果 _switch_chat_template 可能不返回 batch，需要处理。
        # 假设 _switch_chat_template 返回的 DataProto 对象总是包含 batch 属性。
        rm_data.batch = rm_data.batch.cuda()

        # 执行前向计算，在 Ulysses Sharding Manager 的上下文中进行，
        # 这会处理数据在序列并行维度上的分发和收集。
        with self.ulysses_sharding_manager:
            # 对 RM 输入数据进行预处理（例如，根据序列并行策略进行切分）
            rm_data = self.ulysses_sharding_manager.preprocess_data(data=rm_data)
            # 对原始输入数据也进行预处理，因为后续 _expand_to_token_level 需要原始数据的 attention_mask 等信息。
            # 这一步确保了原始 data 和 rm_data 都经过了与 sharding manager 一致的处理。
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            # 获取是否使用动态批次大小的配置
            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                # 如果使用动态批次大小，根据每个 GPU 的最大 token 长度和序列并行大小计算总的最大 token 长度。
                # forward_max_token_len_per_gpu 应该是 RM 的配置项。
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                # 使用 rearrange_micro_batches 将 rm_data.batch 动态地重新排列成微批次，
                # 以便每个微批次的总 token 数大致均衡，并记录原始样本的索引。
                micro_batches, indices = rearrange_micro_batches(batch=rm_data.batch, max_token_len=max_token_len)
            else:
                # 如果不使用动态批次大小，则按固定的 micro_batch_size 将 rm_data.batch 切分成微批次。
                micro_batches = rm_data.batch.split(self.config.micro_batch_size)

            output_scores = [] # 初始化一个列表来存储每个微批次计算得到的 RM 得分
            # 遍历每个微批次
            for micro_batch in micro_batches:
                # 调用 _forward_micro_batch 方法计算当前微批次的 RM 得分。
                # _forward_micro_batch 内部会处理模型的前向传播，并提取每个序列的单个标量得分。
                rm_score_micro_batch = self._forward_micro_batch(micro_batch)
                output_scores.append(rm_score_micro_batch)
            # 将所有微批次的得分在批次维度上拼接起来，得到整个批次的 RM 得分。
            scores = torch.cat(output_scores, dim=0)  # 形状为 (batch_size)

            # 如果使用了动态批次大小，需要将打乱顺序的得分恢复到原始顺序。
            if use_dynamic_bsz:
                # 将 rearrange_micro_batches 返回的嵌套索引列表展平。
                indices = list(itertools.chain.from_iterable(indices))
                # 断言检查，确保展平后的索引数量与计算得到的得分数量一致。
                assert len(indices) == scores.size(0), f"{len(indices)} vs. {scores.size()}"
                # 获取反向索引，用于将得分恢复到原始顺序。
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                # 根据反向索引对得分进行重新排序。
                scores = scores[revert_indices]

            # 调用 _expand_to_token_level 方法将每个序列的标量 RM 得分扩展为 token 级别的得分。
            # 通常是将标量得分赋给响应序列的最后一个有效 token (或 EOS token) 的位置。
            # 使用原始的 data 对象，因为它包含了原始的 attention_mask 和 responses 信息。
            token_level_scores = self._expand_to_token_level(data, scores)

            # 注意：这里的 scores 只是 RM 模型直接输出的得分，可能不是最终用于 RL 训练的奖励。
            # 例如，可能还需要进行归一化、与 KL 惩罚结合等后处理。
            # 创建一个新的 DataProto 对象来存储计算得到的 token 级别 RM 得分。
            output = DataProto.from_dict(tensors={'rm_scores': token_level_scores})
            # 对输出数据进行后处理（例如，从序列并行设备收集数据）
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        # 将最终的输出数据移动到 CPU
        output = output.to('cpu')
        # 清空 CUDA 缓存以释放未使用的 GPU 内存
        torch.cuda.empty_cache()
        return output