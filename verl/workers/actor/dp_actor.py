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
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对单个微批次数据执行前向传播，计算熵和对数概率。
        这个方法主要用于 Actor 模型的推理（计算 log_prob）和 PPO 算法中的策略评估与更新。

        Args:
            micro_batch (TensorDict): 包含输入数据的微批次。
                                      期望包含 'input_ids', 'attention_mask', 'position_ids', 'responses'。
                                      'input_ids' 是 prompt 和 response 的拼接。
            temperature (float): 用于 logits 缩放的温度参数。较高的温度使概率分布更平滑。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                entropy (torch.Tensor): 计算得到的策略熵，形状为 (batch_size, response_length)。
                                        熵衡量了模型输出概率分布的不确定性。
                log_probs (torch.Tensor): 对应于 micro_batch['responses'] 中 token 的对数概率，
                                          形状为 (batch_size, response_length)。
        """
        # breakpoint() # 调试断点，通常在开发和调试时使用。

        # 获取响应序列的长度，用于后续的切片操作，以确保只处理响应部分的 logits 和概率。
        response_length = micro_batch['responses'].size(-1)

        # 使用 torch.autocast 进行混合精度计算（在 CUDA 设备上使用 bfloat16 类型）。
        # 这可以在保持数值稳定性的同时加速计算并减少内存使用。
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # 从微批次中提取输入数据
            input_ids = micro_batch['input_ids']  # (batch_size, sequence_length)，其中 sequence_length = prompt_length + response_length
            batch_size, seqlen = input_ids.shape  # 获取当前微批次的批次大小和完整序列长度
            attention_mask = micro_batch['attention_mask']  # (batch_size, sequence_length)，标记哪些是真实 token，哪些是 padding
            position_ids = micro_batch['position_ids']  # (batch_size, sequence_length)，为每个 token 提供位置信息

            # 如果配置了使用移除填充 (remove_padding) 的优化。
            # 这种优化通常与 FlashAttention 结合使用，通过移除序列间的 padding token 来加速注意力计算。
            if self.use_remove_padding:
                # 使用 flash_attn.bert_padding.unpad_input 移除输入序列中的填充部分。
                # input_ids.unsqueeze(-1) 增加一个维度以匹配 unpad_input 的期望输入形状 (B, S, H) -> (total_nnz, H)。
                # indices 保存了原始有效 token 在展平后的序列中的位置，用于后续恢复原始形状。
                # cu_seqlens 是累积序列长度，用于 FlashAttention 的变长接口。
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                                      attention_mask)  # input_ids_rmpad 形状为 (total_nnz, 1)，total_nnz 是所有样本中非填充 token 的总数
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # 转换为 (1, total_nnz)，以便作为模型输入 (通常模型期望批次维度在前)

                # 类似地，移除 position_ids 中的填充，以确保与 input_ids_rmpad 对齐。
                # 这对于依赖精确位置信息的机制（如旋转位置编码 Rotary Positional Embedding）非常重要。
                # rearrange 将 (b, s, ...) 转换为 (b*s, ...)，然后 index_first_axis 根据 indices 选择有效部分。
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)  # 转换为 (1, total_nnz)

                # 为了计算对数概率，我们需要目标 token。
                # 对于语言模型，在给定前文 input_ids[..., t] 的情况下，目标是预测 input_ids[..., t+1]。
                # 因此，我们将 input_ids_rmpad 向左滚动一位 (shifts=-1) 来创建目标序列。
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                # 注意：最后一个 token 的 "rolled" 对应的是序列之外的，通常在计算损失时会被 mask 掉或不使用。

                # 如果使用了 Ulysses 序列并行 (ulysses_sequence_parallel_size > 1)。
                # Ulysses 是一种将长序列在多个设备上进行拆分和并行处理的技术。
                if self.use_ulysses_sp:
                    # 对移除填充后的 input_ids 和 position_ids 进行 Ulysses 序列并行的填充和切片。
                    # pad_size 是为了确保序列长度能被序列并行大小整除而添加的填充大小。
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad,
                        position_ids_rmpad,
                        sp_size=self.ulysses_sequence_parallel_size
                    )
                    # 同样处理 rolled 的 input_ids，确保对齐。
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled, None, # position_ids_rmpad_rolled 不是必需的
                        self.ulysses_sequence_parallel_size
                    )

                # 移除 input_ids_rmpad_rolled 的第一个维度（如果存在），使其形状为 ( (total_nnz_after_sp_slice) )。
                # total_nnz_after_sp_slice = (total_nnz / sp_size) + pad_size_for_sp
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

                # 将处理后的 input_ids_rmpad 和 position_ids_rmpad 传递给 actor_module (模型)。
                # 当使用 FlashAttention 的变长接口 (flash_attn_varlen) 时，通常不需要显式的 attention_mask，
                # 因为 token 间的填充已被移除，cu_seqlens (或类似的机制) 会告知注意力模块每个序列的边界。
                # use_cache=False 防止模型认为我们正在进行自回归生成（这会启用 KV 缓存，在训练或计算完整序列 log_prob 时通常不需要）。
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None, # 假设模型内部能处理无 mask 的情况或使用 cu_seqlens
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)
                # 获取模型的输出 logits。对于移除填充的输入，输出 logits 的批次维度通常也是1或被压缩。
                logits_rmpad = output.logits.squeeze(0)  # 形状为 (total_nnz_after_sp_slice, vocab_size)

                # 将 logits 除以温度参数。温度参数可以调整输出概率分布的平滑度。
                # temperature > 1 使分布更平滑（更随机），temperature < 1 使分布更尖锐（更确定）。
                # 在计算 log_prob 时，这个温度应该与生成时使用的温度一致（如果目标是重现生成时的 log_prob）。
                logits_rmpad.div_(temperature)

                # 从处理后的 logits 计算熵。
                # self.compute_entropy_from_logits 是 torch.compile 编译过的 verl_F.entropy_from_logits。
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # 形状为 (total_nnz_after_sp_slice)

                # 从处理后的 logits 和对应的目标 token (input_ids_rmpad_rolled) 计算对数概率。
                # logprobs_from_logits 通常会执行 log_softmax 然后根据 labels 收集对应的值。
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled) # 形状为 (total_nnz_after_sp_slice)

                # 如果使用了 Ulysses 序列并行
                if self.use_ulysses_sp:
                    # 从不同设备收集输出（log_probs 和 entropy_rmpad），并移除 Ulysses 引入的填充。
                    # gather_dim 和 unpad_dim 指定了在哪个维度上进行收集和移除填充。
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                    # 此时 log_probs 和 entropy_rmpad 的形状恢复到 (total_nnz)

                # 将移除填充并可能经过序列并行处理的熵和对数概率，使用原始的 indices 填充回原始的批次和序列长度形状。
                # unsqueeze(-1) 是为了匹配 pad_input 的期望输入形状 (total_nnz, H)。
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size, # 原始微批次的 batch_size
                                         seqlen=seqlen) # 原始微批次的 seqlen
                                         # full_entropy 形状为 (batch_size, seqlen, 1)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)
                                           # full_log_probs 形状为 (batch_size, seqlen, 1)

                # 只返回响应部分的熵和对数概率。
                # `[:, -response_length - 1:-1]` 这个切片逻辑是为了提取对应于 `responses` 部分的 token 的预测结果。
                # 假设 `input_ids` 是 `prompt + response`。
                # `log_probs` 的第 `t` 个位置的值是 `log P(input_ids[t+1] | input_ids[0...t])`。
                # 我们需要的是对应于 `responses` 部分的 `log_probs`。
                # `responses` 对应于 `input_ids` 的最后 `response_length` 个 token。
                # 因此，我们需要 `log_probs` 中对应于 `input_ids` 中 `prompt_len-1` 到 `seq_len-2` 这些位置的预测。
                # `seq_len - 1 - response_length` 是 prompt 的最后一个 token 的索引。
                # `seq_len - 2` 是 response 的倒数第二个 token 的索引。
                # 切片 `[:, -response_length - 1:-1]` 提取的是从倒数第 `response_length + 1` 个 token 的预测
                # (即预测 response 的第一个 token) 到倒数第 2 个 token 的预测 (即预测 response 的最后一个 token)。
                # 结果的长度是 `(-1) - (-response_length - 1) = response_length`。
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (batch_size, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (batch_size, response_length)

            else:  # 如果不使用移除填充 (use_remove_padding is False)，并且没有 Ulysses 序列并行
                # 直接将包含填充的原始 input_ids, attention_mask, position_ids 传递给模型。
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False) # 同样，禁用 KV 缓存以获取整个序列的 logits
                logits = output.logits # 获取原始 logits，形状为 (batch_size, seqlen, vocab_size)

                # 对 logits 进行温度缩放
                logits.div_(temperature)

                # 从完整序列的 logits 中提取对应于响应部分前一个 token 的 logits。
                # `logits[:, -response_length - 1:-1]` 这个切片提取了从序列倒数 `response_length + 1` 个位置开始的 logits
                # (这些 logits 用于预测 response 的第一个 token)，到倒数第 2 个位置的 logits (用于预测 response 的最后一个 token)。
                # 提取出的 logits 的序列长度维度是 `response_length`。
                # 形状变为 (batch_size, response_length, vocab_size)
                logits_for_responses = logits[:, -response_length - 1:-1]

                # 根据提取出的 logits 和实际的响应 token (micro_batch['responses']) 计算对数概率。
                # micro_batch['responses'] 的形状是 (batch_size, response_length)
                log_probs = logprobs_from_logits(logits_for_responses, micro_batch['responses']) # (batch_size, response_length)

                # 根据提取出的 logits (对应响应部分的) 计算熵。
                entropy = verl_F.entropy_from_logits(logits_for_responses)  # (batch_size, response_length)

            # 返回计算得到的熵和对数概率 (只包含响应部分)
            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        # breakpoint()
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for batch_idx, data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

            self.actor_optimizer.zero_grad()

            for data in micro_batches:
                data = data.cuda()  # actor device is cpu when using offload
                responses = data['responses']
                response_length = responses.size(1)
                attention_mask = data['attention_mask']
                response_mask = attention_mask[:, -response_length:]
                old_log_prob = data['old_log_probs']
                advantages = data['advantages']

                clip_ratio = self.config.clip_ratio
                entropy_coeff = self.config.entropy_coeff

                # all return: (bsz, response_length)
                entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                              log_prob=log_prob,
                                                                              advantages=advantages,
                                                                              eos_mask=response_mask,
                                                                              cliprange=clip_ratio)
                # compute entropy loss from entropy
                entropy_loss = verl_F.masked_mean(entropy, response_mask)

                # compute policy loss
                policy_loss = pg_loss - entropy_loss * entropy_coeff

                if self.config.use_kl_loss:
                    ref_log_prob = data['ref_log_prob']
                    # compute kl loss
                    kld = core_algos.kl_penalty(logprob=log_prob,
                                                ref_logprob=ref_log_prob,
                                                kl_penalty=self.config.kl_loss_type)
                    kl_loss = masked_mean(kld, response_mask)

                    policy_loss = policy_loss - kl_loss * self.config.kl_loss_coef
                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef

                loss = policy_loss / self.gradient_accumulation
                loss.backward()

                data = {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                }
                append_to_dict(metrics, data)

            grad_norm = self._optimizer_step()
            data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
