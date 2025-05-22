

### 整体task框架
比较抽象的是，不管啥算法在Verl中全部都用这个函数，然后这个文件还叫`main_ppo.py`，不清楚为啥

**main_task 函数完成的事情**：
这个 `main_task` 函数可以大致分为以下几个部分：

1.  **环境初始化与配置加载**:
    *   导入必要的库和模块。
    *   打印并解析传入的 `config` 配置对象。
    *   设置断点（`breakpoint()`）用于调试。

2.  **模型与分词器准备**:
    *   从 HDFS 下载预训练模型的检查点到本地。
    *   根据下载的模型路径实例化分词器 (`tokenizer`)。

3.  **Worker 类定义与选择**:
    *   根据配置 (`config.actor_rollout_ref.actor.strategy`) 判断使用 `fsdp` 还是 `megatron` 策略。
    *   根据策略导入相应的 `ActorRolloutRefWorker`、`CriticWorker` 以及 `RayWorkerGroup` (或其 Megatron 版本)。

4.  **角色与资源池配置**:
    *   定义 `Role` (如 `ActorRollout`, `Critic`, `RefPolicy`)。
    *   创建 `role_worker_mapping`，将角色映射到具体的 Ray远程 actor 类。
    *   定义 `resource_pool_spec`，指定全局资源池的 GPU 配置。
    *   创建 `mapping`，将角色映射到资源池 ID。

5.  **奖励模型 (Reward Model) 配置 (条件性)**:
    *   检查 `config.reward_model.enable` 是否启用奖励模型。
    *   如果启用，根据 `config.reward_model.strategy` 选择 `fsdp` 或 `megatron` 版本的 `RewardModelWorker`。
    *   将 `RewardModel` 角色及其 worker 添加到 `role_worker_mapping` 和 `mapping` 中。

6.  **奖励函数 (Reward Function) 实例化**:
    *   实例化 `RewardManager` 作为训练用的 `reward_fn`。
    *   实例化 `RewardManager` 作为验证用的 `val_reward_fn` (通常会打印更多信息)。

7.  **资源池管理器实例化**:
    *   使用之前定义的 `resource_pool_spec` 和 `mapping` 实例化 `ResourcePoolManager`。

8.  **PPO 训练器 (Trainer) 实例化与执行**:
    *   实例化 `RayPPOTrainer`，传入所有必要的配置、对象（如分词器、worker 映射、资源管理器、奖励函数等）。
    *   设置断点。
    *   调用 `trainer.init_workers()` 初始化所有 worker。
    *   调用 `trainer.fit()` 开始 PPO 训练流程。


```python
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
```

其中**init_workers**会设置好每个worker接下来所需的一些设置
它会：

1. 创建资源池
    * 这个资源池我也不太熟悉，主要似乎是一些关于计算资源（CPU，GPU）的规格和调用方式
    * 通过RayResourcePool对象管理
    ```
    class RayResourcePool(ResourcePool):
        def __init__(self,
                    process_on_nodes: List[int] = None,
                    use_gpu: bool = True,
                    name_prefix: str = "",
                    max_colocate_count: int = 5,
                    detached=False) -> None:
            super().__init__(process_on_nodes, max_colocate_count)
            self.use_gpu = use_gpu
            # print(f"in RayProcessDispatchConfiguration: name_prefix = {name_prefix}")
            self.name_prefix = name_prefix
            self.pgs = None
            self.detached = detached
    ```

2. 具体配置每个角色
    1. 构建角色到资源池的映射（如用Actor对象索引到一个资源池对象）
    2. 根据配置设定一些属性
        * 给Actor分资源池（如果使用引擎混合模式，Actor和Rollout就合并在一个Worker内）
        * 如果 `self.config.algorithm.adv_estimator` 使用GAE，就配一个Critic对象
        * 如果 `self.config.algorithm.adv_estimator` 使用GRPO，就不使用Critic对象（`self.use_critic = False`）
        * 给Ref分资源池

3. 初始化WorkGroup
4. 将WorkGroup分配给各个角色模型，初始化模型


主要训练逻辑在末尾的fit()中

**fit 函数详细介绍**

fit函数的主要流程如下：
1. 验证一遍测试集，获取测试集分数
    * 这里会对整个训练集跑分，获取模型最初在训练集上的评分

2. 开始正式训练循环
    0. 预设数据：
        ```python
        actor_rollout_ref.rollout.n=5 
        total_epochs = 15
        train_batch_size = 32
        ppo_mini_batch_size=16
        ppo_micro_batch_size=8
        data.max_response_length=1024
        ```
    1. 循环条件：
        * 循环epochs，每个epochs循环使用`train_dataloader`
        * `train_dataloader`设置好了size是`train_batch_size`，因此每个batch_dict里包含`train_batch_size`个对象。

        ```python
        # 外层循环：遍历总的训练轮数 (epochs)
        for epoch in range(self.config.trainer.total_epochs):
            # 内层循环：遍历训练数据加载器中的每个批次 (batch)
            for batch_dict in self.train_dataloader:

        ```

    2. 循环内容：
        1. 调用`generate_sequences`函数，传入一个变量，包含
        `train_batch_size`个prompt，用于当前循环。
            * `generate_sequences`: 函数init的时候，会查找config里的变量`n`
            ```
            kwargs = dict(
                n=1,
                logprobs=1,  # can be set to 0 and let actor to recompute
                max_tokens=config.response_length,
            )
            ```

            于是对于输入的每个prompt，会生成n个response，即总共5 * 32个1024长度的response。
            
            ```
            -> len(prompts)
            32
            -> response.shape
            torch.Size([160, 1024])
            ```
            
            最后计算old_log_probs，加入output中一并返回。这里的old_log_probs指的是当前response生成时，具体每个token的对数概率；它是通过`forward_micro_batch`函数得到的。
        
        2. 检查是否使用Ref策略
            * GRPO需要使用，于是forward获得`ref_log_prob`，用于后续计算KL散度

        3. 检查是否使用Critic网络
            * 不使用，跳过

        4. 计算优势 (Advantage) 和奖励 (Reward)
            1. 如果使用奖励模型：
                * 使用compute_rm_score计算分数，添加到reward tensor
                * 合并reward_tensor至batch中
            2. 使用奖励规则算分，获取reward_tensor，合并至batch中。
            ```py
            batch.batch['token_level_scores'] = reward_tensor
            ```
            3. 检查是否使用KL散度
                * GRPO使用

            4. 使用奖励计算优势
                ```py
                batch = compute_advantage(  
                    batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=self.config.actor_rollout_ref.rollout.n
                )
                ```

        5. 更新Critic网络
            * 我们没使用critic网络，跳过

        6. 更新Actor网络（如果Critic预热完毕）

```python
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
```

**关于计算log_prob**
这里涉及到几个概念，分别是
1. Logits：
    * 模型对每个词的原始、未归一化的预测分数。对于一个response，他的logits.shape应是[seq_len, vocab_len]。每一个logit包含了这个位置对应vocab中每个词的预测分数。
2. 熵：
    * 先将 logits 转换为概率分布（使用 Softmax），然后根据熵的定义（-sum(p * log(p))）计算得到的，它衡量了模型预测的不确定性。具体在verl中是这么计算的：
    ```python
    def entropy_from_logits(logits: torch.Tensor):
        """Calculate entropy from logits."""
        pd = torch.nn.functional.softmax(logits, dim=-1)
        entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
        return entropy
    ```
    这是一个等价的更稳定的实现。

3. 对数概率（log_prob）：
    * 对于这个位置，最后选到的token的对数概率


**关于计算奖励**

这里的reward_tensor是一个token level的张量

```
-> reward_tensor.shape
torch.Size([160, 1024])
```

代表了对Response中每一个token的reward

使用奖励模型的计算过程如下（这里GRPO没有使用，仅作学习）：
**compute_rm_score**
1. 检查是否用了动态批次，如果用了，按照序列并行大小计算最大token长途
2. 切分batch
    * 这里输入的batch是最大的train_batch_size（32）
    1. 如果使用动态批次：
        * 按照最大token数切分
    2. 如果没使用：
        * 按照micro_batch_size切分train_batch_size
    3. 对于每个microbatch：
        * forward计算得分
    
3. 如果使用动态批次大小，将打乱的得分恢复到原始顺序

具体注释版本如下
```py
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
```

如果不使用奖励模型，则使用对应的**reward_fn**进行打分，**二者互斥**；
这里**reward_fn**是一个函数指针，根据具体的数据集获取对应的奖励函数，比如我这里使用的是CountDown任务，给出一组数（通常3-4个），给出一个target值，要求模型通过加减乘除这一组数来获得target值。这个任务的奖励函数就是答对了给1分，答错了0分。

**reward_fn**：
这里给出伪代码  
```py
初始化一个reward_tensor，shape是 [rollout.n * train_batch_size, max_response_length]，这里我的实际上是[160, 1024]。这个reward_tensor记录response的每个位置应该有的奖励
for i in range(len(data)):
    1. 通过掩码提取出有效的prompt和response的ids
    2. 将对应ids解码为文字形式，获得人类语言的prompt + response
    3. 从数据集的元数据中查看是哪个数据集，选取对应的 **compute_score_fn**
    4. 调用上面选出的函数计算分数
    # 计算出的分数只赋给当前轮的对话的最后一个有效token
    5. reward_tensor[i, valid_response_length - 1] = score
    

return reward_tensor
```
这里比较难理解的是第五步，按照我之前的直观理解，获得的分数应该是赋给整个response的每个token的，但实际上只**赋给最后一个有效token**。  

但其实仔细想想也可以明白，计算的 score 通常是对整个生成的 sequences_str (prompt + response) 的一个整体评估。例如，在 GSM8K（数学问题解答）任务中，score 可能是 1（如果答案正确）或 0（如果答案错误）。这个分数是针对整个解决方案的，而不是针对解决方案中的某一个词或数字。


update_actor