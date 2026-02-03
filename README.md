# Daily AI Infra Report

## 往期回顾
- [2026-02-01](reports/2026-02-01.md)
- [2026-01-31](reports/2026-01-31.md)

---

## 最新解读 (2026-02-02)
本周AI基础设施领域的研究和开发主要集中在优化大语言模型训练与推理效率、改进网络表示方法、以及提升硬件加速性能。重要进展包括：基于用户重试行为的LLM路由调度算法、4位训练中的异常值分析、多LLM协作的编译优化框架，以及针对RISC-V架构的DNN优化。同时，多个主流项目如FlashAttention、SGLang和vLLM发布了性能优化和兼容性改进。

## 具体内容分析

### 研究论文

1. **Learning to Route and Schedule LLMs from User Retrials via Contextual Queueing Bandits**
   - 通过利用用户重试行为作为隐式反馈，解决了LLM服务中查询路由和调度的关键挑战
   - 链接: https://arxiv.org/abs/2602.02061v1

2. **Dissecting Outlier Dynamics in LLM NVFP4 Pretraining**
   - 分析4位训练中的异常值动态，提出CHON训练配方将损失差距从0.94%降至0.58%
   - 链接: https://arxiv.org/abs/2602.02047v1

3. **DASH: Faster Shampoo via Batched Block Preconditioning and Efficient Inverse-Root Solvers**
   - 通过批处理块预 conditioning 和高效逆根求解器，将Shampoo优化器加速高达4.83倍
   - 链接: https://arxiv.org/abs/2602.02016v1

4. **Logic-Guided Vector Fields for Constrained Generative Modeling**
   - 结合符号逻辑与神经学习，在生成模型中强制执行约束，将违规减少59-82%
   - 链接: https://arxiv.org/abs/2602.02009v1

5. **Position: The Need for Ultrafast Training**
   - 倡导将学习直接引入FPGA fabric，实现非平稳环境中的实时适应
   - 链接: https://arxiv.org/abs/2602.02005v1

6. **IntraSlice: Towards High-Performance Structural Pruning with Block-Intra PCA for LLMs**
   - 提出基于PCA的剪枝方法，完全融合变换矩阵，提高LLM压缩性能
   - 链接: https://arxiv.org/abs/2602.01975v1

### 代码仓库更新

**NVIDIA/cutlass**
- 修复了CuTeDSL示例中SM100块规模gemm重叠累加器问题
- 链接: https://github.com/NVIDIA/cutlass/commit/1cfbb53a23cf009973a6a6ea9e8275c8a691d411

**flashinfer-ai/flashinfer**
- 修复了fused_moe模块中的存根生成目录问题
- 修复了日志级别设置以启用DEBUG日志
- 修复了_cudnn_gemm_fp4_requirement中的参数类型错误
- 链接: 
  - https://github.com/flashinfer-ai/flashinfer/commit/3fa8905c79fe3b711ff62af5bab5e3cc5ec6d5ea
  - https://github.com/flashinfer-ai/flashinfer/commit/806339fbbd5c77ae01889ddef303e59bb503c7d7
  - https://github.com/flashinfer-ai/flashinfer/commit/9329fb4785fa62399ebf8a3e15beccdaa9baea83

**Dao-AILab/flash-attention**
- 修复了benchmark_mask_mod中compute_block_sparsity的使用
- 链接: https://github.com/Dao-AILab/flash-attention/commit/514e63cc26e90719f9d3332eef33146d8f69e1d2

**sgl-project/sglang**
- 添加了JIT concat MLA内核以提高性能
- 改进了diffusion管道中的日志记录
- 为HiCache添加了DeepSeek v32 CPU卸载支持
- 添加了QKNorm跨头内核
- 优化了基数缓存逐出性能
- 添加了对Step-3.5-Flash模型的支持
- 更新了NPU夜间测试
- 链接:
  - https://github.com/sgl-project/sglang/commit/9b1619c148a0a800e9ddb3d2e42ff32ee2dd2679
  - https://github.com/sgl-project/sglang/commit/62004fd2beb77daa10502e292e66a0d2f5662e3e
  - https://github.com/sgl-project/sglang/commit/180594358b990b2a5ce8140fb64aae90d73910fd
  - https://github.com/sgl-project/sglang/commit/a1bbc892af27867901f91e9a1c485824ff9337a6
  - https://github.com/sgl-project/sglang/commit/fd983b09b68c076d448a55266a29ffdfdff2d06e
  - https://github.com/sgl-project/sglang/commit/980d2936cd9a94a6346fb678a8134d5776fdf996
  - https://github.com/sgl-project/sglang/commit/c781db0f6c236e357a0652a962fdf7d3f528d752

**vllm-project/vllm**
- 添加了CPU镜像上传到Docker Hub的说明
- 修复了密集模型的DPMetadata错误
- 为Voxtral Realtime引入了全局log mel最大值
- 修复了sm103a上的cutlass_3x_gemm_fp8_blockwise
- 修复了在线fp8量化与流式权重加载的内存问题
- 减少了具有多个CUDA图的LoRA的内核开销
- 添加了DeepSeek V3.2夜间评估
- 将分析方法移至MM预算
- 为GB系列添加了Fabric检测支持
- 为Nemotron-H启用了潜在MoE的共享/路由重叠
- 链接:
  - https://github.com/vllm-project/vllm/commit/1b60b45d0d745e0c0be65994507a702a19fd4761
  - https://github.com/vllm-project/vllm/commit/4b3803d18044499962f331db0a1614b726110e2a
  - https://github.com/vllm-project/vllm/commit/5019c59dd2d34feb2bc6e52699a36059eacf64a9
  - https://github.com/vllm-project/vllm/commit/089cd4f002484599aeed366c31629dccf491ce81
  - https://github.com/vllm-project/vllm/commit/0130223bd9900710a0d93e46a4255ec5d1a077a8
  - https://github.com/vllm-project/vllm/commit/ffe1fc7a28841973135b981fb68ce515b409a236
  - https://github.com/vllm-project/vllm/commit/9f8cb81b44ce2433facc60dec709dc8bf116c315
  - https://github.com/vllm-project/vllm/commit/d7e17aaacd5ed1b4b4be6bcfef3a1b7cbc84fc9a
  - https://github.com/vllm-project/vllm/commit/528e9b14900fc8a012f2599172e2a4576caafe1a
  - https://github.com/vllm-project/vllm/commit/0aca8b8c628e9a73ab8758d78c9c721bc703ee66

### 问题与讨论

**NVIDIA/cutlass**
- 新问题：询问`examples/python/CuTeDSL/advanced_compiler_control`的用途
- 影响：需要文档澄清
- 链接: https://github.com/NVIDIA/cutlass/issues/2994

**flashinfer-ai/flashinfer**
- 新问题：
  1. `trtllm_fp8_per_tensor_scale_moe`和`trtllm_fp8_block_scale_moe`不支持DeepSeek以外的routing_logits==float32
     - 影响：与vLLM中某些模型的兼容性问题
     - 链接: https://github.com/flashinfer-ai/flashinfer/issues/2469
  2. 请求将cuteDSL fp4密集gemm集成到flashinfer中
     - 影响：fp4操作的潜在性能改进
     - 链接: https://github.com/flashinfer-ai/flashinfer/issues/2466
  3. 关于强制执行PTX代码而不是预编译SASS的问题
     - 影响：开发人员对内核实现的兴趣
     - 链接: https://github.com/flashinfer-ai/flashinfer/issues/2463

**Dao-AILab/flash-attention**
- 新问题：
  1. Blackwell架构支持
     - 影响：与较新的NVIDIA GPU的兼容性
     - 链接: https://github.com/Dao-AILab/flash-attention/issues/2220
  2. 发布whl文件的导入错误
     - 影响：某些环境的安装问题
     - 链接: https://github.com/Dao-AILab/flash-attention/issues/2219

**sgl-project/sglang**
- 新问题：
  1. GLM 4.6 FP8与flash infer后端的KV缓存卸载问题
     - 影响：特定模型配置的部署限制
     - 链接: https://github.com/sgl-project/sglang/issues/18135
  2. 请求从/v1/score API返回令牌使用情况
     - 影响：监控和分析功能的缺失
     - 链接: https://github.com/sgl-project/sglang/issues/18132
  3. 启用分段Cuda Graph默认设置的TODO列表
     - 影响：具有兼容性考虑的性能改进机会
     - 链接: https://github.com/sgl-project/sglang/issues/18130
  4. dsv32 encode_messages中的错误
     - 影响：DeepSeek v32的模型性能问题
     - 链接: https://github.com/sgl-project/sglang/issues/18125
  5. 关于Ubuntu 22.04 Docker镜像可用性的问题
     - 影响：某些环境的兼容性问题
     - 链接: https://github.com/sgl-project/sglang/issues/18115
  6. CUDA错误：设备上没有可用的内核映像
     - 影响：Qwen-Image模型部署失败
     - 链接: https://github.com/sgl-project/sglang/issues/18108
  7. Qwen3-Next-80B-A3B-Thinking与推测解码中多个工具调用损坏
     - 影响：复杂工具交互的功能限制
     - 链接: https://github.com/sgl-project/sglang/issues/18102
  8. 关于ChatMessage::Assistant.content序列化的问题
     - 影响：影响数据一致性的API设计问题
     - 链接: https://github.com/sgl-project/sglang/issues/18100

**vllm-project/vllm**
- 新问题：
  1. 将reasoning_content重命名为reasoning导致的回归
     - 影响：与期望reasoning_content的聊天模板的兼容性问题
     - 链接: https://github.com/vllm-project/vllm/issues/33616
  2. 自2026年1月28日起失去vLLM与PyTorch nightly信号
     - 影响：与PyTorch nightly构建的潜在兼容性问题
     - 链接: https://github.com/vllm-project/vllm/issues/33603
  3. 在单个实例上使用Ray启动时vLLM副本初始化失败
     - 影响：基于Ray的设置的部署限制
     - 链接: https://github.com/vllm-project/vllm/issues/33601
  4. RFC：改善vLLM与下游生态系统的依赖兼容性
     - 影响：依赖管理的长期策略
     - 链接: https://github.com/vllm-project/vllm/issues/33599
  5. CI失败：mi325_4：Qwen3-30B-A3B-FP8-block准确率(H100)
     - 影响：影响对FP8支持的信心的测试失败
     - 链接: https://github.com/vllm-project/vllm/issues/33598
  6. CI失败：mi325_1：多模态准确率评估(小型模型)
     - 影响：影响对多模态支持的信心的测试失败
     - 链接: https://github.com/vllm-project/vllm/issues/33596
  7. RFC：优化Blackwell上的DeepSeekR1吞吐量
     - 影响：性能优化机会
     - 链接: https://github.com/vllm-project/vllm/issues/33583
  8. 模型支持：GLM-4.7-Flash需要来自git的transformers
     - 影响：特定模型配置的部署限制
     - 链接: https://github.com/vllm-project/vllm/issues/33580

## 总结

本周AI基础设施领域的研究和开发活动非常活跃，涵盖了从算法优化到系统实现的多个层面。研究论文主要集中在提高LLM训练和推理效率、改进网络表示方法以及优化硬件加速性能。代码库方面，多个主流项目发布了重要更新，包括性能优化、兼容性改进和新功能支持。同时，社区也报告了一些需要关注的问题，涉及兼容性、部署限制和API设计等方面。这些进展共同推动了AI基础设施的持续发展和完善。
