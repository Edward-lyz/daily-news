# Daily AI Infra Report

## 往期回顾
- [2026-02-03](reports/2026-02-03.md)
- [2026-02-02](reports/2026-02-02.md)
- [2026-02-01](reports/2026-02-01.md)
- [2026-01-31](reports/2026-01-31.md)

---

## 最新解读 (2026-02-04)
今日的 AI Infra 的新闻如下。

## 摘要
```json
{
  "summary": "NVIDIA CUTLASS发布4.4版本，增强CuTe DSL功能并支持CUDA 13.1与GB300架构。FlashInfer修复了权限错误并添加了新的量化方法支持，同时报告了性能问题。Flash-Attention发布3.0稳定版并简化了Flex Flash文档。SGLang实现了多项性能优化，包括KV偏移计算和线性操作融合。vLLM添加了ColBERT模型支持并改进了聊天完成流式传输性能，同时解决了多个兼容性问题。",
  "conclusion": "需关注CUTLASS中的Blackwell GEMM性能问题和FP8共享内存打印错误。FlashInfer的mxfp4量化内核性能显著低于nvfp4，且存在NaN值生成问题。vLLM的Docker镜像兼容性问题、DeepSeek V3.2基准测试失败以及FA3优化回退可能影响生产环境稳定性。各项目对新型硬件架构的支持仍在发展中，建议在生产环境全面采用前进行充分测试。"
}
```

## 具体内容分析
### deepseek-ai/DeepGEMM
昨日无更新。
### deepseek-ai/FlashMLA
昨日无更新。
### NVIDIA/cutlass
# NVIDIA CUTLASS 更新与问题 (2026-02-04)

## 提交更新

### 版本 4.4 发布更新 (#2999)
- **提交**: [6b3e607](https://github.com/NVIDIA/cutlass/commit/6b3e607b852f1543dc21323155a2ad71473c8642)
- **作者**: Junkai-Wu

### CuTe DSL 新功能
- 支持 CUDA toolkit 13.1
  - 设置命令: `cutlass/python/CuTeDSL/setup.sh --cu13`
  - 参考文档: [Python DSL 快速开始指南](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)
- 支持 GB300 架构与 CTK 13.1
  - 示例内核: [SM103 batched 3xFP4 blockscaled GEMM kernel](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/sm103_dense_blockscaled_gemm_persistent.py)
- 新增 cute.experimental 高级组合层
  - 无片段编程模型: copy/dot API 直接接受 memrefs
  - 自动 TMA 描述符生成和更新插入
  - SIMT 复制的自动向量和谓词
  - 新管道抽象与便捷包装器
  - 新分区操作简化分区逻辑
  - 设备端 TMA 描述符分配、初始化和管理
- 提前编译 (AoT) 现已可用

### 新增示例文件
- `examples/python/CuTeDSL/blackwell/sm103_dense_blockscaled_gemm_persistent.py` (2977 行)
- `examples/python/CuTeDSL/experimental/ampere/memcpy_simt_universal_copy.py` (155 行)
- `examples/python/CuTeDSL/experimental/blackwell/dense_block_scaled_gemm.py` (1027 行)
- `examples/python/CuTeDSL/experimental/blackwell/dense_gemm.py` (1410 行)
- `examples/python/CuTeDSL/experimental/blackwell/dense_gemm_2sm.py` (522 行)
- `examples/python/CuTeDSL/experimental/blackwell/dense_gemm_cute_pipeline.py` (1742 行)
- `examples/python/CuTeDSL/experimental/blackwell/dense_gemm_ptr_array.py` (825 行)

### 内核改进
- 修复网格依赖问题，在调度器初始化前添加 `cutlass::arch::wait_on_dependent_grids()`
- 更新多个内核文件以正确处理网格依赖关系
  - `include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized.hpp`
  - `include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized_input_transform.hpp`
  - `include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized_mma_transform.hpp`
  - `include/cutlass/gemm/kernel/sm90_gemm_array_tma_warpspecialized_cooperative.hpp`

## 问题报告

### 性能开销问题 (#3003)
- **标题**: Blackwell GEMM 循环中频繁的 cuTensorMapEncodeTiled 和 cudaGetDriverEntryPoint 调用导致的性能开销
- **作者**: huye566
- **状态**: 新报告 (0 条评论)
- **问题描述**: 
  - 在 Blackwell 架构上使用 CUTLASS 4.x API 进行 GEMM 内核基准测试时，观察到内核启动之间存在显著开销
  - 分析显示 cuTensorMapEncodeTiled 和 cudaGetDriverEntryPoint 调用频率高
  - 内核执行时间: 91us，循环1: 625us
  - 提供了部分 KernelConfigM128 模板代码

### FP8 共享内存打印问题 (#3002)
- **标题**: 在 fp8 共享内存张量上调用 `cute.print_tensor` 时出现段错误
- **作者**: 16bit-ykiko
- **状态**: 已报告 (1 条评论)
- **问题描述**: 
  - 在调用 `cute.print_tensor` 处理 swizzled 共享内存张量时出现意外段错误
  - 提供了部分 HopperFusedMultiHeadAttentionForward 类代码片段

### 导入错误问题 (#3001)
- **标题**: nvidia-cutlass-dsl v4.4.0.dev1 导入错误
- **作者**: DefTruth
- **状态**: 已报告 (2 条评论)
- **问题描述**: 
  - 尝试导入 nvidia-cutlass-dsl v4.4.0.dev1 时出现错误
  - 提供了 hello_world 示例代码，但错误信息被截断

### V100 支持询问 (#3000)
- **标题**: CUTLASS 是否支持 NVIDIA GPU Tesla V100？
- **作者**: xxxxdddds
- **状态**: 新报告 (0 条评论)
- **问题描述**: 直接询问 CUTLASS 是否支持 Tesla V100 GPU
### flashinfer-ai/flashinfer
# FlashInfer Newsletter - 2026-02-04

## Notable Commits

### [cdbb2c3] Fix permission errors in release workflow (#2488)
- Removed `chown` commands and `--user` flag from docker run in nightly-release.yml and release.yml
- [View commit](https://github.com/flashinfer-ai/flashinfer/commit/cdbb2c3c73598bbb04a0e2dc2d7cf6ee9e5aeda5)

### [e284274] Support Fused MoE non gated Relu2 NVFP4 & FP8 and support Nemotron (#2462)
- Added support for new activation types and quantization methods
- Modified benchmarks/bench_trtllm_gen_fused_moe_autotuner.py to use ActivationType instead of GatedActType
- Updated CUDA source files in csrc/ for MoE kernels
- [View commit](https://github.com/flashinfer-ai/flashinfer/commit/e284274e2eb67538bb9c884f2ed9c0143772d3ac)

### [9bf007d] Add/update multi node/multi GPU test scripts (#2410)
- Added scripts/task_run_unit_tests.sh and scripts/task_test_multi_gpu_comm_kernels.sh
- Removed scripts/task_test_blackwell_kernels.sh
- Added test_utils.sh for common testing utilities
- [View commit](https://github.com/flashinfer-ai/flashinfer/commit/9bf007d7403f0a394fda44abd4170b354cac3f05)

### [567ded1] Rename tests/mamba/test_utils.py to tests/mamba/utils.py (#2481)
- Fixed CI test discovery by renaming utility file
- Updated imports in test_selective_state_update_mtp.py and test_selective_state_update_stp.py
- [View commit](https://github.com/flashinfer-ai/flashinfer/commit/567ded1f612c38bd6a921a4a7ce4ecc6db2b9a9e)

### [f84ac1c] Fix memory bandwidth calculation in MLA benchmarks (#2479)
- Corrected memory bandwidth calculations in bench_trtllm_gen_mla.py
- Updated attention.py benchmark routine
- [View commit](https://github.com/flashinfer-ai/flashinfer/commit/f84ac1c97e2e1a4391150ca73971b973ee569e5a)

### [6ae5bfe] Refactor: reduce hopper's gdn prefill compilation time (#2422)
- Added flat_prefill_kernel_delta_rule_sm90_extern.inc and gdn_prefill_sm90_kernel_inst.jinja
- Renamed prefill_kernel_delta_rule_sm90.cu to flat_prefill_kernel_delta_rule_sm90.cu
- Reduced compilation time for Hopper's GDN prefill kernel
- [View commit](https://github.com/flashinfer-ai/flashinfer/commit/6ae5bfe6a48c591a171f32888e5de28b9ca0207b)

## Open Issues

### Performance Issues
- **#2496**: mxfp4 quantize kernel is 10x slower than nvfp4 quantize kernel (5.597 TFLOPs vs 0.442 TFLOPs)
- **#2493**: Request for performance improvement in CuTe DSL GDN Decode Kernel
- **#2491**: Request to add CuTe DSL backends for RoPE APIs

### Bug Reports
- **#2490**: GDN prefill kernel produces NaN values (body truncated)
- **#2494**: XQA always sets dynamic smem size on rank 0, causing issues in multi-GPU environments
- **#2486**: Illegal memory access in RadixTopKKernel_Unified
- **#2485**: Accuracy issue with GLM-4.7 ModelOpt NVFP4 producing garbage output (body truncated)
### Dao-AILab/flash-attention
## Flash-Attention 更新

### 版本发布
- **v3.0.0 稳定版发布** [PR #2223](https://github.com/Dao-AILab/flash-attention/pull/2223)
  - Luca Wehrstedt 将主版本标记为 v3.0.0 稳定版
  - 修改文件: `hopper/__init__.py`

### 新功能
- **Flex Flash 简化文档** [PR #2231](https://github.com/Dao-AILab/flash-attention/pull/2231)
  - Markus Hoehnerbach 为 flex flash 添加了简短说明文档
  - 新增文件: `flash_attn/cute/README.md` (26行新增)

### 问题修复
- **共享内存竞争修复** [PR #2229](https://github.com/Dao-AILab/flash-attention/pull/2229)
  - Driss Guessous 修复了共享内存竞争问题
  - 修改文件:
    - `flash_attn/cute/block_sparse_utils.py`
    - `flash_attn/cute/flash_fwd_sm100.py`

- **依赖更新** [PR #2155](https://github.com/Dao-AILab/flash-attention/pull/2155)
  - Jane (Yuan) Xu 将 TORCH_STABLE_ONLY 替换为 TORCH_TARGET_VERSION
  - 修改文件: `hopper/setup.py`
### sgl-project/sglang
# SGLang 项目更新 (2026-02-04)

## 主要性能优化

- **优化 get_topk_ragged 性能** (#760ae93): 通过融合 get k 和 k_scale triton 内核，显著提升性能
- **改进 MHA 模型的 KV 偏移计算** (#f730c18): 优化不同张量并行度下的内存访问效率
- **融合 qkvbfg 线性操作** (#37c33cc): 将多个线性操作合并为单个 GEMM，提升计算效率
- **添加快速预热标志** (#d279520): 为 DeepGemm 添加快速预热选项，减少启动时间
- **核融合优化** (#4739f2e): 为 Qwen-Image、WAN 和 HunyuanVideo 实现门控残差层归一化尺度偏移核融合

## 新功能与模型支持

- **支持 InternS1-Pro 模型** (#3e7ecb7): 添加新模型支持，包含 580 行代码变更
- **支持 π/π-FAST VLA 模型推理** (#18266): 添加机器人基础策略模型推理支持
- **支持按请求传递 spaces_between_special_tokens** (#a6f53cc): 增强 API 灵活性
- **添加 MoE 融合配置** (#efbf395): 为 Qwen3-Coder-Next-FP8 在 H100 TP=2 上添加配置

## Bug 修复与优化

- **修复连续动态请求下的服务器缓存 DIT 错误** (#da758ed): 解决缓存管理问题
- **修复多模态 Session 问题** (#c1d529c): 暴露 Session 通过 Engine，修复多模态处理
- **修复 Kimi K2.5 的 MoE GEMM 配置初始化** (#599c5f4): 解决模型配置问题
- **修复注意力测试中的 MockModelRunner** (#2e87c2b): 确保测试正确性
- **修复 GPU-0 上的冗余内存使用** (#4c40304): 优化内存分配
- **修复 AITER 注意力中的 OOM 错误** (#18262): 解决内存管理问题

## 文档更新

- **支持 Markdown/Notebook 友好的文档导出** (#e616d35, #669a9bd): 添加文档格式转换功能
- **修复文档拼写错误** (#de6a032): 更新多个文档文件中的拼写错误
- **添加 SGLANG_MOONCAKE_CUSTOM_MEM_POOL 文档** (#c8212b9): 记录环境变量支持
- **修复 README 拼写错误** (#1f72f66): 更新项目主文档

## 社区问题与请求

- **更新投机解码文档** (#18268): 需要记录新的 SGLANG_ENABLE_SPEC_V2 功能
- **添加分段 CUDA 图文档** (#18267): 需要为性能功能添加文档
- **集成 FA4 SM90 解码** (#18265): 开发 FA4 在 SM90 上的实现
- **验证 FA4 在注意力后端兼容性** (#18264): 更新兼容性文档
- **解释如何使用 NVIDIA ModelOpt 检查点** (#18261): 添加 ModelOpt 工作流程文档
- **支持 ModelOpt 的 MXFP8 加载器** (#18258): 添加 ModelOpt PTQ MXFP8 支持
### vllm-project/vllm
## vLLM Updates - February 4, 2026

### New Features
- **ColBERT late interaction model support** ([#33686](https://github.com/vllm-project/vllm/pull/33686)): Added ColBERT model support with new implementation in `vllm/model_executor/models/colbert.py`

### Performance Improvements
- **Chat completion streaming optimization** ([#33782](https://github.com/vllm-project/vllm/pull/33782)): Enhanced streaming performance in `vllm/entrypoints/openai/chat_completion/serving.py`
- **Spec decoding optimization** ([#33612](https://github.com/vllm-project/vllm/pull/33612)): 1.5% throughput improvement via async scheduler optimizations
- **GDN Attention State Layout optimization** ([#33291](https://github.com/vllm-project/vllm/pull/33291)): Changed layout from [N, HV, K, V] to [N, HV, V, K] for better performance
- **Zero-copy GQA implementation** ([#33732](https://github.com/vllm-project/vllm/pull/33732)): Improved memory efficiency for multimodal and CPU processing

### Bug Fixes
- **DeepSeek R1 MLA fix** ([#33637](https://github.com/vllm-project/vllm/pull/33637)): Fixed DeepSeek R1 with CUTLASS MLA on B200 GPUs
- **TRTLLM attention conflict** ([#33192](https://github.com/vllm-project/vllm/pull/33192)): Disabled TRTLLM attention when KV transfer is enabled
- **MCP tools non-streaming mode** ([#32762](https://github.com/vllm-project/vllm/pull/32762)): Fixed McpCall return for built-in MCP tools
- **Qwen audio-in-video support** ([#33605](https://github.com/vllm-project/vllm/pull/33605)): Fixed audio support for Qwen2.5-Omni and Qwen3-Omni
- **Interns1-pro initialization** ([#33793](https://github.com/vllm-project/vllm/pull/33793)): Fixed initialization and pipeline parallelism issues
- **ROCm float8_e4m3fnuz support** ([#33713](https://github.com/vllm-project/vllm/pull/33713)): Added missing data type to NCCL dispatching

### Model Enhancements
- **Qwen3-Omni transcription** ([#29828](https://github.com/vllm-project/vllm/pull/29828)): Enhanced transcription capabilities
- **RotaryEmbedding CustomOp** ([#33800](https://github.com/vllm-project/vllm/pull/33800)): Added support for gpt-oss models

### Infrastructure Updates
- **XPU unquantized MoE support** ([#33659](https://github.com/vllm-project/vllm/pull/33659)): Added XPU platform support for unquantized MoE
- **Ray device visibility unification** ([#33308](https://github.com/vllm-project/vllm/pull/33308)): Fixed device handling across CUDA and ROCm
- **ORJSONResponse integration** ([#33548](https://github.com/vllm-project/vllm/pull/33548)): Improved request processing efficiency

### Reverted Changes
- **FA3 swizzle optimization** ([#33841](https://github.com/vllm-project/vllm/pull/33841)): Reverted recent changes to flash attention backend
- **torch.compile cold start** ([#33820](https://github.com/vllm-project/vllm/pull/33820)): Reverted cold start time optimizations

### Notable Issues
- Docker image compatibility issues with Qwen3-Next and FP8 GEMM settings
- DeepSeek V3.2 benchmark failures
- Mistral3 multimodal inference errors
- Pipeline parallelism issues with Step3p5ForCausalLM
### NVIDIA/cutile-python
昨日无更新。

## 总结
- Diff 内容已截断以满足 prompt 预算。
- Issue 内容已截断以满足 prompt 预算。
- OpenRouter repo summarize failed for NVIDIA/cutlass with qwen/qwen3-coder:free: OpenRouter 429 Too Many Requests: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"qwen/qwen3-coder:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations","provider_name":"Venice","is_byok":false}},"user_id":"user_2wqU29q2Bhpw2S2iw7Pwn8RHaXB"}
- OpenRouter repo summarize failed for NVIDIA/cutlass with openai/gpt-oss-120b:free: OpenRouter 429 Too Many Requests: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"openai/gpt-oss-120b:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations","provider_name":"OpenInference","is_byok":false}},"user_id":"user_2wqU29q2Bhpw2S2iw7Pwn8RHaXB"}
- OpenRouter repo summarize failed for flashinfer-ai/flashinfer with qwen/qwen3-coder:free: OpenRouter 429 Too Many Requests: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"qwen/qwen3-coder:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations","provider_name":"Venice","is_byok":false}},"user_id":"user_2wqU29q2Bhpw2S2iw7Pwn8RHaXB"}
- OpenRouter repo summarize failed for flashinfer-ai/flashinfer with openai/gpt-oss-120b:free: OpenRouter 429 Too Many Requests: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"openai/gpt-oss-120b:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations","provider_name":"OpenInference","is_byok":false}},"user_id":"user_2wqU29q2Bhpw2S2iw7Pwn8RHaXB"}
- OpenRouter repo summarize failed for Dao-AILab/flash-attention with qwen/qwen3-coder:free: OpenRouter 402 Payment Required: {"error":{"message":"Provider returned error","code":402,"metadata":{"raw":"{\"error\":\"API key USD spend limit exceeded. Your account may still have USD balance, but this API key has reached its configured USD spending limit.\"}","provider_name":"Venice","is_byok":false}},"user_id":"user_2wqU29q2Bhpw2S2iw7Pwn8RHaXB"}
- OpenRouter repo summarize failed for Dao-AILab/flash-attention with openai/gpt-oss-120b:free: OpenRouter 429 Too Many Requests: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"openai/gpt-oss-120b:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations","provider_name":"OpenInference","is_byok":false}},"user_id":"user_2wqU29q2Bhpw2S2iw7Pwn8RHaXB"}
- OpenRouter repo summarize failed for sgl-project/sglang with qwen/qwen3-coder:free: OpenRouter 429 Too Many Requests: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"qwen/qwen3-coder:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations","provider_name":"Venice","is_byok":false}},"user_id":"user_2wqU29q2Bhpw2S2iw7Pwn8RHaXB"}
- OpenRouter repo summarize failed for sgl-project/sglang with openai/gpt-oss-120b:free: OpenRouter 429 Too Many Requests: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"openai/gpt-oss-120b:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations","provider_name":"OpenInference","is_byok":false}},"user_id":"user_2wqU29q2Bhpw2S2iw7Pwn8RHaXB"}
- OpenRouter repo summarize failed for vllm-project/vllm with qwen/qwen3-coder:free: OpenRouter 429 Too Many Requests: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"qwen/qwen3-coder:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations","provider_name":"Venice","is_byok":false}},"user_id":"user_2wqU29q2Bhpw2S2iw7Pwn8RHaXB"}
- OpenRouter repo summarize failed for vllm-project/vllm with openai/gpt-oss-120b:free: OpenRouter 429 Too Many Requests: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"openai/gpt-oss-120b:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations","provider_name":"OpenInference","is_byok":false}},"user_id":"user_2wqU29q2Bhpw2S2iw7Pwn8RHaXB"}
- OpenRouter global summarize failed for qwen/qwen3-coder:free: OpenRouter 429 Too Many Requests: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"qwen/qwen3-coder:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations","provider_name":"Venice","is_byok":false}},"user_id":"user_2wqU29q2Bhpw2S2iw7Pwn8RHaXB"}
- OpenRouter global summarize failed for openai/gpt-oss-120b:free: OpenRouter 429 Too Many Requests: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"openai/gpt-oss-120b:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations","provider_name":"OpenInference","is_byok":false}},"user_id":"user_2wqU29q2Bhpw2S2iw7Pwn8RHaXB"}
