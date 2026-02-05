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
  "summary": "NVIDIA发布了CUTLASS 4.4，增强了CuTe DSL对Blackwell架构和CUDA 13.1的支持，同时解决了GEMM循环中的性能问题。FlashInfer添加了Fused MoE支持和多GPU测试能力，但mxfp4量化性能仍是关注点。SGLang优化了多种模型的内核融合并修复了MoE相关bug。VLLM引入了ColBERT支持并改进了流式性能，同时修复了与多种硬件配置的兼容性问题。Flash-attention解决了共享内存竞争条件并更新了文档。",
  "conclusion": "主要风险包括CUTLASS中Blackwell GEMM性能下降、FlashInfer中mxfp4量化瓶颈以及VLLM中与某些模型的Docker兼容性问题。团队应监控这些回归情况，特别是在多GPU部署和量化模型推理方面。各项目对内核优化的焦点表明性能将持续改进，但需要在生产环境中进行仔细测试。"
}
```

## 具体内容分析
### deepseek-ai/DeepGEMM
昨日无更新。
### deepseek-ai/FlashMLA
昨日无更新。
### NVIDIA/cutlass
## NVIDIA/cutlass v4.4 发布更新

### 提交内容

Junkai-Wu 提交了 v4.4 版本更新 (#2999)，主要更新包括：

#### CuTe DSL 新特性
- 支持 CUDA toolkit 13.1，可通过 `cutlass/python/CuTeDSL/setup.sh --cu13` 设置
- GB300 (Blackwell) 架构支持，新增示例 `examples/python/CuTeDSL/blackwell/sm103_dense_blockscaled_gemm_persistent.py`
- 引入 cute.experimental 高级组合层：
  - 无片段编程模型
  - 自动 TMA 描述符生成
  - SIMT 复制的自动向量和谓词
  - 新管道抽象
  - 新分区操作
  - 设备端 TMA 描述符分配

#### 新增示例
- `examples/python/CuTeDSL/experimental/ampere/memcpy_simt_universal_copy.py`
- `examples/python/CuTeDSL/experimental/blackwell/dense_block_scaled_gemm.py`
- `examples/python/CuTeDSL/experimental/blackwell/dense_gemm.py`
- `examples/python/CuTeDSL/experimental/blackwell/dense_gemm_2sm.py`
- `examples/python/CuTeDSL/experimental/blackwell/dense_gemm_cute_pipeline.py`
- `examples/python/CuTeDSL/experimental/blackwell/dense_gemm_ptr_array.py`

#### 内核改进
- `include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized.hpp`：添加依赖网格等待
- `include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized_input_transform.hpp`：重新排序调度器初始化
- `include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized_mma_transform.hpp`：重新排序调度器初始化
- `include/cutlass/gemm/kernel/sm90_gemm_array_tma_warpspecialized_cooperative.hpp`：重新排序调度器初始化

[查看完整提交](https://github.com/NVIDIA/cutlass/commit/6b3e607b852f1543dc21323155a2ad71473c8642)

### 问题报告

#### 性能问题：Blackwell GEMM 循环中的频繁调用
- **问题**：在 Blackwell 架构上使用 CUTLASS 4.x API 时，内核启动之间存在显著开销
- **原因**：频繁的 `cudaGetDriverEntryPoint` 和 `cuTensorMapEncodeTiled` 调用
- **影响**：GEMM 内核执行 91us，但循环耗时 625us
- [查看详情](https://github.com/NVIDIA/cutlass/issues/3003)

#### Bug：fp8 共享内存 tensor 调用 `cute.print_tensor` 导致段错误
- **组件**：CuTe DSL
- **描述**：在调用 `cute.print_tensor` 时出现意外段错误
- [查看详情](https://github.com/NVIDIA/cutlass/issues/3002)

#### 导入错误：nvidia-cutlass-dsl v4.4.0.dev1
- **组件**：CuTe DSL
- **描述**：导入模块时出现错误
- [查看详情](https://github.com/NVIDIA/cutlass/issues/3001)

#### 支持 Tesla V100 GPU
- **问题**：询问 CUTLASS 是否支持 NVIDIA Tesla V100 GPU
- [查看详情](https://github.com/NVIDIA/cutlass/issues/3000)
### flashinfer-ai/flashinfer
# FlashInfer Newsletter - 2026-02-04

## Notable Commits

### CI Fix: Release Workflow Permissions
- **PR**: #2488 | **SHA**: [cdbb2c3](https://github.com/flashinfer-ai/flashinfer/commit/cdbb2c3c73598bbb04a0e2dc2d7cf6ee9e5aeda5)
- Fixed permission errors in release workflow by removing `chown` commands and `--user` flag in Docker run commands
- Files: `.github/workflows/nightly-release.yml`, `.github/workflows/release.yml`

### Feature: Fused MoE Enhancements
- **PR**: #2462 | **SHA**: [e284274](https://github.com/flashinfer-ai/flashinfer/commit/e284274e2eb67538bb9c884f2ed9c0143772d3ac)
- Added support for Fused MoE non-gated Relu2 NVFP4 & FP8 and Nemotron
- 949 total changes (645 additions, 304 deletions)
- Files: Multiple files including benchmarks, csrc, flashinfer, include, and tests

### Multi-Node/Multi-GPU Testing
- **PR**: #2410 | **SHA**: [9bf007d](https://github.com/flashinfer-ai/flashinfer/commit/9bf007d7403f0a394fda44abd4170b354cac3f05)
- Added comprehensive multi-node/multi-GPU test scripts
- 1061 total changes (629 additions, 432 deletions)
- Files: `scripts/task_run_unit_tests.sh`, `scripts/task_test_multi_gpu_comm_kernels.sh`, `scripts/test_utils.sh`

### CI Test Discovery Fix
- **PR**: #2481 | **SHA**: [567ded1](https://github.com/flashinfer-ai/flashinfer/commit/567ded1f612c38bd6a921a4a7ce4ecc6db2b9a9e)
- Renamed `tests/mamba/test_utils.py` to `tests/mamba/utils.py` to fix CI test discovery
- Files: `tests/mamba/test_selective_state_update_mtp.py`, `tests/mamba/test_selective_state_update_stp.py`

### MLA Benchmarks Memory Bandwidth Fix
- **PR**: #2479 | **SHA**: [f84ac1c](https://github.com/flashinfer-ai/flashinfer/commit/f84ac1c97e2e1a4391150ca73971b973ee569e5a)
- Fixed memory bandwidth calculation in MLA benchmarks
- Files: `benchmarks/bench_trtllm_gen_mla.py`, `benchmarks/routines/attention.py`

### Hopper GDN Prefill Optimization
- **PR**: #2422 | **SHA**: [6ae5bfe](https://github.com/flashinfer-ai/flashinfer/commit/6ae5bfe6a48c591a171f32888e5de28b9ca0207b)
- Reduced Hopper's GDN prefill compilation time and fixed docstring
- 248 total changes (207 additions, 41 deletions)
- Files: Multiple files including csrc, flashinfer, and include directories

## New Issues

### Performance: mxfp4 Quantization Kernel
- **Issue**: #2496
- mxfp4 quantization kernel is an order of magnitude slower than nvfp4 (0.456ms vs 0.036ms)
- Request for CuTe DSL update to improve performance

### Bug: XQA Shared Memory Configuration
- **Issue**: #2494
- XQA always sets dynamic shared memory size on GPU 0, causing issues in multi-GPU environments
- Results in CUDA invalid argument errors when kernel launches on incorrect GPU

### Feature Request: CuTe DSL Enhancements
- **Issue**: #2493 - Performance improvement for CuTe DSL GDN Decode Kernel
- **Issue**: #2491 - Add CuTe DSL backends for RoPE APIs

### Bug: GDN Prefill Kernel Produces NaN
- **Issue**: #2490
- GDN prefill kernel produces NaN values when used in vLLM
- Test case provided in issue description

### Bug: Top-K Page Table Transform
- **Issue**: #2486
- Illegal memory access in RadixTopKKernel_Unified
- Minimal test case provided that reproduces the issue

### Accuracy Issue: GLM-4.7 ModelOpt NVFP4
- **Issue**: #2485
- GLM-4.7 ModelOpt NVFP4 produces garbage output with FlashInfer TRTLLM MoE backend
- Model produces completely garbled output instead of coherent responses
### Dao-AILab/flash-attention
## Dao-AILab/flash-attention 更新

### 提交记录

1. **Markus Hoehnerbach** 添加了 flex flash 的简短说明文档
   - PR: [#2231](https://github.com/Dao-AILab/flash-attention/pull/2231)
   - 修改文件: `flash_attn/cute/README.md`
   - 变更: +26 行

2. **Jane (Yuan) Xu** 更新了 Torch 版本检测逻辑
   - PR: [#2155](https://github.com/Dao-AILab/flash-attention/pull/2155)
   - 修改文件: `hopper/setup.py`
   - 变更: +1 -1 行
   - 代码改动:
     ```python
     # 从 TORCH_STABLE_ONLY 改为 TORCH_TARGET_VERSION
     ```

3. **Driss Guessous** 修复了共享内存竞争问题
   - PR: [#2229](https://github.com/Dao-AILab/flash-attention/pull/2229)
   - 修改文件:
     - `flash_attn/cute/block_sparse_utils.py` (+1 -1)
     - `flash_attn/cute/flash_fwd_sm100.py` (+3 -3)
   - 总变更: +4 -4 行
### sgl-project/sglang
# SGLang Daily Updates - 2026-02-04

## Performance Optimizations

- **Optimized get_topk_ragged** by fusing get k and k_scale triton kernels ([#760ae93](https://github.com/sgl-project/sglang/commit/760ae933bb3878a6897e7e552a746929c29e9d90))
- **Fused qkvbfg linear operations** into single GEMM for better performance ([#37c33cc](https://github.com/sgl-project/sglang/commit/37c33cc0aa6213fd4abcfb40c3e1d71dde484295))
- **Improved KV offset calculation** for MHA models with different TP sizes ([#f730c18](https://github.com/sgl-project/sglang/commit/f730c186799d966a62531269ce46178364c85dc3))
- **Added kernel fusion** for Qwen-Image, WAN and HunyuanVideo models ([#4739f2e](https://github.com/sgl-project/sglang/commit/4739f2e8d5732f7464d1af75d31b4d44c61783b6))

## New Features

- **Added MoE fused config** for Qwen3-Coder-Next-FP8 on H100 TP=2 ([#efbf395](https://github.com/sgl-project/sglang/commit/efbf39583e7a716e0204b071db687145392e41b2))
- **Added support for interns1-pro model** with rotary embedding updates ([#3e7ecb7](https://github.com/sgl-project/sglang/commit/3e7ecb78a60f8e1d889cfe25c88006577783d903))
- **Support for spaces_between_special_tokens per request** in OpenAI protocol ([#a6f53cc](https://github.com/sgl-project/sglang/commit/a6f53cc5e3ac7eb8ae9e4236d9834897684505ad))
- **Added fast warmup flag** for DeepGemm backend ([#d279520](https://github.com/sgl-project/sglang/commit/d279520ba5771e0bd361c6a762b653391bb1bc09))

## Bug Fixes

- **Fixed kimi k2.5's MoE GEMM config initialization** ([#599c5f4](https://github.com/sgl-project/sglang/commit/599c5f4922579742a0c65a4c2fb4503dd63f7ae3))
- **Fixed MockModelRunner in attention tests** ([#2e87c2b](https://github.com/sgl-project/sglang/commit/2e87c2bd5e43bfad57150ff878761bc6cffc0ab8))
- **Fixed redundant memory usage on GPU-0** for diffusion models ([#4c40304](https://github.com/sgl-project/sglang/commit/4c403045ec690dbcf3b63a941356e004201ba337))
- **Fixed server cache-dit bug** under continuous dynamic requests ([#da758ed](https://github.com/sgl-project/sglang/commit/da758ed601270b21e1cfb404306ff0ca5c816a3f))

## Documentation & Infrastructure

- **Fixed misspellings & typos** in 15 documentation files ([#de6a032](https://github.com/sgl-project/sglang/commit/de6a03260f59fd33a9eeb8f67e7e6e2cf235a70f))
- **Documented SGLANG_MOONCAKE_CUSTOM_MEM_POOL** environment variable ([#c8212b9](https://github.com/sgl-project/sglang/commit/c8212b9fac11d7ad3a2aa088946e1a815a618a97))
- **Added support for Markdown/Notebook-friendly documentation export** ([#669a9bd](https://github.com/sgl-project/sglang/commit/669a9bd1809a344fae5b9d962d2cd6f842cb50c1))
- **Added kimi mi35x nightly test** and reorganized test files ([#6fd878b](https://github.com/sgl-project/sglang/commit/6fd878b41df0153bd28f0185920e1b2d9dcc7480))

## Open Issues

- Need to update speculative decoding documentation for new SGLANG_ENABLE_SPEC_V2 and ngram features ([#18268](https://github.com/sgl-project/sglang/issues/18268))
- Request to document piecewise CUDA graph feature ([#18267](https://github.com/sgl-project/sglang/issues/18267))
- Request to add π0/π0-FAST VLA model inference support ([#18266](https://github.com/sgl-project/sglang/issues/18266))
- Need to integrate FA4 SM90 Decode and verify compatibility ([#18265](https://github.com/sgl-project/sglang/issues/18265), [#18264](https://github.com/sgl-project/sglang/issues/18264))
- OOM errors in AITER attention backend ([#18262](https://github.com/sgl-project/sglang/issues/18262))
- Need to document ModelOpt offline quantization ([#18261](https://github.com/sgl-project/sglang/issues/18261))
- Request to support ModelOpt's PTQ MXFP8 quantization ([#18258](https://github.com/sgl-project/sglang/issues/18258))
### vllm-project/vllm
## Commits

### 新功能
- **ColBERT late interaction model support** ([#33686](https://github.com/vllm-project/vllm/pull/33686)): 添加 ColBERT 模型支持，包含新文件和现有组件的修改。
- **Parser for ResponsesAPI** ([#32712](https://github.com/vllm-project/vllm/pull/32712)): 实现 ResponsesAPI 的解析器模块。
- **Unquantized MoE support for XPU** ([#33659](https://github.com/vllm-project/vllm/pull/33659)): 为 XPU 硬件添加非量化 MoE 支持。
- **Enable TRITON_ATTN for Batch Invariance** ([#33688](https://github.com/vllm-project/vllm/pull/33688)): 为批处理不变性启用 Triton attention。

### 性能优化
- **Optimize chat completion streaming** ([#33782](https://github.com/vllm-project/vllm/pull/33782)): 改进聊天完成流式传输性能。
- **Optimize spec decoding + async scheduling** ([#33612](https://github.com/vllm-project/vllm/pull/33612)): 通过优化实现 1.5% 吞吐量提升。
- **Change GDN Attention State Layout** ([#33291](https://github.com/vllm-project/vllm/pull/33291)): 将注意力状态布局从 [N, HV, K, V] 优化为 [N, HV, V, K]。
- **Reduce e2e fusion test time** ([#33293](https://github.com/vllm-project/vllm/pull/33293)): 大幅减少端到端融合测试执行时间。

### Bug 修复
- **Fix DeepSeek R1 with CUTLASS MLA on B200** ([#33637](https://github.com/vllm-project/vllm/pull/33637)): 解决 B200 硬件上 DeepSeek R1 模型问题。
- **Disable TRTLLM attention when KV transfer is enabled** ([#33192](https://github.com/vllm-project/vllm/pull/33192)): 修复 KV 传输兼容性问题。
- **Fix torchrun PP broadcast deadlock with async scheduling** ([#33701](https://github.com/vllm-project/vllm/pull/33701)): 解决流水线并行中的死锁问题。
- **Fix audio-in-video support for Qwen models** ([#33605](https://github.com/vllm-project/vllm/pull/33605)): 增强 Qwen2.5-Omni 和 Qwen3-Omni 的音频视频支持。
- **Fix interns1-pro initialization and PP** ([#33793](https://github.com/vllm-project/vllm/pull/33793)): 修正流水线并行初始化问题。

### 硬件支持
- **Zero-copy GQA for multimodal and CPU** ([#33732](https://github.com/vllm-project/vllm/pull/33732)): 为多模态和 CPU 实现零拷贝 GQA。
- **Split attention dispatch by head_dim alignment** ([#32161](https://github.com/vllm-project/vllm/pull/32161)): 基于 head 维度对齐改进 CPU attention 分发。
- **Unify Ray device visibility handling across CUDA and ROCm** ([#33308](https://github.com/vllm-project/vllm/pull/33308)): 统一 CUDA 和 ROCm 的 Ray 设备可见性处理。

### 重构和清理
- **Remove unused dead code** ([#33718](https://github.com/vllm-project/vllm/pull/33718)): 清理多个文件中的未使用代码。
- **Clean up AOT compile bypass** ([#33578](https://github.com/vllm-project/vllm/pull/33578)): 简化 AOT 编译绕过逻辑。
- **Deprecate profiling environments** ([#33722](https://github.com/vllm-project/vllm/pull/33722)): 弃用未使用的性能分析环境变量。

## Issues

### 关键 Bug
- **Docker image issues with Qwen3-Next** ([#33833](https://github.com/vllm-project/vllm/issues/33833)): 较新 Docker 镜像在运行特定配置的 Qwen3-Next 时出现问题。
- **Deepseek V3.2 benchmark failure** ([#33831](https://github.com/vllm-project/vllm/issues/33831)): Deepseek V3.2 基准测试中的 TypeError。
- **Mistral3 multimodal inference failure** ([#33828](https://github.com/vllm-project/vllm/issues/33828)): Mistral3 离线多模态推理中的提示占位符错误。

### CI 失败
- **Quantized Models Test failure** ([#33816](https://github.com/vllm-project/vllm/issues/33816)): GPTQ 量化模型测试失败。
- **LM Eval Large Models failure** ([#33812](https://github.com/vllm-project/vllm/issues/33812)): 大型模型的 LM 评估 CI 失败。
- **Kernels MoE Test failure** ([#33809](https://github.com/vllm-project/vllm/issues/33809)): MoE 内核测试问题。

### 模型特定问题
- **Step3p5ForCausalLM pipeline parallelism failure** ([#33823](https://github.com/vllm-project/vllm/issues/33823)): 使用流水线并行时 Step3p5 模型问题。
- **llm.score() failure with batched multimodal input** ([#33813](https://github.com/vllm-project/vllm/issues/33813)): qwen3-vl-reranker 批处理多模态输入评分失败。
### NVIDIA/cutile-python
昨日无更新。

## 总结
- Diff 内容已截断以满足 prompt 预算。
- Issue 内容已截断以满足 prompt 预算。
- OpenRouter repo summarize failed for NVIDIA/cutlass with openai/gpt-oss-120b:free: OpenRouter 429 Too Many Requests: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"openai/gpt-oss-120b:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations","provider_name":"OpenInference","is_byok":false}},"user_id":"user_2wqU29q2Bhpw2S2iw7Pwn8RHaXB"}
- OpenRouter repo summarize failed for flashinfer-ai/flashinfer with openai/gpt-oss-120b:free: OpenRouter 429 Too Many Requests: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"openai/gpt-oss-120b:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations","provider_name":"OpenInference","is_byok":false}},"user_id":"user_2wqU29q2Bhpw2S2iw7Pwn8RHaXB"}
- OpenRouter repo summarize failed for Dao-AILab/flash-attention with openai/gpt-oss-120b:free: OpenRouter 429 Too Many Requests: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"openai/gpt-oss-120b:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations","provider_name":"OpenInference","is_byok":false}},"user_id":"user_2wqU29q2Bhpw2S2iw7Pwn8RHaXB"}
- OpenRouter repo summarize failed for sgl-project/sglang with openai/gpt-oss-120b:free: OpenRouter 429 Too Many Requests: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"openai/gpt-oss-120b:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations","provider_name":"OpenInference","is_byok":false}},"user_id":"user_2wqU29q2Bhpw2S2iw7Pwn8RHaXB"}
- OpenRouter repo summarize failed for vllm-project/vllm with openai/gpt-oss-120b:free: OpenRouter 429 Too Many Requests: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"openai/gpt-oss-120b:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations","provider_name":"OpenInference","is_byok":false}},"user_id":"user_2wqU29q2Bhpw2S2iw7Pwn8RHaXB"}
- OpenRouter global summarize failed for openai/gpt-oss-120b:free: OpenRouter 429 Too Many Requests: {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"openai/gpt-oss-120b:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations","provider_name":"OpenInference","is_byok":false}},"user_id":"user_2wqU29q2Bhpw2S2iw7Pwn8RHaXB"}
