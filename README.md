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
  "summary": "NVIDIA发布了Cutlass v4.4版本，增加了对Blackwell架构的支持，但报告了性能开销问题。FlashInfer扩展了MoE功能并添加了多GPU测试，同时修复了多个bug。Flash-attention发布了v3.0.0稳定版并改进了文档。SGLang增强了多模态支持，通过内核优化和新模型集成提升了性能。VLLM专注于性能优化，添加了新模型支持，并修复了各种兼容性问题。",
  "conclusion": "AI基础设施生态系统持续快速发展，针对Blackwell和Hopper等新硬件的优化不断推出。尽管新功能和模型支持不断增加，但性能问题和bug仍然存在，特别是在量化、内存管理和多GPU设置方面。用户在升级到新版本时应保持谨慎，特别是生产环境部署。"
}
```

## 具体内容分析
### deepseek-ai/DeepGEMM
昨日无更新。
### deepseek-ai/FlashMLA
昨日无更新。
### NVIDIA/cutlass
提交：
- v4.4 release update v2. (#2999) [`6b3e607`](https://github.com/NVIDIA/cutlass/commit/6b3e607b852f1543dc21323155a2ad71473c8642)
- 受影响文件: CHANGELOG.md, README.md, examples/python/CuTeDSL/blackwell/dense_gemm_persistent_dynamic.py, examples/python/CuTeDSL/blackwell/sm103_dense_blockscaled_gemm_persistent.py, examples/python/CuTeDSL/experimental/ampere/memcpy_simt_universal_copy.py, examples/python/CuTeDSL/experimental/blackwell/dense_block_scaled_gemm.py, examples/python/CuTeDSL/experimental/blackwell/dense_gemm.py, examples/python/CuTeDSL/experimental/blackwell/dense_gemm_2sm.py, examples/python/CuTeDSL/experimental/blackwell/dense_gemm_cute_pipeline.py, examples/python/CuTeDSL/experimental/blackwell/dense_gemm_ptr_array.py, include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized.hpp, include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized_input_transform.hpp, include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized_mma_transform.hpp, include/cutlass/gemm/kernel/sm100_tile_scheduler_group.hpp, include/cutlass/gemm/kernel/sm103_blockscaled_gemm_array_tma_warpspecialized.hpp, include/cutlass/gemm/kernel/sm90_gemm_array_tma_warpspecialized_cooperative.hpp
- 文件列表已截断。
Issues：
- Performance overhead: Frequent cuTensorMapEncodeTiled and cudaGetDriverEntryPoint calls during Blackwell GEMM loops (https://github.com/NVIDIA/cutlass/issues/3003)
- Issue 内容已截断。
- [BUG] segmentation fault when calling `cute.print_tensor` on fp8 shared-memory tensor (https://github.com/NVIDIA/cutlass/issues/3002)
- Issue 内容已截断。
- [BUG] nvidia-cutlass-dsl v4.4.0.dev1 import error (https://github.com/NVIDIA/cutlass/issues/3001)
- Issue 内容已截断。
- [QST] is cutlass support on nvidia GPU Tesla V100 ? (https://github.com/NVIDIA/cutlass/issues/3000)
### flashinfer-ai/flashinfer
提交：
- ci: fix permission errors in release workflow on ci-infra runner (#2488) [`cdbb2c3`](https://github.com/flashinfer-ai/flashinfer/commit/cdbb2c3c73598bbb04a0e2dc2d7cf6ee9e5aeda5)
- 受影响文件: .github/workflows/nightly-release.yml, .github/workflows/release.yml
- feat: Support Fused MoE non gated Relu2 NVFP4 & FP8 and support Nemotron, fixed (#2462) [`e284274`](https://github.com/flashinfer-ai/flashinfer/commit/e284274e2eb67538bb9c884f2ed9c0143772d3ac)
- 受影响文件: benchmarks/bench_trtllm_gen_fused_moe_autotuner.py, benchmarks/routines/flashinfer_benchmark_utils.py, benchmarks/routines/moe.py, csrc/trtllm_batched_gemm_runner.cu, csrc/trtllm_fused_moe_kernel_launcher.cu, csrc/trtllm_fused_moe_routing_deepseek.cu, csrc/trtllm_fused_moe_runner.cu, flashinfer/__init__.py, flashinfer/fused_moe/__init__.py, flashinfer/fused_moe/core.py, include/flashinfer/trtllm/batched_gemm/KernelRunner.h, include/flashinfer/trtllm/fused_moe/DevKernel.h, include/flashinfer/trtllm/fused_moe/RoutingKernel.h, include/flashinfer/trtllm/fused_moe/runner.h, tests/moe/test_dpsk_fused_moe_fp8.py, tests/moe/test_trtllm_gen_fused_moe.py
- 文件列表已截断。
- Add/update multi node/multi GPU test scripts (#2410) [`9bf007d`](https://github.com/flashinfer-ai/flashinfer/commit/9bf007d7403f0a394fda44abd4170b354cac3f05)
- 受影响文件: scripts/task_run_unit_tests.sh, scripts/task_test_blackwell_kernels.sh, scripts/task_test_multi_gpu_comm_kernels.sh, scripts/task_test_multi_node_comm_kernels.sh, scripts/test_utils.sh
- fix: Rename tests/mamba/test_utils.py to tests/mamba/utils.py to fix CI test discovery (#2481) [`567ded1`](https://github.com/flashinfer-ai/flashinfer/commit/567ded1f612c38bd6a921a4a7ce4ecc6db2b9a9e)
- 受影响文件: tests/mamba/test_selective_state_update_mtp.py, tests/mamba/test_selective_state_update_stp.py, tests/mamba/utils.py
- fix: Fix memory bandwidth calculation in MLA benchmarks (#2479) [`f84ac1c`](https://github.com/flashinfer-ai/flashinfer/commit/f84ac1c97e2e1a4391150ca73971b973ee569e5a)
- 受影响文件: benchmarks/bench_trtllm_gen_mla.py, benchmarks/routines/attention.py
- refactor: reduce hopper's gdn prefill compilation time and fix docstring. (#2422) [`6ae5bfe`](https://github.com/flashinfer-ai/flashinfer/commit/6ae5bfe6a48c591a171f32888e5de28b9ca0207b)
- 受影响文件: csrc/flat_prefill_kernel_delta_rule_sm90_extern.inc, csrc/gdn_prefill_launcher.cu, csrc/gdn_prefill_sm90_kernel_inst.jinja, csrc/prefill_kernel_delta_rule_sm90.cu, flashinfer/gdn_prefill.py, flashinfer/jit/core.py, flashinfer/jit/gdn.py, include/flashinfer/flat/ampere/collective/flat_collective_inverse.hpp, include/flashinfer/flat/ampere/collective/flat_collective_load.hpp, include/flashinfer/flat/common.hpp, include/flashinfer/flat/cute_ext.hpp, include/flashinfer/flat/debug.hpp, include/flashinfer/flat/hopper/collective/flat_collective_load.hpp, include/flashinfer/flat/hopper/collective/flat_collective_store.hpp, include/flashinfer/flat/hopper/collective/flat_collective_tma_warpspecialized_delta_rule.hpp, include/flashinfer/flat/hopper/collective/flat_common.hpp
- 文件列表已截断。
Issues：
- [Perf] mxfp4 quantize kernel is slow (https://github.com/flashinfer-ai/flashinfer/issues/2496)
- XQA always sets dynamic smem size on rank 0 (https://github.com/flashinfer-ai/flashinfer/issues/2494)
- Perf improvement for CuTe DSL GDN Decode Kernel (https://github.com/flashinfer-ai/flashinfer/issues/2493)
- Add CuTe DSL backends for RoPE APIs (https://github.com/flashinfer-ai/flashinfer/issues/2491)
- [Bug] GDN prefill kernel produces NaN (https://github.com/flashinfer-ai/flashinfer/issues/2490)
- Issue 内容已截断。
- [top_k_page_table_transform bug]Illegal memory access in RadixTopKKernel_Unified (https://github.com/flashinfer-ai/flashinfer/issues/2486)
- [bug] Accuracy Issue: GLM-4.7 ModelOpt NVFP4 produces garbage output with FlashInfer TRTLLM MoE backend (https://github.com/flashinfer-ai/flashinfer/issues/2485)
- Issue 内容已截断。
### Dao-AILab/flash-attention
## Flash-attention 更新

### 版本发布
- **v3.0.0 稳定版发布** [e2743ab](https://github.com/Dao-AILab/flash-attention/commit/e2743ab5b3803bb672b16437ba98a3b1d4576c50)
  - Luca Wehrstedt 将主版本标记为 v3.0.0 稳定版 (#2223)
  - 修改: hopper/__init__.py
  - 变更: +1 -1

### 功能改进
- **Flex Flash 文档完善** [24445c0](https://github.com/Dao-AILab/flash-attention/commit/24445c0c177f0455c076b32c41b26eee81c4e7a7)
  - Markus Hoehnerbach 添加了 flex flash 的简短说明文档 (#2231)
  - 修改: flash_attn/cute/README.md
  - 变更: +26 -0

### 问题修复
- **共享内存竞争修复** [188643b](https://github.com/Dao-AILab/flash-attention/commit/188643b82d5b06679662028558d802bcd9acfe6c)
  - Driss Guessous 修复了共享内存竞争问题 (#2229)
  - 修改: flash_attn/cute/block_sparse_utils.py, flash_attn/cute/flash_fwd_sm100.py
  - 变更: +4 -4

### 依赖更新
- **TORCH_TARGET_VERSION 替换** [ef9e6a6](https://github.com/Dao-AILab/flash-attention/commit/ef9e6a644192eb2b90155abe0372542f6d9a27b6)
  - Jane (Yuan) Xu 使用 TORCH_TARGET_VERSION 替换 TORCH_STABLE_ONLY (#2155)
  - 修改: hopper/setup.py
  - 变更: +1 -1
### sgl-project/sglang
# SGLang Newsletter - 2026-02-04

## New Features & Improvements

### Diffusion & Multimodal
- [Support layerwise offload for mova](https://github.com/sgl-project/sglang/commit/dff3ba202ad034630a8faa53ae6550a06b981e90) - Added layerwise offloading capability for MoVA model
- [Kernel fusion for Qwen-Image, WAN and HunyuanVideo](https://github.com/sgl-project/sglang/commit/4739f2e8d5732f7464d1af75d31b4d44c61783b6) - Implemented gated residual layernorm scale shift fusion
- [Clean MOVA codes](https://github.com/sgl-project/sglang/commit/0c9a0adc5329d393435097337fa113174363299e) - Refactored and optimized MoVA implementation
- [Prohibit Chinese characters usage](https://github.com/sgl-project/sglang/commit/f218234e4f19323d09f10c03505a89a82d52ebd4) - Added pre-commit checks to prevent Chinese characters in code

### Performance Optimizations
- [Optimize get_topk_ragged by fusing kernels](https://github.com/sgl-project/sglang/commit/760ae933bb3878a6897e7e552a746929c29e9d90) - Improved performance by fusing get k and k_scale operations
- [Fuse qkvbfg linear into one gemm](https://github.com/sgl-project/sglang/commit/37c33cc0aa6213fd4abcfb40c3e1d71dde484295) - Optimized attention computation with fused operations
- [Improve kv offset calculation for MHA model](https://github.com/sgl-project/sglang/commit/f730c186799d966a62531269ce46178364c85dc3) - Enhanced KV cache management for different tensor parallel sizes

### Model Support
- [Support interns1-pro model](https://github.com/sgl-project/sglang/commit/3e7ecb78a60f8e1d889cfe25c88006577783d903) - Added support for InternS1-Pro model with rotary embedding implementation
- [Add MoE fused config for Qwen3-Coder-Next-FP8](https://github.com/sgl-project/sglang/commit/efbf39583e7a716e0204b071db687145392e41b2) - Added configuration for MoE fusion on H100 TP=2

## Documentation
- [Support Markdown/Notebook-Friendly Documentation Export](https://github.com/sgl-project/sglang/commit/e616d3584737686f6d221ce3f21c67b98a936827) - Added capability to convert rat files to markdown
- [Fix misspellings & typos](https://github.com/sgl-project/sglang/commit/de6a03260f59fd33a9eeb8f67e7e6e2cf235a70f) - Corrected documentation across multiple files
- [Document SGLANG_MOONCAKE_CUSTOM_MEM_POOL](https://github.com/sgl-project/sglang/commit/c8212b9fac11d7ad3a2aa088946e1a815a618a97) - Added documentation for memory pool configuration

## Bug Fixes
- [Fix test_return_routed_experts](https://github.com/sgl-project/sglang/commit/c910829708c7b71f82e646393c6503b17501e396) - Updated to use response-level sglext
- [Fix obvious logic error](https://github.com/sgl-project/sglang/commit/c1d5cc3b24ada6857bc32af13e0a0528a01fcb70) - Corrected pyproject.toml configurations
- [Fix MockModelRunner in attention tests](https://github.com/sgl-project/sglang/commit/2e87c2bd5e43bfad57150ff878761bc6cffc0ab8) - Updated test implementations for flash attention backends
- [Fix CI workflow](https://github.com/sgl-project/sglang/commit/bdaf3de9b3ece4d81b6bb297b75272e08f819720) - Added SGLANG_IS_IN_CI environment variable to release-docs workflow

## Platform Support
- [Add kimi mi35x nightly test and stability fixes](https://github.com/sgl-project/sglang/commit/6fd878b41df0153bd28f0185920e1b2d9dcc7480) - Enhanced AMD GPU support with improved stability

## Open Issues
- [Update speculative decoding document](https://github.com/sgl-project/sglang/issues/18268) - Documentation needed for new SGLANG_ENABLE_SPEC_V2 and ngram support
- [Add doc for piecewise CUDA graph](https://github.com/sgl-project/sglang/issues/18267) - Documentation requested for --enable-piecewise-cuda-graph feature
- [Add π0/π0-FAST VLA model inference support](https://github.com/sgl-project/sglang/issues/18266) - Feature request for robotics foundation policy family
- [Integrate FA4 SM90 Decode](https://github.com/sgl-project/sglang/issues/18265) - Performance optimization needed for FA4 on SM90
- [OOM errors in AITER attention](https://github.com/sgl-project/sglang/issues/18262) - Memory allocation issue with AITER attention backend
- [Explain how to use nvidia modelopt checkpoints](https://github.com/sgl-project/sglang/issues/18261) - Documentation needed for ModelOpt offline quantization workflow
### vllm-project/vllm
## VLLM 项目更新 (2026-02-04)

### 主要提交

**性能优化**
- [优化规范解码与异步调度](https://github.com/vllm-project/vllm/commit/711edaf0d089a15df5fa2b99248c516e53929bd2) - 提升吞吐量 1.5% (Wentao Ye)
- [优化聊天完成流式性能](https://github.com/vllm-project/vllm/commit/f67ee8b859215df4b521c67b9f26e27f30c9739f) (Chauncey)
- [改变 GDN 注意力状态布局](https://github.com/vllm-project/vllm/commit/824058076c56164a3772a5f5829bd9662507e5a3) 从 [N, HV, K, V] 到 [N, HV, V, K] (Vadim Gimpelson)

**新功能**
- [添加 ColBERT 晚期交互模型支持](https://github.com/vllm-project/vllm/commit/439afa4eea14db2be232a9ce78eacc2c7bbfac77) (Ilya Boytsov)
- [为 Qwen3-Omni 添加转录支持](https://github.com/vllm-project/vllm/commit/535de06cb1d90ed1c48246a512e74c87fe1768e4) (Muhammad Hashmi)

**错误修复**
- [修复 w8a8 oneDNN 量化矩阵乘法支持 3D 输入](https://github.com/vllm-project/vllm/commit/fd03538bf97cd7f4fedd6da4584c89635878174f) (Fadi Arafeh)
- [修复 DeepSeek R1 在 B200 上的 CUTLASS MLA 问题](https://github.com/vllm-project/vllm/commit/a7be77beef5f59d9d349818b4f2860483551b255) (Chauncey)
- [修复 Qwen2.5-Omni 和 Qwen3-Omni 的视频内音频支持](https://github.com/vllm-project/vllm/commit/f8516a1ab95febcf131a37478914031f50fdd9db) (Yueqian Lin)
- [修复 Interns1-pro 初始化和流水并行](https://github.com/vllm-project/vllm/commit/192ad4648b2066ebdf1fa04ad84f24bdf0cd6533) (Isotr0py)

**CI/构建改进**
- [并行化 CPU CI 测试](https://github.com/vllm-project/vllm/commit/07daee132b30140bb7c5b28d7f8c856036d2baad) (Li, Jiang)
- [减少 torch.compile e2e 融合测试时间](https://github.com/vllm-project/vllm/commit/4d9513537d00a9b6678a2b1ed3c3566a81f7dd77) (Luka Govedič)
- [确保 ROCm AITER 环境变量设置](https://github.com/vllm-project/vllm/commit/c1395f72cd22d97eb39ecd67d9d22f2af3d20bda) (rasmith)

### 已知问题

1. **Docker 问题** - [v0.15.0 及更新版本 Docker 镜像在运行 Qwen3-Next 时出现问题](https://github.com/vllm-project/vllm/issues/33833) (jasonlizhengjian)

2. **Deepseek V3.2 基准测试失败** - [TypeError: argument 'tokens': 'NoneType' object](https://github.com/vllm-project/vllm/issues/33831) (wzhao18)

3. **Mistral3 多模态推理失败** - [离线多模态推理示例出现提示符占位符错误](https://github.com/vllm-project/vllm/issues/33828) (skavulya)

4. **流水并行问题** - [Step3p5ForCausalLM 在流水并行下失败](https://github.com/vllm-project/vllm/issues/33823) (gregporter)

5. **量化模型测试失败** - [GPTQ Marlin 量化模型测试失败](https://github.com/vllm-project/vllm/issues/33816) (mgoin)

6. **多模态评分失败** - [llm.score() 在 qwen3-vl-reranker 批量多模态输入上失败](https://github.com/vllm-project/vllm/issues/33813) (JiahuiChen-GitHub)
### NVIDIA/cutile-python
昨日无更新。

## 总结
- Diff 内容已截断以满足 prompt 预算。
- Issue 内容已截断以满足 prompt 预算。
- OpenRouter repo summarize failed for NVIDIA/cutlass: OpenRouter 429 Too Many Requests (z-ai/glm-4.5-air:free): {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"z-ai/glm-4.5-air:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations","provider_name":"Z.AI","is_byok":false}},"user_id":"user_2wqU29q2Bhpw2S2iw7Pwn8RHaXB"}
- OpenRouter repo summarize failed for flashinfer-ai/flashinfer: OpenRouter 429 Too Many Requests (z-ai/glm-4.5-air:free): {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"z-ai/glm-4.5-air:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations","provider_name":"Z.AI","is_byok":false}},"user_id":"user_2wqU29q2Bhpw2S2iw7Pwn8RHaXB"}
