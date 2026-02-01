# Daily AI Infra Report

## 往期回顾
- 暂无往期记录

---

## 最新解读 (2026-01-31)
## 摘要

本期重点关注 **FlashInfer** 对 MoE 内核的重大重构（回退了部分不稳定的特性），**SGLang** 对 All-Reduce 算法的可配置化优化及多模态模型支持的扩展，以及 **vLLM** 对 Blackwell 架构 (SM120) 的支持补全和 V1 引擎的关键修复。

---

## 具体内容分析

### 1. FlashInfer
**评估**: 本次更新包含一次大规模的代码回退，旨在提高稳定性，移除了部分实验性支持。

*   **[Revert] MoE 内核重构与 Nemotron 支持移除** ([`87a45d1`](https://github.com/flashinfer-ai/flashinfer/commit/87a45d131dc6518493718e20086cc665ed46da4f))
    *   **变更详情**: 回退了 "Support Fused MoE non gated Relu2 NVFP4 & FP4 and support Nemotron" 的特性。
    *   **代码细节**:
        *   **枚举简化**: 将 `ActivationType` (包含 Gelu, Relu, Silu, Swiglu, Relu2, Identity 等) 简化并重命名为 `GatedActType`，仅保留 `SwiGlu` 和 `GeGlu`。
          ```cpp
          // include/flashinfer/trtllm/fused_moe/runner.h
          enum class GatedActType : int64_t {
            SwiGlu = 0,
            GeGlu = 1,
          };
          ```
        *   **移除 Nemotron 支持**: 删除了 `NumNemotronExperts` (512) 的相关常量定义和逻辑。
        *   **DeepSeek 路由限制**: 将 DeepSeek 路由的 `topK` 上限从 22 重新降回 8。
          ```cpp
          // csrc/trtllm_fused_moe_runner.cu
          FLASHINFER_CHECK(topK <= 8, "For DeepSeek routing method, must have topK <= 8");
          ```
    *   **影响**: 这是一个**破坏性变更**，移除了对 Nemotron 模型和部分非 Gated 激活函数 (如 Relu2) 的支持。这表明之前的实现可能存在稳定性或性能问题，团队选择回退以保持内核健壮性。

*   **CI 流程修复** ([`a52eff1`](https://github.com/flashinfer-ai/flashinfer/commit/a52eff1e837bba6724ebbd7d98445635b8b4835d))
    *   **变更**: 强制 CI 构建时不使用缓存 (`no-cache: true`, `pull: true`)，解决 Docker 构建使用旧状态的问题。

### 2. SGLang
**评估**: 增强了底层通信库的可控性，并快速跟进支持了最新的多模态模型。

*   **优化 Custom All-Reduce** ([`afebb7a`](https://github.com/sgl-project/sglang/commit/afebb7ab7893869ec8cf7ed6dd6f06981bc8ccef))
    *   **变更**: 引入环境变量 `SGLANG_CUSTOM_ALLREDUCE_ALGO`，允许用户强制指定 All-Reduce 算法（`oneshot`/`1stage` 或 `twoshot`/`2stage`），便于针对不同硬件/负载微调性能。
    *   **代码片段**:
      ```cpp
      // sgl-kernel/csrc/allreduce/custom_all_reduce.cuh
      const char* env_algo = std::getenv("SGLANG_CUSTOM_ALLREDUCE_ALGO");
      if (env_algo != nullptr) {
        if (std::strcmp(env_algo, "1stage") == 0 || ...) { force_1stage = true; }
        // ...
      }
      ```

*   **模型支持扩展**
    *   **MiniCPM-V 4.5** ([`4d28cda`](https://github.com/sgl-project/sglang/commit/4d28cda007b0ff34e9f936c4ad1b8ec08ed0574b)): 增加了对最新 MiniCPM-V 4.5 模型的支持，修改了 `minicpmv.py`。
    *   **Qwen3-Next Eagle3** ([`3ca29df`](https://github.com/sgl-project/sglang/commit/3ca29dffc72ac8a1000d3dad988fd96ab3339b9d)): 支持 Qwen3-Next 的 Eagle3 推测解码。
    *   **Ling Flash v2.0** ([`855dd05`](https://github.com/sgl-project/sglang/commit/855dd0546ce7ba460f23bed08c469e63955228e2)): 支持 Ling Flash v2.0 的 Eagle3。

*   **GLM4v RoPE 优化** ([`4ea4f2a`](https://github.com/sgl-project/sglang/commit/4ea4f2a20c4b7d6d78220ac5e1c80aa1d288c9fb))
    *   **变更**: 优化了 GLM4v 的 `get_rope_index` 实现，并添加了基准测试脚本 `benchmark/bench_rope/benchmark_rope_index.py`。

### 3. vLLM
**评估**: 修复了 V1 引擎的关键 Bug，并开始布局下一代硬件支持。

*   **支持 SM120 (RTX Blackwell)** ([`0797811`](https://github.com/vllm-project/vllm/commit/079781177ae4c9fba429bf093cae73cf4cfae7a8))
    *   **变更**: 为 FlashInfer CUTLASS NVFP4 MoE 内核添加了 SM120 (Blackwell 架构) 的支持。
    *   **代码片段**:
      ```python
      # vllm/model_executor/layers/fused_moe/cutlass_moe.py
      # 添加了对 capability == 120 的判断支持
      ```

*   **V1 引擎修复与重构**
    *   **Redo #33110** ([`21997f4`](https://github.com/vllm-project/vllm/commit/21997f45b10c17f44276cf3872e5f85c61dc7dfd)): 重新引入了之前回退的更改，但增加了线程限制，修复了多线程处理输入时的潜在竞争问题。
    *   **Renderer 抽象化** ([`a358e4d`](https://github.com/vllm-project/vllm/commit/a358e4dffe10ebff2c44434958bcc190ad2e5542)): 将 `Renderer` 重构为抽象类，统一了 DeepSeek, Grok, HF, Mistral 等不同模型的渲染接口。

*   **量化重构** ([`b5f8c30`](https://github.com/vllm-project/vllm/commit/b5f8c3092d1e1466b2b9c516fb39e5b2c15e774b))
    *   **变更**: 将所有 W8A8 量化逻辑统一收敛到 `QuantFP8` 类中，清理了 `input_quant_fp8.py` 和 `fp8_utils.py`。

---

## Issues 看点

### FlashInfer
*   **[Bug] CI docker release workflow uses cached states** ([#2453](https://github.com/flashinfer-ai/flashinfer/issues/2453)): 指出 CI 构建过程中使用了 Docker 缓存导致构建内容不是最新的。**已通过 commit [`a52eff1`](https://github.com/flashinfer-ai/flashinfer/commit/a52eff1e837bba6724ebbd7d98445635b8b4835d) 修复。**

### SGLang
*   **[Feature] Support bf16 gemm from DeepGemm** ([#18028](https://github.com/sgl-project/sglang/issues/18028)): 社区请求集成 DeepGemm 的 bf16 gemm 实现，以利用其针对 SM90/SM100 的优化，可能带来显著的性能提升。
*   **[Bug] Qwen-Image-Edit-2511-4bit not work** ([#18029](https://github.com/sgl-project/sglang/issues/18029)): 4-bit 量化版本的 Qwen-Image-Edit 模型加载失败，反映出量化模型在多模态任务上的兼容性问题。

### vLLM
*   **[Draft][RFC]: Introduce ATOM as model implementation backend for AMD GPU** ([#33478](https://github.com/vllm-project/vllm/issues/33478)): 提议引入 AMD 优化的 ATOM 后端，集成 `aiter` (算子库) 和 `mori` (通信库)，旨在显著提升 vLLM 在 AMD GPU 上的竞争力。
*   **[Feature]: Add INT8 Support for KV Cache Quantization** ([#33480](https://github.com/vllm-project/vllm/issues/33480)): 用户请求在 KV Cache 量化中增加 INT8 支持，以适配不支持 FP8 的旧款 GPU (如 A100, RTX 4090)。
*   **[Bug]: V1 Engine TypeError for Qwen2** ([#33470](https://github.com/vllm-project/vllm/issues/33470)): V1 引擎在加载 Qwen2 时报错 `FlashAttentionImpl.__init__() got an unexpected keyword argument 'layer_idx'`，这是一个阻碍性的兼容性 Bug。

---

## 总结

1.  **FlashInfer 收缩战线**: 通过 Commit [`87a45d1`](https://github.com/flashinfer-ai/flashinfer/commit/87a45d131dc6518493718e20086cc665ed46da4f)，FlashInfer 移除了 Nemotron 支持并简化了激活函数类型，这表明在追求新特性支持的同时，团队开始重视核心 MoE 内核的稳定性与代码维护性。
2.  **SGLang 拥抱新模型与可控性**: SGLang 在一天内密集增加了 MiniCPM-V 4.5 ([`4d28cda`](https://github.com/sgl-project/sglang/commit/4d28cda007b0ff34e9f936c4ad1b8ec08ed0574b)) 和 Qwen3-Next Eagle3 ([`3ca29df`](https://github.com/sgl-project/sglang/commit/3ca29dffc72ac8a1000d3dad988fd96ab3339b9d)) 的支持，同时通过暴露 All-Reduce 算法配置 ([`afebb7a`](https://github.com/sgl-project/sglang/commit/afebb7ab7893869ec8cf7ed6dd6f06981bc8ccef)) 给高级用户提供了更强的性能调优能力。
3.  **vLLM 布局 Blackwell 与 AMD**: vLLM 不仅通过 Commit [`0797811`](https://github.com/vllm-project/vllm/commit/079781177ae4c9fba429bf093cae73cf4cfae7a8) 早早支持了 SM120 (RTX 50 系列) 的 MoE 量化算子，还在 Issue [#33478](https://github.com/vllm-project/vllm/issues/33478) 中讨论引入 AMD ATOM 后端，显示出对多硬件生态的积极支持。
