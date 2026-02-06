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
## 本周关键更新概览

- **NVIDIA/CUTLASS** 迎来 **4.4 版**，在 CuTe DSL 中加入对 **CUDA 13.1**、**GB300** 与 **cute.experimental** 的原生支持，并实现 **AoT 编译**。多处 kernel 加入 `wait_on_dependent_grids()` 防止前置 kernel 的全局内存未刷新导致错误，同时对 Blackwell SM103 的 block‑scaled GEMM 示例进行扩展。\
- **flashinfer‑ai/flashinfer** 完成 **0.6.3** 版本升级，新增对 **DeepSeek V3** 路由的检查、**NO_SMEM** epilogue 对齐校验、以及多 GPU/多节点测试脚本。大量 **MoE**、**GDN**、**Diffusion** 相关 kernel 通过 Jinja 模板和外部 inc 文件优化，显著降低 Hopper 上的编译时间并提升 FP8/FP4 量化性能。\
- **Dao‑AILab/flash‑attention** 主要修复了 **hdim=192** 场景下的共享内存计算错误，并将主分支标记为 **v3.0.0**，同步更新了文档与依赖声明。\
- **sgl‑project/sglang** 本期提交量最大，涵盖 **Speculative Decoding 文档完善**、**Qwen3‑VL 位置编码插值重构**、**NixlKVManager** 与 **Mooncake** KV 管理优化、**XPU MoE** 支持、以及 **Diffusion** 中的层级 offload 与冗余显存清理。新增 **Markdown/Notebook‑友好文档导出** 与 **piecewise CUDA graph** 说明，提升了可维护性与用户体验。\
- **vllm‑project/vllm** 继续强化 **分布式与 FP8** 能力：修复 DeepSeek v3.2 tokenizer 返回 `None`、在 ROCm 上加入 **skinny‑gemm padding**、实现 **safetensors 排序** 以保证加载确定性、以及 **NCCL 权重同步 API**（支持 RLHF 场景）。此外，统一了 **Spec‑Decode 并行草稿模型**、完善了 **LoRA FP8**、以及在 KV 连接器中加入跨层缓存布局和调试日志。

## 具体内容分析
### deepseek-ai/DeepGEMM
昨日无更新。
### deepseek-ai/FlashMLA
昨日无更新。
### NVIDIA/cutlass
提交  
- **CUTLASS 4.4 版本更新**：在 CuTe DSL 中加入对 CUDA Toolkit 13.1 的支持、GB300 兼容、实验性 `cute.experimental` 高层抽象以及 AoT 编译功能。对应 PR **[#2999](https://github.com/NVIDIA/cutlass/pull/2999)**，核心提交 **[`6b3e607`](https://github.com/NVIDIA/cutlass/commit/6b3e607b852f1543dc21323155a2ad71473c8642)**。  

  **文件变更概览**  
  - `CHANGELOG.md`、`README.md`：新增 “CuTe DSL 现在支持 CUDA 13.1”、`GB300`、`cute.experimental` 等特性说明。  
    ```diff
    + CuTe DSL now supports CUDA toolkit 13.1!
    + GB300 is now supported in CuTe DSL with CTK 13.1
    + cute.experimental: introduce a higher‑level, composable layer …
    ```
  - `examples/python/CuTeDSL/blackwell/sm103_dense_blockscaled_gemm_persistent.py`（新增 3 KB）：演示 Blackwell SM103 上的 3×FP4 block‑scaled GEMM。  
  - `examples/python/CuTeDSL/experimental/blackwell/` 目录下新增多套实验示例（`dense_gemm.py`、`dense_gemm_2sm.py`、`dense_gemm_cute_pipeline.py`、`dense_gemm_ptr_array.py`、`dense_block_scaled_gemm.py`），展示基于 `cute.experimental` 的管线、指针数组、双 SM 等新特性。  
    ```python
    @cute.jit
    def dense_gemm(...):
        # 使用 cute.experimental 的 pipeline 抽象
        pipeline = cute_ext.Pipeline(...)
    ```
  - `examples/python/CuTeDSL/experimental/ampere/memcpy_simt_universal_copy.py`（新增 155 行）：提供 SIMT 复制的参考实现。  
  - `include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized.hpp`、`..._input_transform.hpp`、`..._mma_transform.hpp`、`..._tile_scheduler_group.hpp`、`..._blockscaled_gemm_array_tma_warpspecialized.hpp`：在多个 kernel 中加入 `cutlass::arch::wait_on_dependent_grids();` 以防止前置 kernel 的全局内存未刷新导致的错误，并添加相应注释。  
    ```cpp
    // Ensure that the prefetched kernel does not touch
    // unflushed global memory prior to this instruction.
    cutlass::arch::wait_on_dependent_grids();
    ```
  - `include/cutlass/gemm/kernel/sm90_gemm_array_tma_warpspecialized_cooperative.hpp`：重构调度器构造顺序，确保在调用 `wait_on_dependent_grids()` 之前完成调度器初始化。  

Issues  
- **性能开销**：Blackwell GEMM 循环中频繁调用 `cuTensorMapEncodeTiled` 与 `cudaGetDriverEntryPoint`，导致循环时间显著增加。详情请见 **[#3003](https://github.com/NVIDIA/cutlass/issues/3003)**。  
- **Segmentation fault**：在 FP8 共享内存张量上调用 `cute.print_tensor` 时出现段错误。详情请见 **[#3002](https://github.com/NVIDIA/cutlass/issues/3002)**。  
- **导入错误**：`nvidia-cutlass-dsl v4.4.0.dev1` 在导入时抛出异常，影响基本示例运行。详情请见 **[#3001](https://github.com/NVIDIA/cutlass/issues/3001)**。  
- **兼容性询问**：用户询问 CUTLASS 是否支持 NVIDIA Tesla V100 GPU。详情请见 **[#3000](https://github.com/NVIDIA/cutlass/issues/3000)**。
### flashinfer-ai/flashinfer
## 提交

| SHA | PR | 关键文件 | 变更概览 |
|-----|----|----------|----------|
| `d0886ce` | [#2497](https://github.com/flashinfer-ai/flashinfer/pull/2497) | `version.txt` | 将版本号从 **0.6.2** 更新为 **0.6.3**。<br>`-0.6.2` → `+0.6.3` |
| `1e9b237` | [#2502](https://github.com/flashinfer-ai/flashinfer/pull/2502) | `csrc/trtllm_fused_moe_kernel_launcher.cu` | 新增对非 DeepSeekV3 路由的检查：<br>`if (routing != DeepSeekV3) { TVM_FFI_ICHECK(args->n_group <= 1); TVM_FFI_ICHECK(args->topk_group <= 1); }`<br>并在 DeepSeekV3 场景下强制 `n_group != 0`。 |
| `8655234` | [#2495](https://github.com/flashinfer-ai/flashinfer/pull/2495) | `csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch.h` | 为 **NO_SMEM** epilogue 添加输出 N 对齐检查，确保 `N` 为 256‑bit 对齐。<br>（patch 完整，未截断） |
| `1748eb5` | [#2484](https://github.com/flashinfer-ai/flashinfer/pull/2484) | `benchmarks/README.md`、`benchmarks/flashinfer_benchmark.py`、`benchmarks/routines/rope.py`、`benchmarks/routines/sampling.py` 等 | README 中加入 **Sampling** 与 **RoPE** 支持描述；`flashinfer_benchmark.py` 扩展入口以调用新 routine；新增 `rope.py`、`sampling.py` 实现对应基准。 |
| `cdbb2c3` | [#2488](https://github.com/flashinfer-ai/flashinfer/pull/2488) | `.github/workflows/nightly-release.yml`、`.github/workflows/release.yml` | 删除多余的文件权限修改，修复 CI 在 `ci‑infra` runner 上的权限错误。 |
| `e284274` | [#2462](https://github.com/flashinfer-ai/flashinfer/pull/2462) | 多个文件（如 `include/flashinfer/trtllm/fused_moe/DevKernel.h`、`csrc/trtllm_fused_moe_kernel_launcher.cu` 等） | 支持 **非门控 ReLU2** 的 NVFP4/FP8 Fused MoE；加入对 **Nemotron** 的兼容；更新 kernel 调度与 runner 实现。 |
| `9bf007d` | [#2410](https://github.com/flashinfer-ai/flashinfer/pull/2410) | `scripts/task_run_unit_tests.sh`、`scripts/task_test_multi_gpu_comm_kernels.sh`、`scripts/task_test_multi_node_comm_kernels.sh`、`scripts/test_utils.sh` | 新增多节点/多 GPU 测试脚本；删除已废弃的 `task_test_blackwell_kernels.sh`。 |
| `567ded1` | [#2481](https://github.com/flashinfer-ai/flashinfer/pull/2481) | `tests/mamba/utils.py`（文件重命名） | 将 `tests/mamba/test_utils.py` 重命名为 `utils.py`，解决 CI 测试发现问题。 |
| `f84ac1c` | [#2479](https://github.com/flashinfer-ai/flashinfer/pull/2479) | `benchmarks/bench_trtllm_gen_mla.py`、`benchmarks/routines/attention.py` | 修正 MLA 基准中的内存带宽计算公式，提升报告的准确性。 |
| `6ae5bfe` | [#2422](https://github.com/flashinfer-ai/flashinfer/pull/2422) | `csrc/flat_prefill_kernel_delta_rule_sm90_extern.inc`、`csrc/gdn_prefill_launcher.cu`、`include/flashinfer/flat/hopper/collective/flat_collective_load.hpp` 等 | 通过新增外部 inc 文件和 Jinja 模板，显著降低 Hopper 架构上 GDN Prefill 编译时间；同步更新文档字符串。 |

> **说明**：所有列出的 PR 均指向对应的 Pull Request 页面，确保读者可直接查看完整代码变更。若某个文件的 diff 被截断（`patch_truncated: true`），已在表格中注明并提供关键代码片段的概览。

## Issues

| 编号 | 标题 | 作者 | 创建时间 | 简要描述 |
|------|------|------|----------|----------|
| [#2496](https://github.com/flashinfer-ai/flashinfer/issues/2496) | **[Perf] mxfp4 quantize kernel is slow** | bkryu | 2026‑02‑04 22:21:14 | mxfp4 量化核的执行时间约为 0.456 ms，明显慢于 nvfp4（0.036 ms），希望通过 CuTe DSL 优化提升性能。 |
| [#2494](https://github.com/flashinfer-ai/flashinfer/issues/2494) | **XQA always sets dynamic smem size on rank 0** | zackangelo | 2026‑02‑04 20:17:11 | 在 `csrc/xqa/mha.cu` 中动态共享内存大小始终在 GPU 0 上设置，导致多 GPU 环境下出现 `CUDA_INVALID_ARGUMENT` 错误。 |
| [#2493](https://github.com/flashinfer-ai/flashinfer/issues/2493) | **Perf improvement for CuTe DSL GDN Decode Kernel** | kahyunnam | 2026‑02‑04 19:21:43 | 通过 CuTe DSL 重写 `gated_delta_rule_decode_pretranspose`，已在 PR #2370 中展示显著性能提升。 |
| [#2491](https://github.com/flashinfer-ai/flashinfer/issues/2491) | **Add CuTe DSL backends for RoPE APIs** | kahyunnam | 2026‑02‑04 17:53:04 | 计划将 RoPE 相关 API 从 CUDA kernel 迁移到 CuTe DSL，以便统一后端并潜在提升性能。 |
| [#2490](https://github.com/flashinfer-ai/flashinfer/issues/2490) | **[Bug] GDN prefill kernel produces NaN** | ZJY0516 | 2026‑02‑04 15:35:14 | 在 vLLM 中调用 `flashinfer.gdn_prefill.chunk_gated_delta_rule` 时输出出现 NaN，已提供调试数据（链接在 Issue 中）。 |
| [#2486](https://github.com/flashinfer-ai/flashinfer/issues/2486) | **[top_k_page_table_transform bug] Illegal memory access in RadixTopKKernel_Unified** | huangzhilin-hzl | 2026‑02‑04 03:18:03 | `top_k_page_table_transform` 在特定大序列长度下触发非法内存访问，复现代码已给出。 |
| [#2485](https://github.com/flashinfer-ai/flashinfer/issues/2485) | **[bug] Accuracy Issue: GLM-4.7 ModelOpt NVFP4 produces garbage output with FlashInfer TRTLLM MoE backend** | nvyutwu | 2026‑02‑04 00:24:47 | 使用 FlashInfer TRTLLM MoE 后端加载 GLM‑4.7 NVFP4 检查点时，模型输出全是乱码，影响 vLLM 与 SGLang 两个推理框架。 |

> **备注**：部分 Issue 的正文被截断或缺失（如 #2490、#2485），已在表格中标明，后续可补充完整信息。
### Dao-AILab/flash-attention
提交：

- `f1284cf` – 修复 **hdim=192** 时共享内存计算错误。涉及 `flash_attn/cute/flash_fwd_sm100.py`（+7 行，‑1 行）。  
  [PR #2235](https://github.com/Dao-AILab/flash-attention/pull/2235) | [commit](https://github.com/Dao-AILab/flash-attention/commit/f1284cff5d2b2ad4160ceefaf096a800502d16fd)

- `e2743ab` – 将当前主分支标记为 **v3.0.0** 稳定版。修改 `hopper/__init__.py`（+1 行，‑1 行）。  
  [PR #2223](https://github.com/Dao-AILab/flash-attention/pull/2223) | [commit](https://github.com/Dao-AILab/flash-attention/commit/e2743ab5b3803bb672b16437ba98a3b1d4576c50)

- `24445c0` – 为 **Flex Flash** 添加简短 README。更新 `flash_attn/cute/README.md`（+26 行）。  
  [PR #2231](https://github.com/Dao-AILab/flash-attention/pull/2231) | [commit](https://github.com/Dao-AILab/flash-attention/commit/24445c0c177f0455c076b32c41b26eee81c4e7a7)

- `ef9e6a6` – 使用 `TORCH_TARGET_VERSION` 替代 `TORCH_STABLE_ONLY`。修改 `hopper/setup.py`（+1 行，‑1 行）。  
  [PR #2155](https://github.com/Dao-AILab/flash-attention/pull/2155) | [commit](https://github.com/Dao-AILab/flash-attention/commit/ef9e6a644192eb2b90155abe0372542f6d9a27b6)

- `188643b` – 修复共享内存竞争问题。涉及 `flash_attn/cute/block_sparse_utils.py`（+1 行，‑1 行）和 `flash_attn/cute/flash_fwd_sm100.py`（+3 行，‑3 行）。  
  [PR #2229](https://github.com/Dao-AILab/flash-attention/pull/2229) | [commit](https://github.com/Dao-AILab/flash-attention/commit/188643b82d5b06679662028558d802bcd9acfe6c)

Issues：暂无。
### sgl-project/sglang
**本期代码更新**  

- **[Doc] add a summary section for spec decode document**  
  - PR: <https://github.com/sgl-project/sglang/commit/ef1d0ea8854155983796ba200addd4c6db0e2e41>  
  - 作者: shuwenn，2026‑02‑05  
  - 变更: `docs/advanced_features/speculative_decoding.ipynb`（+38 / –1 行）  
  - 说明: 为 Speculative Decoding 文档新增 “Summary” 小节，帮助快速了解功能概览。  
  - 代码片段: *patch 已截断，未展示具体代码*  

- **[Doc] refine spec decode docs for SpecV2/STANDALONE/NGRAM**  
  - PR: <https://github.com/sgl-project/sglang/commit/8b21dd4b774a4fd79662d11cfa3f0478bfb33f3f>  
  - 作者: shuwenn，2026‑02‑05  
  - 变更: 同上 `speculative_decoding.ipynb`（+202 / –1 行）  
  - 说明: 对 SpecV2、STANDALONE、N‑GRAM 三种推理模式的文档进行细化，补充使用示例。  
  - 代码片段: *patch 已截断*  

- **Refactor(qwen3‑vl) optimize position encoding interpolation**  
  - PR: <https://github.com/sgl-project/sglang/commit/6a4b81e2d9fc0da998d296b86fbacc7e1594988d>  
  - 作者: aaaandychen，2026‑02‑05  
  - 变更: `python/sglang/srt/models/qwen3_vl.py`（+123 / –23 行），`test/srt/test_embed_interpolate_unittest.py`（+3 / –2 行）  
  - 说明: 重构 Qwen3‑VL 位置编码插值实现，提升多尺度视觉特征对齐精度。  
  - 代码片段（`qwen3_vl.py`）:  
    ```python
    # 新增的插值函数
    def interpolate_position_encoding(...):
        # 具体实现省略
    ```  

- **Fix flaky test_frequency_penalty_reduces_word_repetition**  
  - PR: <https://github.com/sgl-project/sglang/commit/d22163eb8c02169f5e39de1f5f3d09023c1e3c25>  
  - 作者: Alison Shao，2026‑02‑05  
  - 变更: `test/registered/sampling/test_penalty.py`（+13 / –3 行）  
  - 说明: 为频率惩罚测试加入确定性随机种子，消除间歇性失败。  

- **NixlKVManager optimizations**  
  - PR: <https://github.com/sgl-project/sglang/commit/498d8d068096dca5f17f28d54c5464706ec22d7c>  
  - 作者: ovidiusm，2026‑02‑05  
  - 变更: `python/sglang/srt/disaggregation/nixl/conn.py`（+80 / –60 行）  
  - 说明: 优化 KV 管理器的连接逻辑，降低内存占用并提升并发吞吐。  

- **[diffusion] feat: allow T5's TP Group to reuse the transformer's SP Group**  
  - PR: <https://github.com/sgl-project/sglang/commit/b639779dd89aea042d4a274f79190126cf841330>  
  - 作者: wxy，2026‑02‑05  
  - 变更: 多个文件涉及配置与运行时层（共 242 行改动）  
  - 说明: 在 Diffusion 任务中，T5 编码器的 Tensor‑Parallel 组可共享 Transformer 的 Sharding‑Parallel 组，简化部署配置。  

- **throw error if got adapter with added_tokens**  
  - PR: <https://github.com/sgl-project/sglang/commit/3f32a5831d329320972403931819ca037b071830>  
  - 作者: Glen Liu，2026‑02‑05  
  - 变更: `python/sglang/srt/lora/lora_manager.py`（+4 行）  
  - 说明: 当 LoRA 适配器携带 `added_tokens` 时抛出明确错误，防止隐式冲突。  

- **[Kernel] Add JIT apply_rope_with_cos_sin_cache_inplace**  
  - PR: <https://github.com/sgl-project/sglang/commit/2eb4359ada96d94a9f4047afec0fae76722766f7>  
  - 作者: pansicheng，2026‑02‑05  
  - 变更: 新增 `rope.cuh`（+656 行）和 `rope.py`（+236 行），以及对应单元测试。  
  - 说明: 实现基于缓存的原位 RoPE 计算，显著降低旋转嵌入的算子开销。  
  - 代码片段（`rope.py`）:  
    ```python
    @torch.jit.script
    def apply_rope_with_cos_sin_cache_inplace(q, k, cos, sin):
        # 直接在原位修改 q/k
        ...
    ```  

- **docker: add patch to increase GPU deepep timeout**  
  - PR: <https://github.com/sgl-project/sglang/commit/8f8c1724ae2f4dd4ba40b18d246766da3545e4eb>  
  - 作者: ishandhanani，2026‑02‑05  
  - 变更: `docker/Dockerfile`（+3 行）  
  - 说明: 将容器内部的 DeepSpeed 超时阈值提升，防止长序列推理被意外中止。  

- **[PD] Minor code cleanup for mooncake backend**  
  - PR: <https://github.com/sgl-project/sglang/commit/afae4c7178f57a60f45c0685f41e1ac42ae4b607>  
  - 作者: Shangming Cai，2026‑02‑05  
  - 变更: `python/sglang/srt/disaggregation/mooncake/conn.py`（+27 / –26 行）  
  - 说明: 清理 Mooncake 后端连接实现，提升代码可读性。  

- **[piecewise graph]: support MiniMax-M2**  
  - PR: <https://github.com/sgl-project/sglang/commit/079fc8f3c591a43316d98fae6f108ce03d0eeeb3>  
  - 作者: zhangheng，2026‑02‑05  
  - 变更: `fp8_kernel.py` 与 `minimax_m2.py`（共 35 行）  
  - 说明: 为 MiniMax‑M2 模型加入分段 CUDA 图支持，提升大模型推理的图调度效率。  

- **[FIX] Always support TP > 4 for FP4 Gemm**  
  - PR: <https://github.com/sgl-project/sglang/commit/3f1df322f9df60e228a567bbd2fa3064d4b5f269>  
  - 作者: danielafrimi，2026‑02‑05  
  - 变更: `python/sglang/srt/layers/quantization/modelopt_quant.py`（+154 / –6 行）  
  - 说明: 修复 FP4 GEMM 在 Tensor‑Parallel 大于 4 时的兼容性问题。  

- **[XPU] Integrate MoE and minor improvements in XPU attention backend**  
  - PR: <https://github.com/sgl-project/sglang/commit/368936a62bdd533cdfe919d8a055c73cc2e34712>  
  - 作者: Meng, Hengyu，2026‑02‑05  
  - 变更: 多文件涉及 MoE Triton 实现、XPU 公共工具及新增 XPU 测试（共 240 行）  
  - 说明: 在 XPU 后端加入混合专家（MoE）支持，并优化注意力算子。  

- **[Diffusion] Support layerwise offload for mova**  
  - PR: <https://github.com/sgl-project/sglang/commit/dff3ba202ad034630a8faa53ae6550a06b981e90>  
  - 作者: Xiaoyu Zhang，2026‑02‑05  
  - 变更: `python/sglang/multimodal_gen/runtime/server_args.py`（+4 / –3 行）  
  - 说明: 为 MOVA 扩散模型提供分层显存卸载选项，降低单卡显存需求。  

- **Fix test_return_routed_experts to use response-level sglext**  
  - PR: <https://github.com/sgl-project/sglang/commit/c910829708c7b71f82e646393c6503b17501e396>  
  - 作者: Alison Shao，2026‑02‑05  
  - 变更: `test/registered/rl/test_return_routed_experts.py`（+6 / –8 行）  
  - 说明: 测试改为基于响应级别的 SGLang 扩展，提升覆盖率。  

- **Support Markdown/Notebook‑Friendly Documentation Export**  
  - PR: <https://github.com/sgl-project/sglang/commit/e616d3584737686f6d221ce3f21c67b98a936827>  
  - 作者: Kun Lin，2026‑02‑05  
  - 变更: `docs/Makefile`（+9 / –3 行）  
  - 说明: 新增 Make 目标，将 `.rat` 文档自动转换为 Markdown/Notebook，便于下游集成。  

- **[docs] fix misspellings & typos**  
  - PR: <https://github.com/sgl-project/sglang/commit/de6a03260f59fd33a9eeb8f67e7e6e2cf235a70f>  
  - 作者: rinbaro，2026‑02‑05  
  - 变更: 多个文档文件（共 46 行）  
  - 说明: 统一拼写、纠正错别字，提升文档专业度。  

- **[PD] doc: Document SGLANG_MOONCAKE_CUSTOM_MEM_POOL**  
  - PR: <https://github.com/sgl-project/sglang/commit/c8212b9fac11d7ad3a2aa088946e1a815a618a97>  
  - 作者: Teng Ma，2026‑02‑05  
  - 变更: `docs/advanced_features/pd_disaggregation.md` 与环境变量文档（+4 行）  
  - 说明: 说明 Mooncake 自定义内存池变量及可选取值。  

- **[PD] improve kv offset calculation for MHA model with different tp size**  
  - PR: <https://github.com/sgl-project/sglang/commit/f730c186799d966a62531269ce46178364c85dc3>  
  - 作者: Ch3ngY1，2026‑02‑05  
  - 变更: `python/sglang/srt/disaggregation/mooncake/conn.py`（+31 / –75 行）  
  - 说明: 修正多卡 MHA 场景下 KV 偏移计算，避免跨 TP 维度的错位。  

- **[diffusion] prohibit Chinese characters usage**  
  - PR: <https://github.com/sgl-project/sglang/commit/f218234e4f19323d9f10c03505a89a82d52ebd4>  
  - 作者: Mick，2026‑02‑05  
  - 变更: `.pre-commit-config.yaml`（+8 行）以及 T5 编码器文件（–3 行）  
  - 说明: 在代码库中强制禁止中文字符，防止潜在编码问题。  

- **fix kimi k2.5's moe gemm config init**  
  - PR: <https://github.com/sgl-project/sglang/commit/599c5f4922579742a0c65a4c2fb4503dd63f7ae3>  
  - 作者: yinghui，2026‑02‑05  
  - 变更: `python/sglang/srt/managers/scheduler.py`（+6 / –1 行）  
  - 说明: 修正 Kimi‑K2.5 MoE GEMM 配置初始化的路径错误。  

- **Add MoE fused config for Qwen3‑Coder‑Next‑FP8 on H100 TP=2**  
  - PR: <https://github.com/sgl-project/sglang/commit/efbf39583e7a716e0204b071db687145392e41b2>  
  - 作者: Mohammad Miadh Angkad，2026‑02‑04  
  - 变更: 新增 `configs/triton_3_5_1/...json`（+146 行）  
  - 说明: 为 Qwen3‑Coder‑Next‑FP8 在 H100（TP=2）提供预设 MoE 配置文件。  

- **model: support interns1‑pro**  
  - PR: <https://github.com/sgl-project/sglang/commit/3e7ecb78a60f8e1d889cfe25c88006577783d903>  
  - 作者: RunningLeon，2026‑02‑04  
  - 变更: 新增模型 `interns1pro.py`（+252 行），以及对应处理器 `interns1pro.py`（+118 行），并在配置与 RotaryEmbedding 中做少量改动（+199 行）  
  - 说明: 引入 Interns1‑Pro 机器人策略模型，支持新一代 VLA 推理。  

- **entrypoint: support passing spaces_between_special_tokens per request**  
  - PR: <https://github.com/sgl-project/sglang/commit/a6f53cc5e3ac7eb8ae9e4236d9834897684505ad>  
  - 作者: RunningLeon，2026‑02‑04  
  - 变更: `python/sglang/srt/entrypoints/openai/protocol.py`（+8 行）  
  - 说明: API 现在可以在单次请求中自定义特殊 token 之间的空格间隔。  

- **[diffusion] fix: fix the bug of redundant memory usage on GPU‑0**  
  - PR: <https://github.com/sgl-project/sglang/commit/4c403045ec690dbcf3b63a941356e004201ba337>  
  - 作者: wxy，2026‑02‑04  
  - 变更: `python/sglang/multimodal_gen/runtime/platforms/cuda.py`（+3 行）  
  - 说明: 修复 GPU‑0 上的冗余显存分配，提升多卡部署的显存利用率。  

- **[diffusion] chore: clean MOVA codes**  
  - PR: <https://github.com/sgl-project/sglang/commit/0c9a0adc5329d393435097337fa113174363299e>  
  - 作者: Zhang Yiyang (SII)，2026‑02‑04  
  - 变更: `mova_audio_dit.py`（+1 / –59 行），`mova_video_dit.py`（+1 / –61 行），以及调度器 `flow_match_pair.py`（+75 / –47 行）  
  - 说明: 大幅删减冗余代码，简化 MOVA 音视频模型实现。  

**本期 Issue 关注**  

- **[Docs] Update speculative decoding document**  
  - 链接: <https://github.com/sgl-project/sglang/issues/18268>  
  - 简述: 需求在文档中补全 `SGLANG_ENABLE_SPEC_V2`、`ngram` 等新特性的使用说明。  

- **[Docs] Add doc for piecewise CUDA graph**  
  - 链接: <https://github.com/sgl-project/sglang/issues/18267>  
  - 简述: 计划新增 `--enable-piecewise-cuda-graph` 的设计与使用文档，提升特性可见度。  

- **[Feature] Add π/π‑FAST VLA model inference support**  
  - 链接: <https://github.com/sgl-project/sglang/issues/18266>  
  - 简述: 讨论在
### vllm-project/vllm
- **91a07ff** – *Fix DeepSeek v3.2 tokenizer outputting None issue*  
  <https://github.com/vllm-project/vllm/commit/91a07ff6187e7308794b2a4863ab9e1f821ed464>  
  - **文件**: `vllm/tokenizers/detokenizer_utils.py`（+4 行）  
  - 关键改动：在 `detokenize` 流程中加入空值检查，防止返回 `None`。  

- **d5c4800** – *Adds padding and perf improvements to wvSplitK_fp8*  
  <https://github.com/vllm-project/vllm/commit/d5c4800112c12bbcd4955858ef1b415c16ae16e7>  
  - **文件**: `csrc/rocm/skinny_gemms.cu`（+126 / -174 行）  
  - **文件**: `tests/kernels/quantization/test_rocm_skinny_gemms.py`（+40 / -52 行）  
  - **文件**: `vllm/model_executor/layers/quantization/kernels/scaled_mm/rocm.py`（±3 行）  
  - 关键改动：在 ROCm skinny‑gemm 实现中加入 `padding`，提升 FP8 分割‑K 的吞吐；相应单元测试同步更新。  

- **42d5d70** – *Sort safetensors files to ensure deterministic loading order*  
  <https://github.com/vllm-project/vllm/commit/42d5d705f93b254179e062003e8504fbe04f1b30>  
  - **文件**: `vllm/model_executor/model_loader/weight_utils.py`（+11 / -2 行）  
  - 关键改动：对 `safetensors` 文件列表进行排序，保证模型加载顺序可复现。  

- **116880a** – *Make MM batching more robust*  
  <https://github.com/vllm-project/vllm/commit/116880a5a0af3a226e29f4716c484a4cc8422fc1>  
  - **文件**: 多处 CI 配置（`.buildkite/*.yaml`）微调（共 +6 / -3 行）  
  - **新增**: `tests/multimodal/media/test_connector.py`（+320 行）  
  - **修改**: `tests/multimodal/*`、`vllm/model_executor/models/*`、`vllm/multimodal/*` 等，累计 **+1053 / -428 行**，强化多模态模型的矩阵乘批处理逻辑。  

- **4145e50** – *Fix DSV3.2 NVFP4*  
  <https://github.com/vllm-project/vllm/commit/4145e50d854e3182c22bad99ec011c283b9c493f>  
  - **文件**: `vllm/model_executor/layers/attention/mla_attention.py`（+4 / -2 行）  
  - 关键改动：修正 NVFP4 计算路径中的索引错误。  

- **20f5d18** – *Rename `translations` to `speech_to_text` for OAI serving component*  
  <https://github.com/vllm-project/vllm/commit/20f5d185a6f570b74ab403577cd4eabe44d14496>  
  - **文件**: `vllm/entrypoints/openai/api_server.py`、`vllm/entrypoints/openai/engine/serving.py`（各 ±4 行）  
  - **重命名目录**: `speech_to_text`（原 `translations`），对应路由、协议文件均迁移。  

- **1887acc** – *Fix tokenizer test for renamed attr on Transformers v5*  
  <https://github.com/vllm-project/vllm/commit/1887acca9e2ceacea8f7b1770bd0a0fd9b6a3b02>  
  - **文件**: `tests/entrypoints/openai/test_serving_tokens.py`（+9 / -1 行）  
  - 关键改动：更新对 `tokenizer` 属性的引用，以兼容 Transformers 5.x。  

- **92e7562** – *Suppress non‑TTY color output on the process name part of the log*  
  <https://github.com/vllm-project/vllm/commit/92e7562a994038c904fea859d90462c7e84a3246>  
  - **文件**: `vllm/utils/system_utils.py`（+6 / -1 行）  
  - 关键改动：在非交互式终端下关闭日志前缀的颜色渲染，提升 CI 可读性。  

- **87d0d17** – *Consolidate Deepseek‑OCR2 processor*  
  <https://github.com/vllm-project/vllm/commit/87d0d17ab583740bce777f334a6281edf9822e78>  
  - **文件**: `vllm/model_executor/models/deepseek_ocr2.py`（删除 320 行）  
  - **文件**: `vllm/model_executor/models/deepseek_ocr.py`（+12 / -2 行）  
  - **文件**: `vllm/transformers_utils/processors/deepseek_ocr.py`（+29 / -9 行）  
  - 关键改动：将 OCR2 合并入 OCR，统一处理逻辑并删除冗余实现。  

- **a57c822** – *Make Inplace Flag for FusedMoEModularKernel part of the constructor*  
  <https://github.com/vllm-project/vllm/commit/a57c8228ffb3ff82b983b32b71ff62a837255129>  
  - **文件**: `vllm/model_executor/layers/fused_moe/config.py`（+6 / -3 行）  
  - **测试**: 多个 `tests/kernels/moe/*`（累计 +132 / -109 行）  
  - 关键改动：将 `inplace` 参数显式化，提升 API 可预测性。  

- **1ee9584** – *Fix swapped engine_ids in NIXL Llama‑4 local attention path*  
  <https://github.com/vllm-project/vllm/commit/1ee95841bd251f9081c3a317984c4dcaa003b3c0>  
  - **文件**: `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py`（±6 行）  
  - 关键改动：纠正 `engine_id` 交换导致的注意力映射错误。  

- **7d8c680** – *Add debug logs*  
  <https://github.com/vllm-project/vllm/commit/7d8c6804e2654873cb25d0b23fef178fa5f37237>  
  - **文件**: `vllm/distributed/kv_transfer/kv_connector/utils.py`（+2 行）  
  - **文件**: `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py`（+3 行）  
  - 关键改动：在 KV 连接器路径中加入调试日志，便于排查分布式 KV 同步问题。  

- **af3162d** – *Unified Parallel Drafting*  
  <https://github.com/vllm-project/vllm/commit/af3162d3aaa559a738396baf5b5134c1ab0742f5>  
  - **文件**: `vllm/v1/spec_decode/eagle.py`（+269 / -47 行）  
  - **文件**: `vllm/v1/spec_decode/draft_model.py`（+23 / -222 行）  
  - **文件**: `vllm/v1/spec_decode/utils.py`（+248 行）等，累计 **+1085 / -392 行**。  
  - 关键改动：实现统一的并行草稿（draft）模型，提升 Spec‑Decode 的吞吐与一致性。  

- **5b2a942** – *Fix LoRA Fp8*  
  <https://github.com/vllm-project/vllm/commit/5b2a9422f0f4cbdd69a9fed1dc1605838314ff81>  
  - **文件**: `vllm/lora/layers/fused_moe.py`（+14 / -8 行）  
  - 关键改动：修正 LoRA 在 FP8 环境下的数值溢出问题。  

- **c1858b7** – *Native Weight Syncing API: NCCL* (1/2)  
  <https://github.com/vllm-project/vllm/commit/c1858b7ec8aa571dc0c0e00aded01019cca6a7e6>  
  - **新增**: `examples/offline_inference/new_weight_syncing/rlhf.py`（+208 行）  
  - **新增**: `examples/offline_inference/new_weight_syncing/rlhf_async_new_apis.py`（+283 行）  
  - **新增**: `examples/online_serving/rlhf_http.py`（+241 行）  
  - **新增测试**: `tests/distributed/test_weight_transfer.py`（+346 行）等，累计 **+2974 行**（几乎全部为新增）。  
  - 关键改动：引入基于 NCCL 的权重同步 API，支持 RLHF 场景的高效分布式权重传输。  

- **82914d2** – *Fix step3p5 parser when using mtp*  
  <https://github.com/vllm-project/vllm/commit/82914d2ae8d0362be06700222f4cd4c5f6b0dc36>  
  - **新增**: `tests/tool_parsers/test_step3p5_tool_parser.py`（+1435 行）  
  - **修改**: `vllm/tool_parsers/step3p5_tool_parser.py`（+20 / -5 行）  
  - 关键改动：完善 `step3p5` 解析器对 MTP 参数的兼容性，新增完整单元测试。  

- **81a90e5** – *Add bart‑plugin to docs*  
  <https://github.com/vllm-project/vllm/commit/81a90e52776503c6cbdccd30fbe53f61c9179bdf>  
  - **文件**: `docs/models/supported_models.md`（+10 行）  
  - **文件**: `docs/usage/v1_guide.md`（+6 / -3 行）  
  - 关键改动：文档中加入 BART 插件的使用说明。  

- **1c3a221** – *Fix corner case of sparse embedding*  
  <https://github.com/vllm-project/vllm/commit/1c3a221d3b0f7a82cd9a6d56e10ea360e2435a1c>  
  - **文件**: `vllm/model_executor/layers/pooler/special.py`（±2 行）  
  - **文件**: `tests/models/language/pooling/test_bge_m3.py`（+10 行）  
  - 关键改动：处理稀疏嵌入在极端输入下的异常行为。  

- **7bd42e6** – *Clean up input preprocessing*  
  <https://github.com/vllm-project/vllm/commit/7bd42e609d24501f59a8b405229ed91f4ca8037c>  
  - **文件**: `vllm/inputs/preprocess.py`（+68 / -203 行）等，累计 **+91 / -204 行**。  
  - 关键改动：重构输入预处理流水线，删除冗余代码，提升可维护性。  

- **a252283** – *Fix Kimi‑K2.5 NVFP4 checkpoints weight loading*  
  <https://github.com/vllm-project/vllm/commit/a2522839d87d2b81b57458dfdbbcb27afb8191ae>  
  - **文件**: `vllm/model_executor/models/kimi_k25.py`（+14 / -4 行）  
  - 关键改动：修正 Kimi‑K2.5 NVFP4 权重加载时的维度错位。  

- **59a5cb3** – *Integrate flashinfer concat_mla_k*  
  <https://github.com/vllm-project/vllm/commit/59a5cb387ae4c11c73855d505adb0b2c7cd3861d>  
  - **文件**: `vllm/model_executor/layers/attention/mla_attention.py`（+17 / -3 行）  
  - **文件**: `vllm/utils/flashinfer.py`（+47 行）  
  - 关键改动：在 FlashInfer 中加入 `concat_mla_k` 支持，提升多模态注意力的拼接效率。  

- **8322d4e** – *Enable Cross layers KV cache layout at NIXL Connector V2*  
  <https://github.com/vllm-project/vllm/commit/8322d4e47f89f7985b9b3b808fc4ba8549d6afcd>  
  - **文件**: `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py`（+76 / -34 行）  
  - **文件**: `vllm/distributed/kv_transfer/kv_connector/utils.py`（+45 / -4 行）  
  - 关键改动：在 NIXL 连接器 V2 中实现跨层 KV 缓存布局，配套文档与测试同步更新。  

- **3e472e8** – *Fix hybrid models and their tests (Mamba/Jamba/Bamba) on ROCm*  
  <https://github.com/vllm-project/vllm/commit/3e472e81f99b5bcf494369ee2d26ee9d6ceeffe3>  
  - **文件**: `vllm/model_executor/layers/mamba/mamba_mixer.py`（+6 行）  
  - **文件**: `tests/models/language/generation/test_hybrid.py`（+5 行）  
  - 关键改动：修复 ROCm 环境下混合模型的编译与测试错误。  

- **038914b** – *Move `task` outside of `PoolingParams.verify`*  
  <https://github.com/vllm-project/vllm/commit/038914b7c891c0b5b2853ec0574062dc3bea8073>  
  - **文件**: 多处 `vllm/entrypoints/pooling/*` 与 `vllm/entrypoints/llm.py`（累计 +186 / -216 行）  
  - 关键改动：将 `task` 参数抽离验证层，简化池化 API，提升代码可读性。  

- **d2f4a71** – *Kimi‑K2 grouped_topk usage for Flashinfer monolithic kernels*  
  <https://github.com/vllm-project/vllm/commit/d2f4a71cd54418369f617a174e6c839a71a47ed8>  
  - **文件**: `vllm/model_executor/models/deepseek_v2.py`（+3 / -11 行）  
  - 关键改动：在 FlashInfer 单体 kernel 中启用 `grouped_topk`，优化 Kimi‑K2 的查询效率。  

- **2abd975** – *Do not count local prefix cache hits in connector queries*  
  <https://github.com/vllm-project/vllm/commit/2abd97592f947c041ba70329532f0cf62dd8971f>  
  - **文件**: `vllm/v1/core/sched/scheduler.py`（+19 / -20 行）  
  - **测试**: `tests/v1/core/test_scheduler.py`（+69 / -10 行）等，累计 **+115 / -31 行**。  
  - 关键改动：在 KV 连接器统计中排除本地前缀缓存命中，提升指标准确性。  

- **6abb045** – *Optimize the performance of structured output + reasoning*  
  <https://github.com/vllm-project/vllm/commit/6abb0454adb531de0b081bbf65ccf907e4bd560d>  
  - **文件**: `vllm/entrypoints/openai/chat_completion/serving.py`（+41 / -60 行）  
  - **文件**: `vllm/v1/structured_output/__init__.py`（+4 / -1 行）等，累计 **+51 / -61 行**。  
  - 关键改动：在结构
### NVIDIA/cutile-python
昨日无更新。

## 总结
整体来看，CUTLASS 与 FlashInfer 正在加速对最新 CUDA 工具链和 FP8/FP4 量化的原生支持，显著提升了高性能 GEMM 与 MoE 的可用性。SGLang 与 vLLM 则聚焦于推理层面的可扩展性：从 Speculative Decoding、XPU 后端到分布式权重同步，均展示出对多模态、跨卡协同以及大模型训练/推理的深度耕耘。下一步，社区可关注 **CuTe DSL 在实际模型中的落地**、**FlashInfer 与 vLLM 的跨框架 MoE 整合**，以及 **XPU 与 Hopper 上的 FP8 优化**，这些方向有望进一步压缩推理时延并降低算力成本。
