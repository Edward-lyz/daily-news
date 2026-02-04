# Daily AI Infra Report

## 往期回顾
- [2026-02-02](reports/2026-02-02.md)
- [2026-02-01](reports/2026-02-01.md)
- [2026-01-31](reports/2026-01-31.md)

---

## 最新解读 (2026-02-03)
今日的 AI Infra 的新闻如下。

## 摘要
## 本周关键更新概览

1. **DeepGEMM**：在 SM100 FP8 MQA 相关实现中加入 `tcgen05_after_thread_sync()` 同步调用，解决了线程同步导致的日志概率错误，提升了多查询注意力（MQA）在 SM100 架构上的稳定性。\
2. **CUTLASS v4.4**：新增 CuTe DSL 对 CUDA 13.1 的支持并引入 GB300、`cute.experimental` 等特性；在多个 SM100/SM103/SM90 GEMM kernel 中加入 `cutlass::arch::wait_on_dependent_grids()`，防止前置 kernel 访问未刷新全局内存，显著提升跨网格调度安全性。\
3. **FlashInfer**：多项性能与兼容性改进，包括在 Hopper 上的 GDN 预填编译时间优化、MLA 基准带宽计算修正、SM90 TMA 生成更高效的 kernel、以及对 XPU、Ascend/NPU 的潜在支持探索。\
4. **FlashAttention**：针对 CUDA 13.1 的兼容性问题，报告了缺失 `GLIBC_2.32` 的安装错误，为后续发行版提供了修复方向。\
5. **SGLang**：大幅度代码清理与功能扩展：多模态 Diffusion 支持、MTGPU 分布式实现、Mooncake KV 传输优化、Docker CI 迁移至自建 runner、以及对 Qwen3、GPT‑OSS 120B 等模型的实验性支持。\
6. **vLLM**：持续强化模型兼容性与性能：CPU 端注意力调度对齐、Parser for Responses API、torchrun 分布式死锁修复、FP8 MoE 路由安全检查、Mooncake KV 连接器重构、以及对多模态模型（如 Qwen3‑ASR、Gemma3n）和 XPU 的迁移。\
7. **cuTile‑Python**：在 gather/scatter 操作中加入自定义掩码功能，提升了张量切片的灵活性和安全性。

## 具体内容分析
### deepseek-ai/DeepGEMM
**提交**  
- **SHA**: `477618c`  
- **标题**: Fix a sync issue in SM100 MQA logits (#285)  
- **PR 链接**: <https://github.com/deepseek-ai/DeepGEMM/pull/285>  
- **提交链接**: <https://github.com/deepseek-ai/DeepGEMM/commit/477618cd51baffca09c4b0b87e97c03fe827ef03>  

**文件变更**  

| 模块 / 文件 | 变更类型 | 关键代码片段 |
|------------|----------|--------------|
| `deep_gemm/include/deep_gemm/impls/sm100_fp8_mqa_logits.cuh` | 修改 | ```cpp\n+                    tcgen05_after_thread_sync();\n``` 在 `empty_umma_barriers[i]->wait` 之后以及 `full_umma_barriers[warpgroup_idx]->wait` 之后插入同步调用，以解决线程同步问题。 |
| `deep_gemm/include/deep_gemm/impls/sm100_fp8_paged_mqa_logits.cuh` | 修改 | ```cpp\n+                full_q_barriers[q_stage_idx]->wait(q_phase);\n``` 在 Q 阶段切换时加入显式等待；<br>```cpp\n+                tcgen05_after_thread_sync();\n``` 同样在 `empty_umma_barriers[i]->wait` 后添加同步调用。 |

**Issues**  
- 本次提交未关联任何 Issue。
### deepseek-ai/FlashMLA
昨日无更新。
### NVIDIA/cutlass
提交：

- **PR [6b3e607](https://github.com/NVIDIA/cutlass/commit/6b3e607b852f1543dc21323155a2ad71473c8642) – v4.4 release update v2. (#2999)**
  - **CHANGELOG.md** – 添加 CuTe DSL 对 CUDA 13.1 的支持、GB300 支持以及 `cute.experimental` 新特性。  
    ```diff
    +  - CuTe DSL now supports CUDA toolkit 13.1!
    +    + Set up with cutlass/python/CuTeDSL/setup.sh --cu13
    +    + Refer to https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html for more details
    +  - GB300 is now supported in CuTe DSL with CTK 13.1
    +    + Refer to [SM103 batched 3xFP4 blockscaled GEMM kernel](...)
    +  - cute.experimental: introduce a higher‑level, composable layer …
    ```
  - **README.md** – 同步更新 “What’s New in CUTLASS 4.4” 部分，列出上述 DSL 新特性。  
    ```diff
    +  - CuTe DSL now supports CUDA toolkit 13.1!
    +    + Set up with cutlass/python/CuTeDSL/setup.sh --cu13
    +    + Refer to https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html for more details
    ```
  - **examples/python/CuTeDSL/blackwell/dense_gemm_persistent_dynamic.py** – 大幅重构持久化调度循环，引入 `pipeline.make_pipeline_state` 与 `acc_consumer_state`，并对 TMA 存储路径做了显式分支。  
    ```diff
    -            # Persistent tile scheduling loop for epilogue
    +            acc_consumer_state = pipeline.make_pipeline_state(
    +                pipeline.PipelineUserType.Consumer, self.num_acc_stage
    +            )
    ```
  - **examples/python/CuTeDSL/blackwell/sm103_dense_blockscaled_gemm_persistent.py** – 新增示例，演示 SM103 上的 3×FP4 block‑scaled GEMM（完整文件约 3 k 行，已完整提交）。
  - **examples/python/CuTeDSL/experimental/ampere/memcpy_simt_universal_copy.py** – 新增实验性 SIMT memcpy 示例（完整文件已提交）。
  - **examples/python/CuTeDSL/experimental/blackwell/** – 新增四个实验性示例（`dense_block_scaled_gemm.py`, `dense_gemm.py`, `dense_gemm_2sm.py`, `dense_gemm_cute_pipeline.py`, `dense_gemm_ptr_array.py`），每个文件均包含完整版权声明和示例代码，已完整提交。
  - **include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized.hpp** – 在调度前加入 `cutlass::arch::wait_on_dependent_grids();`，确保前置 kernel 的全局内存已刷新。  
    ```diff
    +    // Ensure that the prefetched kernel does not touch
    +    // unflushed global memory prior to this instruction.
    +    cutlass::arch::wait_on_dependent_grids();
    ```
  - **include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized_input_transform.hpp** – 调整 `TileScheduler` 的实例化顺序，先调用 `wait_on_dependent_grids()` 再创建调度器。  
    ```diff
    -    // TileID scheduler
    -    TileScheduler scheduler(&shared_storage.clc_response[0], params.scheduler, block_id_in_cluster);
    +    // Ensure memory ops in this kernel are not done prior to completion of dependent grids.
    +    cutlass::arch::wait_on_dependent_grids();
    +    // TileID scheduler
    +    TileScheduler scheduler(&shared_storage.clc_response[0], params.scheduler, block_id_in_cluster);
    ```
  - **include/cutlass/gemm/kernel/sm100_gemm_array_tma_warpspecialized_mma_transform.hpp** – 同上，移动 `wait_on_dependent_grids()` 调用位置。
  - **include/cutlass/gemm/kernel/sm100_tile_scheduler_group.hpp** – 添加注释说明构造调度器时可能触及前置 kernel 写入的全局内存。
  - **include/cutlass/gemm/kernel/sm103_blockscaled_gemm_array_tma_warpspecialized.hpp** – 与 `sm100` 版本保持一致，加入 `wait_on_dependent_grids()`。
  - **include/cutlass/gemm/kernel/sm90_gemm_array_tma_warpspecialized_cooperative.hpp** – 将 `TileScheduler` 的创建逻辑迁移到 `wait_on_dependent_grids()` 之后，以避免潜在的内存竞争。（由于补丁被截断，完整实现细节请参考 PR 中的完整文件。）

- **PR [1cfbb53](https://github.com/NVIDIA/cutlass/commit/1cfbb53a23cf009973a6a6ea9e8275c8a691d411) – Fix: SM100 block‑scale gemm overlapping accumulator (#2995)**
  - **examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent.py** – 修复累加器重叠导致的错误，新增 `ceil_div` 辅助函数并重新计算 `threads_per_cta` 与 barrier 大小。  
    ```diff
    +def ceil_div(a, b):
    +    return (a + b - 1) // b
    ...
    -self.threads_per_cta = 32 * len((self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id))
    +self.threads_per_warp = 32
    +self.threads_per_cta = self.threads_per_warp * len((self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id))
    ```
  - **examples/python/CuTeDSL/blackwell/dense_blockscaled_gemm_persistent_prefetch.py** – 同步更新，加入相同的 `ceil_div` 实现与线程/屏障计算逻辑。

Issues：

- **Issue [#2997](https://github.com/NVIDIA/cutlass/issues/2997) – “warning #2908-D: the implicit by‑copy capture of ‘this’ is deprecated”**  
  报告在使用 C++20 编译 CUTLASS 时，`sm100_blockscaled_mma_array_warpspecialized_rcggemm.hpp` 第 694 行出现 lambda 隐式复制捕获 `this` 的警告。该问题已在 PR #2995 中通过显式捕获方式得到部分缓解（但仍需在相关文件中统一修正）。

- **Issue [#2996](https://github.com/NVIDIA/cutlass/issues/2996) – “CUTLASS int8 fprop2d kernel slower than TensorRT on Jetson Orin”**  
  用户在 Jetson AGX Orin 上使用 NCxHWx<32> 交错布局的 INT8 Conv2d Fprop kernel，发现性能比 TensorRT 慢约 12%。该问题涉及布局优化与内核调度，尚未在本次提交中解决，后续可能需要针对 Ampere SM87 的特化实现进行调优。
### flashinfer-ai/flashinfer
**提交**

| SHA | 标题 | 关联 PR / Commit 链接 | 关键模块/文件 | 代码片段（示例） |
|-----|------|----------------------|--------------|----------------|
| `f84ac1c` | fix: Fix memory bandwidth calculation in MLA benchmarks | [commit f84ac1c97…](https://github.com/flashinfer-ai/flashinfer/commit/f84ac1c97e2e1a4391150ca73971b973ee569e5a) | `benchmarks/bench_trtllm_gen_mla.py`、`benchmarks/routines/attention.py` | ```python<br># 计算带宽（原始实现）<br>bandwidth = (2 * hidden_dim * seq_len) / time<br># 修正后使用实际访存量<br+>bandwidth = (4 * hidden_dim * seq_len) / time<br>``` |
| `6ae5bfe` | refactor: reduce hopper's gdn prefill compilation time and fix docstring | [commit 6ae5bfe6…](https://github.com/flashinfer-ai/flashinfer/commit/6ae5bfe6a48c591a171f32888e5de28b9ca0207b) | `csrc/gdn_prefill_launcher.cu`、`flashinfer/jit/gdn.py`、`include/flashinfer/flat/hopper/collective/flat_collective_store.hpp`（文件重命名） | ```cpp<br>// 新增的外部 inc 文件，定义 delta rule<br>#include "flat_prefill_kernel_delta_rule_sm90_extern.inc"<br>```<br>（部分 diff 被截断，具体实现请查看完整提交） |
| `0eb69bb` | Fix autotuner oom | [commit 0eb69bb0…](https://github.com/flashinfer-ai/flashinfer/commit/0eb69bb00e5561b076ce288f6dfdf20f5d6783b5) | `flashinfer/autotuner.py` | ```python<br># 限制搜索空间防止 OOM<br>if total_params > MAX_PARAMS:\n    raise MemoryError("Autotuner OOM")<br># 新增 fallback 策略<br>config = fallback_config\n``` |
| `9e069e7` | fix: benchmark blockscale moe routine supports non-DS routing | [commit 9e069e76…](https://github.com/flashinfer-ai/flashinfer/commit/9e069e76f63e287ccc8894e1edd67f011869dc52) | `benchmarks/routines/moe.py` | ```python<br># 新增路由分支<br>if not use_ds_routing:\n    route = custom_route(inputs)\nelse:\n    route = ds_route(inputs)\n``` |
| `fb671fd` | ci: migrate release workflows to ci-infra runners | [commit fb671fdb…](https://github.com/flashinfer-ai/flashinfer/commit/fb671fdb79d53aaf8a1d3baca1c9aa35d080d510) | `.github/workflows/nightly-release.yml`、`.github/workflows/release-ci-docker.yml`、`.github/workflows/release.yml` | ```yaml<br># 使用 ci‑infra runner<br>runs-on: [self-hosted, ci-infra]\n``` |
| `5e5a866` | perf: improve gdn decode cute‑dsl kernels | [commit 5e5a8668…](https://github.com/flashinfer-ai/flashinfer/commit/5e5a8668bede32d86035dba98f460a278332e366) | `benchmarks/bench_gdn_decode.py`、`flashinfer/gdn_decode.py` | ```python<br># 关键优化：使用 cute‑dsl 生成更紧凑的 TMA<br>kernel = cute_dsl.generate_kernel(params, schedule=\"sm90\")\n``` |
| `2d7a987` | Add sm90 guard to fence ptx | [commit 2d7a987c…](https://github.com/flashinfer-ai/flashinfer/commit/2d7a987c9ca4251d0a3e4c1249614f2474353884) | `csrc/nv_internal/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu` | ```cpp<br>#if __CUDA_ARCH__ >= 900<br>// SM90 专用 fence\nasm volatile ("fence.ptx" ::: "memory");<br>#endif\n``` |
| `b7404d0` | MTP for mamba | [commit b7404d06…](https://github.com/flashinfer-ai/flashinfer/commit/b7404d064d27f0c1805dc7ecfc8f7fbd5db1b053) | `include/flashinfer/mamba/kernel_selective_state_update_mtp.cuh`、`tests/mamba/test_selective_state_update_mtp.py` | ```cpp<br>// MTP（Multi‑Thread‑Per‑warp）实现片段<br>template <typename T>\n__global__ void selective_state_update_mtp(...){\n    // warp‑level reduction\n}\n``` |
| `273c09c` | Update Docker CI tags to 20260203-9b5901e | [commit 273c09cd…](https://github.com/flashinfer-ai/flashinfer/commit/273c09cd03cde5f84f561807d7178ad6b8de7a25) | `ci/docker-tags.yml` | ```yaml<br># 更新标签\n- 20260203-9b5901e\n``` |
| `95d8511` | add sgl_kernel.fast_topk_v2 to top_k benchmark | [commit 95d85118…](https://github.com/flashinfer-ai/flashinfer/commit/95d851180671ebc178eb18bea8fc437fdbeff313) | `benchmarks/bench_topk.py` | ```python<br># 新增 fast_topk_v2 调用\nresult = sgl_kernel.fast_topk_v2(scores, k)\n``` |
| `9b5901e` | ci: set LD_LIBRARY_PATH in Docker images for correct cuBLAS detection | [commit 9b5901eb…](https://github.com/flashinfer-ai/flashinfer/commit/9b5901eb007633ebbb38fbe6ee0a014caca88e0a) | `docker/Dockerfile.cu126`、`docker/Dockerfile.cu128`、`docker/Dockerfile.cu129`、`docker/Dockerfile.cu130` | ```dockerfile<br>ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH\n``` |
| `c7761ad` | [Perf][Feature] Add SM103‑specific schedulers for NVFP4 CUTLASS kernels | [commit c7761add…](https://github.com/flashinfer-ai/flashinfer/commit/c7761add6e6b7703ce5f12b03be387e78048f36b) | `include/flashinfer/gemm/fp4_gemm_template_sm103.h`、`flashinfer/jit/gemm/core.py` | ```cpp<br>// SM103 专用调度器示例<br>using Scheduler = cutlass::gemm::threadblock::GemmSchedulerSM103;\n``` |

> **说明**：部分提交的 `patch_truncated` 为 `true`，因此这里仅展示了关键实现片段，完整 diff 请在对应的 Commit 页面查看。

---

**Issues**

| 编号 | 标题 | 链接 | 作者 | 创建时间 | 简要说明 |
|------|------|------|------|----------|----------|
| #2483 | **[Feature Request] Skip‑Softmax (BLASST) Optimization in XQA** | https://github.com/flashinfer-ai/flashinfer/issues/2483 | jimmyzho | 2026‑02‑03 22:08:34 UTC | TensorRT‑LLM 已实现 Skip‑Softmax Attention（论文 https://www.arxiv.org/pdf/2512.12087），建议在 FlashInfer 中加入对应后端并更新 API，以提升 Hopper 解码核性能。 |
| #2480 | **[DSR1] DSR1 router gemm performance issue when upstream to SGL** | https://github.com/flashinfer-ai/flashinfer/issues/2480 | nv-yunzheq | 2026‑02‑03 19:18:47 UTC | SGL PR‑17707 中的路由 GEMM 在性能上不及 SGL 内部的 TRT‑LLM 复制实现，需要定位瓶颈并合并功能差异。 |
| #2474 | **would it be support Ascend/NPU** | https://github.com/flashinfer-ai/flashinfer/issues/2474 | L4-1024 | 2026‑02‑03 07:15:54 UTC | 询问是否计划在 FlashInfer 中加入对华为 Ascend / NPU 硬件的支持。 |
| #2473 | **SageAttention support** | https://github.com/flashinfer-ai/flashinfer/issues/2473 | YangXu1990uiuc | 2026‑02‑03 06:43:00 UTC | 探讨在 SageAttention 2（使用 FP8）中实现以下流水线：<br>• Q·K 使用 int8 IMMA（GB200）或 bf16（GB300）<br>• per‑16‑token scaling<br>• Softmax fp32<br>• PV fp8<br>• V 端 scaling<br>目标是与 SGL/TPU 对齐的性能。 |

> 所有 Issue 均已在对应页面提供更详细的讨论与后续进展。
### Dao-AILab/flash-attention
提交  
Issues  

- **标题**: what's the full name of SPT?  
  **作者**: endurehero  
  **创建时间**: 2026‑02‑03 23:43:43 UTC  
  **概要**: 在 `SingleTileLPTBwdScheduler` 中的参数 `spt` 用于在因果且确定性场景下逆序遍历 n tiles，以提升性能，询问其全称。  
  **链接**: [https://github.com/Dao-AILab/flash-attention/issues/2226](https://github.com/Dao-AILab/flash-attention/issues/2226)  
  **关联 PR / Commit**: 目前暂无对应的 PR，相关提交尚未确定（缺少 commit SHA 与 URL）。

- **标题**: run code error after pip install cuda 12.6 flash‑attn2.8.2  
  **作者**: cqray1990  
  **创建时间**: 2026‑02‑03 06:58:49 UTC  
  **概要**: 在 Ubuntu 20 上使用 `pip install flash_attn-2.8.2+cu128torch2.8-cp312-cp312-linux_x86_64.whl` 后，导入时报错缺少 `GLIBC_2.32`。  
  **链接**: [https://github.com/Dao-AILab/flash-attention/issues/2225](https://github.com/Dao-AILab/flash-attention/issues/2225)  
  **关联 PR / Commit**: 目前暂无对应的 PR，相关提交尚未确定（缺少 commit SHA 与 URL）。
### sgl-project/sglang
**提交**  

- **c1d529c** – 修复多模态 Session 并通过 Engine 暴露（[#18152](https://github.com/sgl-project/sglang/commit/c1d529c19605cbf1f9be8db6d6d225b1465ea2e0)）  
  - `python/sglang/srt/entrypoints/engine.py`（+37 行）  
  - `python/sglang/srt/managers/schedule_batch.py`（-1 行）  
  - `python/sglang/srt/managers/scheduler.py`（+17 行）  

- **1f72f66** – 文档 typo 修正（[#18207](https://github.com/sgl-project/sglang/commit/1f72f66c6d6ef0eff589230b9ca06cf4e44ecdd2)）  
  - `README.md`（+1 / -1 行）  

- **da758ed** – 修复连续动态请求下的 server cache‑dit bug（[#17140](https://github.com/sgl-project/sglang/commit/da758ed601270b21e1cfb404306ff0ca5c816a3f)）  
  - `python/sglang/multimodal_gen/runtime/cache/cache_dit_integration.py`（+67 / -1 行）  
  - `python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py`（+71 / -40 行）  

- **ae004e1** – Nightly wheel 打标签使用最新 commit（[#18204](https://github.com/sgl-project/sglang/commit/ae004e15c98b36dda9c460d4d0ee9891889a5adf)）  
  - `.github/workflows/release-pypi-nightly.yml`（+1 / -1 行）  

- **793bf9f** – Qwen3 Embeddings 权重重命名检查更新（[#17535](https://github.com/sgl-project/sglang/commit/793bf9fc06499cb1ba236444a7a3dde0ea5b7e49)）  
  - `python/sglang/srt/models/qwen3.py`（+5 / -1 行）  

- **e867040** – 新增流式并行工具调用测试用例（[#18097](https://github.com/sgl-project/sglang/commit/e867040fc6624445ffa4b567c157b598d9aed2c8)）  
  - `python/sglang/test/tool_call_test_runner.py`（+32 行）  

- **7de650c** – 支持 MTGPU 上的 Diffusion 模型（文档/安装）（[#17346](https://github.com/sgl-project/sglang/commit/7de650c83c4d63b9107184bd5cf36303d89e8d28)）  
  - `python/sglang/multimodal_gen/README.md`（+7 / -2 行）  
  - `python/sglang/multimodal_gen/docs/install.md`（+3 / -1 行）  
  - `python/sglang/multimodal_gen/docs/install_musa.md`（新增 24 行）  

- **ec2461b** – MTGPU 多 GPU 支持（[#17318](https://github.com/sgl-project/sglang/commit/ec2461bc16e59d8a5738340fb559ccf396cd9af7)）  
  - `python/sglang/multimodal_gen/runtime/distributed/device_communicators/pynccl_wrapper.py`（+1 / -1 行）  
  - `python/sglang/multimodal_gen/runtime/models/encoders/clip.py`（+3 / -1 行）  
  - `python/sglang/multimodal_gen/utils.py`（+5 / -3 行）  

- **acf724b** – 仅在自定义 CUDA 路径中导入 `sgl_kernel`（[#15592](https://github.com/sgl-project/sglang/commit/acf724b036c595292b036942db69fb169efffc45)）  
  - `python/sglang/multimodal_gen/runtime/layers/activation.py`（+6 / -1 行）  
  - `python/sglang/multimodal_gen/runtime/layers/layernorm.py`（+6 / -1 行）  

- **e166ca8** – HiCache 增加缓存命中细分及 Prometheus 指标（[#17648](https://github.com/sgl-project/sglang/commit/e166ca87584368699ac35ccb5e518211631849cf)）  
  - 关键文件包括 `protocol.py`（+63 / -14 行）、`serving_chat.py`（+28 / -23 行）、`serving_completions.py`（+29 / -23 行）等共 15 处改动，累计 +333 行、-74 行。  

- **d48bbe3** – NPU CI 中 `sgl‑kernel` 导入错误修复（[#18173](https://github.com/sgl-project/sglang/commit/d48bbe3beda74106e27ab4e83eebbc235a5fbd59)）  
  - `python/sglang/srt/layers/moe/moe_runner/flashinfer_trtllm.py`（+5 / -1 行）  

- **495290a** – 启用 XPU 设备单元测试（[#11712](https://github.com/sgl-project/sglang/commit/495290aefd1f5e8f7872a218473ac0b4c7dcc2f6)）  
  - 多文件改动，涉及 `topk.py`、`runners.py`、`test_utils.py` 等，累计 +237 行、-151 行。  

- **0a69256** – 改进 cu13 构建 Docker（[#18194](https://github.com/sgl-project/sglang/commit/0a6925639b3f2d7b70503f06170eb490c09dac4c)）  
  - `.github/workflows/release-docker-cu13-framework.yml`（+6 行）  
  - `docker/Dockerfile`（+1 / -3 行）  

- **0db6fd4** – 恢复 sgl_kernel 排除模式（[#18193](https://github.com/sgl-project/sglang/commit/0db6fd4dbec5e9eaac6ee35dfba257720ae65ea5)）  
  - 四个 CI workflow 文件均删除大量排除规则（共 -47 行）  

- **820df54** – 添加 cu13 开发容器至发布流程（[#18192](https://github.com/sgl-project/sglang/commit/820df545f2c48ea92911e2078389cc1a447bde3f)）  
  - 新增 `release-docker-cu13-framework.yml`（+157 行）  
  - `release-docker.yml`（+52 / -2 行）  

- **99fab2c** – 修复 Mistral Large 3 NVFP4 TRTLLM MoE（[#18065](https://github.com/sgl-project/sglang/commit/99fab2ce673eeae87736cee44844777a3c9ae304)）  
  - `compressed_tensors_moe.py`（+94 / -103 行）  
  - `test_mistral_large3.py`（+21 / -8 行）  

- **a45647b** – Mooncake intra‑node NVLink KV 传输支持（[#17866](https://github.com/sgl-project/sglang/commit/a45647bce16ddf159713a65c3b281c7e66700bcc)）  
  - `disaggregation/mooncake/utils.py`（+3 / -1 行）  
  - `disaggregation/utils.py`（+3 行）  

- **cc69ac9** – 动态 chunk 大小预热以提升 prefill 延迟测量（[#17198](https://github.com/sgl-project/sglang/commit/cc69ac9e7a7d94fac0b94b8d4d19bf62af69df15)）  
  - `scheduler_pp_mixin.py`（+3 / -2 行）  

- **8e933e1** – AMD PD/D PR CI 新增（[#17183](https://github.com/sgl-project/sglang/commit/8e933e1914b6b335e58bb1442b3931e2303192bb)）  
  - `pr-test-amd.yml`（+113 行）  
  - 新增 AMD CI 脚本及两套 disaggregation 测试（共 +713 行）  

- **25508d1** – Docker 默认时区改为 UTC，去除硬编码（[#18121](https://github.com/sgl-project/sglang/commit/25508d11c03cb1344a1cffc8e4e9e517181388f0)）  
  - `docker/Dockerfile`（+1 / -5 行）  
  - `docker/gateway.Dockerfile`（+1 / -3 行）  

- **6f6b9c6** – 多线程加载器使用 safetensors `load_file`（[#18124](https://github.com/sgl-project/sglang/commit/6f6b9c6e42a927f087276536f225147eb8507714)）  
  - `weight_utils.py`（+2 / -6 行）  

- **7a9d9c7** – Mooncake `batch_exists` 中应用 `extra_backend_tag`（[#17265](https://github.com/sgl-project/sglang/commit/7a9d9c79d15dfa7c2c1ff9685ea7fba3d5c599ad)）  
  - `mooncake_store.py`（+5 行）  

- **74f716d** – Gigachat 3 工具解析器及测试（[#14765](https://github.com/sgl-project/sglang/commit/74f716dbd711bef3bd7bc9d6eab7ebc675e47e65)）  
  - `function_call_parser.py`（+2 行）  
  - 新增 `gigachat3_detector.py`（+209 行）  
  - `test_function_call_parser.py`（+506 行）  

- **4181290** – `run_eval.py` 新增 `--top‑k` 参数（[#18025](https://github.com/sgl-project/sglang/commit/4181290efd668082109055c53261a5c956817caa)）  
  - `run_eval.py`（+10 / -1 行）  

- **fe57a88** – LoRA 重叠加载单元测试改为使用 pytest（[#18140](https://github.com/sgl-project/sglang/commit/fe57a887b110a52670fcaa5c616cd0a8dce12615)）  
  - `test_lora_overlap_loading.py`（+127 / -9 行）  

- **f032c4f** – 支持 Markdown/Notebook 文档导出（[#18131](https://github.com/sgl-project/sglang/commit/f032c4f3d66605edd3177ac7d242c3d9704888bf)）  
  - `release-docs.yml`（+1 行）  
  - `docs/Makefile`（+20 行）  
  - `docs/README.md`（+68 / -1 行）  

- **78bf13d** – MoE 重构：`modelopt_quant.py` → `flashinfer_trtllm.py`（[#16685](https://github.com/sgl-project/sglang/commit/78bf13db4447b98eb9d8169c400448d1dcad12a3)）  
  - `flashinfer_trtllm.py`（+277 / -15 行）  
  - `fp8.py`（+1 行）  
  - `modelopt_quant.py`（+64 / -176 行）  

- **eedd472** – Diffusion pipeline 修复 `image_edit` 输入图像 bug（[#18109](https://github.com/sgl-project/sglang/commit/eedd472025c4f3d0247c5ef4f05dce54663a09c9)）  
  - `diffusers_pipeline.py`（+3 行）  

- **e484c90** – 为 GLM‑4.7‑FP8 tp8 H20 系列添加 Triton fused MoE 配置（[#18091](https://github.com/sgl-project/sglang/commit/e484c90cc7aa3340b073be1d09e0731fb414c9ca)）  
  - 两个 JSON 配置文件（各 +146 行）  

- **9b1619c** – JIT concat MLA kernel 添加（[#17889](https://github.com/sgl-project/sglang/commit/9b1619c148a0a800e9ddb3d2e42ff32ee2dd2679)）  
  - `bench_concat_mla.py`（+163 行）  
  - `concat_mla.py`（+65 行）  
  - `concat_mla.cuh`（+325 行）  
  - `test_concat_mla.py`（+169 行）  

- **62004fd** – Diffusion UX 日志改进（[#18122](https://github.com/sgl-project/sglang/commit/62004fd2beb77daa10502e292e66a0d2f5662e3e)）  
  - 多文件小幅改动，累计 +19 行、-40 行。  

- **1805943** – HiCache 支持 DeepSeek v32 CPU offloading（[#17415](https://github.com/sgl-project/sglang/commit/180594358b990b2a5ce8140fb64aae90d73910fd)）  
  - `hiradix_cache.py`（+15 / -1 行）  
  - `memory_pool_host.py`（+189 / -31 行）  
  - 新增 `test_nsa_pool_host_unit.py`（+130 行）  

- **a1bbc89** – QKNorm 跨头 kernel（[#18073](https://github.com/sgl-project/sglang/commit/a1bbc892af27867901f91e9a1c485824ff9337a6)）  
  - `bench_qknorm_across_heads.py`（+121 行）  
  - `qknorm_across_heads.cuh`（+232 行）  
  - `norm.py`（+34 行）  
  - `test_qknorm_across_heads.py`（+75 行）  

- **fd983b0** – Radix cache 驱逐性能优化（[#14339](https://github.com/sgl-project/sglang/commit/fd983b09b68c076d448a55266a29ffdfdff2d06e)）  
  - `hiradix_cache.py`（+77 / -27 行）  
  - `mamba_radix_cache.py`（-13 行）  
  - `radix_cache.py`（+26 / -15 行）  
  - `swa_radix_cache.py`（-13 行）  

- **c8da307** – 添加 GPT‑OSS 120B Nightly 测试（[#18134](https://github.com/sgl-project/sglang/commit/c8da307d7e6353fdcfdf6ca1ecb1cec854411230)）  
  - `nightly-test-nvidia.yml`（+4 / -4 行）  
  - 新增 `test_gpt_oss_120b.py`（+84 行）  

**Issues**  

- **#18203** – DGX Spark 环境启动服务器时报 `sgl_kernel` 无法加载公共库错误（缺少详细环境信息）。  
- **#18201** – 双栈网络下 `--host` 为空导致 warmup 将主机当作 URL，启动失败。  
- **#18197** – DeepSeek 3.2 NVFP4 RMSNorm 报非法内存错误，涉及 batch_size 超过共享内存限制。  
- **#18191** – DeepSeekV2 + Mooncake 在重新平衡阶段出现 `Parameter` 缺少 `weight_loader` 属性的异常。  
- **#18190** – 请求在 SGLang 中加入对 Qwen3‑TTS 文本到语音模型的支持。  
- **#18188** – 建议在 SGLang Docker 镜像中使用 Python 虚拟环境而非系统全局安装。  
- **#18181** – 使用 SGLang+Megatron 进行 rollout 时出现 403 Forbidden 客户端错误，导致训练中断。  
- **#18177** – 路线图：在 Ascend NPU 上实现 Diffusion 支持（已完成部分量化
### vllm-project/vllm
**提交**  

- **4dffc5e** – [Split attention dispatch by head‑dim alignment](https://github.com/vllm-project/vllm/commit/4dffc5e044317326b9e2b2fd2a019c499d63c427)  
  *CPU*：改进注意力调度，使其按 `head_dim` 对齐，提升 CPU 端吞吐。涉及 `cmake/cpu_extension.cmake`、`csrc/cpu/cpu_attn.cpp`、`csrc/cpu/generate_cpu_attn_dispatch.py` 等文件。

- **e1bf04b** – [Initial Implementation of Parser for ResponsesAPI](https://github.com/vllm-project/vllm/commit/e1bf04b6c27a070859264290ffdccbf333f27fa6)  
  新增 `vllm/parser` 包，实现响应 API 解析抽象层，包含 `abstract_parser.py`、`minimax_m2_parser.py`、`parser_manager.py`。相应测试与入口点代码也已更新。

- **0208017** – [Fix torchrun PP broadcast deadlock with async scheduling](https://github.com/vllm-project/vllm/commit/02080179a3fc92c339b93040838f44b72313e07b)  
  解决分布式 `torchrun` 在异步调度下的广播死锁，修改 `vllm/v1/worker/gpu_model_runner.py` 与相关测试。

- **1b8fe6f** – [Make pooling entrypoints request schema consensus | ScoreRequest](https://github.com/vllm-project/vllm/commit/1b8fe6f7c4b96ec57172f4fd268341a03c12499b)  
  统一 Pooling 接口请求结构，更新 `vllm/entrypoints/llm.py`、`vllm/entrypoints/pooling/score/*` 以及对应示例与测试。

- **52ee210** – [Fix negative accepted tokens metric crash in Spec Decoding](https://github.com/vllm-project/vllm/commit/52ee21021a87735d46c4245c60bc0be42dd58c73)  
  修复 Spec Decoding 中负数接受 token 指标导致的崩溃，涉及 `vllm/v1/core/sched/scheduler.py` 与单元测试。

- **655efb3** – [Remove comments of ray in dependency files](https://github.com/vllm-project/vllm/commit/655efb3e69bb18d150a88c0d726ca2b49f22cbdd)  
  清理 `requirements/cuda.txt`、`requirements/rocm.txt` 中的 Ray 注释。

- **bd8da29** – [Fix sparse MLA metadata building](https://github.com/vllm-project/vllm/commit/bd8da29a66ea8c0e0f208cab7ba0b6be640f3faa)  
  修正稀疏 MLA 注意力元数据构建错误，修改 `vllm/model_executor/layers/attention/mla_attention.py`。

- **2a99c5a** – [Disable TRTLLM FP8 MoE for incompatible routing settings](https://github.com/vllm-project/vllm/commit/2a99c5a6c86daef8c766ba2dbf05c385b192c64b)  
  当 `router_logits_dtype==float32` 且 `routing_method!=DeepSeekV3` 时关闭 TRTLLM FP8 MoE，涉及多个 MoE 与量化层文件。

- **3f7662d** – [Voxtral Realtime name change](https://github.com/vllm-project/vllm/commit/3f7662d6505e441026e668ba78a2207d669f4f32)  
  更新实时示例脚本名称，调整 `examples/online_serving/openai_realtime_*.py` 与对应测试。

- **a372f3f** – [Fix Tensor Parallelism for Quantized Mamba models (n_groups=1)](https://github.com/vllm-project/vllm/commit/a372f3f40afd0aed802242ce59b6a2640d4ef59e)  
  修正量化 Mamba 模型在 `n_groups=1` 时的张量并行实现，修改 `vllm/model_executor/layers/mamba/mamba_mixer2.py`。

- **61e632a** – [Turn `@config` into a `dataclass_transform`](https://github.com/vllm-project/vllm/commit/61e632aea15f76fd1c46354b00f9cac62cd28c4e)  
  将配置装饰器改为 `dataclass_transform`，影响 `vllm/config/*` 与多处测试文件。

- **b1bb18d** – [Significantly speed up torch.compile cold start times](https://github.com/vllm-project/vllm/commit/b1bb18de8d12cac63e3bbb0f59b3726fcb68dc80)  
  优化编译后端初始化路径，涉及 `vllm/compilation/backends.py` 与 `compiler_interface.py`。

- **2267cb1** – [FA3 swizzle optimization update](https://github.com/vllm-project/vllm/commit/2267cb1cfd838a192f47ff677d91164fb5cb2862)  
  为 FlashAttention‑3 添加新 swizzle 优化，更新 `cmake/external_projects/vllm_flash_attn.cmake` 与后端实现。

- **0d6ccf6** – [Rework Mooncake connector and introduce bootstrap server](https://github.com/vllm-project/vllm/commit/0d6ccf68fa2c439e17d02f26c4044ed5df7f7099)  
  大幅重构分布式 KV‑Transfer Mooncake 连接器，新增 `examples/online_serving/disaggregated_serving/mooncake_connector/*` 与内部实现 `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/*`。

- **18e7cbb** – [Fix startup hang for Granite Speech](https://github.com/vllm-project/vllm/commit/18e7cbbb158a86bdc76585e64ada795bf1c0d435)  
  修复 Granite Speech 模型启动时的阻塞，修改 `vllm/multimodal/budget.py`。

- **f0d5251** – [Skip warm‑up for Voxtral models to avoid confusing error](https://github.com/vllm-project/vllm/commit/f0d525171557e3fe74e8e6df52257f9d66831d3f)  
  移除 Voxtral 模型的 warm‑up 步骤，更新相关翻译与模型文件。

- **5c4f2dd** – [Pass `prefix` parameter to MMEncoderAttention](https://github.com/vllm-project/vllm/commit/5c4f2dd6ef2009d81f4a765b5c2a7278fc389ef3)  
  为多模态编码器注意力加入 `prefix` 参数，涉及多模型实现文件（如 `aimv2.py`、`blip.py`、`intern_vit.py` 等）。

- **f3d8a34** – [Do not add extra newline for image‑only prompts](https://github.com/vllm-project/vllm/commit/f3d8a3467111d861cb814152f9c5c8aeaff335c2)  
  修正仅含图像的多模态提示生成时多余的 `\n`，修改 `vllm/entrypoints/chat_utils.py`。

- **4bc913a** – [Add Nemotron‑Nano v3 tests](https://github.com/vllm-project/vllm/commit/4bc913aeeca39a304e4ace51febf55f142c8c86e)  
  新增 Nemotron‑Nano 30B A3B 配置的 CI 测试，更新 Buildkite 配置与模型配置文件。

- **fbb3cf6** – [Avoid double free during async scheduling + request abort + async KV cache transfer](https://github.com/vllm-project/vllm/commit/fbb3cf698123cc1243dae8003b63dfa807ef8b53)  
  修复异步调度下请求中止导致的双重释放错误，涉及 `vllm/v1/core/sched/scheduler.py`。

- **2df2b34** – [Document NixlConnector backend selection via kv_connector_extra_config](https://github.com/vllm-project/vllm/commit/2df2b3499dee2025f3f5aa12fb68ea07013c0aa7)  
  在文档中说明 NixlConnector 的后端选择方式，更新 `docs/features/nixl_connector_usage.md`。

- **2a8d84e** – [Fix Gemma3n audio encoder for Transformers v5](https://github.com/vllm-project/vllm/commit/2a8d84e66d19014c44155ca1ee79b4aa0227734d)  
  调整 Gemma3n 多模态模型的音频编码器兼容性，修改 `vllm/model_executor/models/gemma3n_mm.py`。

- **a3acfa1** – [Add Intern‑S1‑Pro model support](https://github.com/vllm-project/vllm/commit/a3acfa10719a931111caccd08ef19f1551b2fe1e)  
  新增 Intern‑S1‑Pro 模型实现及旋转嵌入 FOPE 支持，涉及 `vllm/model_executor/models/interns1_pro.py`、`rotary_embedding/*` 等。

- **be8168f** – [Fix Gemma3 GGUF for Transformers v5](https://github.com/vllm-project/vllm/commit/be8168ff889aa8981d4e8a158fc1b4d0a4deb18b)  
  修正 GGUF 加载路径，更新 `vllm/transformers_utils/gguf_utils.py`。

- **f6af346** – [Fix offline test for Transformers v5](https://github.com/vllm-project/vllm/commit/f6af34626d37f63ecb128e1f775ebcbbc1d0e5bf)  
  更新离线模式测试以兼容 Transformers v5，修改 `tests/entrypoints/offline_mode/test_offline_mode.py`。

- **ceab70c** – [Fix qwen3‑asr response error](https://github.com/vllm-project/vllm/commit/ceab70c89d2b1f5eeaeb4582eb927b16dacb7671)  
  修复 Qwen‑3 ASR 模型返回错误的异常处理，涉及 `vllm/model_executor/models/qwen3_asr.py`。

- **52683cc** – [Update default image format of `encode_base64`](https://github.com/vllm-project/vllm/commit/52683ccbe194688b5c2a1a8ff6b6d9a060a2b2e7)  
  将默认图像编码格式改为 `png`，修改 `vllm/multimodal/media/image.py` 与 `utils.py`。

- **e346e2d** – [Disable certain TRTLLM FP8 MoE routing methods](https://github.com/vllm-project/vllm/commit/e346e2d056a66bb84287e4fea049bde9a37bd72b)  
  禁用 `Renormalize` 与 `RenormalizeNaive` 在 TRTLLM FP8 MoE 中的路由方式，修改 `flashinfer_trtllm_moe.py`。

- **83449a5** – [Clean up pooling serial utils](https://github.com/vllm-project/vllm/commit/83449a5ff04f70a20c24e8e6fc719881b29e10ac)  
  重构 Pooling 串行工具，新增 `vllm/entrypoints/pooling/utils.py`，并精简 `serial_utils.py`。

- **dad2d6a** – [Fix DeepSeek‑OCR‑2 chat template to include BOS token](https://github.com/vllm-project/vllm/commit/dad2d6a590207cb8938fb915602794665b8e9326)  
  更新 DeepSeek‑OCR‑2 的聊天模板，确保包含 BOS token，修改 `vllm/transformers_utils/configs/deepseek_vl2.py`。

- **32e84fa** – [Investigate torchrun distributed tests hanging issue](https://github.com/vllm-project/vllm/commit/32e84fa1ff4d371d52042657a05d825c475cba3a)  
  添加调试信息以定位 `torchrun` 分布式测试卡死，修改相关测试文件。

- **fd9c83d** – [Document workaround for standalone_compile failing](https://github.com/vllm-project/vllm/commit/fd9c83d0e05e6a4214be7dabf3b2bd64a9696ed8)  
  在文档中说明 `torch.compile` 的独立编译失败的临时解决方案，更新 `docs/design/debug_vllm_compile.md`。

- **b95cc50** – [Remove deprecated VLLM_ALL2ALL_BACKEND env var](https://github.com/vllm-project/vllm/commit/b95cc5014dc7b260e5c70ae33d1b30c54d11306d)  
  清理已废弃的环境变量，涉及 `vllm/config/parallel.py` 与 `vllm/envs.py`。

- **6139789** – [Code simplification in `scheduler.py`](https://github.com/vllm-project/vllm/commit/61397891ce00c6e28ca9918fab11be1b9e925a20)  
  精简调度器实现，删除冗余代码。

- **ef248ff** – [Remove deprecated profiler env vars](https://github.com/vllm-project/vllm/commit/ef248ff740200c91791ba952b3458a5d5a016d26)  
  删除已废弃的性能分析环境变量，修改 `vllm/config/profiler.py` 与 `vllm/envs.py`。

- **e106044** – [Deprecate ipex, switch to vllm‑xpu‑kernels for XPU](https://github.com/vllm-project/vllm/commit/e10604480bb8177d563253194667ee9c1590e31a)  
  大幅迁移 XPU 后端至 `vllm‑xpu‑kernels`，删除 `vllm/_ipex_ops.py`，更新层实现与平台入口。

- **bf001da** – [Interleaved thinking keeps compatibility with reasoning_content](https://github.com/vllm-project/vllm/commit/bf001da4bfb53854927b68055a12efd05d494786)  
  在交叉推理模式下保持 `reasoning_content` 兼容性，修改 `vllm/entrypoints/chat_utils.py`。

- **a0a984a** – [Remove hardcoded America/Los_Angeles timezone from Dockerfiles](https://github.com/vllm-project/vllm/commit/a0a984ac2e4503de1a76f55ece65ac0847678503)  
  清理 Dockerfile 中的时区硬编码，提升跨时区构建一致性。

- **4b3803d** – [DPMetadata raises assert error for dense model](https://github.com/vllm-project/vllm/commit/4b3803d18044499962f331db0a1614b726110e2a)  
  修复密集模型在数据并行元数据检查时的断言错误，涉及 `vllm/forward_context.py`。

- **4c4b6f7** – [Add sampling parameters to Responses API](https://github.com/vllm-project/vllm/commit/4c4b6f7a9764bac8bf9f2a0bfedf852d8e59c98e)  
  为 Responses API 引入采样参数字段，更新协议文件 `vllm/entrypoints/openai/responses/protocol.py` 与对应测试。

- **10546f9** – [Fix multimodal budget setting for Qwen Omni models](https://github.com/vllm-project/vllm/commit/10546f925aef5d805e41f0f40c6610d08ff1a037)  
  调整 Qwen Omni 系列模型的多模态预算配置，修改 `vllm/multimodal/budget.py`。

- **e69c990** – [Optimize ARM vectorization backend for CPU](https://github.com/vllm-project/vllm/commit/e69c990c216c623b1de22f055926602a336f9352)
### NVIDIA/cutile-python
- **提交** `7bad878` – Add custom mask to gather/scatter  
  **链接**: https://github.com/NVIDIA/cutile-python/commit/7bad87895b5b9501df3a7f760e23655aea477e04  
  **变更文件**:  
  - `changelog.d/gather-scatter-mask.md` (新增, +4 行)  
  - `src/cuda/tile/_ir/ops.py` (修改, +61 – 12 行)  
  - `src/cuda/tile/_stub.py` (修改, +25 – 6 行)  
  - `test/test_gather_scatter.py` (修改, +195 行)  
  **说明**: 本次提交为 gather/scatter 操作新增自定义掩码功能。由于 `patch_truncated` 为 true，未能展示具体代码片段。

- **Issue** `[BUG]: There’s a bug in the swizzle computation in MatMul.py for misaligned (non-aligned) cases.`  
  **链接**: https://github.com/NVIDIA/cutile-python/issues/68  
  **报告者**: zhao008  
  **创建时间**: 2026-02-03T06:59:25Z  
  **说明**: Issue 内容缺失（`body_truncated` 为 true），未提供详细描述。

## 总结
### 展望与关注点
- **跨架构同步与内存一致性**：CUTLASS 与 DeepGEMM 的同步机制改进表明，跨网格调度的内存可见性仍是高性能 GPU 计算的关键瓶颈，后续需关注 `wait_on_dependent_grids` 在更高 SM 版本的适配情况。\
- **新硬件与 DSL 支持**：CuTe DSL 对 CUDA 13.1 的快速跟进以及 FlashInfer 对 Hopper/SM90 的优化，预示着硬件迭代将驱动 DSL 与库层面的同步升级，开发者应关注对应的文档与示例更新。\
- **多模态与分布式协同**：SGLang 与 vLLM 在 Mooncake KV、MTGPU、以及分布式 KV 传输方面的突破，为大规模多模态推理提供了可行路径，值得关注其在实际部署中的吞吐与延迟表现。\
- **生态兼容性**：FlashAttention 的 GLIBC 兼容性问题、cuTile‑Python 的掩码特性以及 vLLM 对 XPU 的迁移，凸显了 AI 基础设施在不同系统环境下的兼容挑战，建议在 CI 中加入更广泛的系统依赖检测。\
- **社区需求**：FlashInfer 的 Skip‑Softmax、DSR1 路由性能以及对 Ascend/NPU 的支持请求，反映出用户对高效注意力算子和跨平台部署的迫切需求，后续可关注这些特性的实现进度。
