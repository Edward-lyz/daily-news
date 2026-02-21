# Daily AI Infra Report

## 往期回顾
- [2026-02-18](reports/2026-02-18.md)
- [2026-02-17](reports/2026-02-17.md)
- [2026-02-16](reports/2026-02-16.md)
- [2026-02-15](reports/2026-02-15.md)
- [2026-02-13](reports/2026-02-13.md)
- [2026-02-12](reports/2026-02-12.md)
- [2026-02-10](reports/2026-02-10.md)

---

## 最新解读 (2026-02-19)
今日的 AI Infra 的新闻如下。

## 摘要
```json
{
  "summary": "FlashInfer 发布了 0.6.4 版本，新增 BF16 GEMM 基准测试支持并改进了 CI/CD 流程，同时修复了 MLA 的 skip softmax 稀疏性问题。Flash Attention 在 SM100 后向传播方面进行了重大优化，但出现了 GB200 上 cute DSL 的段错误问题。SGLang 扩展了对 AMD 的支持，添加了 FP8 在线量化、SWA 在 trtllm_mha 中的支持，并增强了多模态生成能力。VLLM 更新了 FlashInfer 版本并重新启用了 DeepSeek NVFP4 AR+Norm 融合，同时修复了 torch.compile 时间不一致问题并添加了 MLA 后端的 FP8 KV 缓存测试。",
  "conclusion": "各框架持续优化性能和硬件兼容性，但需关注潜在风险：FlashInfer 的 Hopper FP8 NaN 问题可能影响 Qwen3.5-397B-A17B-FP8 模型稳定性；VLLM 与 HuggingFace 嵌入结果不一致问题需要进一步调查；Flash Attention 在 GB200 上的段错误可能影响特定部署场景。建议用户密切关注这些问题的发展，并在生产环境中谨慎采用相关更新。"
}
```

## 具体内容分析
### deepseek-ai/DeepGEMM
昨日无更新。
### deepseek-ai/FlashMLA
昨日无更新。
### NVIDIA/cutlass
昨日无更新。
### flashinfer-ai/flashinfer
## FlashInfer 更新摘要 (2026-02-19)

### 主要提交

1. **BF16 GEMM 基准测试支持** (#2525)
   - 添加了对 SM10.0+ GPU 的 BF16 矩阵乘法支持
   - 新增 `mm_bf16` 和 `bmm_bf16` 基准测试例程
   - [查看详情](https://github.com/flashinfer-ai/flashinfer/commit/1d3ac82b01d608c235a81cf5ef24b659ba1e9be6)

2. **版本更新至 0.6.4** (#2565)
   - 修复了多项问题并改进了性能
   - [查看详情](https://github.com/flashinfer-ai/flashinfer/commit/f1e6fdcb8f65104047697f022b5d055ef022d763)

3. **TRTLLM-Gen 跳过 Softmax 注意力支持** (#2547)
   - 为 MLA (Multi-Head Latent Attention) 添加了跳过 Softmax 稀疏性支持
   - 可通过 `skip_softmax_threshold_scale_factor` 参数启用
   - [查看详情](https://github.com/flashinfer-ai/flashinfer/commit/11537c7c8ad9b6c89b021c269c3144bb71897a36)

4. **改进 CI/CD 流程**
   - 修复了 H100 清理问题 (#2590)
   - 为夜间发布作业添加清理步骤 (#2510)
   - 添加了 Hopper 架构支持 (#2552)
   - [查看详情](https://github.com/flashinfer-ai/flashinfer/commit/9d733dd3eafa24ea8b4eaf7749d3a382bafd678f)

5. **外部贡献者问题认领工作流** (#2586)
   - 添加了 `!claim` 命令让任何人认领未分配的问题
   - 添加了 `!assign @username` 命令让维护者分配特定用户
   - [查看详情](https://github.com/flashinfer-ai/flashinfer/commit/f98a52d75b3c8d649b3ae9132e9ca1854a419381)

### 问题追踪

1. **缺少 TRTLLM-GEN 不规则内核** (#2596)
   - MLA prefill (head_dim_qk=192, head_dim_v=128) 在 SM103 上缺少 BF16/FP16 -> FP8 输出内核
   - 影响 vLLM 中的 MLA 注意力实现

2. **Hopper 上 CUTLASS FP8 内核的 NaN 问题** (#2595)
   - 当 FP8 块缩放极小时 (~1e-23)，FlashInfer CUTLASS MoE 内核产生 NaN
   - 影响 Qwen3.5-397B-A17B-FP8 模型中的"死亡"专家

3. **NVFP4 MoE 的稀疏性 cubins API** (#2589)
   - 需要将稀疏性 cubins 和 trtllm-gen cubins 添加到 MOE Python 和 C++ 接口中

### 性能改进

- 修复了多 CTA 协作中 max_len > length 时的 chunk_end 计算问题 (#2489)
- 为 DGX Spark (SM121) 启用了 fmha_v2_prefill_deepseek 支持 (#2559)
### Dao-AILab/flash-attention
**提交**  

| 短 SHA | 信息 | 作者 | 日期 | PR 链接 |
|--------|------|------|------|--------|
| `6079a9b` | [Bwd,Sm100] Fix num reg variables | Tri Dao | 2026‑02‑20 04:22 UTC | https://github.com/Dao-AILab/flash-attention/commit/6079a9bf4cfd7af8e7586afea6c49a97ebddf46e |
| `710d3cc` | BWD sm100 2cta (#2202) | Ted Zadouri | 2026‑02‑20 01:44 UTC | https://github.com/Dao-AILab/flash-attention/commit/710d3cc239eb5171e8b87bcde9e51349d4affe8b |

**代码变更概览**  

- **`flash_attn/cute/flash_bwd_sm100.py`**  
  - 第一次提交：仅 2 行增删（patch 未提供，具体改动不可见）。  
  - 第二次提交：大幅改动，新增 922 行，删除 343 行，合计 1 265 行变更。由于 `patch_truncated` 为 `false`，但实际补丁内容未在数据中给出，故无法展示代码片段。  

- **`flash_attn/cute/blackwell_helpers.py`**  
  - 增加 21 行，删除 7 行，合计 28 行改动。具体 diff 未提供。  

- **`flash_attn/cute/copy_utils.py`**  
  - 新增 32 行，未删除。具体实现细节缺失。  

- **`flash_attn/cute/flash_bwd_postprocess.py`**  
  - 大幅重构：新增 195 行，删除 65 行，合计 260 行。代码细节未展示。  

- **`flash_attn/cute/mask.py`**  
  - 仅 1 行增删（可能是微调注释或条件）。  

**总体统计**  
- 总提交数：2  
- 新增行数：1 173  
- 删除行数：418  

---

**Issues**  

| 标题 | 作者 | 创建时间 | 链接 | 简要说明 |
|------|------|----------|------|----------|
| [Cute][GB200] Segfault on test with cute dsl 4.4.0 | henrylhtsang | 2026‑02‑19 20:53 UTC | https://github.com/Dao-AILab/flash-attention/issues/2264 | 在 GB200 GPU 上使用 cute DSL 4.4.0 运行 `pytest -x -s test_flash_attn.py` 时出现段错误，B200 上未复现。 |
| [Question] Numerics of e2e | henrylhtsang | 2026‑02‑19 01:43 UTC | https://github.com/Dao-AILab/flash-attention/issues/2263 | 询问 exp2 仿真是否位相等价，尤其在 BF16 输出时出现 1 ULP 偏差，怀疑可能是 FMA 误差。 |

*注：两个 Issue 的正文均被截断，详细内容未提供。*
### sgl-project/sglang
提交：
- Upd: CODEOWNERS (#19055) [`b2573fe`](https://github.com/sgl-project/sglang/commit/b2573fe4267ae3a6e3cda521fad15e8efb07dc0c)
- 受影响文件: .github/CODEOWNERS
- [GPT-OSS] support fp8 online quantization for gpt-oss bf16 (#18988) [`4bffd3a`](https://github.com/sgl-project/sglang/commit/4bffd3a2323a332506ce16f0fbd2ce7e96db2204)
- 受影响文件: python/sglang/srt/layers/quantization/fp8.py, python/sglang/srt/server_args.py
- Add generated-shared-prefix dataset in bench_one_batch (#18986) [`96bae23`](https://github.com/sgl-project/sglang/commit/96bae2355e7010773729d4ca5d2b84e4621ab658)
- 受影响文件: python/sglang/test/bench_one_batch_server_internal.py
- [feat] feat: support swa in trtllm_mha (#18970) [`ab18734`](https://github.com/sgl-project/sglang/commit/ab18734375accf3dbcd043f5b6244d4082ee84f5)
- 受影响文件: python/sglang/srt/layers/attention/trtllm_mha_backend.py
- [AMD] support two batch overlapping for mori ep (#17953) [`fbb6098`](https://github.com/sgl-project/sglang/commit/fbb60984872ee69d868b595f81c0155b15601de5)
- 受影响文件: docs/advanced_features/server_arguments.md, python/sglang/srt/batch_overlap/operations_strategy.py, python/sglang/srt/batch_overlap/two_batch_overlap.py, python/sglang/srt/layers/attention/aiter_backend.py, python/sglang/srt/layers/moe/ep_moe/layer.py, python/sglang/srt/layers/moe/fused_moe_triton/layer.py, python/sglang/srt/layers/moe/token_dispatcher/__init__.py, python/sglang/srt/layers/moe/token_dispatcher/moriep.py, python/sglang/srt/models/deepseek_v2.py, python/sglang/srt/server_args.py, python/sglang/test/bench_one_batch_server_internal.py
- Fix adjust_num_token_non_padded_for_attn_tp returning CPU tensor (#19051) [`38ee749`](https://github.com/sgl-project/sglang/commit/38ee749dd90cdab572d393bb856430ef92c981d2)
- 受影响文件: python/sglang/srt/model_executor/forward_batch_info.py
- [Fix] Run FlashInfer autotune on non-default stream for NCCL 2.29+ compatibility (#18987) [`3358ba8`](https://github.com/sgl-project/sglang/commit/3358ba894588f25da97c1c29fea9e32569669d60)
- 受影响文件: python/sglang/srt/distributed/device_communicators/pynccl_allocator.py, python/sglang/srt/model_executor/model_runner.py
- [Fix] DO NOT skip save_kv_cache for dllm (#19020) [`5285240`](https://github.com/sgl-project/sglang/commit/52852404c869d7be291b386073baca850f69aaef)
- 受影响文件: python/sglang/srt/layers/attention/flashinfer_backend.py
- Fix NSA FP8 KV cache path for both-trtllm MHA one-shot (#18931) [`f23a23c`](https://github.com/sgl-project/sglang/commit/f23a23cc05fca67d296426e4acb9ce69524f26e4)
- 受影响文件: python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py
- [diffusion] feat: support nunchaku for Z-Image-Turbo and flux.1 (int4) (#18959) [`8d789b5`](https://github.com/sgl-project/sglang/commit/8d789b5c3d7ed3f585dd27c6052b2e12fb5b96d0)
- 受影响文件: python/sglang/multimodal_gen/configs/models/dits/flux.py, python/sglang/multimodal_gen/runtime/layers/quantization/configs/nunchaku_config.py, python/sglang/multimodal_gen/runtime/loader/component_loaders/transformer_loader.py, python/sglang/multimodal_gen/runtime/loader/fsdp_load.py, python/sglang/multimodal_gen/runtime/loader/utils.py, python/sglang/multimodal_gen/runtime/models/dits/base.py, python/sglang/multimodal_gen/runtime/models/dits/flux.py, python/sglang/multimodal_gen/runtime/models/dits/qwen_image.py, python/sglang/multimodal_gen/runtime/models/dits/zimage.py
- [jit kernel] Support per_token_group_quant_8bit jit kernel (#18905) [`7d95344`](https://github.com/sgl-project/sglang/commit/7d953440ec965032b6b5af386d56c427ae4e6da1)
- 受影响文件: python/sglang/jit_kernel/benchmark/bench_per_token_group_quant_8bit.py, python/sglang/jit_kernel/csrc/gemm/per_token_group_quant_8bit.cuh, python/sglang/jit_kernel/per_token_group_quant_8bit.py, python/sglang/jit_kernel/tests/test_per_token_group_quant_8bit.py, python/sglang/jit_kernel/utils.py, python/sglang/srt/layers/quantization/fp8_kernel.py
- [diffusion] logging: log available mem when each stage starts in debug level (#18998) [`38a6965`](https://github.com/sgl-project/sglang/commit/38a69652e62b5b9682916bacd676afeea9f60614)
- 受影响文件: python/sglang/multimodal_gen/runtime/entrypoints/cli/generate.py, python/sglang/multimodal_gen/runtime/entrypoints/diffusion_generator.py, python/sglang/multimodal_gen/runtime/entrypoints/openai/utils.py, python/sglang/multimodal_gen/runtime/entrypoints/utils.py, python/sglang/multimodal_gen/runtime/loader/component_loaders/component_loader.py, python/sglang/multimodal_gen/runtime/managers/gpu_worker.py, python/sglang/multimodal_gen/runtime/managers/scheduler.py, python/sglang/multimodal_gen/runtime/pipelines_core/composed_pipeline_base.py, python/sglang/multimodal_gen/runtime/pipelines_core/executors/pipeline_executor.py, python/sglang/multimodal_gen/runtime/pipelines_core/schedule_batch.py, python/sglang/multimodal_gen/runtime/pipelines_core/stages/base.py, python/sglang/multimodal_gen/runtime/pipelines_core/stages/decoding.py, python/sglang/multimodal_gen/runtime/pipelines_core/stages/decoding_av.py, python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising.py, python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising_av.py, python/sglang/multimodal_gen/runtime/pipelines_core/stages/denoising_dmd.py
- 文件列表已截断。
- Fix lint on main (#19054) [`0d20cf5`](https://github.com/sgl-project/sglang/commit/0d20cf5a66531eed4db554589dd80c045b5c9e17)
- 受影响文件: python/sglang/srt/debug_utils/dumper.py, python/sglang/srt/managers/scheduler.py
- feature: docker patch workflow (#19025) [`77fdb6a`](https://github.com/sgl-project/sglang/commit/77fdb6af8146947095186c9f9d4629f2a372af4d)
- 受影响文件: .github/workflows/patch-docker-dev.yml
- fix lint on main (#19052) [`b59a22f`](https://github.com/sgl-project/sglang/commit/b59a22f781ef2f7090367b76d58e8ae722ed5ace)
- 受影响文件: python/sglang/srt/debug_utils/dumper.py
- [AMD] Replace msgpack with msgspec in MORI-IO (#19007) [`b0786cd`](https://github.com/sgl-project/sglang/commit/b0786cdf94935f0dec701079ea7e66d008560746)
- 受影响文件: python/sglang/srt/disaggregation/mori/conn.py
- [Fix][Qwen3.5] Pass max_mamba_cache_size to mamba pool in disaggregation decode path (#19002) [`8541b11`](https://github.com/sgl-project/sglang/commit/8541b1118d6cb396fe9c8f580ea597996635f8fb)
- 受影响文件: python/sglang/srt/disaggregation/decode.py, python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py
- Feature/sdar support (#19044) [`295bc17`](https://github.com/sgl-project/sglang/commit/295bc175760a1d690ca3f8e7c963dd5221df3815)
- 受影响文件: docs/supported_models/text_generation/diffusion_language_models.md, python/sglang/srt/dllm/config.py, python/sglang/srt/models/sdar.py, python/sglang/srt/models/sdar_moe.py
- Support using SGLang port in dumper (#19038) [`046ef0a`](https://github.com/sgl-project/sglang/commit/046ef0aa355fce3722502ea46e6aa5e5ee67458d)
- 受影响文件: python/sglang/srt/debug_utils/dumper.py, python/sglang/srt/entrypoints/http_server.py, python/sglang/srt/managers/io_struct.py, python/sglang/srt/managers/scheduler.py, python/sglang/srt/managers/tokenizer_communicator_mixin.py, test/registered/debug_utils/test_dumper.py
- Support resetting and enhance HTTP endpoints for dumper (#19046) [`2fecc2c`](https://github.com/sgl-project/sglang/commit/2fecc2c075a18d2e6ec8165aeaa8e7ecdb71e77c)
- 受影响文件: python/sglang/srt/debug_utils/dumper.py, test/registered/debug_utils/test_dumper.py
- Enhance configure and env parsing in dumper (#19034) [`503bf30`](https://github.com/sgl-project/sglang/commit/503bf3047a51b398b9f8c57d70a9216216b5b1d9)
- 受影响文件: python/sglang/srt/debug_utils/dumper.py, test/registered/debug_utils/test_dump_comparator.py, test/registered/debug_utils/test_dumper.py
- Support filtering labels in dumper (#19018) [`df995aa`](https://github.com/sgl-project/sglang/commit/df995aab565db64a2ae99aafd4b0dacf52274c5e)
- 受影响文件: python/sglang/srt/debug_utils/dumper.py, test/registered/debug_utils/test_dumper.py
- Support captured dump output and console output control in dumper (#19017) [`261bca3`](https://github.com/sgl-project/sglang/commit/261bca3c58c617def40aa226b92be137167397d4)
- 受影响文件: python/sglang/srt/debug_utils/dumper.py, test/registered/debug_utils/test_dumper.py
- Hint users when wrongly execute it with partial ranks in dumper (#19014) [`fc1500a`](https://github.com/sgl-project/sglang/commit/fc1500adc628e657d19a94d9d997a5176c623753)
- 受影响文件: python/sglang/srt/debug_utils/dumper.py, test/registered/debug_utils/test_dumper.py
- Support cleanup previous dumps in dumper (#19013) [`b41d412`](https://github.com/sgl-project/sglang/commit/b41d412c3d9ec606edadfb4bdc61f3076faf1ef2)
- 受影响文件: python/sglang/srt/debug_utils/dumper.py, test/registered/debug_utils/test_dumper.py
- Fix flashinfer autotune to only wrap run_once() (#19004) [`13a4a04`](https://github.com/sgl-project/sglang/commit/13a4a0406eecb8bcd85c1a3994fc89ec386b42b0)
- 受影响文件: python/sglang/srt/model_executor/model_runner.py
- Fix long prompt KV allocation by falling back to torch native APIs when exceeding Triton tensor limit (#18250) [`64bca53`](https://github.com/sgl-project/sglang/commit/64bca5315f59ae6b7e57cf4d8a13180f0e7e518f)
- 受影响文件: python/sglang/srt/mem_cache/allocator.py
- Register tensors with symmetric memory for qwen (#18643) [`99df920`](https://github.com/sgl-project/sglang/commit/99df920cdbba20f033ba58765f5da04f59efe4fe)
- 受影响文件: python/sglang/srt/models/qwen2_moe.py
- Revert "Add SDAR model support" (#19032) [`73a7f0d`](https://github.com/sgl-project/sglang/commit/73a7f0d04997aa7528abacf18c90c07ec4819ef3)
- 受影响文件: docs/supported_models/text_generation/diffusion_language_models.md, python/sglang/srt/dllm/config.py, python/sglang/srt/models/sdar.py, python/sglang/srt/models/sdar_moe.py, test/registered/dllm/test_sdar.py
- Tiny remove duplicate coredump env injection (#19023) [`db34c1c`](https://github.com/sgl-project/sglang/commit/db34c1cbfbf1f74152cee4965fcd9365d1d69aa9)
- 受影响文件: python/sglang/srt/debug_utils/cuda_coredump.py
- [spec v2]Fix torch gc of future indices (#18958) [`5ff5aa6`](https://github.com/sgl-project/sglang/commit/5ff5aa6923b914b03b4592b841ee9d69bfde664e)
- 受影响文件: python/sglang/srt/managers/overlap_utils.py
- Add SDAR model support (#18318) [`44ab752`](https://github.com/sgl-project/sglang/commit/44ab752b7aeaf4a26ce6cbab972a74372126879b)
- 受影响文件: docs/supported_models/text_generation/diffusion_language_models.md, python/sglang/srt/dllm/config.py, python/sglang/srt/models/sdar.py, python/sglang/srt/models/sdar_moe.py, test/registered/dllm/test_sdar.py
Issues：
- [Feature] ROCm nightly in upstream lmsysorg docker org (https://github.com/sgl-project/sglang/issues/19031)
- [Bug] GLM5 nightly Mi355 broken due to transformer dependency (https://github.com/sgl-project/sglang/issues/19028)
- Issue 内容已截断。
- [Bug] Missing saving kv for LLaDA2 (https://github.com/sgl-project/sglang/issues/19019)
- Issue 内容已截断。
- [Bug] cutlass_fp4_group_mm does not support SM 110 (Jetson AGX Thor) (https://github.com/sgl-project/sglang/issues/18994)
- Issue 内容已截断。
- [Feature] Support NVFP4 kernels for SM110 (Jetson AGX Thor) (https://github.com/sgl-project/sglang/issues/18993)
### vllm-project/vllm
**提交**

- **ea5f903** – 更新 Flashinfer 版本并重新启用 DeepSeek NVFP4 AR+Norm 融合。涉及 `docker/Dockerfile*`、`docker/versions.json`、`requirements/cuda.txt`、`vllm/model_executor/models/config.py`（删除 24 行，新增 1 行）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/ea5f903f80fec5afd4960a3846b8a84b0e53ca6e)

- **0632ed8** – 修复 AMD CI 中 `test_custom_allreduce` 对 A100 测试组的兼容性。修改 `tests/distributed/test_custom_all_reduce.py`（新增 2 行）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/0632ed8778cab44de6152eb873d09fa40c241962)

- **aaefc58** – 回滚 PR #34818 与 #33600，修正多项测试与配置错误。涉及 `tests/*`、`vllm/config/*`、`vllm/engine/arg_utils.py`、`vllm/model_executor/layers/attention/*`、`vllm/platforms/cuda.py`（共 536 行改动）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/aaefc58ee0f023ec7bd3671ca83aae1b8a8f271d)

- **f24b2de** – 为 MLA 后端添加 FP8 KV 缓存测试。修改 `tests/v1/attention/test_mla_backends.py`（新增 68 行，删除 27 行）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/f24b2de3d3812301932645e3002adba46a8c0055)

- **fac1507** – 移除失效的 Prime‑RL 集成测试。删除 `.buildkite/scripts/run-prime-rl-test.sh`，并在 CI 配置中删减 42 行。  
  [查看提交](https://github.com/vllm-project/vllm/commit/fac1507f03c78d8717853c8a15ad1d887d71cc1d)

- **f863994** – 修正 `torch.compile` 日志中时间不一致的问题。修改 `vllm/compilation/*`（共 25 行改动）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/f8639940844bcc10e3f374d2bb5aa33ae52a2624)

- **e4a5d8c** – 将 `torch_aot_compile` 目录迁入 `torch_compile_cache`。修改 `vllm/compilation/decorators.py`（新增 5 行，删除 4 行）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/e4a5d8c653fc00adb06922bddcb7fec14b01a62b)

- **a6d0299** – Helion kernel 添加 `num_tokens` 维度至 `silu_mul_fp8` 自动调优与分发。涉及 `vllm/kernels/helion/*`（新增 55 199 行配置，修改 93 行代码）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/a6d0299c75f2c0687334d50d302801ade083c784)

- **6ce80f7** – 防止 MkDocs v2 被误装。修改 `requirements/docs.txt`（行数微调）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/6ce80f7071b009badaa2c473e96ec55a134790d2)

- **1fe4621** – 在 `mamba_get_block_table_tensor` 中避免 dtype 提升同步，提升性能。修改 `vllm/v1/attention/backends/utils.py`（新增 6 行）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/1fe462168c381f604a5ef9d491a230a3dd861d2c)

- **ed31a02** – 将 SSE 事件构建抽离至 `streaming_events.py`，简化 `serving.py`。新增 897 行实现文件。  
  [查看提交](https://github.com/vllm-project/vllm/commit/ed31a020ee5e383a069a59750261a307bd8ddde4)

- **f9ac192** – 移除 V0 中未使用的 MM 占位符。修改 `vllm/outputs.py`（删除 5 行，仅保留 1 行）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/f9ac19204f0c4c3041d0afbe7d5eb4d63e73f15c)

- **59965af** – 修复 `_dummy_run` 缺少 `prepare_inputs_event` 同步导致的错误。修改 `vllm/v1/worker/gpu_model_runner.py`（新增 31 行）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/59965affbd6e652a3c8ed229b66ef34a681e5693)

- **b1c4f0b** – 优化分组 Top‑K kernel 实现。新增 `csrc/moe/moeTopKFuncs.cuh`，修改 `csrc/moe/grouped_topk_kernels.cu`（共 743 行改动）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/b1c4f0b26548d36fca304b298957e4791eafa09b)

- **8de7c63** – 修复 ROCm 上 AITER_FA 推理的投机执行支持。修改 `vllm/v1/attention/backends/rocm_aiter_fa.py`（新增 35 行）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/8de7c636cc02a8306441af868b9c1d0e6d64799f)

- **0597792** – 为 MXFP4/MXFP8 TRTLLM 后端添加日志输出。修改 `vllm/model_executor/layers/quantization/mxfp4.py`（新增 3 行）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/059779231f158b8b570e71aaa5c66f49b41b2fb1)

- **ea37530** – 为 LFM2 模型加入 LoRA 支持。修改 `vllm/model_executor/models/lfm2*.py`（共 52 行改动）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/ea37530b474fa738a99a53a8975af4e389b968c7)

- **f5432e3** – 放宽 ROCm CI 中 RemoteOpenAIServer 启动超时限制。修改测试文件 `tests/entrypoints/openai/test_serving_chat.py` 与 `tests/utils.py`（各 1 行改动）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/f5432e35a3a4f0bd6e7d49c51a35a0a01bc32452)

- **07cab21** – 添加已废弃的环境变量工具函数。修改 `vllm/config/utils.py`（新增 65 行）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/07cab212f0dcc51cfe4e4f93b58935e8079f26b7)

- **0c1dc42** – 为 `test_moriio_connector.py` 添加 `default_vllm_config`，确保 AMD CI 通过。修改 `tests/v1/kv_connector/unit/test_moriio_connector.py`（新增 8 行）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/0c1dc42748760fc75aef68e973c9ff7a47501337)

- **676f82a** – 在系统消息中加入非文本内容校验，防止错误输入。修改 `vllm/entrypoints/openai/chat_completion/protocol.py`（新增 49 行）并更新对应测试。  
  [查看提交](https://github.com/vllm-project/vllm/commit/676f82ae8140a512dae73bcae6c6d23907f55e0e)

- **81bfc21** – 改进 FP8 Oracle，实现针对特定配置的 kernel 选择。修改 `vllm/model_executor/layers/fused_moe/oracle/fp8.py`（新增 46 行）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/81bfc21a6ad0cb498dbe5466ccf2987624efbba5)

- **4e2c7ca** – 为 MoE 量化配置在 `torch.compile` 下添加回归测试。修改 `tests/quantization/test_compressed_tensors.py`（新增 23 行）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/4e2c7caf2d11444ce6c1e4895bc921c93610bd7c)

- **d9e62c0** – 修复 Quark 在 MI300 上的 fp8 激活尺度处理。修改 `vllm/model_executor/layers/quantization/quark/quark_moe.py`（微调 6 行）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/d9e62c03eb98e3adcf82a2177f4a8b8f851406e4)

- **a1a2d79** – 更新 CPU arm64 镜像构建脚本使用正确标签。修改 `.buildkite/image_build/image_build_cpu_arm64.sh`（行数对调）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/a1a2d79442ed00284e70b829e07cadbb887bdf73)

- **ac900c8** – 在 LLM 接口实现输出类型检查。修改 `vllm/entrypoints/llm.py` 与 `vllm/v1/engine/llm_engine.py`（共 96 行改动）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/ac900c89bba77f69ed42e8a19a5006bd215eeb80)

- **76df607** – 修正 `pause_scheduler()` 中的状态名称错误。修改 `vllm/v1/engine/core.py`（前后各 9 行）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/76df6072ff4829980ad71764191fc970a873275a)

- **16f24e8** – 为 H100 添加 GPT‑OSS 评估 CI 作业。修改 `.buildkite/test_areas/misc.yaml`（新增 13 行）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/16f24e87975ef4cd2c12879425062913ef62f6fd)

- **40b2f1c** – CPU 端微调模型运行器的共享内存广播与异步工具。修改 `vllm/distributed/device_communicators/shm_broadcast.py`、`vllm/v1/worker/gpu/*`（共 38 行改动）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/40b2f1c3d9c1dbcec185e8b6911fd273524f5b88)

- **648951a** – 修复 `benchmark_fused_collective` 在自定义算子初始化时的崩溃。修改 `benchmarks/kernels/benchmark_fused_collective.py`（新增 11 行）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/648951a9c3ab7d8ade25b80edb55eb4018acfd58)

- **f72061a** – 为 MoE 提供更具描述性的 `is_supported_config` 错误信息。修改 `vllm/model_executor/layers/fused_moe/*` 与量化工具文件（共 48 行改动）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/f72061a19ae7fbb7f193c31f0abea355fab41892)

- **662205d** – 修复基础模型测试失败。涉及 `tests/*`、`vllm/config/*`、`vllm/model_executor/layers/attention/*`、`vllm/platforms/cuda.py` 等（共 394 行改动）。  
  [查看提交](https://github.com/vllm-project/vllm/commit/662205d34eb1bb42228768d7a69a1ac4abf38c89)

**Issues**

- **[#34910](https://github.com/vllm-project/vllm/issues/34910)** – **Embedding 结果不一致**：用户报告 vllm 的嵌入向量与 HuggingFace `sentence-transformers` 的输出差异显著。涉及 `vllm` 的 `LLM.embed` 接口与 HF `SentenceTransformer.encode` 的对比实验。当前已在社区讨论 4 条评论，待进一步定位差异根源。  
  作者：ehsankf（2026‑02‑19）  

---  

*以上为 2026‑02‑19 当日 vllm 项目关键提交与热点 Issue 概览。*
### NVIDIA/cutile-python
昨日无更新。

## 总结
- Diff 内容已截断以满足 prompt 预算。
- OpenRouter repo summarize failed for sgl-project/sglang: OpenRouter 429 Too Many Requests (z-ai/glm-4.5-air:free): {"error":{"message":"Provider returned error","code":429,"metadata":{"raw":"z-ai/glm-4.5-air:free is temporarily rate-limited upstream. Please retry shortly, or add your own key to accumulate your rate limits: https://openrouter.ai/settings/integrations","provider_name":"Z.AI","is_byok":false}},"user_id":"user_2wqU29q2Bhpw2S2iw7Pwn8RHaXB"}
