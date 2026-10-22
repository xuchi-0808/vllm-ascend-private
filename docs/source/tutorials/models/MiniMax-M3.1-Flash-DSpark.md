# MiniMax-M3.1-Flash + DSpark

This tutorial records an experimental single-node MiniMax-M3.1-Flash W8A8
integration with its five-layer DSpark checkpoint. The recorded A3 run used
16 NPUs, TP4 × DP4, MRV2, a `FULL_DECODE_ONLY` target and an eager draft with
seven speculative tokens. Text, one synthetic image, one synthetic video and
one `tool_choice: auto` request completed with measured speculative acceptance.

The Flash checkpoint uses the standalone adapter under
[`examples/spec_decode/minimax_m3_flash_dspark`](https://github.com/ycl-ai-infra/vllm-ascend-private/tree/releases/v0.30.0rc2/examples/spec_decode/minimax_m3_flash_dspark).
Its A3 CPU FP8-to-BF16 loading path is selected explicitly through a separate
draft view with architecture `MiniMaxM31FlashA3DSparkModel`. The source checkpoint
keeps `DSparkMiniMaxDraftModel`; the plugin does not replace that architecture's
native registration and preserves an existing `AutoConfig` mapping. A5 retains
its native draft loading. No A5 change or A5 validation is part of this example.
The older
[MiniMax-M3 QuaRot + DSpark tutorial](MiniMax-M3-DSpark.md) describes a different
checkpoint and vocabulary/rotation conversion. Do not apply that conversion to
this Flash checkpoint.

The seven-request A3 serving snapshot below predates this alias isolation: it
used the same adapter algorithm under the old `DSparkMiniMaxDraftModel`
registration. The updated preparation and preflight commands target the new
isolated alias. Alias-isolation preflight is a configuration check; it is not a
repeat of the seven full-model serving requests under the new alias.

## 1. Prepare an isolated environment

Create a separate container using the A3 device, driver and shared-memory mounts
in the [MiniMax-M3 installation guide](MiniMax-M3.md#41-docker-image-installation).
Mount the original checkpoint read-only and use a separate writable directory
for source, build output, draft views, media and logs. A container clone does not
copy files hidden behind host bind mounts; reproduce those mounts explicitly.

Use the same Python environment for both native builds, the adapter and serving.
The recorded environment had Python 3.11.10, CANN 9.2.0, Torch 2.10.0 and
torch_npu 2.10.0.post7 on `Ascend910_9382`. Use the framework revision in
`.github/vllm-main-verified.commit`; do not substitute an arbitrary vLLM head.
See also the [private source reproduction notes](MiniMax-M3-private-reproduction.md).

Set deployment inputs before running the commands below. These paths are
examples and must be replaced with the paths inside the new container.

```bash
export WORK_DIR=/path/to/dspark-work
export TARGET_MODEL=/path/to/MiniMax-M3.1-Flash-W8A8
export DRAFT_SOURCE="$TARGET_MODEL/dspark"
export DRAFT_VIEW="$WORK_DIR/draft-flash-view"
export RUN_DIR="$WORK_DIR/run"
export MEDIA_DIR="$WORK_DIR/media"
export ASCEND_SRC="$WORK_DIR/vllm-ascend-private"
export CORE_SRC="$WORK_DIR/vllm"
export FA_SRC="$WORK_DIR/flash-attention-npu"
export CANN_ENV=/path/to/ascend-toolkit/set_env.sh
export ATB_ENV=/path/to/atb/set_env.sh
mkdir -p "$WORK_DIR" "$RUN_DIR" "$MEDIA_DIR"
source "$CANN_ENV"
source "$ATB_ENV"
```

## 2. Integrate and rebuild both operator sources

The required changes are in two repositories:

- [flash-attention-npu PR #348](https://github.com/MinghuasLab/flash-attention-npu/pull/348)
  fixes device/thread mask initialization and workspace sizing for Ascend 910 FA3.
- [vLLM Ascend private PR #9](https://github.com/ycl-ai-infra/vllm-ascend-private/pull/9)
  supplies the DSpark/FA3 integration on `releases/v0.30.0rc2`.

Fetch the current default branch and PR head, inspect their relationship and
integrate the PR when it is not already included. Rebuild after the final source
update: replacing Python files or reusing an old native extension is insufficient.
Record `git rev-parse HEAD` and submodule identities for every new run.

### 2.1. Ecosystem FA3 build

Follow the [ecosystem README](https://github.com/MinghuasLab/flash-attention-npu#build-from-source)
for its prerequisites, including `packaging`, `psutil`, the CANN compiler and a
Torch/torch_npu pair compatible with the container. The exact recorded input was
main `920da315f077ab712675644212915549669f6d09` plus PR #348 head
`e76e24b67145aa42ac51e84055b195fe5b3264b5`:

```bash
git clone https://github.com/MinghuasLab/flash-attention-npu.git "$FA_SRC"
git -C "$FA_SRC" checkout 920da315f077ab712675644212915549669f6d09
git -C "$FA_SRC" fetch origin pull/348/head
git -C "$FA_SRC" cherry-pick e76e24b67145aa42ac51e84055b195fe5b3264b5
git -C "$FA_SRC" submodule update --init --recursive
cd "$FA_SRC"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_FORCE_REBUILD=TRUE
export FLASH_ATTN_BUILD_VERSION=v3
export FLASH_ATTN_BUILD_NPU=910
export FLASH_ATTN_SKIP_SUBMODULE_INIT=1
python -m pip install . --no-build-isolation --no-deps --no-cache-dir -v
```

The integrated build used CATLASS
`769cd40a8716b28650b6bebb08db4834eea4462f` and produced
`flash_attn_npu-0.4.2-cp311-cp311-linux_aarch64.whl`.
`MAX_JOBS` does not limit this version's ecosystem build; do not infer its CPU
concurrency from that variable.

### 2.2. vLLM Ascend full native rebuild

```bash
git clone --branch releases/v0.30.0rc2 \
  https://github.com/ycl-ai-infra/vllm-ascend-private.git "$ASCEND_SRC"
git -C "$ASCEND_SRC" fetch origin pull/9/head:review-dspark
git -C "$ASCEND_SRC" merge --no-edit review-dspark
git -C "$ASCEND_SRC" submodule update --init --recursive
```

At the time of the recorded deployment, the default branch was
`75b3f4bf5d233c0627af86de1fc0181534ecfadc`, PR #9 was
`a03189c183ac1c280ad0c9721e07ef177a906550`, the Ascend CATLASS gitlink was
`41bf90da655bba3c66d0acd7e00abe33960ecfd6`, and paired vLLM was
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`. These identities describe the
measured source snapshot, not future branch heads.

Install that paired vLLM source using the empty-backend source-build procedure
from the [installation guide](../../getting_started/installation.md). Preserve
the container's matched Torch, torch_npu and Triton versions. Then build all
Ascend operators and the Torch extension:

```bash
cd "$ASCEND_SRC"
export SOC_VERSION=ascend910_9382
export MAX_JOBS=32
export CMAKE_BUILD_PARALLEL_LEVEL=32
export SETUPTOOLS_SCM_PRETEND_VERSION=0.30.0rc2
export COMPILE_CUSTOM_KERNELS=1
python -m pip install -e . --no-build-isolation --no-deps --no-cache-dir -v
```

Select `SOC_VERSION` for the actual hardware and bound build concurrency to the
container's resources. The recorded build completed all 714 configured native
build tasks and installed `vllm_ascend==0.30.0rc2`. Keep the complete generated
`_cann_ops_custom` vendor package, including AICPU libraries and operator JSON;
copying only the Torch extension is insufficient. The reproduction notes cover
competing OPP environment hooks and vendor initialization.

## 3. Install the Flash adapter and apply the core prerequisite

```bash
export EXAMPLE_DIR="$ASCEND_SRC/examples/spec_decode/minimax_m3_flash_dspark"
python -m pip install -e "$EXAMPLE_DIR" --no-build-isolation --no-deps
export CORE_SPECULATIVE_PY="$CORE_SRC/vllm/config/speculative.py"
python "$EXAMPLE_DIR/patch_core_guard.py" "$CORE_SPECULATIVE_PY"
python "$EXAMPLE_DIR/patch_core_guard.py" "$CORE_SPECULATIVE_PY" --apply
```

Point `CORE_SPECULATIVE_PY` at the `speculative.py` used by the installed vLLM,
not a different checkout. The first invocation displays the proposed diff.
The explicit `--apply` invocation is required before preflight or launch: it adds
only `MiniMaxM31FlashA3DSparkModel` beside the existing `K3DSparkModel` architecture
guard and creates a `.py.before-minimax-dspark-flash` backup. The script refuses
an unexpected guard layout. Review a changed core version instead of bypassing
the guard broadly.

Installing vLLM Ascend or this example does not automatically modify another
repository's vLLM core. The example's setup installs only the adapter package;
run the preparation, patch and preflight scripts through their example paths.
If `VLLM_PLUGINS` is set, include `minimax_dspark_flash` in its allowlist alongside
the other plugins required by the environment.
The plugin registers only the A3 experimental alias. Original
`DSparkMiniMaxDraftModel` registration and any existing `minimax_m3_text`
`AutoConfig` mapping remain available for their native loading paths, including
A5. Installing the plugin alone does not select the A3 dequantization path.

### 3.1. Prepare an independent draft view

```bash
python "$EXAMPLE_DIR/prepare_draft_view.py" \
  --source "$DRAFT_SOURCE" \
  --destination "$DRAFT_VIEW" \
  --hidden-norm gemma \
  --final-norm gemma
```

The two norm choices are explicit experimental inputs. The recorded run used
Gemma semantics for both auxiliary hidden norms and final norm. The tool checks
the 97-tensor checkpoint schema, writes a separate normalized configuration and
symlinks source files. It rejects a nonempty destination, a destination inside
the source, duplicate tensor names and its reserved preparation filename. It
preserves the original checkpoint configuration and tensors.

The source must declare `DSparkMiniMaxDraftModel`. Only the independent view is
rewritten to `MiniMaxM31FlashA3DSparkModel`; pass that view as the speculative
model to opt into this A3 experiment. That alias loads FP8 E4M3 weights with U8
E8M0 scales, dequantizes on CPU to BF16, and shares the target embedding and LM
head. Its draft has five dense
layers, partial rotary factor 0.5 and the checkpoint's SwiGLU parameters.
Pipeline parallelism is restricted to one stage in this example.
Do not prepare this A3 experimental view merely to use the existing A5 draft
loader. A5 continues to use its original checkpoint and native loading path;
this example does not establish A5 behavior or accuracy.

The checkpoint's original auxiliary/final-norm training semantics remain
unproven. Raw hidden-norm weights near zero motivate the Gemma trial, while the
final-norm distribution is ambiguous. No norm ablation was run. Target
verification can correct draft errors, so correct final answers alone do not
prove the draft's training semantics.

### 3.2. Run preflight

Configure the newly built vendor before importing or initializing NPU code:

```bash
export VENDOR="$ASCEND_SRC/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
test -f "$VENDOR/op_api/lib/libcust_opapi.so"
export ASCEND_CUSTOM_OPP_PATH="$VENDOR${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$VENDOR/op_api/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ASCEND_SRC:$CORE_SRC:${PYTHONPATH:-}"
python "$EXAMPLE_DIR/preflight_runtime_config.py" \
  --target "$TARGET_MODEL" --draft "$DRAFT_VIEW" \
  --target-quantization ascend --tp 4 \
  --output "$RUN_DIR/preflight.json"
```

Preflight checks the installed plugin entry point, the new
`MiniMaxM31FlashA3DSparkModel` alias, real `MiniMaxM3TextConfig`, signatures,
safetensors headers and real `SpeculativeConfig`. It checks alias isolation
separately from the earlier serving snapshot's old architecture registration.
It calls Ascend platform pre-registration before constructing model configs so
that the `ascend` quantization method is registered. Checkpoint custom code is
disabled by default; `--trust-remote-code` is an explicit opt-in for a trusted
checkpoint that requires it. No model, worker or forward pass is created.
Passing preflight is a prerequisite, not a serving result.

## 4. Start the single-node service

Set the host's actual communication interface and IP as deployment variables.
The example binds its API to loopback; select the API binding for the intended
network access. The recorded topology uses all 16 NPUs with TP4 × DP4.

```bash
export HCCL_SOCKET_IFNAME=communication-interface
export GLOO_SOCKET_IFNAME="$HCCL_SOCKET_IFNAME"
export HCCL_IF_IP=communication-ip
export VLLM_HOST_IP="$HCCL_IF_IP"
export API_HOST=127.0.0.1
export PORT=8000
export TP_SIZE=4
export DP_SIZE=4
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_BUFFSIZE=1024
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
export HCCL_CONNECT_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=600
export VLLM_CACHE_ROOT="$RUN_DIR/cache/vllm"
export TRITON_CACHE_DIR="$RUN_DIR/cache/triton"
export TORCH_EXTENSIONS_DIR="$RUN_DIR/cache/torch_extensions"
export ASCEND_CACHE_PATH="$RUN_DIR/cache/ascend"
ulimit -c 0

python -m vllm.entrypoints.openai.api_server --model "$TARGET_MODEL" \
  --host "$API_HOST" --port "$PORT" --served-model-name minimax-m3.1-flash \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --per-request-spec-decode-metrics summary \
  --trust-remote-code --quantization ascend --max-model-len 32768 \
  --tensor-parallel-size "$TP_SIZE" --data-parallel-size "$DP_SIZE" \
  --api-server-count 1 --max-num-batched-tokens 16384 \
  --long-prefill-token-threshold 4096 \
  --enable-expert-parallel --max-num-seqs 8 \
  --distributed-executor-backend mp --gpu-memory-utilization 0.92 \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice --tool-call-parser minimax_m3 \
  --limit-mm-per-prompt '{"image":1,"video":1}' \
  --allowed-local-media-path "$MEDIA_DIR" \
  --enable-chunked-prefill --no-enable-prefix-caching \
  --disable-hybrid-kv-cache-manager \
  --speculative-config "{\"model\":\"$DRAFT_VIEW\",\"method\":\"dspark\",\"num_speculative_tokens\":7,\"enforce_eager\":true,\"quantization\":null}" \
  --additional-config '{"enable_cpu_binding":true,"enable_flashcomm1":true,"ascend_compilation_config":{"fuse_norm_quant":false,"enable_static_kernel":false},"multistream_overlap_shared_expert":true,"enable_shared_expert_dp":true,"weight_nz_mode":2}' \
  > "$RUN_DIR/serve.log" 2>&1
```

Use `--trust-remote-code` only for the reviewed target checkpoint. Configure HCCS
or RoCE transport for the actual host before launch, following its communication
setup. Keep the vendor/Python environment from preflight in the serving process.
The recorded run disabled prefix caching. Leave `dspark_draft_topk` unset;
dynamic confidence and adaptive verification were not enabled. Draft eager
execution does not disable the target's full-decode graph mode.

## 5. Verify output and actual speculative acceptance

```bash
curl -fsS "http://127.0.0.1:$PORT/health"
curl -fsS "http://127.0.0.1:$PORT/v1/models"
curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"minimax-m3.1-flash","messages":[{"role":"user","content":"What is 17 + 25?"}],"temperature":0,"max_tokens":256,"chat_template_kwargs":{"thinking_mode":"disabled"}}'
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$RUN_DIR/metrics.txt"
```

Check the final content, finish reason, per-request speculative summary and
Prometheus counter changes. HTTP 200 alone does not show that DSpark was used or
that the answer is correct. Let A be accepted draft tokens, D proposed draft
tokens and S draft rounds. Aggregate acceptance is `A / D`; mean acceptance
length including the bonus token is `1 + A / S`. Sum the counters before
dividing, rather than averaging per-request percentages.

### 5.1. Recorded Flash smoke results

These are the original A3 deployment results using the adapter algorithm under
`DSparkMiniMaxDraftModel`, before the PR isolated it as
`MiniMaxM31FlashA3DSparkModel`. They do not represent a second serving run under
the new alias or an A5 test. The current alias-specific preparation/core guard
and preflight are separate configuration checks.

All seven requests returned HTTP 200. Six completed with `finish_reason: stop`;
the tool request completed with `finish_reason: tool_calls`.

| Case | A | D | S | Acceptance | Mean acceptance length | Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact color copy | 10 | 21 | 3 | 47.619% | 4.3333 | 34.028 |
| Exact paragraph copy 1 | 55 | 63 | 9 | 87.302% | 7.1111 | 1.560 |
| Arithmetic | 8 | 49 | 7 | 16.327% | 2.1429 | 3.039 |
| Exact paragraph copy 2 | 57 | 63 | 9 | 90.476% | 7.3333 | 1.092 |
| Image | 25 | 70 | 10 | 35.714% | 3.5000 | 4.239 |
| Video | 82 | 532 | 76 | 15.414% | 2.0789 | 11.265 |
| Tool auto | 38 | 98 | 14 | 38.776% | 3.7143 | 3.114 |
| **All seven** | **275** | **896** | **128** | **30.692%** | **3.1484** | — |

The image response identified a red square, blue circle and green triangle. The
video response described a red circle moving from left to right with slight
vertical jitter. These were synthetic basic media, not a standardized accuracy
dataset. The tool request initiated `add_numbers` with `{"a":17,"b":25}` under
`tool_choice: auto`. Tool execution/result-turn completion, `required` and
named-function constraints were not tested in this run.

The five service cases (arithmetic, paragraph copy 2, image, video and tool)
totaled **A210/D812/S116**. Each DP engine's Prometheus delta agreed with the
corresponding per-request summaries:

| DP engine | A | D | S |
| --- | ---: | ---: | ---: |
| 0 | 25 | 70 | 10 |
| 1 | 82 | 532 | 76 |
| 2 | 46 | 147 | 21 |
| 3 | 57 | 63 | 9 |

The remaining two copy requests belong to an earlier capture window, so their
tokens are included in the seven-request total but not in this five-case counter
delta. Per-request acceptance histograms independently matched the reported
counts. No counter reset or removed series was observed in the five-case window.
Final health/model checks passed and the serving-log snapshot contained no
`ERROR` or `Traceback`.

### 5.2. Operator check and limits

The newly built FA3 extension passed four BF16 forwards: causal packed prefill
and noncausal draft attention, each called directly and through scheduler
metadata. The checks used GQA 16/2, head dimension 128, page size 128 and a
permuted page table against a CPU FP32 reference on BF16 inputs. At absolute and
relative tolerance 0.025, maximum absolute errors were 0.004383 for causal
prefill and 0.000602 for noncausal draft. This check did not cover KV append,
cache graph replay or full-model accuracy.

These are single-request smoke observations. There is no matched baseline,
throughput comparison, norm ablation or broad text/multimodal accuracy result.
Acceptance rate and mean acceptance length are not speedup factors. The older
M3 QuaRot GPQA results belong to their own checkpoint and configuration and do
not establish Flash accuracy.
