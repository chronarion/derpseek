#!/bin/bash
# Expert-pinning bench, KLD part: dsv41's decode-path harness (engine/harness/decode_kld.py: teacher-forced verify
# rows through the served decode path) on a suite, with a Derpseek profile and any DERPSEEK_* / DSV41_* overrides set
# in the environment (an "approach"). Takes the GPU: stop the Derpseek server first.
#   MODEL=/path/to/DeepSeek-V4.1-Flash-1017 DSV41=/path/to/dsv41/engine GPU=nvidia.com/gpu=<GB300 UUID> \
#   CORPUS=suite.json OUT=out/exact-b16 [PROFILE=exact] [TICK=1] bench/deb/run_kld.sh --batch 16 --rows 240 \
#       --routes full --ws-decode
# OUT/full holds one safetensors file per window (logits, target ids, positions); report.py compares two such dirs.
set -u
: "${MODEL:?checkpoint dir}" "${DSV41:?dsv41 engine dir}" "${GPU:?CDI device, e.g. nvidia.com/gpu=GPU-...}"
: "${CORPUS:?suite JSON}" "${OUT:?output dir}"
D=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../engine" && pwd); RT=$DSV41/runtime
PROFILE=${PROFILE:-exact}; STATE=${STATE:-$PWD/state}; CDIR=$(cd "$(dirname "$CORPUS")" && pwd); CF=$(basename "$CORPUS")
mkdir -p "$OUT" $STATE/{vllm,flashinfer} $STATE/jit/{tilelang,triton,nv,dj}
set -a; . $D/profiles/$PROFILE.env; set +a
export DSV41_KLD_CORPUS=/corpus/$CF
# the harness drives vLLM's runner with a fixed spec (3 drafts: 4 rows per request per step)
export DSV41_SPEC='{"method":"dspark","num_speculative_tokens":3,"num_speculative_tokens_per_batch_size":[[1,32,3]],"draft_sample_method":"probabilistic","rejection_sample_method":"block"}'
unset DSV41_SUFFIX DSV41_DYNK DSV41_CONF_WIDTH DSV41_CONF_WIDTH_B DSV41_CONF_WIDTH_MODE
docker rm -f derpseek-kld >/dev/null 2>&1
timeout ${TMO:-3600} docker run --rm --name derpseek-kld --device $GPU --ipc host --network none \
  --cap-add SYS_NICE --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 --ulimit core=0 --security-opt label=disable \
  -v $MODEL:/model:ro -v $STATE/vllm:/root/.cache/vllm -v $STATE/flashinfer:/root/.cache/flashinfer \
  -v $STATE/jit/tilelang:/root/.tilelang -v $STATE/jit/triton:/root/.triton -v $STATE/jit/nv:/root/.nv -v $STATE/jit/dj:/root/.dj \
  -v $RT/hook:/w:ro -v $D/hook/mega_peer_hook.py:/w/mega_peer_hook.py:ro \
  -v $RT/fastpath/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro -v $RT/fastpath:/fastpath:ro \
  -v $D/overlay/fastpath_boundary.py:/fastpath/fastpath_boundary.py:ro -v $D/overlay/megamoe_fork:/fastpath/megamoe_fork:ro \
  -v $DSV41:/engine:ro -v $D:/derpseek:ro -v "$(cd "$OUT" && pwd)":/out -v $CDIR:/corpus:ro \
  -v $DSV41/patches/vllm/gumbel.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/sample/gumbel.py:ro \
  -v $DSV41/patches/vllm/rejection_sampler_utils.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:ro \
  -e PYTHONPATH=/engine:/derpseek -e HF_HUB_OFFLINE=1 -e VLLM_LOGGING_LEVEL=INFO -e PYTHONUNBUFFERED=1 \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  $(env | grep -o "^DSV41_[A-Z0-9_]*" | sed "s/^/-e /" | tr "\n" " ") $(env | grep -o "^DERPSEEK_[A-Z0-9_]*" | sed "s/^/-e /" | tr "\n" " ") \
  -e PYTORCH_CUDA_ALLOC_CONF=pinned_max_round_threshold_mb:512 -e DERPSEEK_SOLO=1 \
  -e MEGA_ROWMAP=/derpseek/rowmaps/$ROWMAP -e DS41_FASTPATH=$FASTPATH -e DS41_GPU_RESERVE_MIB=0 \
  --entrypoint /usr/bin/numactl vllm/vllm-openai:v0.32.0-aarch64 --membind=0 \
  python3 $([ "${TICK:-1}" = 1 ] && echo /derpseek/derpseek_engine/harness_tick.py) /engine/harness/decode_kld.py /out "$@"
rc=$?
[ $rc = 124 ] && { echo "TIMEOUT: killing the container"; docker kill derpseek-kld >/dev/null 2>&1; }
echo "harness exited rc=$rc"
exit $rc
