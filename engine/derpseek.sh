#!/bin/bash
# Derpseek: serve DeepSeek-V4.1-Flash-1017 on one DGX Station GB300 alone (no second GPU) as an OpenAI-compatible
# endpoint. vLLM 0.32 with the dsv41 engine's kernels and whole-step graphs; the experts that do not fit in HBM stay
# exact in pinned host memory and MegaMoE reads them over NVLink-C2C (profiles/*.env: ROWMAP, DERPSEEK_MODE=exact).
#   derpseek.sh up [PROFILE]    start the server (PROFILE = profiles/PROFILE.env, default: exact)
#   derpseek.sh down | status | logs
# Needs: a dsv41 checkout with its runtime built (DSV41=path, ./build.sh run once there), the checkpoint at MODEL,
# Docker with the NVIDIA CDI device of the GB300, the vLLM 0.32 aarch64 image.
set -u
D=$(cd "$(dirname "$0")" && pwd)
DSV41=${DSV41:-$D/../../dsv41/engine}
RUNTIME=${RUNTIME:-$DSV41/runtime}
MODEL=${MODEL:-/models/DeepSeek-V4.1-Flash-1017}
STATE=${STATE:-$D/state}
IMG=${IMG:-vllm/vllm-openai:v0.32.0-aarch64}
gpu_uuid() { nvidia-smi --query-gpu=name,uuid --format=csv,noheader | awk -F', ' -v n="$1" 'index($1, n) { print $2; exit }'; }
GB=${GB:-$(gpu_uuid GB300)}
PORT=${PORT:-8112}
MAXLEN=${MAXLEN:-1048576}
NAME=${NAME:-derpseek}
log() { echo "[derpseek $(date +%T)] $*"; }

down() {
  docker stop -t 30 $NAME >/dev/null 2>&1
  if docker inspect $NAME >/dev/null 2>&1; then   # keep the stopped server's log (step and residency stats)
    mkdir -p $STATE/logs; docker logs $NAME > $STATE/logs/$NAME-$(date +%Y%m%d-%H%M%S).log 2>&1
  fi
  docker rm -f $NAME >/dev/null 2>&1
  log "down"
}

main() {
  [ -f "$D/rowmaps/$ROWMAP" ] || { log "no rowmap $D/rowmaps/$ROWMAP"; return 1; }
  mkdir -p $STATE/{vllm,flashinfer} $STATE/jit/{tilelang,triton,nv,dj}
  log "starting on the GB300 alone: profile $PROFILE, rowmap $ROWMAP, port $PORT, max model length $MAXLEN"
  docker run -d --name $NAME --restart no --device nvidia.com/gpu=$GB --ipc host --network host \
    --cap-add SYS_NICE --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 --ulimit core=0 --security-opt label=disable \
    -v $MODEL:/model:ro -v $STATE/vllm:/root/.cache/vllm -v $STATE/flashinfer:/root/.cache/flashinfer \
    -v $STATE/jit/tilelang:/root/.tilelang -v $STATE/jit/triton:/root/.triton -v $STATE/jit/nv:/root/.nv -v $STATE/jit/dj:/root/.dj \
    -v $DSV41:/engine:ro -v $D:/derpseek:ro -e PYTHONPATH=/engine:/derpseek \
    -v $DSV41/patches/vllm/gumbel.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/sample/gumbel.py:ro \
    -v $DSV41/patches/vllm/rejection_sampler_utils.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:ro \
    -v $DSV41/patches/vllm/deepseek_v41_encoding.py:/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v41_encoding.py:ro \
    -v $DSV41/patches/vllm/sampling_params.py:/usr/local/lib/python3.12/dist-packages/vllm/sampling_params.py:ro \
    -v $DSV41/patches/vllm/deepseek_v41.py:/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v41.py:ro \
    -v $DSV41/patches/vllm/detokenizer.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/detokenizer.py:ro \
    $(env | grep -o "^DSV41_[A-Z0-9_]*" | sed "s/^/-e /" | tr "\n" " ") \
    $(env | grep -o "^DERPSEEK_[A-Z0-9_]*" | sed "s/^/-e /" | tr "\n" " ") \
    -v $RUNTIME/hook:/w:ro -v $D/hook/mega_peer_hook.py:/w/mega_peer_hook.py:ro \
    -v $RUNTIME/fastpath/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro \
    -v $RUNTIME/fastpath:/fastpath:ro -v $D/overlay/fastpath_boundary.py:/fastpath/fastpath_boundary.py:ro \
    -v $D/overlay/megamoe_fork:/fastpath/megamoe_fork:ro \
    -e HF_HUB_OFFLINE=1 -e VLLM_LOGGING_LEVEL=INFO -e CUDA_LOG_FILE=stderr -e PYTORCH_CUDA_ALLOC_CONF=pinned_max_round_threshold_mb:512 \
    -e MEGA_ROWMAP=/derpseek/rowmaps/$ROWMAP -e DS41_FASTPATH=$FASTPATH -e DS41_GPU_RESERVE_MIB=0 -e DERPSEEK_SOLO=1 \
    --entrypoint /usr/bin/numactl $IMG --membind=0 \
    vllm serve --model /model --served-model-name deepseek-v4.1-flash deepseek DeepSeek-V4.1-Flash derpseek --trust-remote-code \
    --tensor-parallel-size 1 --moe-backend deep_gemm_mega_moe --enable-expert-parallel \
    --worker-cls derpseek_engine.worker.DerpseekWorker \
    --engram-config '{"cpu_offload": true, "use_thp": true}' --max-model-len $MAXLEN --max-num-seqs ${MAX_NUM_SEQS:-32} \
    --max-num-batched-tokens ${MAX_BATCHED:-8192} --gpu-memory-utilization ${GPU_MEM_UTIL:-0.95} --kv-cache-dtype nvfp4_ds_mla --block-size 128 \
    --attention-config '{"indexer_kv_dtype": "mxfp4", "indexer_sparse_logits": true}' \
    --kernel-config '{"enable_flashinfer_autotune": false}' \
    --tokenizer-mode deepseek_v41 --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice \
    --long-prefill-token-threshold ${LONG_PREFILL:-6144} --enable-prefix-caching \
    --speculative-config "$SPEC" \
    --cudagraph-capture-sizes ${CAPTURE_SIZES:-1 2 4 6 8 12 16 18 20 24 28 32 36 40 48 56 64 72 80 96 128 144 160 192 224 256} ${EXTRA_CAPTURE_SIZES:-} \
    ${KV_OFFLOAD_GB:+--kv-offloading-backend native --kv-offloading-size $KV_OFFLOAD_GB} \
    --override-generation-config '{"temperature": 1.0, "top_p": 0.95}' \
    --default-chat-template-kwargs '{"reasoning_effort":"high"}' \
    --host 0.0.0.0 --port $PORT >/dev/null || return 1
  for i in $(seq 1 240); do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)" = 200 ]; then
      log "ready on :$PORT after $((i*15)) s"
      if [ "${DSV41_WARM:-1}" = 1 ]; then
        python3 -I "$DSV41/bench/warm.py" "http://127.0.0.1:$PORT" >/dev/null 2>&1 && log "warmed" || log "warm-up failed (serving anyway)"
      fi
      return 0
    fi
    [ "$(docker inspect -f '{{.State.Status}}' $NAME 2>/dev/null)" = running ] || { log "server died"; docker logs $NAME 2>&1 | tail -30; return 1; }
    sleep 15
  done
  log "not ready after 60 min"; return 1
}

PROFILE=${2:-exact}
case "$PROFILE" in   # renamed profiles: exact-fast -> lossy-fast in 1.4 (exact-fast-kv was an alias of it), skip-fast -> wavefront-fast in 1.6
  exact-fast|exact-fast-kv) echo "profile $PROFILE is called lossy-fast since 1.4"; PROFILE=lossy-fast ;;
  exact-fast-prefill) echo "profile $PROFILE is called lossy-fast-prefill since 1.4"; PROFILE=lossy-fast-prefill ;;
  skip-fast) echo "profile $PROFILE is replaced by wavefront-fast since 1.6"; PROFILE=wavefront-fast ;;
esac
case "${1:-}" in
  up|main)
    P=$D/profiles/$PROFILE.env
    [ -f "$P" ] || { echo "no profile $P"; exit 2; }
    [ -d "$RUNTIME/fastpath" ] || { echo "no dsv41 runtime at $RUNTIME: run build.sh in the dsv41 checkout first"; exit 2; }
    [ -f "$MODEL/model.safetensors.index.json" ] || { echo "no checkpoint at MODEL=$MODEL"; exit 2; }
    set -a; . "$P"; set +a
    down; main ;;
  down) down ;;
  status) docker ps --filter name=$NAME --format '{{.Names}} {{.Status}}'; curl -s http://127.0.0.1:$PORT/v1/models | head -c 300; echo ;;
  logs) docker logs --tail 50 $NAME ;;
  *) echo "usage: $0 up [PROFILE] | down | status | logs"; exit 2 ;;
esac
