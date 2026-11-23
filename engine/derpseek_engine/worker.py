"""vLLM worker for Derpseek: dsv41's worker (whole-step graphs and its options) with the GB300-alone changes.
`vllm serve ... --worker-cls derpseek_engine.worker.DerpseekWorker` with both engine directories on PYTHONPATH."""
import os

from dsv41_engine.worker import DSV41Worker


class DerpseekWorker(DSV41Worker):
    def _make_model_runner(self):
        runner = super()._make_model_runner()
        from . import solo
        if os.environ.get("DERPSEEK_SOLO", "0") == "1":
            rowmap = os.environ["MEGA_ROWMAP"]
            orig_load = runner.load_model

            def load_model(*a, _orig=orig_load, **k):
                out = _orig(*a, **k)
                solo.penalize_left_out(runner.model, rowmap)
                return out

            runner.load_model = load_model
        if os.environ.get("DERPSEEK_EMBED_HOST", "0") == "1":
            from . import hbm
            orig_load2 = runner.load_model

            def load_model(*a, _orig=orig_load2, **k):
                out = _orig(*a, **k)
                hbm.embed_to_host(runner)      # the embedding tables to pinned host memory, before any capture
                return out

            runner.load_model = load_model
        if os.environ.get("DERPSEEK_W3_MAX_T", "0") not in ("", "0"):
            from . import prefill_w3
            prefill_w3.install()
        if os.environ.get("DERPSEEK_HEAD12", "0") == "1":
            from . import head12
            orig_load3 = runner.load_model

            def load_model(*a, _orig=orig_load3, **k):
                out = _orig(*a, **k)
                head12.install(runner)     # the packed verify head, before warm-up and graph capture
                return out

            runner.load_model = load_model
        if os.environ.get("DERPSEEK_DRAFT_SHAPE"):
            from . import draftshape
            draftshape.install()
        if os.environ.get("DERPSEEK_K3B_SHRINK_VEC"):
            from . import k3cal
            k3cal.install()
        if os.environ.get("DERPSEEK_GRAPH_TIMING", "1") == "1":
            from . import timing
            timing.install()
        if os.environ.get("DERPSEEK_COPY_FENCE", "1") == "1":
            solo.install_copy_fence(runner)
        if os.environ.get("DERPSEEK_MODE", "") == "exact":
            from . import residency
            inner = runner.sample_tokens

            def sample_tokens(grammar_output, _inner=inner):
                out = _inner(grammar_output)
                residency.tick()      # promote host-row experts that got picks (between steps, never blocking)
                return out

            runner.sample_tokens = sample_tokens
        if os.environ.get("DERPSEEK_ACC2"):
            from . import acc2        # measurement only: exact acceptance of drafting alternatives at batch 1
            acc2.STATE["runner"] = runner
            acc2.install()
        if os.environ.get("DERPSEEK_ACC3"):
            from . import acc3        # measurement only: sparse p / q and drafter hidden states at batch 1
            acc3.set_runner(runner)
            acc3.install()   # with DERPSEEK_ACC3_TARGET=1 it wraps runner.load_model: keep it after the other wrappers
        return runner
