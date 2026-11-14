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
        return runner
