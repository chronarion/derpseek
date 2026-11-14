"""Run one of dsv41's engine harnesses (engine/harness/*.py) with Derpseek's residency manager ticking after every CUDA
graph replay, as the served engine ticks after every step, so the HBM cache rows follow the text the harness feeds
instead of staying at the rowmap's initial cache. Needed to price DERPSEEK_MISS=skip, where a miss is dropped: with a
frozen cache every pick on a host row would be dropped, far more than in serving.
   python3 /derpseek/derpseek_engine/harness_tick.py /engine/harness/decode_kld.py OUT [harness args]"""
import os
import runpy
import sys

import torch

sys.path.insert(0, "/derpseek")


def _install():
    from derpseek_engine import exact, residency
    orig = torch.cuda.CUDAGraph.replay
    state = {"n": 0}

    def replay(self, _orig=orig):
        _orig(self)
        if exact.LAYERS and not torch.cuda.is_current_stream_capturing():
            residency.tick()
            state["n"] += 1
            if state["n"] in (1, 1000, 10000):
                sys.stderr.write(f"DERPSEEK harness: residency ticks after graph replays ({state['n']} so far)\n")

    torch.cuda.CUDAGraph.replay = replay


if __name__ == "__main__":
    target = sys.argv[1]
    _install()
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(target)))
    runpy.run_path(target, run_name="__main__")
