import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import server_load  # noqa: E402

# The load tool's figures against a server of its own whose tokens arrive at known times: the arithmetic exactly on made-up times, then time to first token, gaps, time per token, end-to-end latency and the span of a level on every API in both loads, with failures counted by reason, a timeout and a short reply.
# It needs no model and no llmx binary, so it runs in every job.


def run():
    return server_load.self_test()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
