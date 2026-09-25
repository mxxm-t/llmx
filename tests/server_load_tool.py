import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import server_load  # noqa: E402

# The load tool's own check: the stream reading of every API and the figures exactly on made-up times, then every API in both loads against a server of its own, with failures counted by reason, both timeouts and a short reply, then the whole tool's table, notes and exit status.
# The checks against the server bound times only from below and loosely from above, so a busy machine does not fail them.
# It needs no model and no llmx binary, so it runs in every job.


def run():
    return server_load.self_test()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
