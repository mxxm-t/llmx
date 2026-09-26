# Inside the gate container: the suite's server component of one tree, its servers started with --threads 6 to match the container's CPU quota, since the automatic count oversubscribes it and the uncapped check then passes its 600 s limit.
import os, sys, time
tree = sys.argv[1]
sys.path.insert(0, os.path.join(tree, "tests"))
import common, server
common.EXE = os.path.join(tree, "b-cpu", "llmx")
init = server.Server.__init__
server.Server.__init__ = lambda self, model, *extra: init(self, model, *extra, "--threads", "6")
start = time.time()
try:
    ok = server.run()
except Exception as e:
    print("  error:", repr(e))
    ok = False
print("%s server component %s in %.0f s" % (tree, "PASS" if ok else "FAIL", time.time() - start), flush=True)
