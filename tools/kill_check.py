"""Ring resets at a killed process: llmx generate over the given devices, killed with SIGKILL mid-reply several times, and the GPU ring timeouts the kernel log gives after each kill.

A device that waits on its host between submissions leaves a job only that process can end, which the driver resets every ring for once its job timeout passes (docs/src/backends-vulkan.md).
It reads the kernel log through dmesg, so it needs root or CAP_SYSLOG, and runs on Linux; it exits 1 where any kill was followed by a ring timeout.
"""
import argparse
import re
import signal
import subprocess
import sys
import time

TIMEOUT = re.compile(r"amdgpu (\S+): amdgpu: ring \S+ timeout")


def kernel_log():
    """The kernel log's lines as dmesg gives them now."""
    return subprocess.run(["dmesg"], capture_output=True, text=True, errors="replace", check=True).stdout.splitlines()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--exe", required=True, help="the llmx binary")
    p.add_argument("--model", required=True, help="the GGUF file")
    p.add_argument("--device", default="vulkan:0,cpu", help="the devices, a list over several for a placement that holds between submissions")
    p.add_argument("--layer-shares", default="", help="passed to generate where given")
    p.add_argument("--kills", type=int, default=2, help="the runs killed")
    p.add_argument("--after", type=float, default=20.0, help="seconds from the start to the kill, past the load and into the reply")
    p.add_argument("--settle", type=float, default=20.0, help="seconds after a kill for the driver's job timeout to pass")
    p.add_argument("--pci", default="", help="the PCI addresses of the devices run on, comma separated, as the kernel log names them; others' timeouts are reported apart")
    args = p.parse_args()
    command = [args.exe, "generate", args.model, "Write a long story about a lighthouse keeper.", "-n", "8000", "--ignore-eos", "--device", args.device]
    if args.layer_shares:
        command += ["--layer-shares", args.layer_shares]
    ours = [a for a in args.pci.split(",") if a]
    resets = 0
    for k in range(args.kills):
        before = set(kernel_log())
        run = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace")
        time.sleep(args.after)
        if run.poll() is not None:
            print("kill_check: run %d ended before its kill: %s" % (k + 1, run.stderr.read().strip()[-400:]))
            return 2
        run.send_signal(signal.SIGKILL)
        run.wait()
        time.sleep(args.settle)
        found = [TIMEOUT.search(line).group(1) for line in kernel_log() if line not in before and TIMEOUT.search(line)]
        mine = [card for card in found if not ours or card in ours]
        others = sorted(set(found) - set(mine))
        print("kill_check: run %d killed after %.0f s, %d ring timeouts%s%s" % (k + 1, args.after, len(mine), (" on " + ", ".join(sorted(set(mine)))) if mine else "",
                                                                          ("; others' devices " + ", ".join(others)) if others else ""), flush=True)
        resets += len(mine)
    print("kill_check: %d ring timeouts after %d kills over %s" % (resets, args.kills, args.device))
    return 1 if resets else 0


if __name__ == "__main__":
    sys.exit(main())
