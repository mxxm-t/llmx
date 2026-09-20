"""Record Windows performance counters beside a benchmark (stdlib only)."""

import argparse
import ctypes as ct
from ctypes import wintypes as wt
import datetime
import json
import math
import os
from pathlib import Path
import sys
import time


COUNTERS = {
    "cpu_percent": r"\Processor(_Total)\% Processor Time",
    "disk_idle_percent": r"\PhysicalDisk(*)\% Idle Time",
    "disk_bytes_per_second": r"\PhysicalDisk(*)\Disk Bytes/sec",
    "gpu_engine_percent": r"\GPU Engine(*)\Utilization Percentage",
}


class CounterValue(ct.Structure):
    _fields_ = [("status", wt.DWORD), ("value", ct.c_double)]


class CounterItem(ct.Structure):
    _fields_ = [("name", wt.LPWSTR), ("value", CounterValue)]


class WindowsMonitor:
    def __init__(self):
        if sys.platform != "win32":
            raise RuntimeError("Windows PDH counters are required")
        self.api = ct.WinDLL("pdh")
        self.kernel = ct.WinDLL("kernel32", use_last_error=True)
        process_signatures = {
            "K32EnumProcesses": [ct.POINTER(wt.DWORD), wt.DWORD, ct.POINTER(wt.DWORD)],
            "OpenProcess": [wt.DWORD, wt.BOOL, wt.DWORD],
            "GetProcessTimes": [wt.HANDLE] + [ct.POINTER(wt.FILETIME)] * 4,
            "QueryFullProcessImageNameW": [wt.HANDLE, wt.DWORD, wt.LPWSTR,
                                          ct.POINTER(wt.DWORD)],
            "CloseHandle": [wt.HANDLE],
        }
        for name, args in process_signatures.items():
            fn = getattr(self.kernel, name)
            fn.argtypes = args
            fn.restype = wt.HANDLE if name == "OpenProcess" else wt.BOOL
        self.previous_processes = {}
        signatures = {
            "PdhOpenQueryW": [wt.LPCWSTR, ct.c_size_t, ct.POINTER(wt.HANDLE)],
            "PdhAddEnglishCounterW": [wt.HANDLE, wt.LPCWSTR, ct.c_size_t,
                                     ct.POINTER(wt.HANDLE)],
            "PdhCollectQueryData": [wt.HANDLE],
            "PdhGetFormattedCounterArrayW": [wt.HANDLE, wt.DWORD,
                ct.POINTER(wt.DWORD), ct.POINTER(wt.DWORD), ct.c_void_p],
            "PdhCloseQuery": [wt.HANDLE],
        }
        for name, args in signatures.items():
            fn = getattr(self.api, name)
            fn.argtypes = args
            fn.restype = wt.DWORD
        self.query = wt.HANDLE()
        self.handles = {}
        self.unavailable = {}
        self.check(self.api.PdhOpenQueryW(None, 0, ct.byref(self.query)))
        try:
            for name, path in COUNTERS.items():
                handle = wt.HANDLE()
                status = self.api.PdhAddEnglishCounterW(
                    self.query, path, 0, ct.byref(handle))
                if status:
                    self.unavailable[name] = hex(status)
                else:
                    self.handles[name] = handle
            before = time.perf_counter()
            self.check(self.api.PdhCollectQueryData(self.query))
            self.previous_collection = [before, time.perf_counter()]
            self.processes()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(status):
        if status:
            raise RuntimeError(f"PDH status {status:#x}")

    def values(self, handle):
        # Wildcard membership can grow between the sizing and data calls.
        for _ in range(3):
            size, count = wt.DWORD(), wt.DWORD()
            status = self.api.PdhGetFormattedCounterArrayW(
                handle, 0x200 | 0x8000, ct.byref(size), ct.byref(count), None)
            if status != 0x800007d2:
                return {"error": hex(status), "items": []}
            buffer = ct.create_string_buffer(size.value)
            status = self.api.PdhGetFormattedCounterArrayW(
                handle, 0x200 | 0x8000, ct.byref(size), ct.byref(count), buffer)
            if status == 0x800007d2:
                continue
            if status:
                return {"error": hex(status), "items": []}
            items = ct.cast(buffer, ct.POINTER(CounterItem))
            return {"items": [
                {"name": items[i].name, "status": items[i].value.status,
                 "value": items[i].value.value
                 if items[i].value.status in (0, 1)
                 and math.isfinite(items[i].value.value) else None}
                for i in range(count.value)]}
        return {"error": "wildcard membership kept changing", "items": []}

    def sample(self):
        start = time.perf_counter()
        status = self.api.PdhCollectQueryData(self.query)
        collected = time.perf_counter()
        result = {"kind": "sample", "utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(), "monotonic": start,
            "counter_interval_start_bracket": self.previous_collection,
            "counter_interval_end_bracket": [start, collected],
            "collect_status": status,
            "counters": {name: self.values(handle)
                         for name, handle in self.handles.items()} if not status else {}}
        result["processes"] = self.processes()
        self.previous_collection = [start, collected]
        result["collection_seconds"] = time.perf_counter() - start
        return result

    def processes(self):
        # PID plus creation time avoids same-name counter churn and PID reuse.
        capacity = 1024
        while True:
            pids, used = (wt.DWORD * capacity)(), wt.DWORD()
            if not self.kernel.K32EnumProcesses(pids, ct.sizeof(pids), ct.byref(used)):
                return {"error": ct.get_last_error(), "items": []}
            if used.value < ct.sizeof(pids):
                break
            capacity *= 2
        items, current = [], {}
        for pid in pids[:used.value // ct.sizeof(wt.DWORD)]:
            if pid == 0:
                continue
            handle = self.kernel.OpenProcess(0x1000, False, pid)
            if not handle:
                items.append({"pid": pid, "error": ct.get_last_error()})
                continue
            try:
                created, exited, kernel, user = [wt.FILETIME() for _ in range(4)]
                if not self.kernel.GetProcessTimes(handle, ct.byref(created), ct.byref(exited),
                                                   ct.byref(kernel), ct.byref(user)):
                    items.append({"pid": pid, "error": ct.get_last_error()})
                    continue
                def ticks(value):
                    return (value.dwHighDateTime << 32) | value.dwLowDateTime
                now = time.perf_counter()
                identity = (pid, ticks(created))
                cpu = (ticks(kernel) + ticks(user)) / 1e7
                previous = self.previous_processes.get(identity)
                current[identity] = (now, cpu)
                name, size = ct.create_unicode_buffer(32768), wt.DWORD(32768)
                named = self.kernel.QueryFullProcessImageNameW(handle, 0, name, ct.byref(size))
                items.append({"pid": pid, "created_filetime": identity[1],
                              "name": Path(name.value).name if named else None,
                              "name_error": None if named else ct.get_last_error(),
                              "cpu_seconds": cpu, "monotonic": now,
                              "cpu_interval_start": previous[0] if previous else None,
                              "cpu_percent": 100 * (cpu - previous[1]) / (now - previous[0])
                              if previous else None})
            finally:
                self.kernel.CloseHandle(handle)
        self.previous_processes = current
        return {"items": items}

    def close(self):
        if self.query:
            self.api.PdhCloseQuery(self.query)
            self.query = wt.HANDLE()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()
    if not all(math.isfinite(x) and x > 0 for x in (args.seconds, args.interval)):
        parser.error("seconds and interval must be positive finite values")
    if args.stop_file and args.stop_file.exists():
        parser.error("stop file already exists")
    with args.output.open("x", encoding="ascii") as output:
        started, cpu_started = time.perf_counter(), time.process_time()
        monitor = WindowsMonitor()
        def write(value):
            output.write(json.dumps(value, ensure_ascii=True, allow_nan=False) + "\n")
            output.flush()
        try:
            write({"kind": "metadata", "pid": os.getpid(),
                   "logical_cpus": os.cpu_count(), "interval_seconds": args.interval,
                   "initial_counter_collection_bracket": monitor.previous_collection,
                   "counters": COUNTERS, "unavailable": monitor.unavailable,
                   "process_cpu_units": "100 percent equals one logical CPU",
                   "limitations": "Sampled intervals may miss short-lived processes; "
                   "inaccessible/new processes have unknown CPU deltas; GPU engine values are not additive."})
            deadline = started + args.seconds
            next_sample = started + args.interval
            while time.perf_counter() < deadline:
                time.sleep(max(0, min(next_sample, deadline) - time.perf_counter()))
                write(monitor.sample())
                if args.stop_file and args.stop_file.exists():
                    break
                next_sample = max(next_sample + args.interval, time.perf_counter())
            write({"kind": "end", "elapsed_seconds": time.perf_counter() - started,
                   "monitor_cpu_seconds": time.process_time() - cpu_started})
        finally:
            monitor.close()


if __name__ == "__main__":
    main()
