#!/bin/bash
# package.sh: the investigation's record for docs/benchmarks/server-investigation-20260929: the summaries, analyses, flags and microbench tables as files,
# every lane's raw records (load-tool tables, server logs, levels, health, progress, clocks, the mix checks' ids) and the monitor's samples in raw.tar.xz, the load-tool JSON records in records.tar.xz, the scripts and the probe trees' diffs.
# The dispatch traces (*.trace, 30 to 60 MB each) stay on the machine that ran them.
W=/opt/claude-work/llmx-p2-perf2
P=$W/package/server-investigation-20260929
rm -rf $W/package; mkdir -p $P/analysis $P/micro $P/scripts $P/probe-src
cd $W/jobs
python3 $W/summ2.py lane-* --json $P/summary.json > $P/summary.txt
python3 $W/flags.py monitor.jsonl lane-* witness-* --json $P/flags.json > $P/flags.txt
cp previous-20260928.json previous-20260928.txt $P/
python3 $W/microtab.py micro micro3 micro4 micro5 micro6 micro7 $P/micro/microtab.json > $P/micro/microtab.md
for d in micro micro3 micro4 micro5 micro6 micro7; do mkdir -p $P/micro/$d; cp $d/*.txt $d/progress $P/micro/$d/ 2>/dev/null; done
for t in lane-*; do for a in $(ls $t/*.trace 2>/dev/null | xargs -n1 basename | sed 's/\.trace$//'); do python3 $W/analyze2.py $t $a --json $P/analysis/$t-$a.json > $P/analysis/$t-$a.txt 2>&1; done; done
cp $W/*.sh $W/*.py $W/FLAGS.txt $P/scripts/
cp probe-src/*.diff $P/probe-src/
tar -cJf $P/raw.tar.xz --exclude="*.trace" --exclude="*.trace.names" --exclude="lane-*/*[0-9].json" --exclude="witness-*/*.json" lane-* witness-* micro* monitor.jsonl build-*.out fetch.out
# The load tool's records, every request with its inter-token gaps, with every time rounded to 0.1 ms, which keeps them a few MB; the full-precision files stay beside the traces.
rm -rf /tmp/perf2-records; mkdir -p /tmp/perf2-records
python3 - <<EOF
import glob, json, os
def rnd(x):
    if isinstance(x, float): return round(x, 4)
    if isinstance(x, list): return [rnd(v) for v in x]
    if isinstance(x, dict): return {k: rnd(v) for k, v in x.items()}
    return x
for f in glob.glob("lane-*/*.json") + glob.glob("witness-*/*.json"):
    if f.endswith(".ids.json"): continue
    os.makedirs("/tmp/perf2-records/" + os.path.dirname(f), exist_ok=True)
    json.dump(rnd(json.load(open(f))), open("/tmp/perf2-records/" + f, "w"), separators=(",", ":"))
EOF
(cd /tmp/perf2-records && tar -cJf $P/records.tar.xz .)
sha256sum $W/main/b/llmx $W/probe*/b/llmx $W/probe*/b/llmx-perf-matmul-probe > $P/binaries.sha256
cat $W/models/SHA256SUMS > $P/models.sha256
du -sh $P; ls $P
