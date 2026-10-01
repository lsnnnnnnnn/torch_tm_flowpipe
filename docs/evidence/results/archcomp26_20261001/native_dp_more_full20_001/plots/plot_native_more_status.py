"""Plot the saved DP more native prefix and undecided period; no digests."""

from collections import defaultdict
import json
from pathlib import Path
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
RANGES = RUN / "ranges.bin"
LOG = RUN / "native.log"
H = 0.005
EXPECTED_LANES = 225
EXPECTED_STEPS = 80
RECORD = struct.Struct("<QQd16d")

log = LOG.read_text()
assert log.splitlines()[-2] == "UNKNOWN"
assert log.count("Unknown.") == 45 and "Unsafe." not in log
periods = [int(line.split()[1]) for line in log.splitlines() if line.startswith("Step ")]
assert periods == list(range(16))
accepted_steps = periods[-1] * 4  # Step 15 is the first period with UNKNOWN.

blob = RANGES.read_bytes()
assert len(blob) % RECORD.size == 0
by_step = defaultdict(list)
seen = set()
for values in RECORD.iter_unpack(blob):
    lane, step, h, *bounds = values
    assert 0 <= lane < EXPECTED_LANES and 1 <= step <= EXPECTED_STEPS
    assert abs(h - H) < 1e-12 and (lane, step) not in seen
    seen.add((lane, step))
    by_step[step].append((lane, bounds))
assert set(by_step) == set(range(1, 65))
assert all(len(by_step[step]) == EXPECTED_LANES for step in by_step)

union = {}
outside = {}
for step, records in by_step.items():
    # theta1_dot is the third physical state, each with tube/endpoint lo/hi.
    union[step] = (min(v[8] for _, v in records), max(v[9] for _, v in records))
    outside[step] = sorted(lane for lane, v in records if v[8] < -1.5 or v[9] > 1.5)
assert not any(outside[s] for s in range(1, accepted_steps + 1))
assert [len(outside[s]) for s in range(61, 65)] == [3, 22, 43, 45]

fig, ax = plt.subplots(figsize=(11, 6.2))
fig.subplots_adjust(left=0.09, right=0.985, top=0.92, bottom=0.17)
ax.axhspan(-1.5, 1.5, color="#cfe9d7", alpha=0.55, zorder=0)
ax.axvspan(0.32, 0.4, color="#e4e4e4", alpha=0.8, zorder=0)
for step, (lo, hi) in union.items():
    color = "#4386b5" if step <= accepted_steps else "#dc7a28"
    ax.add_patch(Rectangle(((step - 1) * H, lo), H, hi - lo,
                           facecolor=color, edgecolor=color, linewidth=0.2,
                           alpha=0.58, zorder=2))
ax.plot([0, 0], [1, 1.3], color="black", linewidth=2.2, zorder=3)
ax.axvline(0.30, color="#a85313", linestyle="--", linewidth=1.2)
ax.axvline(0.32, color="#666666", linestyle="--", linewidth=1.2)
ax.text(0.318, -1.64, "UNKNOWN begins", ha="right", va="bottom", fontsize=9,
        color="#9d4d10")
ax.text(0.36, 0.08, "steps 65–80\nno records", ha="center", va="center",
        fontsize=11, color="#555555")
ax.set(xlim=(0, 0.4), ylim=(-1.68, 1.62), xlabel="t [s]",
       ylabel=r"$\dot{\theta}_1$", title="Double Pendulum more-robust: native 225-box tube union")
ax.grid(alpha=0.2)
ax.legend(handles=[
    Patch(facecolor="#cfe9d7", label="Safe band ±1.5, all continuous time"),
    Patch(facecolor="#4386b5", label="Steps 1–60: author checker accepted through t=0.30"),
    Patch(facecolor="#dc7a28", label="Steps 61–64: interval enclosures, property UNKNOWN"),
    Patch(facecolor="#e4e4e4", label="Steps 65–80: no saved data"),
], loc="upper right", fontsize=8)
fig.text(0.02, 0.025,
         "Source: native ranges.bin (225 lanes/step) and native.log (45 Unknown., no Unsafe.).\n"
         "Interval crossing is not a certified counterexample; checker and NN soundness are not independently proved here.",
         fontsize=8, va="bottom")

prefix = HERE / "dp_more_native_theta1dot_status"
fig.savefig(prefix.with_suffix(".png"), dpi=180)
fig.savefig(prefix.with_suffix(".pdf"))
plt.close(fig)
(prefix.with_suffix(".json")).write_text(json.dumps({
    "source_ranges": str(RANGES), "source_native_log": str(LOG),
    "remote_ranges": "/srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs/archcomp26_20261001/native_dp_more_full20_001/ranges.bin",
    "record_count": len(seen), "lanes_per_observed_step": EXPECTED_LANES,
    "accepted_prefix_steps": [1, 60], "uncertain_steps": [61, 64],
    "unobserved_steps": [65, 80], "outside_band_lane_counts":
    {str(s): len(outside[s]) for s in range(61, 65)},
    "status_basis": "native.log period 15 returned 45 Unknown.; ranges.bin contains no status field",
    "content_digest": "not calculated",
}, indent=2) + "\n")
print(prefix)
