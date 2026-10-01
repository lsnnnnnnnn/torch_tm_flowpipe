"""Plot the saved numerical corner trajectory; no checksum or solver rerun."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
summary = json.loads((HERE / "data/summary.json").read_text())
with (HERE / "data/trajectory.csv").open(newline="") as stream:
    rows = list(csv.DictReader(stream))
t = [float(row["t"]) for row in rows]
v = [float(row["theta1_dot"]) for row in rows]
crossing = summary["first_crossing_DOP853"]

fig, ax = plt.subplots(figsize=(10.5, 5.8))
fig.subplots_adjust(left=0.1, right=0.985, top=0.89, bottom=0.19)
ax.axhspan(-1.5, 1.5, color="#d5eadb", label="Official continuous safe band ±1.5")
ax.axvspan(0.30, 0.32, color="#f3d6b8", alpha=0.8,
           label="Native interval checker UNKNOWN, 0.30–0.32 s")
ax.axvspan(0.32, 0.4, color="#e6e6e6", alpha=0.7,
           label="No native range records after 0.32 s")
ax.plot(t, v, color="#b2262c", linewidth=2.1,
        label="Numerical corner point, DOP853 + float32 ONNX")
ax.scatter([crossing["time"]], [-1.5], color="#b2262c", edgecolor="black",
           zorder=5, s=48)
ax.annotate(f"candidate crossing\nt={crossing['time']:.9f} s",
            xy=(crossing["time"], -1.5), xytext=(0.22, -1.72),
            arrowprops={"arrowstyle": "->", "color": "#7c171b"}, fontsize=9,
            color="#7c171b")
ax.set(xlim=(0, 0.4), ylim=(-1.84, 1.58), xlabel="t [s]",
       ylabel=r"$\dot{\theta}_1$",
       title="Double Pendulum more-robust: official ONNX corner trajectory")
ax.grid(alpha=0.2)
ax.legend(loc="upper right", fontsize=8)
fig.text(0.02, 0.025,
         "Initial state (1.3,1.3,1.3,1.3); controller sampled every 0.02 s and held; "
         "DOP853 rtol=1e-12, atol=1e-14.\n"
         "The crossing is a numerical candidate, not a certified counterexample. "
         "Gray indicates absence of native interval records, not absence of point data.",
         fontsize=8, va="bottom")
out = HERE / "corner_theta1dot_candidate"
fig.savefig(out.with_suffix(".png"), dpi=180)
fig.savefig(out.with_suffix(".pdf"))
plt.close(fig)
print(out)
