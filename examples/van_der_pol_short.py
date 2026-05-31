from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from torch_tm_flowpipe import Interval, flowpipe_step
from torch_tm_flowpipe.ode_examples import van_der_pol_ode


if __name__ == "__main__":
    segment = flowpipe_step(
        van_der_pol_ode,
        [Interval(1.0, 1.05), Interval(0.0, 0.05)],
        h=0.01,
        order=4,
    )
    print(f"status={segment.status}")
    print(f"attempts={segment.validation_attempts}")
    print("flowpipe_range=" + str([iv.to_tuple() for iv in segment.tm.range_box()]))
    print("final_range=" + str([iv.to_tuple() for iv in segment.final_tm.range_box()]))
