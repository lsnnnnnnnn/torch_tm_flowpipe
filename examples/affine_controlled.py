from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from torch_tm_flowpipe import Interval, flowpipe_step
from torch_tm_flowpipe.ode_examples import affine_controlled_ode


if __name__ == "__main__":
    segment = flowpipe_step(
        affine_controlled_ode,
        [Interval(-0.1, 0.1), Interval(-0.1, 0.1)],
        h=0.02,
        order=3,
        affine_u={"A": [[0.5, -0.25]], "b": [0.0], "error": [0.01]},
    )
    print(f"status={segment.status}")
    print(f"attempts={segment.validation_attempts}")
    print("flowpipe_range=" + str([iv.to_tuple() for iv in segment.tm.range_box()]))
    print("final_range=" + str([iv.to_tuple() for iv in segment.final_tm.range_box()]))
