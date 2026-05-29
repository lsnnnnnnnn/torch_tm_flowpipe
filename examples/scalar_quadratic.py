from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from torch_tm_flowpipe import Interval, flowpipe_step
from torch_tm_flowpipe.ode_examples import scalar_quadratic_ode


if __name__ == "__main__":
    segment = flowpipe_step(
        scalar_quadratic_ode,
        [Interval(0.0, 0.5)],
        h=0.05,
        order=4,
    )
    print(f"status={segment.status}")
    print(f"attempts={segment.validation_attempts}")
    print(f"flowpipe_range={segment.tm.range_box()[0].to_tuple()}")
    print(f"final_range={segment.final_tm.range_box()[0].to_tuple()}")
