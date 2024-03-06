"""External Ego Control

This is the Python port of the [External control of
Ego](https://github.com/esmini/esmini/tree/master//EnvironmentSimulator/code-examples/hello_world/hw4_external_ego_control.cpp)
example.
"""

from datetime import datetime
from pathlib import Path

import esmini
from esmini import PositionMode, PositionModeType

RESOURCES_DIR = esmini.get_default_resources_dir()
DT = 0.001


def main() -> None:
    log_dir = Path.cwd() / "logs"
    log_dir.mkdir(exist_ok=True)
    esmini.set_logfile_path(log_dir / f"log-{datetime.now().isoformat(timespec="seconds")}.txt")

    osc_filename = RESOURCES_DIR / "xosc/cut-in_external.xosc"
    print(osc_filename)
    assert osc_filename.is_file()

    esmini.init_scenario_engine(osc_filename=osc_filename)
    ego_id = esmini.get_entity_id(0)
    z = 0.0

    for i in range(500):
        if esmini.is_sim_quitting():
            break
        esmini.report_object_position(ego_id, 0.0, 8.0, float(i), z, float(1.57 + 0.01 * i), 0.0, 0.0)
        esmini.step_sim()

        state = esmini.get_object_state(ego_id)
        assert state is not None
        print(
            "road_id: {} s: {:.3f} lane_id {} lane_offset: {:.3f} z: {:.2f}".format(
                state.road_id,
                state.s,
                state.lane_id,
                state.lane_offset,
                state.z,
            )
        )

        if i == 100:
            z = 10.0
            print(f"Release relative road alignment, set absolute {z=:.2f}")
            esmini.set_object_position_mode(
                ego_id,
                PositionModeType.SE_SET,
                PositionMode.SE_Z_ABS,  # release relative alignment to road surface
            )

        if i == 200:
            z = 0.0
            print(f"Restore relative road alignment, set absolute {z=:.2f}")
            esmini.set_object_position_mode(
                ego_id,
                PositionModeType.SE_SET,
                PositionMode.SE_Z_REL,  # restore relative alignment to road surface
            )

    esmini.close_sim()


if __name__ == "__main__":
    main()
