"""Hello world - load and play a scenario

This corresponds to this [Hello World C++
example](https://github.com/esmini/esmini/tree/master/Hello-World_coding-example/main.cpp).
"""

from datetime import datetime
from pathlib import Path

import esmini

RESOURCES_DIR = esmini.get_default_resources_dir()


DT = 0.1


def main() -> None:
    log_dir = Path.cwd() / "logs"
    log_dir.mkdir(exist_ok=True)
    esmini.set_logfile_path(log_dir / f"log-{datetime.now().isoformat(timespec="seconds")}.txt")
    scenario_file = RESOURCES_DIR / "xosc/cut-in.xosc"
    esmini.init_scenario_engine(osc_filename=scenario_file)

    for _ in range(500):
        esmini.step_sim(DT)

    esmini.close_sim()


if __name__ == "__main__":
    main()
