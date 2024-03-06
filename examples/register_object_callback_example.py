"""This is a port of the example: https://github.com/ebadi/pyesmini/blob/main/examples/esmini_example1.py"""

import esmini

RESOURCES_DIR = esmini.get_default_resources_dir()
DT = 0.001


def main() -> None:
    xosc_file = RESOURCES_DIR / "xosc/follow_ghost.xosc"
    print(f"{xosc_file=}")
    assert xosc_file.is_file()

    esmini.log_to_console(False)
    esmini.init_scenario_engine(osc_filename=xosc_file)
    esmini.step_sim(DT)
    s1 = esmini.add_object_sensor(0, 2.0, 1.0, 0.5, 1.57, 1.0, 50.0, 1.57, 10)
    assert s1 is not None
    s2 = esmini.add_object_sensor(0, 3.0, 4.0, 1.5, 2.57, 2.0, 51.0, 2.57, 10)
    assert s2 is not None

    print("ObjectGhostState::::", esmini.get_object_ghost_state(0))
    print("ObjectState::::", esmini.get_object_state(0))
    print("RoadInfoAlongGhostTrail::::", esmini.get_road_info_along_ghost_trail(0, 100))
    esmini.clear_search_paths()
    esmini.step_sim(DT)

    # NOTE: Must keep alive
    cb_handle = esmini.register_object_callback(0, object_callback)
    print(cb_handle)

    esmini.step_sim()

    sensed_objects = esmini.fetch_sensed_object_list(s1)
    print("Sensed Objects:::::::", sensed_objects)

    for _ in range(10000):
        esmini.step_sim()


TRIGGERED = False


def object_callback(state: esmini.ScenarioObjectState) -> None:
    global TRIGGERED
    if (time := esmini.get_sim_time()) > 2.5 and not TRIGGERED:
        TRIGGERED = True
        print("Condition Triggered")
        print("Simulation time: ", time)
        print("CALLBACK for obj {}: x={} y={}".format(state.id, state.x, state.y))


if __name__ == "__main__":
    main()
