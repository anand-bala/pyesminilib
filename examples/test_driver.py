"""This is a Python port of the [test driver example](https://github.com/esmini/esmini/blob/master/EnvironmentSimulator/code-examples/test-driver/test-driver.cpp)."""

import logging
from datetime import datetime
from pathlib import Path

import esmini

DEFAULT_TARGET_SPEED = 13.0
CURVE_WEIGHT = 30.0
THROTTLE_WEIGHT = 0.5
DURATION = 32.0


CURRENT_DIR = Path(__file__).parent
RESOURCES_DIR = esmini.get_default_resources_dir()

if __name__ == "__main__":
    logging.basicConfig(
        format="[{levelname:8s}] [{name}]: {message}",
        style="{",
        level=logging.INFO,
    )
    log_dir = Path.cwd() / "logs"
    log_dir.mkdir(exist_ok=True)
    esmini.set_logfile_path(log_dir / f"log-{datetime.now().isoformat(timespec="seconds")}.txt")


def run_sim(ghost: bool) -> None:
    log = logging.getLogger(Path(__file__).stem)
    log.info(f"Ghost mode: {ghost}")

    esmini.add_search_path(RESOURCES_DIR)
    # esmini.init_scenario_engine(osc_filename=CURRENT_DIR / "test-driver.xosc")
    esmini.init_scenario_engine(osc_filename=RESOURCES_DIR / "xosc/cut-in_external.xosc")
    log.info(esmini.get_speed_unit())
    # Lock object to the original lane
    # If setting to false, the object road position will snap to closest lane
    esmini.set_lock_on_lane(0, True)

    # Initialize the vehicle model, fetch initial state from the scenario
    object_state = esmini.get_object_state(0)
    assert object_state is not None
    vehicle_model = esmini.SimpleVehicle(object_state.x, object_state.y, object_state.h, 4.0, object_state.speed)
    vehicle_model.set_steering_rate(6.0)

    # show some road features, including road sensor
    esmini.viewer_toggle_node(esmini.ViewerNodeMask.TRAIL_DOTS | esmini.ViewerNodeMask.ODR_FEATURES, True)

    while esmini.get_sim_time() < DURATION and not esmini.is_sim_quitting():
        dt = esmini.get_sim_time_step()
        object_state = esmini.get_object_state(0)
        assert object_state is not None

        info = esmini.get_road_info_at_distance(0, 5 + 1.75 * object_state.speed, esmini.LookAheadMode.LANE_CENTER, True)
        target_speed = DEFAULT_TARGET_SPEED / (1 + CURVE_WEIGHT * abs(info.angle))

        steer_angle = info.angle
        throttle = THROTTLE_WEIGHT * (target_speed - object_state.speed)
        # esmini.log_message(f"Steering angle: {steer_angle}")
        # esmini.log_message(f"Object speed: {object_state.speed}")
        # esmini.log_message(f"Target speed: {target_speed}")
        # esmini.log_message(f"Throttle: {throttle}")

        # Step vehicle model with driver input, but wait until time > 0
        if esmini.get_sim_time() > 0 and not esmini.is_sim_paused():
            vehicle_model.control_continuous(dt, throttle, steer_angle)

        # Fetch updated state and report to scenario engine
        vehicle_state = vehicle_model.get_state()

        # Report updated vehicle position and heading. z, pitch and roll will be aligned to the road
        esmini.report_object_position(
            object_id=0,
            timestamp=0,
            x=vehicle_state.x,
            y=vehicle_state.y,
            heading=vehicle_state.heading,
        )

        # The following values are not necessary to report.
        # If not reported, esmini will calculate based on motion over time
        # but for accuracy it's recommendeded to report if available.

        # wheel status (revolution and steering angles)
        esmini.report_object_wheel_status(0, vehicle_state.wheel_rotation, vehicle_state.wheel_angle)
        # speed (along vehicle longitudinal (x) axis)
        esmini.report_object_speed(0, vehicle_state.speed)

        # Finally, update scenario using same time step as for vehicle model
        esmini.step_sim(dt)
    esmini.close_sim()


def main() -> None:
    ghosts = [False]
    for ghost in ghosts:
        run_sim(ghost)


if __name__ == "__main__":
    main()
