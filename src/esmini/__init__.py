import importlib.util
import math
import os
import warnings
from enum import IntEnum, IntFlag
from pathlib import Path
from typing import Optional, Union

from typing_extensions import TypeAlias, overload

if importlib.util.find_spec("esmini._esmini_cffi") is None:
    warnings.warn("esmini extension module not present. Doing so now.", UserWarning, stacklevel=0)
    from esmini._build_esmini import ffibuilder

    ffibuilder.compile(verbose=True)


import esmini._esmini_cffi as _esmini_cffi


def get_default_resources_dir() -> Path:
    """Get the path to the default resources bundled with esmini"""
    current_file = Path(__file__).absolute()
    current_dir = current_file.parent
    resources_dir = current_dir / "resources"
    assert resources_dir.is_dir()
    return resources_dir


class PositionMode(IntFlag):
    SE_Z_SET = 1  # 0001
    SE_Z_DEFAULT = 1  # 0001
    SE_Z_ABS = 3  # 0011
    SE_Z_REL = 7  # 0111
    SE_H_SET = SE_Z_SET << 4
    SE_H_ABS = SE_Z_ABS << 4
    SE_H_REL = SE_Z_REL << 4
    SE_H_DEFAULT = SE_Z_DEFAULT << 4
    SE_P_SET = SE_Z_SET << 8
    SE_P_ABS = SE_Z_ABS << 8
    SE_P_REL = SE_Z_REL << 8
    SE_P_DEFAULT = SE_Z_DEFAULT << 8
    SE_R_SET = SE_Z_SET << 12
    SE_R_DEFAULT = SE_Z_DEFAULT << 12
    SE_R_ABS = SE_Z_ABS << 12
    SE_R_REL = SE_Z_REL << 12


class PositionModeType(IntFlag):
    SE_SET = 1  # Used by explicit set functions
    SE_UPDATE = 2  # Used by controllers updating the position


class ScenarioObjectState(object):
    def __init__(self) -> None:
        self._ptr = _esmini_cffi.ffi.new("SE_ScenarioObjectState*")

    @property
    def id(self) -> int:
        return self._ptr.id  # type: ignore

    @property
    def model_id(self) -> int:
        return self._ptr.model_id  # type: ignore

    @property
    def ctrl_type(self) -> int:
        return self._ptr.ctrl_type  # type: ignore

    @property
    def timestamp(self) -> float:
        return self._ptr.timestamp  # type: ignore

    @property
    def x(self) -> float:
        return self._ptr.x  # type: ignore

    @property
    def y(self) -> float:
        return self._ptr.y  # type: ignore

    @property
    def z(self) -> float:
        return self._ptr.z  # type: ignore

    @property
    def h(self) -> float:
        return self._ptr.h  # type: ignore

    @property
    def p(self) -> float:
        return self._ptr.p  # type: ignore

    @property
    def r(self) -> float:
        return self._ptr.r  # type: ignore

    @property
    def road_id(self) -> int:
        return self._ptr.roadId  # type: ignore

    @property
    def t(self) -> float:
        return self._ptr.t  # type: ignore

    @property
    def lane_id(self) -> int:
        return self._ptr.laneId  # type: ignore

    @property
    def lane_offset(self) -> float:
        return self._ptr.laneOffset  # type: ignore

    @property
    def s(self) -> float:
        return self._ptr.s  # type: ignore

    @property
    def speed(self) -> float:
        return self._ptr.speed  # type: ignore

    @property
    def center_offset_x(self) -> float:
        return self._ptr.centerOffsetX  # type: ignore

    @property
    def center_offset_y(self) -> float:
        return self._ptr.centerOffsetY  # type: ignore

    @property
    def center_offset_z(self) -> float:
        return self._ptr.centerOffsetZ  # type: ignore

    @property
    def width(self) -> float:
        return self._ptr.width  # type: ignore

    @property
    def length(self) -> float:
        return self._ptr.length  # type: ignore

    @property
    def height(self) -> float:
        return self._ptr.height  # type: ignore


class RoadInfo(object):
    def __init__(self) -> None:
        self._ptr = _esmini_cffi.ffi.new("SE_RoadInfo*")

    @property
    def global_pos_x(self) -> float:
        return self._ptr.global_pos_x  # type: ignore

    @property
    def global_pos_y(self) -> float:
        return self._ptr.global_pos_y  # type: ignore

    @property
    def global_pos_z(self) -> float:
        return self._ptr.global_pos_z  # type: ignore

    @property
    def local_pos_x(self) -> float:
        return self._ptr.local_pos_x  # type: ignore

    @property
    def local_pos_y(self) -> float:
        return self._ptr.local_pos_y  # type: ignore

    @property
    def local_pos_z(self) -> float:
        return self._ptr.local_pos_z  # type: ignore

    @property
    def angle(self) -> float:
        return self._ptr.angle  # type: ignore

    @property
    def road_heading(self) -> float:
        return self._ptr.road_heading  # type: ignore

    @property
    def road_pitch(self) -> float:
        return self._ptr.road_pitch  # type: ignore

    @property
    def road_roll(self) -> float:
        return self._ptr.road_roll  # type: ignore

    @property
    def trail_heading(self) -> float:
        return self._ptr.trail_heading  # type: ignore

    @property
    def curvature(self) -> float:
        return self._ptr.curvature  # type: ignore

    @property
    def speed_limit(self) -> float:
        return self._ptr.speed_limit  # type: ignore


class RouteInfo(object):
    def __init__(self) -> None:
        self._ptr = _esmini_cffi.ffi.new("SE_RouteInfo*")

    @property
    def x(self) -> float:
        """Route point in the global coordinate system"""
        return self._ptr.x  # type: ignore

    @property
    def y(self) -> float:
        """Route point in the global coordinate system"""
        return self._ptr.y  # type: ignore

    @property
    def z(self) -> float:
        """Route point in the global coordinate system"""
        return self._ptr.z  # type: ignore

    @property
    def road_id(self) -> int:
        """Route point, road ID"""
        return self._ptr.roadId  # type: ignore

    @property
    def junction_id(self) -> int:
        """Route point, junction ID (-1 if not in a junction)"""
        return self._ptr.junctionId  # type: ignore

    @property
    def lane_id(self) -> int:
        """Route point, lane ID"""
        return self._ptr.laneId  # type: ignore

    @property
    def osi_lane_id(self) -> int:
        """Route point, osi lane ID"""
        return self._ptr.osiLaneId  # type: ignore

    @property
    def lane_offset(self) -> float:
        """Route point, lane offset (lateral distance from lane center)"""
        return self._ptr.laneOffset  # type: ignore

    @property
    def s(self) -> float:
        """Route point, s (longitudinal distance along reference line)"""
        return self._ptr.s  # type: ignore

    @property
    def t(self) -> float:
        """Route point, t (lateral distance from reference line)"""
        return self._ptr.t  # type: ignore


class LaneBoundaryId(object):
    def __init__(self) -> None:
        self._ptr = _esmini_cffi.ffi.new("SE_LaneBoundaryId*")

    @property
    def far_left_lb_id(self) -> int:
        return self._ptr.far_left_lb_id  # type: ignore

    @property
    def left_lb_id(self) -> int:
        return self._ptr.left_lb_id  # type: ignore

    @property
    def right_lb_id(self) -> int:
        return self._ptr.right_lb_id  # type: ignore

    @property
    def far_right_lb_id(self) -> int:
        return self._ptr.far_right_lb_id  # type: ignore


class PositionDiff(object):
    def __init__(self) -> None:
        self._ptr = _esmini_cffi.ffi.new("SE_PositionDiff*")

    @property
    def ds(self) -> float:
        """delta s (longitudinal distance)"""
        return self._ptr.ds  # type: ignore

    @property
    def dt(self) -> float:
        """delta t (lateral distance)"""
        return self._ptr.dt  # type: ignore

    @property
    def d_lane_id(self) -> int:
        """delta laneId (increasing left and decreasing to the right)"""
        return self._ptr.dLaneId  # type: ignore

    @property
    def dx(self) -> float:
        """delta x (world coordinate system)"""
        return self._ptr.dx  # type: ignore

    @property
    def dy(self) -> float:
        """delta y (world coordinate system)"""
        return self._ptr.dy  # type: ignore

    @property
    def opposite_lanes(self) -> bool:
        """true if the two position objects are in opposite sides of reference lane"""
        return self._ptr.oppositeLanes  # type: ignore


class SimpleVehicleState(object):
    def __init__(self) -> None:
        self._ptr = _esmini_cffi.ffi.new("SE_SimpleVehicleState*")

    @property
    def x(self) -> float:
        return self._ptr.x  # type: ignore

    @property
    def y(self) -> float:
        return self._ptr.y  # type: ignore

    @property
    def z(self) -> float:
        return self._ptr.z  # type: ignore

    @property
    def heading(self) -> float:
        return self._ptr.h  # type: ignore

    @property
    def pitch(self) -> float:
        return self._ptr.p  # type: ignore

    @property
    def speed(self) -> float:
        return self._ptr.speed  # type: ignore

    @property
    def wheel_rotation(self) -> float:
        return self._ptr.wheel_rotation  # type: ignore

    @property
    def wheel_angle(self) -> float:
        return self._ptr.wheel_angle  # type: ignore


class RoadSign(object):
    def __init__(self) -> None:
        self._ptr = _esmini_cffi.ffi.new("SE_RoadSign*")

    @property
    def id(self) -> int:
        """just an unique identifier of the sign"""
        return self._ptr.id  # type: ignore

    @property
    def x(self) -> float:
        """global x coordinate of sign position"""
        return self._ptr.x  # type: ignore

    @property
    def y(self) -> float:
        """global y coordinate of sign position"""
        return self._ptr.y  # type: ignore

    @property
    def z(self) -> float:
        """global z coordinate of sign position"""
        return self._ptr.z  # type: ignore

    @property
    def z_offset(self) -> float:
        """z offset from road level"""
        return self._ptr.z_offset  # type: ignore

    @property
    def h(self) -> float:
        """global heading of sign orientation"""
        return self._ptr.h  # type: ignore

    @property
    def road_id(self) -> int:
        """road id of sign road position"""
        return self._ptr.roadId  # type: ignore

    @property
    def s(self) -> float:
        """longitudinal position along road"""
        return self._ptr.s  # type: ignore

    @property
    def t(self) -> float:
        """lateral position from road reference line"""
        return self._ptr.t  # type: ignore

    @property
    def name(self) -> str:
        """sign name, typically used for 3D model filename"""
        name: bytes = self._ptr.name  # type: ignore
        return name.decode()

    @property
    def orientation(self) -> int:
        """1=facing traffic in road direction, -1=facing traffic opposite road direction"""
        return self._ptr.orientation  # type: ignore

    @property
    def length(self) -> float:
        """length as specified in OpenDRIVE"""
        return self._ptr.length  # type: ignore

    @property
    def height(self) -> float:
        """height as specified in OpenDRIVE"""
        return self._ptr.height  # type: ignore

    @property
    def width(self) -> float:
        """width as specified in OpenDRIVE"""
        return self._ptr.width  # type: ignore


class RoadObjValidity(object):
    def __init__(self) -> None:
        self._ptr = _esmini_cffi.ffi.new("SE_RoadObjValidity*")

    @property
    def from_lane(self) -> int:
        return self._ptr.fromLane  # type: ignore

    @property
    def to_lane(self) -> int:
        return self._ptr.toLane  # type: ignore


class ImageType(IntEnum):
    RGB = 0x1907
    BGR = 0x80E0


class Image(object):
    def __init__(self) -> None:
        self._ptr = _esmini_cffi.ffi.new("SE_Image*")

    @property
    def width(self) -> int:
        return self._ptr.width  # type: ignore

    @property
    def height(self) -> int:
        return self._ptr.height  # type: ignore

    @property
    def pixel_size(self) -> int:
        """Number of channels in the image.
        3 for RGB/BGR.
        """
        return self._ptr.pixelSize  # type: ignore

    @property
    def pixel_format(self) -> ImageType:
        """ """
        format: int = self._ptr.pixelFormat  # type: ignore
        return ImageType(format)

    @property
    def data(self) -> bytes:
        return self._ptr.data  # type: ignore


class Center(object):
    def __init__(self) -> None:
        self._ptr = _esmini_cffi.ffi.new("SE_Center*")

    @property
    def x_(self) -> float:
        """Center offset in x direction."""
        return self._ptr.x_  # type: ignore

    @property
    def y_(self) -> float:
        """Center offset in y direction."""
        return self._ptr.y_  # type: ignore

    @property
    def z_(self) -> float:
        """Center offset in z direction."""
        return self._ptr.z_  # type: ignore


class Dimensions(object):
    def __init__(self) -> None:
        self._ptr = _esmini_cffi.ffi.new("SE_Dimensions*")

    @property
    def width_(self) -> float:
        """Width of the entity's bounding box. Unit: m; Range: [0..inf[."""
        return self._ptr.width_  # type: ignore

    @property
    def length_(self) -> float:
        """Length of the entity's bounding box. Unit: m; Range: [0..inf[."""
        return self._ptr.length_  # type: ignore

    @property
    def height_(self) -> float:
        """Height of the entity's bounding box. Unit: m; Range: [0..inf[."""
        return self._ptr.height_  # type: ignore


class Parameter(object):
    def __init__(self) -> None:
        self._ptr = _esmini_cffi.ffi.new("SE_Parameter*")

    @property
    def name(self) -> str:
        """Name of the parameter as defined in the OpenSCENARIO"""
        name: bytes = self._ptr.name  # type: ignore
        return name.decode()

    @property
    def value(self):  # noqa: ANN201
        """Pointer to value which can be an integer, double, bool or string (const char*) as defined in the OpenSCENARIO file"""
        # return self._ptr.value  # type: ignore
        raise NotImplementedError


class Variable(object):
    def __init__(self) -> None:
        self._ptr = _esmini_cffi.ffi.new("SE_Variable*")

    @property
    def name(self) -> str:
        """Name of the variable as defined in the OpenSCENARIO"""
        name: bytes = self._ptr.name  # type: ignore
        return name.decode()

    @property
    def value(self):  # noqa: ANN201
        """Pointer to value which can be an integer, double, bool or string (const char*) as defined in the OpenSCENARIO file"""
        # return self._ptr.value  # type: ignore
        raise NotImplementedError


class ViewerFlag(IntFlag):
    NO_VIEWER = 0
    WINDOWED = 1
    OFF_SCREEN_ONLY = 2
    CAPTURE_TO_FILE = 4
    DISABLE_INFO_TEXT = 8


@overload
def init_scenario_engine(
    *,
    osc_filename: Union[str, bytes, os.PathLike],
    disable_ctrls: bool = False,
    use_viewer: ViewerFlag = ViewerFlag.WINDOWED,
    viewer_thread: bool = False,
    record: bool = False,
) -> None: ...


@overload
def init_scenario_engine(
    *,
    xml_specification: Union[str, bytes],
    disable_ctrls: bool = False,
    use_viewer: ViewerFlag = ViewerFlag.WINDOWED,
    viewer_thread: bool = False,
    record: bool = False,
) -> None: ...


def init_scenario_engine(**kwargs) -> None:
    """Initialize the scenario engine

        Parameters
        ----------
    osc_filename
            Path to the OpenSCENARIO file
        xml_specification
            OpenSCENARIO XML as string
        disable_ctrls
            If `False`, controllers will be applied according to OSC file. Otherwise, all controller will be disabled.
        use_viewer
            Flag to control the viewer. See `ViewerFlag`
        viewer_thread
            If `True`, will run the viewer in a separate thread.
        record
            If `True`, will create a recording for later playback.

    """
    xml_specification: Union[str, bytes, None] = kwargs.get("xml_specification")
    osc_filename: Optional[Union[str, bytes, os.PathLike]] = kwargs.get("osc_filename")

    disable_ctrls: bool = kwargs.get("disable_ctrls", False)
    use_viewer: ViewerFlag = kwargs.get("use_viewer", ViewerFlag.WINDOWED)
    viewer_thread: bool = kwargs.get("viewer_thread", False)
    record: bool = kwargs.get("record", False)

    if (xml_specification is not None) == (osc_filename is not None):
        # This checks for XOR NONE
        raise ValueError("Either one of `xml_specification` or `osc_filename`, and not both")

    if xml_specification is not None:
        if isinstance(xml_specification, str):
            xml_specification = xml_specification.encode()
        # Run InitWithString
        spec_string = _esmini_cffi.ffi.new("const char[]", xml_specification)
        ok: int = _esmini_cffi.lib.SE_InitWithString(
            spec_string,
            disable_ctrls,
            int(use_viewer),
            viewer_thread,
            record,
        )
    elif osc_filename is not None:
        if not isinstance(osc_filename, bytes):
            filename = str(osc_filename).encode()
        else:
            filename = osc_filename
        # Run with Init
        spec_filename = _esmini_cffi.ffi.new("const char[]", filename)
        ok: int = _esmini_cffi.lib.SE_Init(
            spec_filename,
            disable_ctrls,
            int(use_viewer),
            viewer_thread,
            record,
        )
    else:
        raise ValueError("Either one of `xml_specification` or `osc_filename`, and not both")

    if ok != 0:
        raise RuntimeError("Unable to initialize esmini scenario engine")


def add_search_path(path: Union[str, bytes, os.PathLike]) -> None:
    """Add a search path for OpenDRIVE and 3D model files.
    Needs to be called before `init_scenario_engine`.
    """
    if not isinstance(path, bytes):
        search_path = str(path).encode()
    else:
        search_path = bytes(path)

    path_str = _esmini_cffi.ffi.new("const char[]", search_path)
    if _esmini_cffi.lib.SE_AddPath(path_str) != 0:
        raise RuntimeError("Unable to add search path")


def clear_search_paths() -> None:
    """
    Clear all search paths for OpenDRIVE and 3D model files.
    Needs to be called prior to `init_scenario_engine`.
    """
    _esmini_cffi.lib.SE_ClearPaths()


def set_logfile_path(path: Union[str, bytes, os.PathLike]) -> None:
    """
    Specify scenario logfile (.txt) file path, optionally including directory path and/or filename.
    Specify only directory (end with "/" or "\") to let esmini set default filename.
    Specify only filename (no leading "/" or "\") to let esmini set default directory.
    Set "" to disable logfile
    """
    if not isinstance(path, bytes):
        logfile_path = str(path).encode()
    else:
        logfile_path = bytes(path)

    logfilepath_str = _esmini_cffi.ffi.new("const char[]", logfile_path)
    _esmini_cffi.lib.SE_SetLogFilePath(logfilepath_str)


def set_datfile_path(path: Union[str, bytes, os.PathLike]) -> None:
    """
    Specify scenario recording (.dat) file path, optionally including directory path and/or filename.
    Specify only directory (end with "/" or "\") to let esmini set default filename.
    Specify only filename (no leading "/" or "\") to let esmini set default directory.
    Set "" to use default .dat filename
    """
    if not isinstance(path, bytes):
        datfile_path = str(path).encode()
    else:
        datfile_path = bytes(path)

    datfilepath_str = _esmini_cffi.ffi.new("const char[]", datfile_path)
    _esmini_cffi.lib.SE_SetDatFilePath(datfilepath_str)


def get_seed() -> int:
    """
    Get seed that esmini uses for current session. It can then be re-used
    in order to achieve repeatable results (for actions that involes some
    degree of randomness, e.g. TrafficSwarmAction).
    """
    return _esmini_cffi.lib.SE_GetSeed()


def set_seed(seed: int) -> None:
    """
    Set seed that will be used by esmini random number generator.


    Notes
    -----
    Using same seed will ensure same result, but also note that timesteps have to be equal. Make sure to use `set_step_dt` with
    fixed timestep, or at least same sequence of `dt` each run.
    """
    _esmini_cffi.lib.SE_SetSeed(seed)


def set_window(x: int, y: int, width: int, height: int) -> None:
    """Set the window position and size.
    Must be called before `init_scenario_engine`.

    Parameters
    ----------
    x
        Screen coordinate in pixels for left side of window
    y
        Screen coordinate in pixels for top of window
    w
        Width in pixels
    h
        Height in pixels
    """
    _esmini_cffi.lib.SE_SetWindowPosAndSize(x, y, width, height)


# TODO: need to register a trampoline
def register_parameter_declaration_callback() -> None:
    raise NotImplementedError


def set_osi_tolerances(max_longitudinal_dist: float = 50, max_lateral_deviation: float = 0.05) -> None:
    """
    Configure tolerances/resolution for OSI road features

    Parameters
    ----------
    max_longitudinal_dist
        Maximum distance between OSI points, even on straight road. Default=50(m)
    max_lateral_deviation
        Control resolution w.r.t. curvature default=0.05(m)
    """
    if _esmini_cffi.lib.SE_SetOSITolerances(max_longitudinal_dist, max_lateral_deviation) != 0:
        raise RuntimeError("unable to set OSI tolerances")


def set_parameter_distribution(path: Union[str, bytes, os.PathLike]) -> None:
    """Specify OpenSCENARIO parameter distribution file.
    Must be called before `init_scenario_engine`.
    """
    if not isinstance(path, bytes):
        param_file = str(path).encode()
    else:
        param_file = bytes(path)

    param_file_str = _esmini_cffi.ffi.new("const char[]", param_file)
    if _esmini_cffi.ffi.SE_SetParameterDistribution(param_file_str) != 0:
        raise RuntimeError("unable to set parameter distribution file")


def reset_parameter_distribution() -> None:
    """
    Reset and disable parameter distribution.
    """
    _esmini_cffi.lib.SE_ResetParameterDistribution()


def get_number_of_permutations() -> int:
    """
    Get the number of parameter value permutations.
    Call after `init_scenario_engine`.
    """
    n_perm: int
    if (n_perm := _esmini_cffi.lib.SE_GetNumberOfPermutations()) >= 0:
        return n_perm
    else:
        raise RuntimeError("couldn't get number of parameter permutations")


def select_permutation(index: int) -> None:
    """
    Select parameter value permutation.
    Call before `init_scenario_engine`, e.g. during or after preceding run.
    """
    _esmini_cffi.lib.SE_SelectPermutation(index)


def get_permutation_index() -> int:
    """Get current parameter permutation index."""
    return _esmini_cffi.lib.SE_GetPermutationIndex()


def step_sim(dt: Optional[float] = None) -> None:
    """Step the simulation forward.

    Parameters
    ----------
    dt
        Time step in seconds. If `None`, time step will be elapsed system/world time since last step.
        Useful for interactive/realtime use cases
    """
    if dt is not None:
        ok: int = _esmini_cffi.lib.SE_StepDT(dt)
    else:
        ok: int = _esmini_cffi.lib.SE_Step()
    if ok != 0:
        raise RuntimeError("unable to step simulation forward")


def close_sim() -> None:
    """Stop the simulation gracefully.

    Useful for:
    1. Releasing memory;
    2. Prepare for next simulation, e.g., reset object lists.
    """
    _esmini_cffi.lib.SE_Close()


def log_to_console(enable: bool) -> None:
    """Enable or disable log to stdout"""
    _esmini_cffi.lib.SE_LogToConsole(enable)


def enable_collision_detection(enable: bool) -> None:
    """Enable or disable global collision detection"""
    _esmini_cffi.lib.SE_CollisionDetection(enable)


def get_sim_time(double: bool = True) -> float:
    """Get simulation time in seconds.

    Parameters
    ----------
    double
        If `True`, returned value has double (64-bit) precision
        Otherwise, float (32-bit) precision
    """
    if double:
        return _esmini_cffi.lib.SE_GetSimulationTimeDouble()
    else:
        return _esmini_cffi.lib.SE_GetSimulationTime()


def get_sim_time_step() -> float:
    """Get simulation time step in seconds.

    The time step is calculated as difference since last call to same funtion.
    Clamped to some reasonable values. First call returns smallest delta (typically 1 ms).
    """
    return _esmini_cffi.lib.SE_GetSimTimeStep()


def is_sim_quitting() -> bool:
    ok: int = _esmini_cffi.lib.SE_GetQuitFlag()
    if ok == -1:
        raise RuntimeError("error when checking if esmini is about to quit")
    return bool(ok)


def is_sim_paused() -> bool:
    ok: int = _esmini_cffi.lib.SE_GetPauseFlag()
    if ok == -1:
        raise RuntimeError("error when checking if esmini is paused")
    return bool(ok)


def get_odr_filename() -> Path:
    """Get name of currently referred and loaded OpenDRIVE file"""
    filename: bytes = _esmini_cffi.lib.SE_GetODRFilename()
    return Path(filename.decode())


def get_scenegraph_filename() -> Path:
    """Get name of currently referred and loaded SceneGraph file"""
    filename: bytes = _esmini_cffi.lib.SE_GetSceneGraphFilename()
    return Path(filename.decode())


def get_number_of_parameters() -> int:
    """Get the number of named parameters within the current scenario"""
    return _esmini_cffi.lib.SE_GetNumberOfParameters()


class ParameterType(IntEnum):
    INT = 0
    DOUBLE = 1
    STRING = 2
    BOOL = 3


VariableType: TypeAlias = ParameterType


def get_parameter_name(index: int, ptype: ParameterType) -> str:
    """Get the name of a named parameter"""
    type_ptr = _esmini_cffi.ffi.new("int *", int(ptype))
    name: bytes = _esmini_cffi.lib.SE_GetParameterName(index, type_ptr)
    return name.decode()


def get_num_properties(index: int) -> int:
    """Get the number of vehicle properties by index

    Parameters
    ----------
    index
        Index of the vehicle
    """
    num: int
    if (num := _esmini_cffi.lib.SE_GetNumberOfProperties(index)) >= 0:
        return num
    raise RuntimeError("Unable to get number of properties")


def set_parameter(name: Union[str, bytes], value: Union[bool, int, float, str, bytes]) -> bool:  # noqa: PYI041
    """Set the parameter with the given `name`. Returns `False` if not successful."""
    if isinstance(value, bool):
        ok = _esmini_cffi.ffi.SE_SetParameterBool(name, value)
    elif isinstance(value, int):
        ok = _esmini_cffi.ffi.SE_SetParameterInt(name, value)
    elif isinstance(value, float):
        ok = _esmini_cffi.ffi.SE_SetParameterDouble(name, value)
    elif isinstance(value, (str, bytes)):
        ok = _esmini_cffi.ffi.SE_SetParameterString(name, value)
    else:
        raise TypeError(f"unsupported parameter type {type(value)}")

    if ok == 0:
        return True
    return False


def get_parameter(name: Union[str, bytes], ptype: ParameterType) -> Union[bool, int, float, str]:  # noqa: PYI041
    """Get the typed value of the named parameter.

    Raises
    ------
    TypeError
        Incorrect type of parameter
    """
    if ptype == ParameterType.BOOL:
        bool_ptr = _esmini_cffi.ffi.new("bool *")
        ok = _esmini_cffi.lib.SE_GetParameterBool(name, bool_ptr)
        if ok == -1:
            raise TypeError("Incorrect type of parameter")
        return bool_ptr[0]  # type: bool
    elif ptype == ParameterType.INT:
        int_ptr = _esmini_cffi.ffi.new("int *")
        ok = _esmini_cffi.lib.SE_GetParameterInt(name, int_ptr)
        if ok == -1:
            raise TypeError("Incorrect type of parameter")
        return int_ptr[0]  # type: int
    elif ptype == ParameterType.DOUBLE:
        double_ptr = _esmini_cffi.ffi.new("double *")
        ok = _esmini_cffi.lib.SE_GetParameterDouble(name, double_ptr)
        if ok == -1:
            raise TypeError("Incorrect type of parameter")
        return double_ptr[0]  # type: float
    else:
        assert ptype == ParameterType.STRING
        str_ptr = _esmini_cffi.ffi.new("char **")
        ok = _esmini_cffi.lib.SE_GetParameterString(name, str_ptr)
        if ok == -1:
            raise TypeError("Incorrect type of parameter")
        ret: bytes = _esmini_cffi.ffi.string(str_ptr[0])
        return ret.decode()


def set_variable(name: Union[str, bytes], value: Union[bool, int, float, str, bytes]) -> bool:  # noqa: PYI041
    """Set the variable with the given `name`. Returns `False` if not successful."""
    if isinstance(value, bool):
        ok = _esmini_cffi.ffi.SE_SetVariableBool(name, value)
    elif isinstance(value, int):
        ok = _esmini_cffi.ffi.SE_SetVariableInt(name, value)
    elif isinstance(value, float):
        ok = _esmini_cffi.ffi.SE_SetVariableDouble(name, value)
    elif isinstance(value, (str, bytes)):
        ok = _esmini_cffi.ffi.SE_SetVariableString(name, value)
    else:
        raise TypeError(f"unsupported variable type {type(value)}")

    if ok == 0:
        return True
    return False


def get_variable(name: Union[str, bytes], ptype: VariableType) -> Union[bool, int, float, str]:  # noqa: PYI041
    """Get the typed value of the named variable.

    Raises
    ------
    TypeError
        Incorrect type of variable
    """
    if ptype == VariableType.BOOL:
        bool_ptr = _esmini_cffi.ffi.new("bool *")
        ok = _esmini_cffi.lib.SE_GetVariableBool(name, bool_ptr)
        if ok == -1:
            raise TypeError("Incorrect type of variable")
        return bool_ptr[0]  # type: bool
    elif ptype == VariableType.INT:
        int_ptr = _esmini_cffi.ffi.new("int *")
        ok = _esmini_cffi.lib.SE_GetVariableInt(name, int_ptr)
        if ok == -1:
            raise TypeError("Incorrect type of variable")
        return int_ptr[0]  # type: int
    elif ptype == VariableType.DOUBLE:
        double_ptr = _esmini_cffi.ffi.new("double *")
        ok = _esmini_cffi.lib.SE_GetVariableDouble(name, double_ptr)
        if ok == -1:
            raise TypeError("Incorrect type of variable")
        return double_ptr[0]  # type: float
    else:
        assert ptype == VariableType.STRING
        str_ptr = _esmini_cffi.ffi.new("char **")
        ok = _esmini_cffi.lib.SE_GetVariableString(name, str_ptr)
        if ok == -1:
            raise TypeError("Incorrect type of variable")
        ret: bytes = _esmini_cffi.ffi.string(str_ptr[0])
        return ret.decode()


def set_object_position_mode(
    object_id: int, position_mode_type: PositionModeType, mode: Optional[PositionMode] = None
) -> None:
    """
    Specify if and how position object will align to the road. The setting is done for individual components:
    Z (elevation), Heading, Pitch, Roll and separately for set- and update operation. Set operations represents
    when position is affected by API calls, e.g. updateObjectWorldPos(). Update operations represents when the
    position is updated implicitly by the scenarioengine, e.g. default controller moving a vehicle along the lane.

    Parameters
    ----------
    object_id
        Id of the object
    position_mode_type
        Type of operations the setting applies to
    mode
        Bitmask combining values from `PositionMode` enum.
        example: To set relative z and absolute roll: (SE_Z_REL | SE_R_ABS) or (7 | 12288) = (7 + 12288) = 12295
        according to roadmanager::PosModeType
    """
    if mode is not None:
        _esmini_cffi.lib.SE_SetObjectPositionMode(object_id, int(position_mode_type), int(mode))
    else:
        _esmini_cffi.lib.SE_SetObjectPositionModeDefault(object_id, int(position_mode_type))


def add_object(
    name: Union[str, bytes],
    object_type: int,
    object_category: int,
    object_role: int,
    model_id: int,
) -> int:
    if (id := _esmini_cffi.lib.SE_AddObject(name, object_type, object_category, object_role, model_id)) >= 0:
        return id
    raise RuntimeError("Unable to add object")


def delete_object(id: int) -> None:
    ok: int = _esmini_cffi.lib.SE_DeleteObject(id)
    if ok != 0:
        raise RuntimeError("unable to delete object")


def report_object_position(
    object_id: int,
    timestamp: float,
    x: float,
    y: float,
    heading: float,
    z: Optional[float] = None,
    pitch: Optional[float] = None,
    roll: Optional[float] = None,
    mode: Optional[PositionMode] = None,
) -> bool:
    """
    Report object position in cartesian coordinates

    Parameters
    ----------

    object_id
        Id of the object
    timestamp
        Timestamp (not really used yet, OK to set 0)
    x
        X coordinate
    y
        Y coordinate
    z
        Z coordinate
    h
        Heading / yaw
    p
        Pitch
    r
        Roll

    Returns
    -------
    bool
        `True` is successful. `False` otherwise.
    """
    ok: int
    if mode is not None:
        z = z or math.nan
        pitch = pitch or math.nan
        roll = roll or math.nan
        ok = _esmini_cffi.lib.SE_ReportObjectPosMode(object_id, timestamp, x, y, z, heading, pitch, roll, int(mode))
    elif None not in [z, pitch, roll]:
        # Just XYH
        ok = _esmini_cffi.lib.SE_ReportObjectPosXYH(object_id, timestamp, x, y, heading)
    else:
        z = z or math.nan
        pitch = pitch or math.nan
        roll = roll or math.nan
        ok = _esmini_cffi.lib.SE_ReportObjectPos(object_id, timestamp, x, y, z, heading, pitch, roll)

    return ok == 0


__docformat__ = "numpy"
