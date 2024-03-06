# pyright: reportAttributeAccessIssue=false
# mypy: disable_error_code=attr-defined

import importlib.util
import math
import warnings
from enum import IntEnum, IntFlag
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

from typing_extensions import TypeAlias, overload

if importlib.util.find_spec("esmini._esmini_cffi") is None:
    warnings.warn("esmini extension module not present. Doing so now.", UserWarning, stacklevel=0)
    from esmini._build_esmini import ffibuilder

    ffibuilder.compile(verbose=True)


import esmini._esmini_cffi as _esmini_cffi

if TYPE_CHECKING:
    import _cffi_backend


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
    def __init__(self, ptr: Optional["_cffi_backend.FFI.CData"] = None) -> None:
        """@private"""
        if ptr is None:
            self._ptr = _esmini_cffi.ffi.new("SE_ScenarioObjectState*")
        else:
            self._ptr = ptr

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
    osc_filename: Union[str, bytes, Path],
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
    osc_filename: Optional[Union[str, bytes, Path]] = kwargs.get("osc_filename")

    disable_ctrls: bool = kwargs.get("disable_ctrls", False)
    use_viewer: ViewerFlag = kwargs.get("use_viewer", ViewerFlag.WINDOWED)
    viewer_thread: bool = kwargs.get("viewer_thread", False)
    record: bool = kwargs.get("record", False)

    if (xml_specification is not None) == (osc_filename is not None):
        # This checks for XOR NONE
        raise ValueError("Either one of `xml_specification` or `osc_filename`, and not both")

    ok: int
    if xml_specification is not None:
        assert isinstance(xml_specification, (str, bytes))
        # Run InitWithString
        ok = _esmini_cffi.lib.SE_InitWithString(
            xml_specification,
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
        ok = _esmini_cffi.lib.SE_Init(
            filename,
            disable_ctrls,
            int(use_viewer),
            viewer_thread,
            record,
        )
    else:
        raise ValueError("Either one of `xml_specification` or `osc_filename`, and not both")

    if ok != 0:
        raise RuntimeError("Unable to initialize esmini scenario engine")


def add_search_path(path: Union[str, bytes, Path]) -> None:
    """Add a search path for OpenDRIVE and 3D model files.
    Needs to be called before `init_scenario_engine`.
    """
    if not isinstance(path, bytes):
        search_path = str(path).encode()
    else:
        search_path = bytes(path)

    if _esmini_cffi.lib.SE_AddPath(search_path) != 0:
        raise RuntimeError("Unable to add search path")


def clear_search_paths() -> None:
    """
    Clear all search paths for OpenDRIVE and 3D model files.
    Needs to be called prior to `init_scenario_engine`.
    """
    _esmini_cffi.lib.SE_ClearPaths()


def set_logfile_path(path: Union[str, bytes, Path]) -> None:
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

    _esmini_cffi.lib.SE_SetLogFilePath(logfile_path)


def set_datfile_path(path: Union[str, bytes, Path]) -> None:
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

    _esmini_cffi.lib.SE_SetDatFilePath(datfile_path)


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


def set_parameter_distribution(path: Union[str, bytes, Path]) -> None:
    """Specify OpenSCENARIO parameter distribution file.
    Must be called before `init_scenario_engine`.
    """
    if not isinstance(path, bytes):
        param_file = str(path).encode()
    else:
        param_file = bytes(path)

    if _esmini_cffi.ffi.SE_SetParameterDistribution(param_file) != 0:
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
    ok: int
    if dt is not None:
        ok = _esmini_cffi.lib.SE_StepDT(dt)
    else:
        ok = _esmini_cffi.lib.SE_Step()
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

    return ok == 0


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
        return bool_ptr[0]
    elif ptype == ParameterType.INT:
        int_ptr = _esmini_cffi.ffi.new("int *")
        ok = _esmini_cffi.lib.SE_GetParameterInt(name, int_ptr)
        if ok == -1:
            raise TypeError("Incorrect type of parameter")
        return int_ptr[0]
    elif ptype == ParameterType.DOUBLE:
        double_ptr = _esmini_cffi.ffi.new("double *")
        ok = _esmini_cffi.lib.SE_GetParameterDouble(name, double_ptr)
        if ok == -1:
            raise TypeError("Incorrect type of parameter")
        return double_ptr[0]
    else:
        assert ptype == ParameterType.STRING
        str_ptr = _esmini_cffi.ffi.new("char **")
        ok = _esmini_cffi.lib.SE_GetParameterString(name, str_ptr)
        if ok == -1:
            raise TypeError("Incorrect type of parameter")
        ret = _esmini_cffi.ffi.string(str_ptr[0])
        if isinstance(ret, bytes):
            return ret.decode()
        assert isinstance(ret, str)
        return ret


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

    return ok == 0


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
        return bool_ptr[0]
    elif ptype == VariableType.INT:
        int_ptr = _esmini_cffi.ffi.new("int *")
        ok = _esmini_cffi.lib.SE_GetVariableInt(name, int_ptr)
        if ok == -1:
            raise TypeError("Incorrect type of variable")
        return int_ptr[0]
    elif ptype == VariableType.DOUBLE:
        double_ptr = _esmini_cffi.ffi.new("double *")
        ok = _esmini_cffi.lib.SE_GetVariableDouble(name, double_ptr)
        if ok == -1:
            raise TypeError("Incorrect type of variable")
        return double_ptr[0]
    else:
        assert ptype == VariableType.STRING
        str_ptr = _esmini_cffi.ffi.new("char **")
        ok = _esmini_cffi.lib.SE_GetVariableString(name, str_ptr)
        if ok == -1:
            raise TypeError("Incorrect type of variable")
        ret = _esmini_cffi.ffi.string(str_ptr[0])
        if isinstance(ret, bytes):
            return ret.decode()
        assert isinstance(ret, str)
        return ret


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


def report_object_road_position(
    object_id: int, timestamp: float, road_id: int, lane_id: int, lane_offset: int, position: float
) -> bool:
    """Report object position in road coordinates.

    Parameters
    ----------
    object_id
        ID of object
    timestamp
        Timestamp (not really used, OK to set to 0)
    road_id
        ID of the road object
    lane_id
        ID of the lane
    lane_offset
        Lateral offset from center of specified lane
    position
        Longitudinal distance of the position along the specified road

    Returns
    -------
    bool
        `True` if successful.
    """
    return _esmini_cffi.lib.SE_ReportObjectRoadPos(object_id, timestamp, road_id, lane_id, lane_offset, position) == 0


def report_object_speed(object_id: int, speed: float) -> bool:
    """Report object longitudinal speed. Useful for an external longitudinal controller.

    Parameters
    ----------
    object_id
        ID of object
    speed
        Speed in forward/longitudinal direction of the entity.

    Returns
    -------
    bool
        `True` if successful.
    """
    return _esmini_cffi.lib.SE_ReportObjectSpeed(object_id, speed) == 0


def report_object_lateral_position(object_id: int, position: float) -> bool:
    """Report object lateral position relative road centerline. Useful for an external lateral controller.

    Parameters
    ----------
    object_id
        ID of object
    position
        Lateral position

    Returns
    -------
    bool
        `True` if successful.
    """
    return _esmini_cffi.lib.SE_ReportObjectLateralPosition(object_id, position) == 0


def report_object_lateral_lane_position(object_id: int, lane_id: int, lane_offset: float) -> bool:
    """Report object lateral position by lane id and lane offset. Useful for an external lateral controller.

    Parameters
    ----------
    object_id
        ID of object
    lane_id
        ID of the lane
    lane_offset
        Lateral offset from center of specified lane

    Returns
    -------
    bool
        `True` if successful.
    """
    return _esmini_cffi.lib.SE_ReportObjectLateralLanePosition(object_id, lane_id, lane_offset) == 0


def report_object_velocity(object_id: int, timestamp: float, x_vel: float, y_vel: float, z_vel: float) -> bool:
    """
    Report object velocity in cartesian coordinates

    Parameters
    ----------
    object_id
        ID of the object
    timestamp
        Timestamp (not really used yet, OK to set 0)
    x_vel
        X component of linear velocity
    y_vel
        Y component of linear velocity
    z_vel
        Z component of linear velocity

    Returns
    -------
    bool
        `True` if successful.
    """
    return _esmini_cffi.lib.SE_ReportObjectVel(object_id, timestamp, x_vel, y_vel, z_vel) == 0


def report_object_angular_velocity(
    object_id: int, timestamp: float, heading_rate: float, pitch_rate: float, roll_rate: float
) -> bool:
    """
    Report object angular velocity in cartesian coordinates

    Parameters
    ----------
    object_id
        ID of the object
    timestamp
        Timestamp (not really used yet, OK to set 0)
    heading_rate
        Heading component of angular velocity
    pitch_rate
        Pitch component of angular velocity
    roll_rate
        Roll component of angular velocity

    Returns
    -------
    bool
        `True` if successful.
    """
    ok = _esmini_cffi.lib.SE_ReportObjectAngularVel(object_id, timestamp, heading_rate, pitch_rate, roll_rate)
    return ok == 0


def report_object_acceleration(object_id: int, timestamp: float, x_acc: float, y_acc: float, z_acc: float) -> bool:
    """
    Report object acceleration in cartesian coordinates

    Parameters
    ----------
    object_id
        ID of the object
    timestamp
        Timestamp (not really used yet, OK to set 0)
    x_acc
        X component of linear acceleration
    y_acc
        Y component of linear acceleration
    z_acc
        Z component of linear acceleration

    Returns
    -------
    bool
        `True` if successful.
    """
    ok = _esmini_cffi.lib.SE_ReportObjectAcc(object_id, timestamp, x_acc, y_acc, z_acc)
    return ok == 0


def report_object_angular_acceleration(
    object_id: int, timestamp: float, heading_acc: float, pitch_acc: float, roll_acc: float
) -> bool:
    """
            Report object angular acceleration in cartesian coordinates

            Parameters
            ----------
            object_id
                ID of the object
            timestamp
                Timestamp (not really used yet, OK to set 0)
            heading_acc
    Heading component of angular acceleration
            pitch_acc
                Pitch component of angular acceleration
            roll_acc
                Roll component of angular acceleration

            Returns
            -------
            bool
                `True` if successful.
    """
    ok = _esmini_cffi.lib.SE_ReportObjectAngularAcc(object_id, timestamp, heading_acc, pitch_acc, roll_acc)
    return ok == 0


def report_object_wheel_status(object_id: int, rotation: float, angle: float) -> bool:
    """Report object wheel status

    Parameters
    ----------
    object_id
        ID Of the object
    rotation
        Wheel rotation
    angle
        Wheel steering angle

    Returns
    -------
    bool
        `True` if successful.
    """
    ok = _esmini_cffi.lib.SE_ReportObjectWheelStatus(object_id, rotation, angle)
    return ok == 0


class LaneType(IntFlag):
    NONE = 1 << 0
    DRIVING = 1 << 1
    STOP = 1 << 2
    SHOULDER = 1 << 3
    BIKING = 1 << 4
    SIDEWALK = 1 << 5
    BORDER = 1 << 6
    RESTRICTED = 1 << 7
    PARKING = 1 << 8
    BIDIRECTIONAL = 1 << 9
    MEDIAN = 1 << 10
    SPECIAL1 = 1 << 11
    SPECIAL2 = 1 << 12
    SPECIAL3 = 1 << 13
    ROADMARKS = 1 << 14
    TRAM = 1 << 15
    RAIL = 1 << 16
    ENTRY = 1 << 17
    EXIT = 1 << 18
    OFF_RAMP = 1 << 19
    ON_RAMP = 1 << 20
    CURB = 1 << 21
    CONNECTING_RAMP = 1 << 22
    REFERENCE_LINE = 1 << 0
    ANY_DRIVING = DRIVING | ENTRY | EXIT | OFF_RAMP | ON_RAMP | BIDIRECTIONAL
    ANY_ROAD = ANY_DRIVING | RESTRICTED | STOP
    ANY = -1


def set_snap_lane_types(object_id: int, lane_types: LaneType) -> bool:
    """Specify which lane types the given object snaps to (is aware of).

    Returns
    -------
    bool
        `True` is successful
    """
    return _esmini_cffi.lib.SE_SetSnapLaneTypes(object_id, int(lane_types)) == 0


def set_lock_on_lane(object_id: int, enable: bool) -> bool:
    """Controls whether the object stays in a lane regardless of lateral position.

    Parameters
    ----------
    object_id
        ID of the object
    enable
        If `True`, the object will lock onto the lane.
        Otherwise, it will snap to the closest lane (default global behavior).

    Returns
    -------
    bool
        `True` is successful
    """
    ok = _esmini_cffi.lib.SE_SetLockOnLane(object_id, enable)
    return ok == 0


def get_number_of_objects() -> int:
    if (ok := _esmini_cffi.lib.SE_GetNumberOfObjects()) >= 0:
        return ok
    raise RuntimeError("Unable to retrieve number of objects. Has scenario been initialized?")


def get_entity_id(index: int) -> int:
    if (ok := _esmini_cffi.lib.SE_GetId(index)) >= 0:
        return ok
    raise RuntimeError("Unable to retrieve object ID. Has scenario been initialized?")


def get_id_by_name(name: Union[str, bytes]) -> int:
    if (ok := _esmini_cffi.lib.SE_GetIdByName(name)) >= 0:
        return ok
    raise RuntimeError("Unable to retrieve object ID. Has scenario been initialized?")


def get_object_state(object_id: int) -> Optional[ScenarioObjectState]:
    """
    Returns
    -------
    ScenarioObjectState | None
        `None` if there was an error
    """
    state = ScenarioObjectState()
    ok = _esmini_cffi.lib.SE_GetObjectState(object_id, state._ptr)
    if ok == 0:
        return state
    return None


def get_object_type_name(object_id: int) -> str:
    """Get the type name of the specified vehicle/pedestrian/misc object."""
    name: bytes = _esmini_cffi.lib.SE_GetObjectTypeName(object_id)
    return name.decode()


def get_object_name(object_id: int) -> str:
    """Get the name of the specified object."""
    name: bytes = _esmini_cffi.lib.SE_GetObjectName(object_id)
    return name.decode()


def get_object_model_file_name(object_id: int) -> Path:
    """Get the 3D model filename of the specified object."""
    name: bytes = _esmini_cffi.lib.SE_GetObjectModelFileName(object_id)
    filename = Path(name.decode())
    return filename


def object_has_ghost(object_id: int) -> bool:
    """Check whether an object has a ghost (special purpose lead vehicle)."""
    check: int = _esmini_cffi.lib.SE_ObjectHasGhost(object_id)
    if check < 0:
        raise RuntimeError("unable to determine if object has ghost. Have you initialized the scenario engine?")
    return bool(check)


def get_object_ghost_state(object_id: int) -> Optional[ScenarioObjectState]:
    """Get the state of the specified object's ghost."""
    state = ScenarioObjectState()
    ok = _esmini_cffi.lib.SE_GetObjectGhostState(object_id, state._ptr)
    if ok == 0:
        return state
    return None


def get_object_number_of_collisions(object_id: int) -> int:
    """Get the number of collisions the specified object is currently involved in."""
    return _esmini_cffi.lib.SE_GetObjectNumberOfCollisions(object_id)


class SpeedUnit(IntEnum):
    KM_PER_HOUR = 1
    M_PER_SEC = 2
    MILES_PER_HOUR = 3


def get_speed_unit() -> Optional[SpeedUnit]:
    """Get the unit of specified speed (in OpenDRIVE road type element).

    All roads will be looped in search for such an element. First found will be used.
    If speed is specified withouth the optional unit, SI unit m/s (`SpeedUnit.M_PER_SEC`) is assumed.
    If no speed entries is found, `None` will be returned.
    """
    ret = _esmini_cffi.lib.SE_GetSpeedUnit()
    if ret < 0:
        raise RuntimeError("error while getting speed unit")
    elif ret == 0:
        return None
    else:
        return SpeedUnit(ret)


class LookAheadMode(IntFlag):
    LANE_CENTER = 0
    ROAD_CENTER = 1
    CURRENT_LANE_OFFSET = 2


class PositionError(Exception):
    """Error in looking up/setting a position"""

    pass


def get_road_info_at_distance(
    object_id: int,
    lookahead_dist: float,
    lookahead_mode: LookAheadMode,
    in_road_driving_direction: bool,
) -> RoadInfo:
    info = RoadInfo()
    ok = _esmini_cffi.lib.SE_GetRoadInfoAtDistance(
        object_id, lookahead_dist, info._ptr, int(lookahead_mode), in_road_driving_direction
    )
    if ok < 0:
        raise PositionError()
    return info


def get_road_info_along_ghost_trail(
    object_id: int,
    lookahead_dist: float,
) -> Tuple[RoadInfo, float]:
    """Get information suitable for driver modeling of a ghost vehicle driving ahead of the ego vehicle.

    Parameters
    ----------
    object_id
        ID of the Ego vehicle.
    lookahead_dist
        The distance, along the ghost trail, to the point from the current Ego vehicle location

    Returns
    -------
    RoadInfo
        Information about the road.
    float
        Speed that the ghost had at this point along the trail.
    """
    info = RoadInfo()
    speed = _esmini_cffi.ffi.new("float *")
    ok = _esmini_cffi.lib.SE_GetRoadInfoAlongGhostTrail(object_id, lookahead_dist, info._ptr, speed)
    if ok < 0:
        raise RuntimeError("unable to get road info along ghost trail")
    return info, speed[0]


def get_road_info_ghost_trail_time(
    object_id: int,
    time: float,
) -> Tuple[RoadInfo, float]:
    """Get information suitable for driver modeling of a ghost vehicle driving ahead of the ego vehicle.

    Parameters
    ----------
    object_id
        ID of the Ego vehicle.
    time
        Simulation time (subtracting headstart time, i.e., `time=0` gives initial state)

    Returns
    -------
    RoadInfo
        Information about the road.
    float
        Speed that the ghost had at this point along the trail.
    """
    info = RoadInfo()
    speed = _esmini_cffi.ffi.new("float *")
    ok = _esmini_cffi.lib.SE_GetRoadInfoGhostTrailTime(object_id, time, info._ptr, speed)
    if ok < 0:
        raise RuntimeError("unable to get road info along ghost trail")
    return info, speed[0]


def get_distance_to_object(object_id1: int, object_id2: int, free_space: bool) -> Optional[PositionDiff]:
    """Find out the delta between two objects.

    Notes
    -----
    Search range is 1000 meters

    Parameters
    ----------
    object_id1
        ID of the object from which to measure.
    object_id2
        ID of the object to which the distance is measured.
    free_space
        Measure distance between bounding boxes (`True`) or between reference points (`False`).

    Returns
    -------
    None | PositionDiff
        `None` if a route between positions cannot be found.
        The `PositionDiff` object otherwise.

    """

    dist = PositionDiff()
    ok = _esmini_cffi.lib.SE_GetDistanceToObject(object_id1, object_id2, free_space, dist._ptr)
    if ok == -2:
        return None
    elif ok == -1:
        raise RuntimeError("unable to compute distance between two objects")
    else:
        return dist


def add_object_sensor(
    object_id: int,
    x: float,
    y: float,
    z: float,
    heading: float,
    range_near: float,
    range_far: float,
    fov: float,
    max_num_obj: int,
) -> Optional[int]:
    """Create an ideal object sensor and attach to specified vehicle.

    Parameters
    ----------
    object_id
        ID of the object to attach the sensor to
    x
        X-coordinate of the sensor in vehicle local coordinates
    y
        Y-coordinate of the sensor in vehicle local coordinates
    z
        Z-coordinate of the sensor in vehicle local coordinates
    heading
        heading of the sensor in vehicle local coordinates
    fov
        horizontal field of view (in degrees)
    range_near
        near value of the sensor depth range
    range_far
        far value of the sensor depth range
    max_num_obj
        Maximum number of objects that the sensor can track

    Returns
    -------
    None | int
        `None` if unsuccessful. The sensor ID (global index) otherwise.
    """
    id = _esmini_cffi.lib.SE_AddObjectSensor(object_id, x, y, z, heading, range_near, range_far, fov, max_num_obj)
    if id < 0:
        return None
    return id


def get_number_of_object_sensors() -> Optional[int]:
    """Get total number of sensors attached to any objects."""
    ret = _esmini_cffi.lib.SE_GetNumberOfObjectSensors()
    if ret < 0:
        return None
    return ret


def visualize_object_sensor_data(object_id: int) -> None:
    """Allow visualization of detected sensor data for sensors attached to this object."""
    _esmini_cffi.lib.SE_ViewSensorData(object_id)


def fetch_sensed_object_list(sensor_id: int) -> Optional[List[int]]:
    """Get list of IDs of objects detected by this sensor."""
    det_list_ptr = _esmini_cffi.ffi.new("int*")
    num_det = _esmini_cffi.lib.SE_FetchSensorObjectList(sensor_id, det_list_ptr)
    if num_det < 0:
        return None
    else:
        det_list = _esmini_cffi.ffi.unpack(det_list_ptr, num_det)
        assert isinstance(det_list, list)
        assert len(det_list) == num_det
        if num_det > 0:
            assert isinstance(det_list[0], int)
        return det_list


ObjectCallback: TypeAlias = Callable[[ScenarioObjectState], None]


class ObjectCallbackHandle(object):
    """Handle to a callback registered for `register_object_callback`"""

    def __init__(self, object_id: int, fn: ObjectCallback) -> None:
        self._handle = _esmini_cffi.ffi.new_handle(self)

        self.object_id = object_id
        self.callback = fn
        _esmini_cffi.lib.SE_RegisterObjectCallback(
            object_id,
            _esmini_cffi.lib.esmini_object_callback,  # fnPtr,
            self._handle,
        )


@_esmini_cffi.ffi.def_extern(name="esmini_object_callback")  # pyright: ignore[reportOptionalCall]
def esmini_object_callback(c_state: "_cffi_backend.FFI.CData", handle: "_cffi_backend.FFI.CData") -> None:
    """@private Trampoline for object callbacks."""
    concrete_handle: ObjectCallbackHandle = _esmini_cffi.ffi.from_handle(handle)
    state = ScenarioObjectState(ptr=c_state)
    concrete_handle.callback(state)


def register_object_callback(object_id: int, callback: ObjectCallback) -> ObjectCallbackHandle:
    """Register a function to be called back from esmini after each frame (update of scenario).

    Notes
    -----
    * Complete or part of the state can be overriden by calling `report_object_road_position`.
    * Registered callbacks will be cleared between `init_scenario_engine` calls.

    .. warning:: Be Careful!

       Be sure to keep the returned handle safe, as it owns the allocation for the callback.
    """
    return ObjectCallbackHandle(object_id, callback)


def log_message(message: Union[str, bytes]) -> None:
    """Log message via esmini"""
    _esmini_cffi.lib.SE_LogMessage(message)


class ViewerNodeMask(IntFlag):
    NONE = 0
    OBJECT_SENSORS = 1 << 0
    TRAIL_LINES = 1 << 1
    TRAIL_DOTS = 1 << 2
    ODR_FEATURES = 1 << 3
    OSI_POINTS = 1 << 4
    OSI_LINES = 1 << 5
    ENV_MODEL = 1 << 6
    ENTITY_MODEL = 1 << 7
    ENTITY_BB = 1 << 8
    INFO = 1 << 9
    INFO_PER_OBJ = 1 << 10
    ROAD_SENSORS = 1 << 11
    TRAJECTORY_LINES = 1 << 12
    ROUTE_WAYPOINTS = 1 << 13
    SIGN = 1 << 14


def viewer_toggle_node(feature: ViewerNodeMask, enable: bool) -> None:
    """Toggle visualization of specific features."""
    _esmini_cffi.lib.SE_ViewerShowFeature(feature, enable)


class SimpleVehicle(object):
    """A simplistic vehicle based on a 2D bicycle kinematic model.

    Parameters
    ----------
    x
        Initial position X world coordinate
    y
        Initial position Y world coordinate
    heading
        Initial heading
    length
        Length of the vehicle
    initial_speed
        Initial speed
    """

    def __init__(self, x: float, y: float, heading: float, length: float, initial_speed: float) -> None:
        self._handle = _esmini_cffi.ffi.SE_SimpleVehicleCreate(x, y, heading, length, initial_speed)
        if self._handle == _esmini_cffi.ffi.NULL:
            raise RuntimeError("Unable to create SimpleVehicle")

    def __del__(self) -> None:
        # Need to free up the vehicle before cleanup
        _esmini_cffi.lib.SE_SimpleVehicleDelete(self._handle)

    def control_discrete(self, dt: float, throttle: int, steering: int) -> None:
        """Control the speed and steering of the vehicle using discrete input.

        The function also steps the vehicle model, updating its position according to motion state and time step.

        Parameters
        ----------
        dt
            Step time in seconds
        throttle
            * -1: break
            * 0: no-op
            * 1: accelerate
        steering
            * -1: left
            * 0: straignt
            * 1: right
        """
        _esmini_cffi.lib.SE_SimpleVehicleControlBinary(self._handle, dt, throttle, steering)

    def control_continuous(self, dt: float, throttle: float, steering: float) -> None:
        """Control the speed and steering of the vehicle using discrete input.

        The function also steps the vehicle model, updating its position according to motion state and time step.

        Parameters
        ----------
        dt
            Step time in seconds
        throttle
            longitudinal control such that -1 implies maximum brake, 0 implies no acceleration, and 1 implies maximum
            acceleration.
        steering
            steering control such that -1 implies maximum left, 0 implies straight, and 1 implies maximum
            right.
        """
        _esmini_cffi.lib.SE_SimpleVehicleControlAnalog(self._handle, dt, throttle, steering)

    def set_control_target(self, dt: float, target_speed: float, heading_to_target: float) -> None:
        """Control the speed and steering by providing reference targets.

        The function also steps the vehicle model, updating its position according to motion state and time step.
        """
        _esmini_cffi.lib.SE_SimpleVehicleControlTarget(
            self._handle,
            dt,
            target_speed,
            heading_to_target,
        )

    def set_max_speed(self, speed: float) -> None:
        _esmini_cffi.lib.SE_SimpleVehicleSetMaxSpeed(self._handle, speed)

    def set_max_acceleration(self, acceleration: float) -> None:
        _esmini_cffi.lib.SE_SimpleVehicleSetMaxAcceleration(self._handle, acceleration)

    def set_max_deceleration(self, deceleration: float) -> None:
        _esmini_cffi.lib.SE_SimpleVehicleSetMaxDeceleration(self._handle, deceleration)

    def set_engine_brake_factor(self, engine_break_factor: float) -> None:
        """Set the engine break factor, applied when no throttle is applied.

        .. note::
           Recommended range is between [0.0, 0.01]. Global default = 0.001
        """
        _esmini_cffi.lib.SE_SimpleVehicleSetEngineBrakeFactor(self._handle, engine_break_factor)

    def set_steering_scale(self, scale: float) -> None:
        """
        Set steering scale factor, which will limit the steering range as speed increases
        .. note::
           Recommended range = [0.0, 0.1]. Global default = 0.018
        """
        _esmini_cffi.lib.SE_SimpleVehicleSteeringScale(self._handle, scale)

    def set_steering_return_factor(self, factor: float) -> None:
        """
        Set steering return factor, which will make the steering wheel strive to neutral position (0 angle).

        .. note::
           Recommended range = [0.0, 10]. Global default = 4.0
        """
        _esmini_cffi.lib.SE_SimpleVehicleSteeringReturnFactor(self._handle, factor)

    def set_steering_rate(self, rate: float) -> None:
        """Set steering rate, which will affect the angular speed of which the steering wheel will turn

        .. note::
           Recommended range = [0.0, 50.0], default = 8.0
        """
        _esmini_cffi.lib.SE_SimpleVehicleSteeringRate(self._handle, rate)

    def get_state(self) -> SimpleVehicleState:
        state = SimpleVehicleState()
        _esmini_cffi.lib.SE_SimpleVehicleGetState(self._handle, state._ptr)
        return state


def save_images_to_ram(enable: bool) -> bool:
    """
    Capture rendered image to RAM for possible fetch via API, e.g. `fetch_image`.
    Set `True` before calling `init_scenario_engine` to enable fetching first frame at time = 0.

    .. note::
       Setting to `False` might improve performance on some systems.

    Returns
    -------
    bool
        `True` if successful.
    """
    return _esmini_cffi.lib.SE_SaveImagesToRAM(enable) >= 0


def save_images_to_file(num_frames: int) -> bool:
    """Save `num_frames` of rendered images to file.
    .. note:::
       Call after `init_scenario_engine`.

    Parameters
    ----------
    num_frames
        `-1` implies continuously save, `0` implies stop, and any number `> 0` saves the subsequent number of frames.

    Returns
    -------
    bool
        `True` if successful.
    """
    return _esmini_cffi.lib.SE_SaveImagesToFile(num_frames)


def fetch_image() -> Optional[Image]:
    """Fetch captured image from RAM (if successful)."""
    img = Image()
    ok = _esmini_cffi.lib.SE_FetchImage(img)
    if ok == 0:
        return img
    return None


class CameraMode(IntEnum):
    ORBIT = 0
    FIXED = 1
    RUBBER_BAND = 2
    RUBBER_BAND_ORBIT = 3
    TOP = 4
    DRIVER = 5
    CUSTOM = 6
    NUM_MODES = 7


def set_camera_mode(mode: CameraMode) -> bool:
    """Select camera mode."""
    return _esmini_cffi.lib.SE_SetCameraMode(mode) >= 0


def set_camera_object_focus(object_id: int) -> bool:
    """Set camera to focus on given object"""
    return _esmini_cffi.lib.SE_SetCameraObjectFocus(object_id)


__docformat__ = "numpy"
