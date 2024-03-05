import importlib.util
import warnings
from dataclasses import dataclass
from enum import IntFlag

if importlib.util.find_spec("esmini._esmini_cffi") is None:
    warnings.warn("esmini extension module not present. Doing so now.", UserWarning, stacklevel=0)
    from esmini._build_esmini import ffibuilder

    ffibuilder.compile(verbose=True)


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


@dataclass
class ScenarioObjectState(object):
    """The state of a scenario object"""

    id: int
    """Automatically generated unique object id"""
    model_id: int
    """Id to control what 3D model to represent the vehicle - see carModelsFiles_[] in scenarioenginedll.cpp"""
    ctrl_type: int
    """0: DefaultController 1: External. Further values see Controller::Type enum"""
    timestamp: float
    """Not used yet (idea is to use it to interpolate position for increased sync bewtween simulators)"""
    x: float
    """global x coordinate of position"""
    y: float
    """global y coordinate of position"""
    z: float
    """global z coordinate of position"""
    h: float
    """heading/yaw in global coordinate system"""
    p: float
    """pitch in global coordinate system"""
    r: float
    """roll in global coordinate system"""
    road_id: int
    """road ID"""
    t: float
    """lateral position in road coordinate system"""
    lane_id: int
    """lane ID"""
    lane_offset: float
    """lateral offset from lane center"""
    s: float
    """longitudinal position in road coordinate system"""
    speed: float
    """speed"""
    center_offset_x: float
    """x coordinate of bounding box center relative object reference point (local coordinate system)"""
    center_offset_y: float
    """y coordinate of bounding box center relative object reference point (local coordinate system)"""
    center_offset_z: float
    """z coordinate of bounding box center relative object reference point (local coordinate system)"""
    width: float
    """width"""
    length: float
    """length"""
    height: float
    """height"""


@dataclass
class RoadInfo(object):
    global_pos_x: float
    global_pos_y: float
    global_pos_z: float
    local_pos_x: float
    local_pos_y: float
    local_pos_z: float
    angle: float
    road_heading: float
    road_pitch: float
    road_roll: float
    trail_heading: float
    curvature: float
    speed_limit: float
