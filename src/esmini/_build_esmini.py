import logging
from pathlib import Path
from typing import List

import urllib3
from cffi import FFI

_CURRENT_FILE = Path(__file__).absolute()
_CURRENT_DIR = _CURRENT_FILE.parent

_INSTALLED_ESMINI_FILES_TXT = _CURRENT_DIR / "_installed_esmini_files.txt"

_ESMINI_VERSION = "v2.37.1"

if __name__ == "__main__":
    logging.basicConfig(
        format="[{levelname:8s}] [{name}]: {message}",
        style="{",
        level=logging.INFO,
    )


def _get_esmini_platform() -> str:
    import platform

    sys_name = platform.system()
    if sys_name == "Linux":
        return "Linux"
    if sys_name == "Darwin":
        return "macOS"
    if sys_name == "Windows":
        return "Windows"
    raise RuntimeError(f"Unsupported plaform for esmini: {platform.platform()}")


def _are_esmini_files_present() -> bool:
    """Checks if esmini files that are present in the esmini demo are present."""
    # First check if we have our tracker
    log = logging.getLogger(__name__)
    log.info("checking if required esmini files are present")

    if not _INSTALLED_ESMINI_FILES_TXT.is_file():
        log.info(f"NOT PRESENT -> {_INSTALLED_ESMINI_FILES_TXT}")
        return False

    # Now, check if all the files tracked in the tracker are present
    with open(_INSTALLED_ESMINI_FILES_TXT, "r") as tracker:
        for line in tracker:
            to_check = _CURRENT_DIR / line.rstrip()
            if not to_check.is_file():
                log.info(f"NOT PRESENT -> {to_check}")
                return False

    return True


def _download_esmini_lib() -> None:
    import re
    import shutil
    import tempfile
    import zipfile

    if _are_esmini_files_present():
        return

    log = logging.getLogger(__name__)
    dynamic_lib_re = re.compile(r"^esmini-demo/bin/\S*?\.(?:so|dylib|dll)$")
    headers_re = re.compile(r"^esmini-demo/EnvironmentSimulator/Libraries/esminiLib/esminiLib\.hpp$")
    resources_re = re.compile(r"^esmini-demo/resources.*?\.osgb")

    http = urllib3.PoolManager()
    esmini_latest_url = (
        f"https://github.com/esmini/esmini/releases/download/{_ESMINI_VERSION}/esmini-demo_{_get_esmini_platform()}.zip"
    )
    log.info(f"downloading esmini release from {esmini_latest_url}")

    # Create a temporary directory for the files
    with tempfile.TemporaryDirectory() as temp_dir_name:
        log.info(f"downloading artifacts to {temp_dir_name}")
        # temp_dir_name = _PARENT_DIR / "../temp"
        temp_dir = Path(temp_dir_name).absolute()
        temp_dir.mkdir(exist_ok=True, parents=True)
        assert temp_dir.is_dir()

        dl_zip_file = temp_dir / "esmini-demo.zip"
        with http.request("GET", esmini_latest_url, preload_content=False) as req, open(dl_zip_file, "wb") as out_file:
            shutil.copyfileobj(req, out_file)

        log.info(f"downloaded file to {dl_zip_file}")
        names: List[str] = []

        # Now we need to extract just the shared libraries
        with zipfile.ZipFile(dl_zip_file, "r") as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                name = member.filename
                if (dynamic_lib_re.fullmatch(name)) is not None or (headers_re.fullmatch(name) is not None):
                    member.filename = Path(name).name
                    log.info(f"extracting {member.filename}")
                    names.append(member.filename)
                    archive.extract(member, path=_CURRENT_DIR)
                elif resources_re.fullmatch(name) is not None:
                    # Extract the members of the resources directory, but remove the "esmini-demo" part.
                    fullname = Path(name)
                    parts = fullname.parts
                    new_name = Path("/".join(parts[1:]))
                    member.filename = str(new_name)
                    log.info(f"extracting {member.filename}")
                    names.append(member.filename)
                    archive.extract(member, path=_CURRENT_DIR)

        with open(_INSTALLED_ESMINI_FILES_TXT, "w") as tracker:
            tracker.write("\n".join(names))

    log.info(f"extracted esmini libraries to {str(_CURRENT_DIR)}")


_download_esmini_lib()

ffibuilder = FFI()

# cdef() expects a single string declaring the C types, functions and
# globals needed to use the shared object. It must be in valid C syntax.
ffibuilder.cdef(
    """

typedef struct
{
    int   id;
    int   model_id;
    int   ctrl_type;
    float timestamp;
    float x;
    float y;
    float z;
    float h;
    float p;
    float r;
    int   roadId;
    float t;
    int   laneId;
    float laneOffset;
    float s;
    float speed;
    float centerOffsetX;
    float centerOffsetY;
    float centerOffsetZ;
    float width;
    float length;
    float height;
    ...;
} SE_ScenarioObjectState;

typedef struct
{
    float global_pos_x;
    float global_pos_y;
    float global_pos_z;
    float local_pos_x;
    float local_pos_y;
    float local_pos_z;
    float angle;
    float road_heading;
    float road_pitch;
    float road_roll;
    float trail_heading;
    float curvature;
    float speed_limit;
    ...;
} SE_RoadInfo;

typedef struct
{
    float x;           // Route point in the global coordinate system
    float y;           // Route point in the global coordinate system
    float z;           // Route point in the global coordinate system
    int   roadId;      // Route point, road ID
    int   junctionId;  // Route point, junction ID (-1 if not in a junction)
    int   laneId;      // Route point, lane ID
    int   osiLaneId;   // Route point, osi lane ID
    float laneOffset;  // Route point, lane offset (lateral distance from lane center)
    float s;           // Route point, s (longitudinal distance along reference line)
    float t;           // Route point, t (lateral distance from reference line)
} SE_RouteInfo;

typedef struct
{
    int far_left_lb_id;
    int left_lb_id;
    int right_lb_id;
    int far_right_lb_id;
} SE_LaneBoundaryId;

typedef struct
{
    float ds;             // delta s (longitudinal distance)
    float dt;             // delta t (lateral distance)
    int   dLaneId;        // delta laneId (increasing left and decreasing to the right)
    float dx;             // delta x (world coordinate system)
    float dy;             // delta y (world coordinate system)
    bool  oppositeLanes;  // true if the two position objects are in opposite sides of reference lane
} SE_PositionDiff;

typedef struct
{
    float x;
    float y;
    float z;
    float h;
    float p;
    float speed;
    float wheel_rotation;
    float wheel_angle;
} SE_SimpleVehicleState;

typedef struct
{
    int         id;           // just an unique identifier of the sign
    float       x;            // global x coordinate of sign position
    float       y;            // global y coordinate of sign position
    float       z;            // global z coordinate of sign position
    float       z_offset;     // z offset from road level
    float       h;            // global heading of sign orientation
    int         roadId;       // road id of sign road position
    float       s;            // longitudinal position along road
    float       t;            // lateral position from road reference line
    const char *name;         // sign name, typically used for 3D model filename
    int         orientation;  // 1=facing traffic in road direction, -1=facing traffic opposite road direction
    float       length;       // length as specified in OpenDRIVE
    float       height;       // height as specified in OpenDRIVE
    float       width;        // width as specified in OpenDRIVE
} SE_RoadSign;

typedef struct
{
    int fromLane;
    int toLane;
} SE_RoadObjValidity;

typedef struct
{
    int            width;
    int            height;
    int            pixelSize;    // 3 for RGB/BGR
    int            pixelFormat;  // 0x1907=RGB (GL_RGB), 0x80E0=BGR (GL_BGR)
    unsigned char *data;
} SE_Image;  // Should be synked with CommonMini/OffScreenImage

typedef struct
{
    float x_;  // Center offset in x direction.
    float y_;  // Center offset in y direction.
    float z_;  // Center offset in z direction.
} SE_Center;

typedef struct
{
    float width_;   // Width of the entity's bounding box. Unit: m; Range: [0..inf[.
    float length_;  // Length of the entity's bounding box. Unit: m; Range: [0..inf[.
    float height_;  // Height of the entity's bounding box. Unit: m; Range: [0..inf[.
} SE_Dimensions;

typedef struct
{
    SE_Center     center_;      // Represents the geometrical center of the bounding box
    SE_Dimensions dimensions_;  // Width, length and height of the bounding box.
} SE_OSCBoundingBox;

typedef enum
{
    SE_Z_SET     = 1,  // 0001
    SE_Z_DEFAULT = 1,  // 0001
    SE_Z_ABS     = 3,  // 0011
    SE_Z_REL     = 7,  // 0111
    SE_H_SET     = SE_Z_SET << 4,
    SE_H_ABS     = SE_Z_ABS << 4,
    SE_H_REL     = SE_Z_REL << 4,
    SE_H_DEFAULT = SE_Z_DEFAULT << 4,
    SE_P_SET     = SE_Z_SET << 8,
    SE_P_ABS     = SE_Z_ABS << 8,
    SE_P_REL     = SE_Z_REL << 8,
    SE_P_DEFAULT = SE_Z_DEFAULT << 8,
    SE_R_SET     = SE_Z_SET << 12,
    SE_R_DEFAULT = SE_Z_DEFAULT << 12,
    SE_R_ABS     = SE_Z_ABS << 12,
    SE_R_REL     = SE_Z_REL << 12,
} SE_PositionMode;

typedef enum
{
    SE_SET    = 1,  // Used by explicit set functions
    SE_UPDATE = 2   // Used by controllers updating the position
} SE_PositionModeType;

typedef struct
{
    const char *name;   // Name of the parameter as defined in the OpenSCENARIO file
    void       *value;  // Pointer to value which can be an integer, double, bool or string (const char*) as defined in the OpenSCENARIO file
} SE_Parameter;

typedef struct
{
    const char *name;   // Name of the variable as defined in the OpenSCENARIO file
    void       *value;  // Pointer to value which can be an integer, double, bool or string (const char*) as defined in the OpenSCENARIO file
} SE_Variable;

int SE_Init(const char *oscFilename, int disable_ctrls, int use_viewer, int threads, int record);
int SE_InitWithString(const char *oscAsXMLString, int disable_ctrls, int use_viewer, int threads, int record);
int SE_InitWithArgs(int argc, const char *argv[]);

int SE_AddPath(const char *path);
void SE_ClearPaths();
void SE_SetLogFilePath(const char *logFilePath);
void SE_SetDatFilePath(const char *datFilePath);
unsigned int SE_GetSeed();
void SE_SetSeed(unsigned int seed);
void SE_SetWindowPosAndSize(int x, int y, int w, int h);

typedef void (*SE_ParameterDeclarationCallback)(void *user_data);

extern "Python" void esmini_parameter_declaration_callback(void *);

void SE_RegisterParameterDeclarationCallback(SE_ParameterDeclarationCallback fnPtr, void *user_data);

int SE_SetOSITolerances(double maxLongitudinalDistance, double maxLateralDeviation);
int SE_SetParameterDistribution(const char *filename);
void SE_ResetParameterDistribution();
int SE_GetNumberOfPermutations();
int SE_SelectPermutation(int index);
int SE_GetPermutationIndex();

int SE_StepDT(float dt);
int SE_Step();
void SE_Close();
void SE_LogToConsole(bool mode);
void SE_CollisionDetection(bool mode);
float SE_GetSimulationTime();  // Get simulation time in seconds
double SE_GetSimulationTimeDouble();
float SE_GetSimTimeStep();
int SE_GetQuitFlag();
int SE_GetPauseFlag();
const char *SE_GetODRFilename();
const char *SE_GetSceneGraphFilename();
int SE_GetNumberOfParameters();
const char *SE_GetParameterName(int index, int *type);
int SE_GetNumberOfProperties(int index);
const char *SE_GetObjectPropertyName(int index, int propertyIndex);
const char *SE_GetObjectPropertyValue(int index, const char *objectPropertyName);

int SE_SetParameter(SE_Parameter parameter);
int SE_GetParameter(SE_Parameter *parameter);
int SE_GetParameterInt(const char *parameterName, int *value);
int SE_GetParameterDouble(const char *parameterName, double *value);
int SE_GetParameterBool(const char *parameterName, bool *value);
int SE_GetParameterString(const char *parameterName, const char **value);
int SE_SetParameterInt(const char *parameterName, int value);
int SE_SetParameterDouble(const char *parameterName, double value);
int SE_SetParameterBool(const char *parameterName, bool value);
int SE_SetParameterString(const char *parameterName, const char *value);

int SE_SetVariable(SE_Variable variable);
int SE_GetVariable(SE_Variable *variable);
int SE_GetVariableInt(const char *variableName, int *value);
int SE_GetVariableDouble(const char *variableName, double *value);
int SE_GetVariableBool(const char *variableName, bool *value);
int SE_GetVariableString(const char *variableName, const char **value);
int SE_SetVariableInt(const char *variableName, int value);
int SE_SetVariableDouble(const char *variableName, double value);
int SE_SetVariableBool(const char *variableName, bool value);
int SE_SetVariableString(const char *variableName, const char *value);

void *SE_GetODRManager();
    
void SE_SetObjectPositionMode(int object_id, SE_PositionModeType type, int mode);
void SE_SetObjectPositionModeDefault(int object_id, SE_PositionModeType type);

int SE_AddObject(const char *object_name, int object_type, int object_category, int object_role, int model_id);
int SE_AddObjectWithBoundingBox(const char       *object_name,
                                               int               object_type,
                                               int               object_category,
                                               int               object_role,
                                               int               model_id,
                                               SE_OSCBoundingBox bounding_box,
                                               int               scale_mode);

int SE_DeleteObject(int object_id);
int SE_ReportObjectPos(int object_id, float timestamp, float x, float y, float z, float h, float p, float r);
int SE_ReportObjectPosMode(int object_id, float timestamp, float x, float y, float z, float h, float p, float r, int mode);
int SE_ReportObjectPosXYH(int object_id, float timestamp, float x, float y, float h);
int SE_ReportObjectRoadPos(int object_id, float timestamp, int roadId, int laneId, float laneOffset, float s);
int SE_ReportObjectSpeed(int object_id, float speed);
int SE_ReportObjectLateralPosition(int object_id, float t);
int SE_ReportObjectLateralLanePosition(int object_id, int laneId, float laneOffset);
int SE_ReportObjectVel(int object_id, float timestamp, float x_vel, float y_vel, float z_vel);
int SE_ReportObjectAngularVel(int object_id, float timestamp, float h_rate, float p_rate, float r_rate);
int SE_ReportObjectAcc(int object_id, float timestamp, float x_acc, float y_acc, float z_acc);
int SE_ReportObjectAngularAcc(int object_id, float timestamp, float h_acc, float p_acc, float r_acc);
int SE_ReportObjectWheelStatus(int object_id, float rotation, float angle);
int SE_SetSnapLaneTypes(int object_id, int laneTypes);
int SE_SetLockOnLane(int object_id, bool mode);
int SE_GetNumberOfObjects();
int SE_GetId(int index);
int SE_GetIdByName(const char *name);
int SE_GetObjectState(int object_id, SE_ScenarioObjectState *state);
const char *SE_GetObjectTypeName(int object_id);
const char *SE_GetObjectName(int object_id);
const char *SE_GetObjectModelFileName(int object_id);
int SE_ObjectHasGhost(int object_id);
int SE_GetObjectGhostState(int object_id, SE_ScenarioObjectState *state);
int SE_GetObjectNumberOfCollisions(int object_id);
int SE_GetObjectCollision(int object_id, int index);
int SE_GetSpeedUnit();
int SE_GetRoadInfoAtDistance(int          object_id,
                                            float        lookahead_distance,
                                            SE_RoadInfo *data,
                                            int          lookAheadMode,
                                            bool         inRoadDrivingDirection);

int SE_GetRoadInfoAlongGhostTrail(int object_id, float lookahead_distance, SE_RoadInfo *data, float *speed_ghost);
int SE_GetRoadInfoGhostTrailTime(int object_id, float time, SE_RoadInfo *data, float *speed_ghost);
int SE_GetDistanceToObject(int object_a_id, int object_b_id, bool free_space, SE_PositionDiff *pos_diff);
int SE_AddObjectSensor(int object_id, float x, float y, float z, float h, float rangeNear, float rangeFar, float fovH, int maxObj);
int SE_GetNumberOfObjectSensors();
int SE_ViewSensorData(int object_id);
int SE_FetchSensorObjectList(int sensor_id, int *list);

typedef void (*SE_ObjectCallback)(SE_ScenarioObjectState *state, void *user_data);

extern "Python" void esmini_object_callback(SE_ScenarioObjectState *, void *);

void SE_RegisterObjectCallback(int object_id, SE_ObjectCallback fnPtr, void *user_data);


// TODO: Register trampolines for function pointer and void pointer
void SE_RegisterConditionCallback(void (*fnPtr)(const char *name, double timestamp));

void SE_LogMessage(const char *message);
void SE_ViewerShowFeature(int featureType, bool enable);

void *SE_SimpleVehicleCreate(float x, float y, float h, float length, float speed);
void SE_SimpleVehicleDelete(void *handleSimpleVehicle);
void SE_SimpleVehicleControlBinary(void  *handleSimpleVehicle,
                                                  double dt,
                                                  int    throttle,
                                                  int    steering);  // throttle and steering [-1, 0 or 1]
void SE_SimpleVehicleControlAnalog(void  *handleSimpleVehicle,
                                                  double dt,
                                                  double throttle,
                                                  double steering);
void SE_SimpleVehicleControlTarget(void *handleSimpleVehicle, double dt, double target_speed, double heading_to_target);
void SE_SimpleVehicleSetMaxSpeed(void *handleSimpleVehicle, float speed);
void SE_SimpleVehicleSetMaxAcceleration(void *handleSimpleVehicle, float maxAcceleration);
void SE_SimpleVehicleSetMaxDeceleration(void *handleSimpleVehicle, float maxDeceleration);
void SE_SimpleVehicleSetEngineBrakeFactor(void *handleSimpleVehicle, float engineBrakeFactor);
void SE_SimpleVehicleSteeringScale(void *handleSimpleVehicle, float steeringScale);
void SE_SimpleVehicleSteeringReturnFactor(void *handleSimpleVehicle, float steeringReturnFactor);
void SE_SimpleVehicleSteeringRate(void *handleSimpleVehicle, float steeringRate);
void SE_SimpleVehicleGetState(void *handleSimpleVehicle, SE_SimpleVehicleState *state);

int SE_SaveImagesToRAM(bool state);
int SE_SaveImagesToFile(int nrOfFrames);
int SE_FetchImage(SE_Image *image);
void SE_RegisterImageCallback(void (*fnPtr)(SE_Image *, void *), void *user_data);
int SE_WritePPMImage(const char *filename, int width, int height, const unsigned char *data, int pixelSize, int pixelFormat, bool upsidedown);
int SE_WriteTGAImage(const char *filename, int width, int height, const unsigned char *data, int pixelSize, int pixelFormat, bool upsidedown);
int SE_AddCustomCamera(double x, double y, double z, double h, double p);
int SE_AddCustomFixedCamera(double x, double y, double z, double h, double p);
int SE_AddCustomAimingCamera(double x, double y, double z);
int SE_AddCustomFixedAimingCamera(double x, double y, double z);
int SE_AddCustomFixedTopCamera(double x, double y, double z, double rot);
int SE_SetCameraMode(int mode);
int SE_SetCameraObjectFocus(int object_id);
int SE_GetNumberOfRoutePoints(int object_id);
int SE_GetRoutePoint(int object_id, int route_index, SE_RouteInfo *routeinfo);

"""
)

# set_source() gives the name of the python extension module to
# produce, and some C source code as a string.  This C code needs
# to make the declarated functions, types and globals available,
# so it is often just the "#include".
ffibuilder.set_source(
    "esmini._esmini_cffi",
    """
    #include "esminiLib.hpp"
""",
    libraries=["esminiLib"],
    include_dirs=[str(_CURRENT_DIR)],
    library_dirs=[str(_CURRENT_DIR)],
    source_extension=".cpp",
    extra_compile_args=[
        "-std=c++17",
        "-pthread",
        "-fPIC",
        "-Wl,-strip-all",
    ],
    runtime_library_dirs=[
        str(_CURRENT_DIR),
    ],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
