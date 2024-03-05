"""Interface for the global `ScenarioManager` in `esmini`."""

from typing import List, Set, Union

from esmini._esmini_cffi import ffi, lib


class _Singleton(type):
    _instances = {}  # type: ignore

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ScenarioManager(metaclass=_Singleton):
    """The esmini scenario manager.

    This is a global object, and thus requires the singleton pattern.
    """

    def __init__(self) -> None:
        self._inited = False

        self._search_paths: Set[bytes] = set()

    @property
    def inited(self) -> bool:
        # TODO: add logic to check if simulation has ended
        return self._inited

    def add_path(self, path: Union[str, bytes]) -> None:
        """Add a search path for OpenDRIVE and 3D model files.

        Needs to be called before `ScenarioManager.create`.
        """
        if self._inited:
            raise RuntimeError("Cannot add search path while scenario is running")
        if isinstance(path, str):
            path = path.encode()
        self._search_paths.add(path)
        lib.SE_AddPath(path)
