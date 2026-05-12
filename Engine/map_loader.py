import os
import importlib.util

class MapLoader:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /workspace
    maps_root = os.path.join(project_root, "__MAPS__")  # /workspace/__MAPS__

    @staticmethod
    def _resolve_path(map_name: str) -> str:
        normalized_path = map_name.replace(".", os.sep).replace("/", os.sep)
        full_path = os.path.join(MapLoader.maps_root, normalized_path + ".py")

        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Map file not found: {full_path}")
        return full_path

    @staticmethod
    def load(map_name: str):
        """맵 파일을 로드하여 MAP 객체 반환"""
        map_path = MapLoader._resolve_path(map_name)
        spec_name = map_name.replace("/", "_").replace(".", "_")

        spec = importlib.util.spec_from_file_location(spec_name, map_path)
        map_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(map_module)

        if not hasattr(map_module, "MAP"):
            raise AttributeError(
                f"The specified map module '{map_name}' does not define a variable named 'MAP'."
            )
        return map_module.MAP
