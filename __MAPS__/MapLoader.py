import os
import importlib.util

class MapLoader:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /workspace
    maps_root = os.path.join(project_root, "__MAPS__")  # /workspace/__MAPS__
    def __init__(self, map_name: str):
        self.map_name = map_name
        self.map_path = self._resolve_path(map_name)
        self.MAP = self.load()

    def _resolve_path(self, map_name):
        normalized_path = map_name.replace(".", os.sep).replace("/", os.sep)
        full_path = os.path.join(self.maps_root, normalized_path + ".py")

        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Map file not found: {full_path}")
        return full_path

    def load(self):
        spec_name = self.map_name.replace("/", "_").replace(".", "_")
        spec = importlib.util.spec_from_file_location(spec_name, self.map_path)
        map_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(map_module)

        if not hasattr(map_module, "MAP"):
            raise AttributeError(f"The specified map module '{self.map_name}' does not define a variable named 'MAP'.")
        return map_module.MAP
