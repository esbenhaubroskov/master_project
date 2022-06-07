import sys 
from pathlib import Path
from typing import Union

sys.path.append(r"C:\Users\esben\Documents\master-project\RMP\rmp-project")
from environment_generator import EnvironmentGenerator
import shapely
import shapely.geometry
import shapely.ops
import json
from pathlib import Path
import numpy as np

class EnvironmentDictLoader:

    def __init__(self):
        pass

    def load_environment(self, path):
        file = open(path, 'r')
        env = json.load(file)
        env["obstacle_map"] = shapely.geometry.shape(env["obstacle_map"])
        env["obstacle_polygons"] = [shapely.geometry.shape(poly) for poly in env["obstacle_polygons"]]
        
        return env

    def save_environment(self, env, path):
        file = open(path, 'w')
        buffer_dict = {}
        for key in env:
            buffer_dict[key] = env[key]
        buffer_dict["obstacle_map"] = shapely.geometry.mapping(buffer_dict["obstacle_map"])
        buffer_dict["obstacle_polygons"] = [shapely.geometry.mapping(poly) for poly in env["obstacle_polygons"]]

        json.dump(buffer_dict, file, indent="  ")
        
        file.close()


def _format_formation_links(links):
    return np.array(links)
    
def _format_polygons(polygons):
    polygons = [shapely.geometry.Polygon(np.array(poly)) for poly in polygons]
    return polygons

def _format_obstacles(obstacle_centers):
    #obstacles = [shapely.geometry.Point(obs) for obs in obstacle_centers]
    obstacles = obstacle_centers
    return obstacles

class EnvironmentTemplate:

    formatters = dict(
        obstacle_polygons = _format_polygons,
        formation_links = _format_formation_links,
        obstacles = _format_obstacles
    )

    def __init__(self, path: Union[Path, str], base_config: dict):
        self.kwargs = dict()
        self.formation_links = None
        self.goal_links = None
        self.config = base_config
        self.template_dict = None
        self.load_template(path)

    def _load_config_to_kwargs(self):
        self.kwargs["n_drones"] = self.config.uav_set["num_uavs"]
        for key, value in self.config.env_params.items():
            self.kwargs[key] = value

    def load_template(self, path):
        env_file = open(path)
        self.template_dict = json.load(env_file)
        env_file.close()
        self._load_config_to_kwargs()
        
        for key, value in self.template_dict["kwargs"].items():
            # Call necessary formatting functions
            if key in self.__class__.formatters.keys():
                value = self.__class__.formatters[key](value)
            self.kwargs[key] = value

        if "formation_links" in self.template_dict.keys():
            self.formation_links = _format_formation_links(self.template_dict["formation_links"])
        if "goal_links" in self.template_dict.keys():
            self.goal_links = self.template_dict["goal_links"]



if __name__=="__main__":
    gen = EnvironmentGenerator(2, 2, "circles", 2.5)
    env = gen.generate_evironment()

    test_circle_1 = shapely.geometry.Point([0,0]).buffer(1)
    test_circle_2 = shapely.geometry.Point([3,0]).buffer(1)
    test_multipolygon = shapely.ops.unary_union([test_circle_1, test_circle_2])
    env_dict = dict(starting_points = [2,2], obstacle_map=test_multipolygon)

    loader = EnvironmentDictLoader()
    dump_path = Path("./dump.json")
    
    loader.save_environment(env, dump_path)
    loaded_env = loader.load_environment(dump_path)
    print(loaded_env)