import random
import sys
import math
from turtle import shape
from pip import main
import shapely as sly
import numpy as np
import shapely.geometry
import shapely.ops
valid_obstacle_types = ["circles", "polygons", "any", "none"]

class TrialCounter:
    def __init__(self, max_count = 100):
        self.count = 0
        self.max_count = max_count
    
    def inc(self):
        self.count += 1
        if self.count > self.max_count:
            raise Exception("Could not generate environment")
    
    def reset(self):
        self.count = 0

class EnvironmentGenerator:
    def __init__(self, n_drones, n_obstacles, obstacle_type, max_distance, seed=None, xlim=[-5,5], ylim=[-5,5], events=[], n_goals=None, minimum_travel_distance=None, **kwargs) -> None:
        
        assert obstacle_type in valid_obstacle_types
        if seed == None:
            seed = random.randrange(sys.maxsize)
        if not seed == None:
            #random.seed(seed)
            self.random = random.Random(seed)
        self.seed = seed
        self.n_drones = n_drones
        self.n_obstacles = n_obstacles
        self.obstacle_type = obstacle_type
        self.xlim = xlim
        self.ylim = ylim
        self.max_distance = max_distance
        self.clearance = 0.75
        self.obstacle_max_size = 2
        self.obstacle_min_size = 0.5

        self.obstacles = []
        self.obstacles_radius = []
        self.obstacle_polygons = []
        self.starting_positions = []
        self.obstacle_multi_polygon = []
        self.goals = []
        self.events = events
        self.n_goals = n_goals
        self.minimum_travel_distance = minimum_travel_distance
        self.goal_radius = 1
        self.goal_area = None

        for key, value in kwargs.items():
            setattr(self, key, value)
        
        if self.minimum_travel_distance == None:
            self.minimum_travel_distance = self.clearance + 2*self.max_distance
        if self.n_goals == None:
            self.n_goals = self.n_drones

    def _generate_position(self, center):
        angle = 2 * math.pi * self.random.random() 
        radius = self.max_distance * math.sqrt(self.random.random())
        x = radius * math.cos(angle) + center[0]
        y = radius * math.sin(angle) + center[1]
        return [x,y]

    
    def generate_goal_positions(self):
        try_cnt = TrialCounter()
        goal_area = [self.random.uniform(*self.xlim),self.random.uniform(*self.ylim)]
        
        # Enforce clearance between goal positions and starting positions 
        while min([np.linalg.norm(np.array(goal_area) - np.array(start_pos)) for start_pos in self.starting_positions]) < self.minimum_travel_distance:
            goal_area = [self.random.uniform(*self.xlim),self.random.uniform(*self.ylim)]
            try_cnt.inc()
        try_cnt.reset()
        
        # Generate the goal positions 
        for _ in range(self.n_goals):
            pos = self._generate_position(goal_area)
            
            # Enforce clearance between goals
            if self.goals:
                while min([np.linalg.norm(np.array(pos)-np.array(goal)) for goal in self.goals]) < self.clearance:
                    pos = self._generate_position(goal_area)
                    try_cnt.inc()
                try_cnt.reset()
            self.goals.append(pos)
        return self.goals

    def generate_start_positions(self):
        try_cnt = TrialCounter()
        start_center = [self.random.uniform(*self.xlim),self.random.uniform(*self.ylim)]

        # Generate start posistion within a max distance between each other and with a minumum distance to each other 
        for _ in range(self.n_drones):
            pos = self._generate_position(start_center)
            
            if self.starting_positions: # Check if list is empty
                while min([np.linalg.norm(np.array(pos)-np.array(start_pos)) for start_pos in self.starting_positions]) < self.clearance:
                    pos = self._generate_position(start_center)
                    try_cnt.inc()
            try_cnt.reset()
            self.starting_positions.append(pos)
       
        return self.starting_positions

    def generate_circles(self, allow_overlap=False):
        try_cnt = TrialCounter()

        for _ in range(self.n_obstacles):
            point = []
            radius = None
            in_startpositions,  in_goals,  in_obstacle = True, True, True

            while in_startpositions or in_goals or in_obstacle:
                point = [self.random.uniform(*self.xlim),self.random.uniform(*self.ylim)]
                radius = self.random.uniform(self.obstacle_min_size, self.obstacle_max_size)

                # Check if point generated is in startposition, goal and other obstacles
                in_startpositions = min([np.linalg.norm(np.array(point)-np.array(start_pos)) for start_pos in self.starting_positions]) < radius + self.clearance
                in_goals = min([np.linalg.norm(np.array(point)-np.array(goal)) for goal in self.goals]) < radius + self.clearance

                in_obstacle = False
                if self.obstacles:
                    for ob, r in zip(self.obstacles, self.obstacles_radius):
                        new_point = shapely.geometry.Point(*point).buffer(radius)
                        obstacle_shape = shapely.geometry.Point(*ob).buffer(r)
                        if new_point.contains(obstacle_shape):
                            in_obstacle = True
                        elif obstacle_shape.contains(new_point):
                            in_obstacle = True
                        elif new_point.intersects(obstacle_shape):
                            if not allow_overlap:
                                in_obstacle = True 
                
                try_cnt.inc()
            try_cnt.reset()
            self.obstacles.append(point)
            self.obstacles_radius.append(radius)


    def generate_polygon(self):  
              
        pass

    def generate_obstacles(self):
        if self.obstacle_type == 'circles':
            self.generate_circles()

        elif self.obstacle_type == 'polygons':
            pass

        elif self.obstacle_type == 'any':
            pass
        
        else:
            print("No obstacles are generated")


    def generate_environment(self, reset=True):
        if reset:
            self.obstacles = []
            self.obstacles_radius = []
            self.obstacle_polygons = []
            self.obstacle_multi_polygon = []
            if self.n_drones > 0:
                self.starting_positions = []
                self.generate_start_positions()
            if self.n_goals > 0:
                self.goals = []
                self.generate_goal_positions()
        self.generate_obstacles()
        self.as_multipolygon()
        return self.as_dict()
    
    def clear_obstacles(self, circles=True, polygons=True, change_obstacle_map=False):
        if circles:
            self.obstacles = []
            self.obstacles_radius = []
        if polygons:
            self.obstacle_polygons = []
        self.obstacle_multi_polygon = []
        if change_obstacle_map:
            self.as_multipolygon


    def as_dict(self):    
        environment = dict(
            starting_positions = self.starting_positions,
            goals = self.goals,
            obstacles = self.obstacles,
            obstacles_radius = self.obstacles_radius,
            obstacle_polygons = self.obstacle_polygons,
            obstacle_map = self.obstacle_multi_polygon,
            goal_area = self.goal_area,
        )
        return environment
    
    def as_multipolygon(self):
        """
        Returns all obstacles as a single MultiPolygon
        """
        polygons = self.obstacle_polygons
        #polygons = self.obstacle_multi_polygon
        for p, r in zip(self.obstacles, self.obstacles_radius):
            polygons.append(shapely.geometry.Point(*p).buffer(r,resolution=4))
        union = shapely.ops.unary_union(polygons)
        self.obstacle_multi_polygon = union
        return union

    def goal_area_as_multipolygon(self):
        goal_polygons = []
        for goal in self.goals:
            r = self.goal_radius
            goal_polygons.append(shapely.geometry.Point(*goal).buffer(r, resolution=4))

        goal_union = shapely.ops.unary_union(goal_polygons)
        self.goal_area = goal_union
        return goal_union

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    gen = EnvironmentGenerator(2, 2, "any", 4)
    print(gen.generate_start_positions())
    print(gen.generate_goal_positions())
    print(gen.as_dict())

    gen = EnvironmentGenerator(2,2,"circles",4)
    env = gen.generate_environment()
    print(env)
    fig, ax  = plt.subplots()
    
    for ob, r in zip(gen.obstacles, gen.obstacles_radius):
        circle = plt.Circle((ob[0], ob[1]), r, color='k', fill=False)
        ax.add_artist(circle)
    ax.plot(np.array(gen.starting_positions)[:,0], np.array(gen.starting_positions)[:,1], 'or')
    ax.plot(np.array(gen.goals)[:,0], np.array(gen.goals)[:,1], 'og')
    ax.set_xlim([-10,10])
    ax.set_ylim([-10,10])
    plt.show()
    