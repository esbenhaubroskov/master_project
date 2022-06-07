import shapely
import shapely.geometry
import numpy as np
import shapely.ops

class LiDARSensor:
    def __init__(self, obstacle_map, n_rays, max_range) -> None:
        self.obstacle_map = obstacle_map
        self.n_rays = n_rays
        self.max_range = max_range
        self.rays = [] # mostly for debugging
        self.intersection_points = []
        self.intersection_bools =  []
        self.x = [None, None]
        pass

    def get_ranges(self, x, invert=False):
        """
        Get the ranges from point x to obstacles as determined by the 
        intersection of the LiDAR rays and the obstacle map.
        x : position of the LiDAR
        """
        intersections = self.get_intersection_points(x)
        ranges = []
        x = np.array(x)

        for intersection in intersections:
            intersection = np.array(intersection)
            distance = np.linalg.norm(intersection-x)
            ranges.append(distance)
        
        ranges = np.array(ranges)

        if invert:
            ranges = self.max_range - ranges
        ranges = np.absolute(np.around(ranges,10)) # Round to 10 decimals
        return ranges

    def construct_rays(self, x=[0,0]):
        """
        Construct ray-objects emitting from point x.
        x : position of the LiDAR
        """
        self.rays = []
        d_angle = 2*np.pi/(self.n_rays)
        for i in range(self.n_rays):
            angle = i * d_angle
            x_point = shapely.geometry.Point(*x)
            end = x + np.array([np.cos(angle), np.sin(angle)])*self.max_range
            end_point = shapely.geometry.Point(*end)
            ray = shapely.geometry.LineString([x,end])
            self.rays.append(ray)
    
    def get_intersection_points(self, x, get_intersection_bools=False):
        """
        Returns the closest point of the intersection between each LiDAR ray and the obstacle map.
        x : position of the LiDAR
        get_intersection_bools : (bool) Also return a list of bools describing if each ray intersects an obstacle.
        """

        if None in self.x or not np.allclose(x, self.x):
            self.update(x)
            self.x = x
            
        
        if get_intersection_bools:
            return self.intersection_points, self.intersection_bools
        else:
            return self.intersection_points
    
    def get_angles(self):
        """
        Returns the directions of all the LiDAR rays
        """
        angle_list = [i*2*np.pi/(self.n_rays) for i in range(self.n_rays)]
        return np.array(angle_list)
    
    def update(self, x):
        """
        Calculates distances and intersection points of lidar rays emitting from the point x.
        """
        self.construct_rays(x)
        x = shapely.geometry.Point(x)
        self.intersection_points = []
        self.intersection_bools = []
        for i in range(len(self.rays)):
            ray = self.rays[i]
            if ray.intersects(self.obstacle_map):
                intersection = ray.intersection(self.obstacle_map)
                _, nearest = shapely.ops.nearest_points(x, intersection)
                self.intersection_bools.append(True)
            else:
                nearest = ray.coords[1]
                self.intersection_bools.append(False)
            nearest = np.array(nearest)
            self.intersection_points.append(nearest)
        
        return self.intersection_points, self.intersection_bools


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    point1 = shapely.geometry.Point(0,0).buffer(1)
    point2 = shapely.geometry.Point(1,4).buffer(1)
    obstacle_map = shapely.geometry.MultiPolygon([point1,point2])
    x_drone = [2,2]
    sensor = LiDARSensor(obstacle_map, 20, 2.5)
    intersection_points = sensor.get_intersection_points(x_drone)
    ranges = sensor.get_ranges(x_drone)
    sensor.construct_rays(x_drone)

    # -----------------
    # plot
    # -----------------
    cmap = matplotlib.cm.tab10
    fig, ax = plt.subplots()
    for geom in obstacle_map.geoms:
        coords = geom.exterior
        patch = matplotlib.patches.Polygon(np.array(coords.xy).T, facecolor=(0.9,0.9,0.9), edgecolor=(0,0,0))
        ax.add_patch(patch)
    ax.relim()
    ax.set_aspect('equal', 'box')
    ax.autoscale_view()
    print(len(sensor.rays))

    for ray in sensor.rays:
        x,y = ray.xy
        plt.plot(x,y, '-', color=cmap(0))#(0,0.3,0.5))
    
    
    print(ranges)    
    for ip in intersection_points:
        x,y = np.array(ip)
        plt.plot(x,y,marker=(6,2), color=cmap(1), markersize=12)
    
    plt.plot(x_drone[0],x_drone[1], 'ko')
    plt.show()