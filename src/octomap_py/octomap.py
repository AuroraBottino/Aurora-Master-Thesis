from octomap_py.config import octomap_aurora
import numpy as np 

class Octomap:
    def __init__(self, starting_position, radius, resolution, min_points_per_voxel):

        if not isinstance(starting_position, np.ndarray):
            starting_position = np.asarray(starting_position)

        assert starting_position.shape[0] == 3, f"Expected numpy array of size 3, got size {starting_position.shape[0]}"
        
        self.octomap = octomap_aurora.Octomap()
        self.octomap.init(starting_position, radius, resolution, min_points_per_voxel)

        self.updated = False

    def add_point_cloud(self, points, cameras):

        if not isinstance(points, np.ndarray):
            points = np.asarray(points)

        if not isinstance(cameras, np.ndarray):
            cameras = np.asarray(cameras)

        assert points.shape[1] == 3, f"Expected numpy array of size Nx3, got size Nx{points.shape[0]}"
        assert cameras.shape[1] == 3, f"Expected numpy array of size Nx3, got size Nx{cameras.shape[0]}"
        assert points.shape[0] == cameras.shape[0], f"'points' and 'cameras' expected same size"

        for point, camera in zip(points, cameras):
            self.octomap.add_point(point, camera)
        self.octomap.update_voxels()

    def get_all_centroids(self):
        voxels = self.octomap.get_voxels_indices()
        centroids = np.array([self.octomap.get_centroid(voxel) for voxel in voxels])
        return centroids
    
    def get_all_normals(self, indices=False):
        voxels = self.octomap.get_voxels_indices()
        normals = np.array([self.octomap.get_normal(voxel) for voxel in voxels])
        return normals