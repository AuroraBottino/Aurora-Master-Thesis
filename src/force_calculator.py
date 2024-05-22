#!/usr/bin/env python

import rospy
import numpy as np
from cola2_msgs.msg import DVL
import copy

class ForceCalculator:
    """
    This node implements the attractive and repulsive forces acting on the AUV
    """

    def __init__(self):
        ## \brief Variable to store the alpha parameter, which is used to scale the direction of the repulsive force. If alpha = 0, the repulsive force is computed only considering only the direction of motion. If alpha = 1, the repulsive force is computed only considering the direction of the obstacles. By changing the value of alpha, the repulsive force can be computed considering both the direction of motion and the direction of the obstacles.
        self.alpha = 0.75
        self.average_repulsive_forces = None
        ## \brief Variable to store the minimum distance between the AUV and the obstacles used to compute the repulsive force
        self.min_distance = 0.25
        ## \brief Variable to store the repulsive gain, which is used to scale the repulsive force
        self.repulsive_gain = 0.5   
        self.current_velocity = np.zeros(3)
        ## \brief Subscriber to the DVL topic, used to get the current velocity
        self.dvl_subscriber = rospy.Subscriber('/girona500/navigator/dvl', DVL, self.dvl_callback)

        self.closest_distances_to_obstacles = []

    
    def dvl_callback(self, msg): 
        """
        Callback function for the DVL topic, used to get the current velocity
        Args:
            msg: The message containing the DVL data
        """
        velocity_x = msg.velocity.x
        velocity_y = msg.velocity.y
        velocity_z = msg.velocity.z
        self.current_velocity = np.array([velocity_x, velocity_y, velocity_z])

    def attractive_force(self): 
        """
        Function to calculate the attractive force. The attractive force is the current velocity of the AUV
        """
        attractive_force = self.current_velocity
        return attractive_force

        
    def repulsive_force(self, current_velocity, obstacle_centroids, obstacle_normals, points_on_auv, min_distance, alpha, repulsive_gain, beta, max_obstacles=None, risk_factor_all_points=None):
        """
        Function to calculate the repulsive force
        """

        if np.linalg.norm(current_velocity) != 1.0:
            current_velocity = current_velocity / (np.linalg.norm(current_velocity) + 1e-6)

        # normals_norms = np.linalg.norm(obstacle_normals, axis=1)
        normals_norms = np.linalg.norm(obstacle_normals, axis=1)
        normals_norms = normals_norms[:, np.newaxis]  #to get shape (167,1)

        if np.any(normals_norms > 1.0):
            obstacle_normals = obstacle_normals / (normals_norms + 1e-6)

        directions = points_on_auv[:, None] - obstacle_centroids   # num_auv_points x num_obstacles x 3
        distances = np.linalg.norm(directions, axis=2)             # num_auv_points x num_obstacles
        minimum_distances = np.min(distances, axis=0)   # num_obstacles x 1
        sorted_distances = np.sort(minimum_distances)[:10]
        mean_sorted_distances = np.mean(sorted_distances)
        self.closest_distances_to_obstacles.append(mean_sorted_distances)
                       

        dot_products = np.dot(obstacle_normals, current_velocity)  # num_obstacles x 1 

        # calculate repulsive forces and directions 
        repulsive_forces_distance = 1.0 - distances / min_distance # num_auv_points x num_obstacles
        repulsive_forces_angle = (1.0 - dot_products) / 2.0        # num_obstacles x 1 
        repulsive_forces_combined = (repulsive_forces_distance + repulsive_forces_angle) / 2.0  # num_auv_points x num_obstacles
        repulsive_direction_vectors = alpha * obstacle_normals + (1 - alpha) * current_velocity # num_obstacles x 3 

        # check if there are points within the minimum distance 
        if np.sum(distances < min_distance) == 0:
            return np.zeros(3)

        # get the forces and directions of the close enough obstacles 
        repulsive_forces_filtered = copy.deepcopy(repulsive_forces_combined)            # num_auv_points x num_obstacles
        repulsive_forces_filtered[distances >= min_distance] = 0.0                      # num_auv_points x num_obstacles

        if risk_factor_all_points is not None:
            repulsive_forces_filtered = beta*repulsive_forces_filtered +  (1-beta)*risk_factor_all_points.T 

        closest_auv_point_idxs = np.argmax(repulsive_forces_filtered,axis=0)              # num_obstacles x 1
        closest_obstacles = distances[closest_auv_point_idxs,np.arange(len(closest_auv_point_idxs))] < min_distance  # num_obstacles x 1
        closest_distances = distances[closest_auv_point_idxs,np.arange(len(closest_auv_point_idxs))][closest_obstacles] # num_closest_obstacles x 1
        repulsive_forces_final = repulsive_forces_filtered[closest_auv_point_idxs,np.arange(len(closest_auv_point_idxs))][closest_obstacles] # num_closest_obstacles x 1
        repulsive_vectors_final = repulsive_direction_vectors[closest_obstacles,:]      # num_closest_obstacles x 3
    
        # use only a set number of points 
        num_repulsive_points = repulsive_forces_final.shape[0]  # scalar 
        if max_obstacles is not None and num_repulsive_points > max_obstacles: 
            idxs_to_remove = np.argsort(closest_distances)[max_obstacles:]
            repulsive_forces_final[idxs_to_remove] = 0.0
            num_repulsive_points = max_obstacles
    
        # calculate the final repulsive force 
        if num_repulsive_points == 0:
            return np.zeros(3)
        weighted_repulsive_vector = (1.0 / (np.sum(repulsive_forces_final)+1e-6)) * np.sum(repulsive_forces_final[:, None] * repulsive_vectors_final, axis=0) # 3x1
        weighted_repulsive_vector = weighted_repulsive_vector / np.linalg.norm(weighted_repulsive_vector) # 3x1
        average_repulsive_forces =  np.sum(repulsive_forces_final) / num_repulsive_points # scalar
        return repulsive_gain * average_repulsive_forces * weighted_repulsive_vector #  3x1

   
    
   