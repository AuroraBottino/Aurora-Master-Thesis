#!/usr/bin/env python

import numpy as np
import cv2
from geometry_msgs.msg import PoseStamped
import yaml


class FeatureMatchingCalculator:
    """
    This node implements the feature matching algorithm, the PnP RANSAC algorithm and the reprojection error algorithm
    """
    def __init__(self):

        ## \brief Variable to store the BFMatcher, used in the feature matching process (see function matching in featurematching_calculator.py)
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        ## \brief Variable to store the threshold for the Euclidean distance between 3D points, used to filter the matches
        self.threshold_distance = 0.30
        ## \brief Variable to store reprojection threshold, used to filter valid 3D points 
        self.reprojection_threshold = 15.0 

   #YAML FILE
        with open('/home/user/catkin_ws/src/collision_aurora/Settings/calibration_params.yaml', 'r') as file:
            self.camera_params = yaml.safe_load(file)

        self.robot_points = np.array([[self.camera_params['cam_origin_pose_x'],self.camera_params['cam_origin_pose_y'],self.camera_params['cam_origin_pose_z']],
                                  [-0.83,0.5,-1.34],
                                  [-0.83,-0.5,-1.34],
                                  [0.77,-0.5,-1.34],
                                  [0.77,0.5,-1.34],
                                  [-0.83,0,-0.59],
                                  [0.72,0,-0.63]])
     
        # Intrinsic matrices
        self.intrinsic_matrices = {}
        for i in range(1, 7): 
            fx = self.camera_params[f'Camera_pinhole_fx_cam{i}']
            fy = self.camera_params[f'Camera_pinhole_fy_cam{i}']
            cx = self.camera_params[f'Camera_pinhole_px_cam{i}']
            cy = self.camera_params[f'Camera_pinhole_py_cam{i}']
            
            self.intrinsic_matrices[i-1] = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])  

        # Estrinsic matrices
        self.extrinsic_matrices = {}
        for i in range(1, 7):  
            params = [
                self.camera_params[f'CameraSystem_cam{i}_1'],
                self.camera_params[f'CameraSystem_cam{i}_2'],
                self.camera_params[f'CameraSystem_cam{i}_3'],
                self.camera_params[f'CameraSystem_cam{i}_4'],
                self.camera_params[f'CameraSystem_cam{i}_5'],
                self.camera_params[f'CameraSystem_cam{i}_6']
            ]
            self.extrinsic_matrices[i-1] = self.cayley2hom(params)

    def cayley2hom(self, cayleyParams):  
        """
        Converts Cayley parameters to a homogeneous matrix
        return: A 4x4 homogeneous matrix
        """
        cayleyRot = np.array([cayleyParams[0], cayleyParams[1], cayleyParams[2]]) 
        rot = self.cayley2rot(cayleyRot) 

        hom = np.array([
            [rot[0,0], rot[0,1], rot[0,2], cayleyParams[3]],
            [rot[1,0], rot[1,1], rot[1,2], cayleyParams[4]],
            [rot[2,0], rot[2,1], rot[2,2], cayleyParams[5]],
            [0, 0, 0, 1]
        ])
        return hom

    def cayley2rot(self, cayParamIn):
        """
        Converts Cayley parameters to a rotation matrix
        return: A 3x3 rotation matrix
        """
        c1, c2, c3 = cayParamIn 
        c1sqr, c2sqr, c3sqr = c1 ** 2, c2 ** 2, c3 ** 2
        scale = 1 + c1sqr + c2sqr + c3sqr

        R = np.array([
            [1 + c1sqr - c2sqr - c3sqr, 2 * (c1 * c2 - c3), 2 * (c1 * c3 + c2)],
            [2 * (c1 * c2 + c3), 1 - c1sqr + c2sqr - c3sqr, 2 * (c2 * c3 - c1)],
            [2 * (c1 * c3 - c2), 2 * (c2 * c3 + c1), 1 - c1sqr - c2sqr + c3sqr]
        ])

        R /= scale 
        return R
    

    def extract_descriptors(self, msg): 
        """
        Function to extracts descriptors from the message
        Args:
            msg: The message containing descriptors
        Return:  A numpy array of extracted descriptors
        """
        descriptor_length = msg.descriptor_length
        num_descriptors = msg.num_descriptors
        descriptors = np.array(msg.descriptors).reshape((num_descriptors, descriptor_length)).astype(np.float32)
        return descriptors


    def euclidean_distance(self, pt1, pt2): 
        """
        Function to compute the Euclidean distance between two 3D points
        Return: A numpy array of the Euclidean distance
        """
        return np.linalg.norm(pt1 - pt2)
    
    def calculate_mean_distance(self, pt1s, pt2s): 
        """
        Function to compute the mean distance between two sets of 3D points
        Return: A numpy array of the mean distance
        """
        distances = [self.euclidean_distance(pt1, pt2) for pt1, pt2 in zip(pt1s, pt2s)]
        return np.mean(distances)
    
    def calculate_mean_points(self, valid_pt1s, valid_pt2s): 
        """
        Function to compute the mean points between two sets of 3D points
        Return: The mean points
        """
        mean_points = np.asarray(valid_pt1s) + np.asarray(valid_pt2s) / 2
        return mean_points
    
    def odometry_callback(self, msg,): 
        """
        Callback function for the odometry topic, used to get the current position
        Args:
            msg: The message containing the odometry data
        """
        position = msg.pose.pose.position
        self.current_position = (position.x, position.y, position.z)
        self.positions_history.append(self.current_position)
        self.velocities_history.append(self.current_velocity)

        pose_stamped = PoseStamped()
        pose_stamped.header = msg.header  
        pose_stamped.pose = msg.pose.pose
    
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

    
    def matching(self, current_descriptors, current_keypoints, current_points_3d, last_points_3d, last_descriptors, last_keypoints):
        """
        This is the feature matching function. Using a BFMatcher, it matches the current descriptors, the 3D points and keypoints of two 
        consecutive messages and it filters the matches based on the Euclidean distance between the 3D points. If the Euclidean distance
        is smaller than the threshold distance set at 30 cm, the match is considered valid.
        Args:
            current_descriptors: Descriptors from the current frame
            current_keypoints: Keypoints from the current frame
            current_points_3d: 3D points from the current frame
            last_points_3d: 3D points from the previous frame
            last_descriptors: Descriptors from the previous frame
            last_keypoints: Keypoints from the previous frame
        Return: The 2D and 3D points of the current and previous images, the good matches and the number of matches
        """
        matches = self.bf.knnMatch(current_descriptors, last_descriptors, k=2) 

        pt_2d_current_vector = []
        pt_2d_previous_vector = []
        pt_3d_current_vector = []
        pt_3d_previous_vector = []

        good_matches = []
        for m in matches:
            if len(m) >= 2 and m[0].distance < 0.75 * m[1].distance:
                good_matches.append([m[0],m[1]]) 
            #    good_matches.append(m[0])
                pt_3d_current = current_points_3d[m[0].queryIdx]  
                pt_2d_current = current_keypoints[m[0].queryIdx]  
                pt_3d_previous = last_points_3d[m[0].trainIdx]  
                pt_2d_previous = last_keypoints[m[0].trainIdx]  

                distance_pt3d = self.euclidean_distance(pt_3d_current, pt_3d_previous)
                if distance_pt3d > self.threshold_distance:
                    continue 

                pt_2d_current_vector.append(pt_2d_current)
                pt_2d_previous_vector.append(pt_2d_previous)
                pt_3d_current_vector.append(pt_3d_current)
                pt_3d_previous_vector.append(pt_3d_previous)

        return pt_2d_current_vector, pt_2d_previous_vector, pt_3d_current_vector, pt_3d_previous_vector, good_matches, len(matches)
    

    def pnp_ransac(self, pt_2d_current_vector, pt_2d_previous_vector, pt_3d_current_vector, pt_3d_previous_vector, intrinsic_matrices):
        """
        This is the PnP RANSAC function. Computes the R,t vectors to project the 3D points of the previous image into the current image
        Args:
            pt_2d_current_vector: 2D points from the current image
            pt_2d_previous_vector: 2D points from the previous image
            pt_3d_current_vector: 3D points from the current image
            pt_3d_previous_vector: 3D points from the previous image
            intrinsic_matrices: Camera intrinsic matrices
        Return: The 3D and 2D points of the current and previous images, the success of the PnP RANSAC, the rotation and translation vectors, the inliers
        """
   
        model_points_previous = np.asarray(pt_3d_previous_vector) 
        image_points_current = np.asarray(pt_2d_current_vector) 
        dist_coeffs = np.zeros((4,1))
        (success, rvec, tvec, inliers) = cv2.solvePnPRansac(model_points_previous, image_points_current, intrinsic_matrices, dist_coeffs, flags=cv2.SOLVEPNP_EPNP, iterationsCount=500, reprojectionError=8.0)
        inliers = np.ravel(inliers) 
        if success == False:   
            print(f"succes: {success}")
            model_points_current = np.asarray([])
            image_points_previous = np.asarray([])
            return model_points_previous, image_points_current, model_points_current, image_points_previous, success, rvec, tvec, inliers
        model_points_previous = np.asarray(pt_3d_previous_vector)[inliers,:] 
        image_points_current = np.asarray(pt_2d_current_vector)[inliers,:]
        model_points_current = np.asarray(pt_3d_current_vector)[inliers,:]
        image_points_previous = np.asarray(pt_2d_previous_vector)[inliers,:]

        return model_points_previous, image_points_current, model_points_current, image_points_previous, success, rvec, tvec, inliers
    
    
    def reprojection_error(self, model_points_previous, image_points_current, model_points_current, image_points_previous, rvec, tvec, intrinsic_matrices, current_cam_T_world, last_cam_T_world):
        """
        This function calculates the reprojection error of the 3D points. It reprojects the 3D points of the previous image into the current image and 
        the 3D points of the current image into the previous image. If the distance between the reprojected 2D points and the actual 2D points is smaller
        than the reprojection threshold, the 3D point is considered valid.
        Args:
            model_points_previous: 3D points from the previous image
            image_points_current: 2D points from the current image
            model_points_current: 3D points from the current image
            image_points_previous: 2D points from the previous image
            rvec: Rotation vector from PnP RANSAC
            tvec: Translation vector from PnP RANSAC
            intrinsic_matrices: Camera intrinsic matrices
            current_cam_T_world: Current camera transformation matrix 
            last_cam_T_world: Previous camera transformation matrix 
        Return: The valid 3D points, the distances from the camera and the distances between the 2D points
        """
       
        valid_3d_points = []
        distance_pt2d_vector = [] 
        distances_from_camera = []

        
        for i in range(len(model_points_previous)):
            pt_3d_previous = model_points_previous[i] #3D point in the previous image
            pt_2d_current = image_points_current[i] #2D point in the current image
            pt_3d_current = model_points_current[i] #3D point in the current image
            pt_2d_previous = image_points_previous[i] #2D point in the previous image

            distances_pt2d = [] 
                
            #Reprojecting the previous 3D point into the current image
            point_2d_projected, _ = cv2.projectPoints(pt_3d_previous, rvec, tvec, intrinsic_matrices, None)
            distance1_pt2d = np.linalg.norm(point_2d_projected - pt_2d_current) 
            #Reprojecting the current 3D point into the previous image  
            rvec2, _ = cv2.Rodrigues(np.linalg.inv((last_cam_T_world))[:3, :3])
            tvec2 = (np.linalg.inv(last_cam_T_world))[:3, 3]
            point_2d_projected, _ = cv2.projectPoints(pt_3d_current, rvec2, tvec2, intrinsic_matrices, None)
            distance2_pt2d = np.linalg.norm(point_2d_projected - pt_2d_previous) 

            distance_pt2d = (distance1_pt2d + distance2_pt2d) / 2.0
            distance_pt2d_vector.append(distance_pt2d) 

            mean_pt3d = (pt_3d_current+ pt_3d_previous) / 2.0

            mean_pt3d_camera_frame = np.dot(np.linalg.inv(current_cam_T_world), np.array([mean_pt3d[0], mean_pt3d[1], mean_pt3d[2], 1]))[0:3] 
            distance_from_camera = np.linalg.norm(mean_pt3d_camera_frame) 
            distances_from_camera.append(distance_from_camera)

            distances_pt2d.append(distance_pt2d) 
            
            if distance_pt2d < self.reprojection_threshold:
                valid_3d_points.append(mean_pt3d)
        
            #     rospy.loginfo(f"Valid 3D point: {mean_pt3d}")
            # else:
            #     rospy.logwarn(f"Reprojection error too high: {distance}")

        return (valid_3d_points, distances_from_camera, distance_pt2d_vector)


