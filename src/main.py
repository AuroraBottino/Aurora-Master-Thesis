#!/usr/bin/env python

import rospy
import numpy as np
import cv2
import sensor_msgs.point_cloud2 as pc2
from collision_detection_module.msg import DescribedPointCloud
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt 
from octomap_py.octomap import Octomap
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray 
from cola2_msgs.msg import DVL
import yaml
import pandas as pd
from geometry_msgs.msg import PoseArray
from force_calculator import ForceCalculator
from risk_calculator import RiskCalculator
from visualization_class import Visualization
from featurematching_calculator import FeatureMatchingCalculator
from std_msgs.msg import Header, ColorRGBA


class PointCloudFeatureMatcher: 
    
    def __init__(self):

        # Initialize the node
        rospy.init_node('point_cloud_feature_matcher', anonymous=True)
        
        ## \brief Subscriber to the point cloud topic
        # \param topic_name: name of the topic to subscribe to the point cloud data
        topic_name = rospy.get_param('~topic_name', '/collision_detector/local_point_cloud') 
        self.subscriber = rospy.Subscriber(topic_name, DescribedPointCloud, self.callback)

        ## \brief Variable to store the number of processed point clouds
        self.processed_clouds_count = 0
        rospy.loginfo(f"Subscribed to: {topic_name}")

        ## \brief Subscriber to the odometry topic, used to get the current position
        self.odom_subscriber = rospy.Subscriber('/girona500/navigator/odometry', Odometry, self.odometry_callback) 
        ## \brief Subscriber to the DVL topic, used to get the current velocity
        self.dvl_subscriber = rospy.Subscriber('/girona500/navigator/dvl', DVL, self.dvl_callback)
        ## \brief Publisher for the visualization of the centroids and normals of the octomap
        self.centroids_normals_publisher = rospy.Publisher('visualization_centroids_normals', MarkerArray, queue_size=100)
        ## \brief Publisher for the current point cloud
        self.currentpointcloud_publisher = rospy.Publisher('visualization_pointcloud', PointCloud2, queue_size=10)
        ## \brief Publisher for the previous point cloud
        self.previouspointcloud_publisher = rospy.Publisher('visualization_previous_pointcloud', PointCloud2, queue_size=10)
        ## \brief Publisher for the point cloud of valid 3D points
        self.valid_3d_points_publisher = rospy.Publisher('visualization_valid_points', PointCloud2, queue_size=10)
        ## \brief Publisher for the robot's trajectory visualization
        self.visualize_trajectory_publisher = rospy.Publisher('visualization_trajectory', PoseArray, queue_size=10)
        ## \brief Publisher for the risk factor visualization using a colormap
        self.risktocolormap_publisher = rospy.Publisher("color_pointcloud", PointCloud2, queue_size=2)
        ## \brief Publisher for the attractive force visualization
        self.attractive_force_publisher = rospy.Publisher('visualization_attractive_force', MarkerArray, queue_size=100)
        ## \brief Publisher for the repulsive force visualization: this force is computed using the theory of potential fields, without considering the risk factor (beta = 0)
        self.repulsive_force_publisher = rospy.Publisher('visualization_repulsive_force', MarkerArray, queue_size=10)
        ## \brief Publisher for the repulsive force with risk: this force includes the risk factor (beta = 1)
        self.repulsive_force_risk_publisher = rospy.Publisher('visualization_repulsive_force_risk', MarkerArray, queue_size=10)
        ## \brief Publisher for the points on the AUV: this publisher includes all the seven selected points on the AUV
        self.points_on_auv_publisher = rospy.Publisher('visualization_point_on_auv', MarkerArray, queue_size=10)

        ## \brief Publisher for the camera base position on the AUV: the individual publisher can be used if we want to visualize each point separately from the others
        self.point0_pub = rospy.Publisher('point0_marker', MarkerArray, queue_size=10)
        ## \brief Publisher for point 1 on the AUV
        self.point1_pub = rospy.Publisher('point1_marker', MarkerArray, queue_size=10)
        ## \brief Publisher for point 2 on the AUV
        self.point2_pub = rospy.Publisher('point2_marker', MarkerArray, queue_size=10)
        ## \brief Publisher for point 3 on the AUV
        self.point3_pub = rospy.Publisher('point3_marker', MarkerArray, queue_size=10)
        ## \brief Publisher for point 4 on the AUV
        self.point4_pub = rospy.Publisher('point4_marker', MarkerArray, queue_size=10)
        ## \brief Publisher for point 5 on the AUV
        self.point5_pub = rospy.Publisher('point5_marker', MarkerArray, queue_size=10)
        ## \brief Publisher for point 6 on the AUV
        self.point6_pub = rospy.Publisher('point6_marker', MarkerArray, queue_size=10)

        self.last_descriptors = [None, None, None, None, None, None]
        self.current_descriptor_index = 0 
        self.last_points3d =  [None, None, None, None, None, None]
        self.current_point3d_index = 0
        self.last_keypoints = [None, None, None, None, None, None] 
        
        ## \brief Variable to store the BFMatcher, used in the feature matching process (see function matching in featurematching_calculator.py)
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        octomap_resolution = 0.75
        min_points_per_voxel = 5
        octomap_radius = 30.0
        start_position = np.array([0.0, 0.0, 0.0]) 
        self.octomap = Octomap(start_position,octomap_radius, octomap_resolution, min_points_per_voxel)

        self.force_calculator = ForceCalculator()
        self.risk_calculator = RiskCalculator()
        self.visualization_class = Visualization()
        self.featurematching_calculator = FeatureMatchingCalculator()

        self.previous_cloud_points = None
        self.current_position = None
        self.previous_position = None
        self.current_velocity = np.zeros(3)
        ## \brief Variable to store the repulsive gain, which is used to scale the repulsive force
        self.repulsive_gain = 1.0
        ## \brief Variable to store the minimum distance between the AUV and the obstacles used to compute the repulsive force
        self.min_distance = 3.0
        self.current_timestamp = None
        self.previous_timestamp = None
        self.positions_history = []
        self.velocities_history = []
        ## \brief Variable to store the reprojection errors 
        self.distance_pt2d_vector = []
        ## \brief Variable to store the distances of 3D points in camera frame from the camera
        self.distances_from_camera = []
        ## \brief Variable to store the previous transformation between the camera frame and the world frame
        self.last_cam_T_world = [None, None, None, None, None, None]
        ## \brief Variable to store the current pose of the camera in the world frame
        self.cam_world_pose = None
        ## \brief Variable to store the previous pose of the camera in the world frame
        self.last_cam_world_pose = None
        ## \brief Variable to store the 3D points for each camera
        self.cam_3d_points = None
        self.points_on_auv = np.zeros((7, 3))
        ## \brief Variable to store the maximum number of obstacles to consider inside the minimun distance to the robot in order to compute the repulsive force, which is equal to 10
        self.max_obstacles = 10
        
        ## \brief Variable to store the alpha parameter, which is used to scale the direction of the repulsive force. If alpha = 0, the repulsive force is computed only considering only the direction of motion. If alpha = 1, the repulsive force is computed only considering the direction of the obstacles. By changing the value of alpha, the repulsive force can be computed considering both the direction of motion and the direction of the obstacles.
        self.alpha = 0.5 
        ## \brief Variable to store the beta parameter, which is used to scale the risk factor. If beta = 0, the repulsive force is computed without considering the risk factor. If beta = 1, the repulsive force is computed only considering the risk factor. By changing the value of beta, the repulsive force can be computed considering both the risk factor and the repulsive force.
        self.beta = 1
        self.average_repulsive_forces = None

        # Load the camera parameters from the yaml file
        with open('/home/user/catkin_ws/src/collision_aurora/Settings/calibration_params.yaml', 'r') as file:
            ## \brief Variable to store the camera parameters read from the yaml file
            self.camera_params = yaml.safe_load(file)

        ## \brief Variable to store the seven selected points on the AUV
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

    # Function to convert Cayley parameters to a homogeneous matrix
    def cayley2hom(self, cayleyParams):  
        """
        Converts Cayley parameters to a homogeneous matrix
        Return: A 4x4 homogeneous transformation matrix
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

    # Function to convert Cayley parameters to a rotation matrix
    def cayley2rot(self, cayParamIn):
        """
        Converts Cayley parameters to a rotation matrix
        Return: A 3x3 rotation matrix
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

    def odometry_callback(self, msg): 
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


    def callback(self, msg):
        """
        Callback function for the point cloud topic
        Args:
            msg: The message containing the point cloud data
        """

        rospy.loginfo("Callback triggered")
        # Extract the timestamp from the message and update the current and previous timestamps
        new_timestamp = msg.header.stamp
        if self.current_timestamp is not None:
            self.previous_timestamp = self.current_timestamp
        self.current_timestamp = new_timestamp 

        # Extract the camera pose from the message and update the current and last camera poses
        cam_world_pose = np.array(msg.cam_world_pose).reshape((4, 4))
        if self.cam_world_pose is not None:
            self.last_cam_world_pose = self.cam_world_pose 
        self.cam_world_pose = cam_world_pose
        
        # Extract the descriptors, 3D points and keypoints from the message
        descriptors = self.extract_descriptors(msg)
        cloud_points = np.array(list(pc2.read_points(msg.points3d, field_names=("x", "y", "z"))))
        keypoints = np.array(msg.keypoints).reshape(-1, 2) 

        # Initialize the current descriptor index, the current keypoint index, the current velocity and the camera_T_world
        self.current_descriptor_index = 0
        self.current_keypoint_index = 0
        current_velocity = self.current_velocity
        cam_T_world = [None, None, None, None, None, None]

        # Intialize the variables for the valid 3D points, the distance between the 2D points and the 3D points, the distances from the camera and the 3D points for all cameras
        valid_3d_points = [] 
        distance_pt2d_vector = [] 
        distances_from_camera = []
        all_cam_3d_points = []


        #Camera position
        robotbase_T_camerabase = np.eye(4) 
        translation_vector = np.array([self.camera_params['cam_origin_pose_x'], self.camera_params['cam_origin_pose_y'], self.camera_params['cam_origin_pose_z']])
        robotbase_T_camerabase[:3,3] = translation_vector
        point0_T_camerabase = np.dot(cam_world_pose,robotbase_T_camerabase)
        self.points_on_auv[0,:] = point0_T_camerabase[:3, 3]
        
        #Point1 on AUV
        point1_T_camerabase = np.eye(4)
        translation_vector1 = np.array([-0.83, 0.5, -1.34])
        point1_T_camerabase[:3,3] = translation_vector1
        point1_T_world = np.dot(cam_world_pose, np.dot(robotbase_T_camerabase, point1_T_camerabase))
        self.points_on_auv[1,:] = point1_T_world[:3, 3] 
    
        #Point2 on AUV
        point2_T_camerabase = np.eye(4)
        translation_vector2 = np.array([-0.83, -0.5, -1.34])
        point2_T_camerabase[:3,3] = translation_vector2
        point2_T_world = np.dot(cam_world_pose, np.dot(robotbase_T_camerabase, point2_T_camerabase))
        self.points_on_auv[2,:] = point2_T_world[:3, 3]
    
        #Point3 on AUV
        point3_T_camerabase = np.eye(4)
        translation_vector3 = np.array([0.77, -0.5, -1.34])
        point3_T_camerabase[:3,3] = translation_vector3
        point3_T_world = np.dot(cam_world_pose, np.dot(robotbase_T_camerabase, point3_T_camerabase))
        self.points_on_auv[3,:] = point3_T_world[:3, 3]
        
        #Point4 on AUV
        point4_T_camerabase = np.eye(4)
        translation_vector4 = np.array([0.77, 0.5, -1.34])
        point4_T_camerabase[:3,3] = translation_vector4
        point4_T_world = np.dot(cam_world_pose, np.dot(robotbase_T_camerabase, point4_T_camerabase))
        self.points_on_auv[4,:] = point4_T_world[:3, 3]
      
        #Point5 on AUV
        point5_T_camerabase = np.eye(4)
        translation_vector5 = np.array([-0.83, 0.0, -0.59])
        point5_T_camerabase[:3,3] = translation_vector5
        point5_T_world = np.dot(cam_world_pose, np.dot(robotbase_T_camerabase, point5_T_camerabase))
        self.points_on_auv[5,:] = point5_T_world[:3, 3]
    
        #Point6 on AUV
        point6_T_camerabase = np.eye(4)
        translation_vector6 = np.array([0.72, 0.0, -0.63])
        point6_T_camerabase[:3,3] = translation_vector6
        point6_T_world = np.dot(cam_world_pose, np.dot(robotbase_T_camerabase, point6_T_camerabase))
        self.points_on_auv[6,:] = point6_T_world[:3, 3]

        current_position = (cam_world_pose[0, 3], cam_world_pose[1, 3], cam_world_pose[2, 3])
        if self.previous_position is not None:
            current_velocity[0] = current_position[0] - self.previous_position[0]
            current_velocity[1] = current_position[1] - self.previous_position[1]
            current_velocity[2] = current_position[2] - self.previous_position[2]

        self.visualization_class.visualize_trajectory(current_position)

        num_matches = 0
        for i in range(0, 6): 
            if i in self.extrinsic_matrices:
                cam_T_world[i] = np.dot(cam_world_pose, self.extrinsic_matrices[i]) 

        for cam_idx, num_features in enumerate(msg.cam_num_features):
        #    rospy.loginfo(f"Processing camera {cam_idx} with {num_features} features")

            descriptor_start_idx = self.current_descriptor_index
            descriptor_end_idx = num_features
            keypoint_start_idx = self.current_keypoint_index
            keypoint_end_idx = num_features 
            self.current_descriptor_index = descriptor_end_idx 
            self.current_keypoint_index = keypoint_end_idx

            if descriptor_start_idx == descriptor_end_idx: 
                continue

            # Extract the descriptors, 3D points and keypoints for the current camera
            cam_descriptors = descriptors[descriptor_start_idx:descriptor_end_idx, :] 
            cam_3d_points = cloud_points[descriptor_start_idx:descriptor_end_idx, :]
            cam_keypoints = keypoints[keypoint_start_idx:keypoint_end_idx].reshape(-1, 2) 
            # Store the 3D points for each camera
            all_cam_3d_points.append(cam_3d_points) 

            if self.last_descriptors[cam_idx] is None and self.last_points3d[cam_idx] is None: 
                self.last_descriptors[cam_idx] = cam_descriptors
                self.last_points3d[cam_idx] = cam_3d_points
                self.last_keypoints[cam_idx] = cam_keypoints
                self.last_cam_T_world[cam_idx] = cam_T_world[cam_idx]
                self.previous_cloud_points = cloud_points
            else:
                
                # Call the feature matching function
                pt_2d_current_vector, pt_2d_previous_vector, pt_3d_current_vector, pt_3d_previous_vector, good_matches, current_matches = self.featurematching_calculator.matching(cam_descriptors, cam_keypoints, cam_3d_points, self.last_points3d[cam_idx], self.last_descriptors[cam_idx], self.last_keypoints[cam_idx])
                num_matches += current_matches

                if len(pt_2d_current_vector) < 4:
                    continue

                # Call the PnP RANSAC function
                model_points_previous, image_points_current, model_points_current, image_points_previous, success, rvec, tvec, inliers = self.featurematching_calculator.pnp_ransac(pt_2d_current_vector, pt_2d_previous_vector, pt_3d_current_vector, pt_3d_previous_vector, self.intrinsic_matrices[cam_idx])

                if success == True:
                        
                    # Call the reprojection error function
                    current_valid_3d_points, current_distances_from_camera, current_distance_pt2d_vector = self.featurematching_calculator.reprojection_error(model_points_previous, image_points_current, model_points_current, image_points_previous, rvec, tvec, self.intrinsic_matrices[cam_idx], cam_T_world[cam_idx], self.last_cam_T_world[cam_idx])
                    valid_3d_points.extend(current_valid_3d_points)
                    distances_from_camera.extend(current_distances_from_camera)
                    distance_pt2d_vector.extend(current_distance_pt2d_vector)

                    #Update the last descriptors, 3D points, keypoints, total transformation and current position
                    self.last_descriptors[cam_idx] = cam_descriptors
                    self.last_points3d[cam_idx] = cam_3d_points
                    self.last_keypoints[cam_idx] = cam_keypoints
                    self.last_cam_T_world[cam_idx] = cam_T_world[cam_idx]
                    self.previous_position = current_position
   
        # Percentage of valid points 
        percentage_valid_points = (len(valid_3d_points) / (float(num_matches) +1e-6)) * 100.0
        rospy.loginfo(f"Percentage of valid points: {percentage_valid_points:.2f}%")


        if len(valid_3d_points) > 0:
            valid_3d_points = np.asarray(valid_3d_points)
            robot_positions = np.repeat([current_position], valid_3d_points.shape[0], axis=0)
            self.octomap.add_point_cloud(valid_3d_points, robot_positions)
            
        obstacle_centroids = self.octomap.get_all_centroids()
        obstacle_normals = self.octomap.get_all_normals()
    
        if current_position is not None and len(valid_3d_points)>0:
            self.visualization_class.visualize_octomap(obstacle_centroids, obstacle_normals, current_position)

            #ATTRACTIVE FORCE
            attractive_force = self.force_calculator.attractive_force()
            self.visualization_class.visualize_attractive_force(current_position, current_velocity)

            #RISK FACTOR
            delta_t = (msg.header.stamp - self.previous_timestamp).to_sec()
            total_cam_3d_points = np.concatenate(all_cam_3d_points, axis=0)
        
            riskfactor, riskrobot_pts_idx, risk_factor_all_points = self.risk_calculator.computeRiskFactor(self.robot_points, obstacle_centroids, 
            self.last_cam_world_pose, cam_world_pose, 0, float(delta_t), offset=-0.1 ,time_multiplier=1/20,k=1.2,min_time_clip_value=0.3)
     
            #Risk factor using all the 3d points for all the cameras
            #risktocolormap = self.risk_calculator.risk_to_colormap(riskfactor, total_cam_3d_points)

            #Risk factor using only the obstacle centroids
            risktocolormap = self.risk_calculator.risk_to_colormap(riskfactor, obstacle_centroids)

           #REPULSIVE FORCE
            repulsive_force_risk = self.force_calculator.repulsive_force(current_velocity, 
            obstacle_centroids, obstacle_normals, self.points_on_auv, self.min_distance, self.alpha, 
            self.repulsive_gain, 0.0, self.max_obstacles, risk_factor_all_points)
         
            repulsive_force = self.force_calculator.repulsive_force(current_velocity,
            obstacle_centroids, obstacle_normals, self.points_on_auv, self.min_distance, self.alpha,
            self.repulsive_gain, 1.0 , self.max_obstacles, risk_factor_all_points)

     
            if len(repulsive_force) == 2:
                repulsive_force, points_within_min_distance = repulsive_force

            self.visualization_class.visualize_repulsive_force(current_position, repulsive_force, self.repulsive_force_publisher, ColorRGBA(1.0, 1.0, 0.0, 1.0))
            self.visualization_class.visualize_repulsive_force(current_position, repulsive_force_risk, self.repulsive_force_risk_publisher, ColorRGBA(1.0, 0.5, 0.0, 1.0))

            processed_point_cloud = self.visualization_class.publish_pointcloud(cloud_points)
            self.currentpointcloud_publisher.publish(processed_point_cloud)
            processed_previous_point_cloud = self.visualization_class.publish_pointcloud(self.previous_cloud_points)
            self.previouspointcloud_publisher.publish(processed_previous_point_cloud)    

            self.visualization_class.publish_pointcloud(self.previous_cloud_points)
            self.visualization_class.publish_valid_3d_points(valid_3d_points)

        # self.visualization_class.create_points_on_auv_marker(points_on_auv=self.points_on_auv)
            self.visualization_class.create_point0_marker(0, self.points_on_auv)
            self.visualization_class.create_point1_marker(1, self.points_on_auv)
            self.visualization_class.create_point2_marker(2, self.points_on_auv)
            self.visualization_class.create_point3_marker(3, self.points_on_auv)
            self.visualization_class.create_point4_marker(4, self.points_on_auv)
            self.visualization_class.create_point5_marker(5, self.points_on_auv)
            self.visualization_class.create_point6_marker(6, self.points_on_auv)

            self.save_distances()

            # plt.figure()
            # plt.plot(distances_from_camera, distance_pt2d_vector, 'ro')
            # plt.show

        self.processed_clouds_count += 1
        self.previous_cloud_points = cloud_points



    def save_distances(self):
        """
        This function saves in a csv file the distances of the 3D points from the camera and the distances between
        the 2D points. It is used to plot how the reprojection error changes as function of the distances from the camera.
        See script plot.py for the plot.
        """

        distance_pt2d_vector = np.array(self.distance_pt2d_vector)
        distances_from_camera = np.array(self.distances_from_camera)
        total_distances = np.column_stack((distance_pt2d_vector, distances_from_camera))
        df = pd.DataFrame(total_distances, columns=['reprojection_error', 'distances_from_camera'])
        filename = "/home/user/catkin_ws/src/collision_aurora/src/reprojection_error.csv"
        df.to_csv(filename, index=False)


    def run(self):
        rospy.loginfo("PointCloudFeatureMatcher is running...")
        rospy.spin()

    def __del__(self):
        cv2.destroyAllWindows()

if __name__ == '__main__':

    feature_matcher = PointCloudFeatureMatcher()
    feature_matcher.run()

  

                        
  