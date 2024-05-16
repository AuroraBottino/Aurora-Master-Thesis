#!/usr/bin/env python

import rospy
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Path
from octomap_py.octomap import Octomap
from sensor_msgs.msg import PointField
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, PoseArray
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from cola2_msgs.msg import DVL

class Visualization:
    """
    This node implements the visualitazion of: points on the AUV, the previous and the current point clouds, the valid 3d points, the robot's trajectory, the octomap, the attractive and repulsive forces
    """
       
    def __init__(self):
       
       
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
    

        # Initialize the octomap
        octomap_resolution = 0.75
        min_points_per_voxel = 5
        octomap_radius = 30.0
        start_position = np.array([0.0, 0.0, 0.0]) 
        self.octomap = Octomap(start_position,octomap_radius, octomap_resolution, min_points_per_voxel)


        # Initialize the path and the markers
        self.path = Path() 
        self.path.header.frame_id = 'map'  
        self.pose_array = PoseArray()
        self.attractive_marker_array = MarkerArray()
        self.repulsive_marker_array = MarkerArray()
        self.total_force_marker_array = MarkerArray()


    def create_marker(self, marker_type, position, orientation, scale, color, id, namespace, frame_id='map'):
            """
            Create a marker for visualization
            Return: marker
            """
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = rospy.Time.now()
            marker.id = id
            marker.ns = namespace
            marker.type = marker_type
            marker.action = Marker.ADD 
            marker.scale.x = scale[0]
            marker.scale.y = scale[1]
            marker.scale.z = scale[2]
            marker.color = color
            return marker

    # def create_points_on_auv_marker(self, points_on_auv, frame_id='map'):

    #     marker_point1 = MarkerArray()
    #     for i in range(0,7): 
    #         marker = Marker()
    #         marker.header.frame_id = frame_id
    #         marker.header.stamp = rospy.Time.now()
    #         marker.id = i
    #         marker.type = Marker.SPHERE
    #         marker.action = Marker.ADD
    #         marker.pose.position.x = points_on_auv[i,0]
    #         marker.pose.position.y = points_on_auv[i,1]
    #         marker.pose.position.z = points_on_auv[i,2]
    #         marker.scale.x = 0.2
    #         marker.scale.y = 0.2
    #         marker.scale.z = 0.2
    #         marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0) # Red color
    #         marker_point1.markers.append(marker)
    #     self.points_on_auv_publisher.publish(marker_point1)

    def create_point0_marker(self, id, points_on_auv, frame_id='map'):
        """
        Create a marker to visualize the camera base position on the AUV
        """
        marker_point0 = MarkerArray()
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.id = id
        marker.ns = 'point1_marker'
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = points_on_auv[0,0]
        marker.pose.position.y = points_on_auv[0,1]
        marker.pose.position.z = points_on_auv[0,2]
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)
        marker_point0.markers.append(marker)
        self.point0_pub.publish(marker_point0)


    def create_point1_marker(self, id, points_on_auv, frame_id='map'):
        """
        Create a marker to visualize the first point (-0.83,0.5,-1.34) on the AUV
        """
        marker_point1 = MarkerArray()
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.id = id
        marker.ns = 'point1_marker'
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = points_on_auv[1,0]
        marker.pose.position.y = points_on_auv[1,1]
        marker.pose.position.z = points_on_auv[1,2]
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0) 
        marker_point1.markers.append(marker)
        self.point1_pub.publish(marker_point1)

    def create_point2_marker(self, id, points_on_auv, frame_id='map'):
        """
        Create a marker to visualize the second point (-0.83,-0.5,-1.34) on the AUV
        """
        marker_point2 = MarkerArray()
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.id = id
        marker.ns = 'point2_marker'
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = points_on_auv[2,0]
        marker.pose.position.y = points_on_auv[2,1]
        marker.pose.position.z = points_on_auv[2,2]
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0) 
        marker_point2.markers.append(marker)
        self.point2_pub.publish(marker_point2)

    def create_point3_marker(self, id, points_on_auv, frame_id='map'):
        """
        Create a marker to visualize the third point (0.77,-0.5,-1.34) on the AUV
        """
        marker_point3 = MarkerArray()
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.id = id
        marker.ns = 'point3_marker'
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = points_on_auv[3,0]
        marker.pose.position.y = points_on_auv[3,1]
        marker.pose.position.z = points_on_auv[3,2]
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)
        marker_point3.markers.append(marker)
        self.point3_pub.publish(marker_point3)

    def create_point4_marker(self, id, points_on_auv, frame_id='map'):
        """
        Create a marker to visualize the fourth point (0.77,0.5,-1.34) on the AUV
        """
        marker_point4 = MarkerArray()
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.id = id
        marker.ns = 'point4_marker'
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = points_on_auv[4,0]
        marker.pose.position.y = points_on_auv[4,1]
        marker.pose.position.z = points_on_auv[4,2]
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)
        marker_point4.markers.append(marker)
        self.point4_pub.publish(marker_point4)

    def create_point5_marker(self, id, points_on_auv, frame_id='map'):
        """
        Create a marker to visualize the fifth point (-0.83,0,-0.59) on the AUV
        """
        marker_point5 = MarkerArray()
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.id = id
        marker.ns = 'point5_marker'
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = points_on_auv[5,0]
        marker.pose.position.y = points_on_auv[5,1]
        marker.pose.position.z = points_on_auv[5,2]
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)
        marker_point5.markers.append(marker)
        self.point5_pub.publish(marker_point5)
    
    def create_point6_marker(self, id, points_on_auv, frame_id='map'):
        """
        Create a marker to visualize the sixt point (0.72,0,-0.63) on the AUV
        """
        marker_point6 = MarkerArray()
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.id = id
        marker.ns = 'point6_marker'
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = points_on_auv[6,0]
        marker.pose.position.y = points_on_auv[6,1]
        marker.pose.position.z = points_on_auv[6,2]
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)
        marker_point6.markers.append(marker)
        self.point6_pub.publish(marker_point6)
   

    def publish_pointcloud(self, cloud_points): 
        """
        Function to visualize the current and the previous point clouds
        """
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map'  

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]

        pointcloud = pc2.create_cloud_xyz32(header, cloud_points) 
        return pointcloud

    def publish_valid_3d_points(self, valid_3d_points):
        """
        Function to visualize the valid 3d points
        """
        if len(valid_3d_points) == 0:
            rospy.loginfo("No valid points to publish")
            return

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'map' 

        valid_points_cloud = pc2.create_cloud_xyz32(header, valid_3d_points)
        self.valid_3d_points_publisher.publish(valid_points_cloud)


    def visualize_trajectory(self, point):
        """
        Function to visualize the trajectory of the robot
        """
        self.pose_array.header.frame_id = 'map'
        self.pose_array.header.stamp = rospy.Time.now()

        pose = Pose()
        pose.position.x = point[0]
        pose.position.y = point[1]
        pose.position.z = point[2]
        self.pose_array.poses.append(pose)

        self.visualize_trajectory_publisher.publish(self.pose_array)


    def visualize_attractive_force(self, current_position, attractive_force):
        """
        Create arrow marker to visualize the attractive force
        """
        if current_position is None or np.isnan(current_position).any():
            return

        if attractive_force is None or np.isnan(attractive_force).any():
            return

        force_magnitude = np.linalg.norm(attractive_force)
        if force_magnitude > 0:
            normalized_force = attractive_force / force_magnitude
        else:
            normalized_force = attractive_force
            return

        arrow_marker = self.create_marker(
            Marker.ARROW,
            current_position,
            [0, 0, 0, 1],
            [0.05, 0.05, 0.05],
            ColorRGBA(0.0, 0.0, 1.0, 1.0),  #color blue
            len(self.attractive_marker_array.markers),
            'attractive_force',
            frame_id='map'
        )

        start_point = Point(*current_position)
        end_point = Point(*(current_position + normalized_force ))
        arrow_marker.points = [start_point, end_point]

        self.attractive_marker_array.markers.append(arrow_marker)
        self.attractive_force_publisher.publish(self.attractive_marker_array)


    def visualize_repulsive_force(self, current_position, repulsive_force, pub, color):
        """
        Create arrow marker to visualize the repulsive force
        """
        if current_position is None or repulsive_force is None:
            rospy.logwarn("Current position or repulsive force is not defined. Skipping visualization.")
            return
        
        repulsive_marker_array = MarkerArray()

        arrow_marker = self.create_marker(
            Marker.ARROW,
            current_position,
            [0, 0, 0, 1],
            [0.05, 0.05, 0.05],
            color, #colore giallo
            len(repulsive_marker_array.markers), # len(self.repulsive_marker_array.markers),
            'repulsive_force',
            frame_id='map'
        )
        
        start_point = Point(*current_position)
        end_point = Point(*(current_position + repulsive_force))
        arrow_marker.points = [start_point, end_point]

        # self.repulsive_marker_array.markers.append(arrow_marker)
        repulsive_marker_array.markers.append(arrow_marker)
        # self.repulsive_force_publisher.publish(self.repulsive_marker_array)
        pub.publish(repulsive_marker_array)


    def visualize_octomap(self, obstacle_centroids, obstacle_normals, current_position):
        """
        Function to visualize the octomap. Create arrow markers for the normals and sphere markers for the centroids
        """
        delete_markers = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        delete_markers.markers.append(delete_marker)
        self.centroids_normals_publisher.publish(delete_markers)
        rospy.sleep(0.1)
        marker_array = MarkerArray()
        marker_id = 0

        for i, (centroid, normal) in enumerate(zip(obstacle_centroids, obstacle_normals)):
            centroid_to_robot = np.array(current_position) - np.array(centroid)
            normal_normalized = normal / np.linalg.norm(normal)
            centroid_to_robot_normalized = centroid_to_robot / np.linalg.norm(centroid_to_robot) 

            # Arrow for normals, grey color
            normal_color = ColorRGBA(0.5, 0.5, 0.5, 1.0)
            normal_length = 0.5  
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.ns = 'normals'
            marker.header.stamp = rospy.Time.now()
            marker.id = marker_id
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.color = normal_color
            marker.points.append(Point(centroid[0], centroid[1], centroid[2]))  
            marker.points.append(Point(centroid[0] + normal_normalized[0] * normal_length, centroid[1] + normal_normalized[1] * normal_length, centroid[2] + normal_normalized[2] * normal_length))
            marker_array.markers.append(marker)
            marker_id += 1

            # Sphere marker for centroids, grey color
            centroid_color = ColorRGBA(0.5, 0.5, 0.5, 1.0)
            sphere_marker = Marker()
            sphere_marker.header.frame_id = 'map'
            sphere_marker.ns = 'centroids'
            sphere_marker.header.stamp = rospy.Time.now()
            sphere_marker.id = marker_id
            sphere_marker.type = Marker.SPHERE
            sphere_marker.action = Marker.ADD
            sphere_marker.scale.x = 0.05
            sphere_marker.scale.y = 0.05
            sphere_marker.scale.z = 0.05
            sphere_marker.color = centroid_color
            sphere_marker.pose.position.x = centroid[0]
            sphere_marker.pose.position.y = centroid[1]
            sphere_marker.pose.position.z = centroid[2]
            marker_array.markers.append(sphere_marker)
            marker_id += 1

        self.centroids_normals_publisher.publish(marker_array)

