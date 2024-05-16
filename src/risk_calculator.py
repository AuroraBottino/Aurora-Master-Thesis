#!/usr/bin/env python

import rospy
import numpy as np
import cv2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointField
from sensor_msgs.msg import PointCloud2
import struct
from std_msgs.msg import Header


class RiskCalculator:
    """
    This node implements the risk factor for the AUV
    """
    def __init__(self):
        
        ## \brief Publisher for the risk factor visualization using a colormap
        self.risktocolormap_pub = rospy.Publisher("color_pointcloud", PointCloud2, queue_size=2)
    
    def computeRiskFactor(self, robot_points,points3d,world_Tprevious,world_Tcurrent,previous_timestamp,current_timestamp, offset=0.75,
                      time_multiplier=0.5,k=0.5,min_time_clip_value=0.1):
        """
        Function to calculate the risk factor
        """
        # Function variables and parameters
        d_min_temp = np.zeros((points3d.shape[0],robot_points.shape[0])); t_min_temp = np.zeros((points3d.shape[0],robot_points.shape[0]))
        min_d_norm = np.zeros((points3d.shape[0],robot_points.shape[0])); risk_factor_all_points = np.zeros((points3d.shape[0],robot_points.shape[0]))
        delta_t = (current_timestamp-previous_timestamp) * time_multiplier # Time passed between two keyframes

        # Get the relative transformation between frames
        current_Tprevious = np.linalg.inv(world_Tcurrent) @ world_Tprevious

        # Computing displacement vectors between cameras
        c_current_current = np.array([[0],[0],[0]]) # camera current relative location
        c_previous_current = np.array(current_Tprevious[0:3,3:4])
        # When robot only rotates use c_previous_current(?)
        v = c_current_current - c_previous_current
        norm_factor = np.linalg.norm(v)/delta_t

        c1 = np.transpose(robot_points)
            
        # Converting world point to camera current coordinate system
        points3d_hom = np.transpose(points3d)
        points3d_hom = np.vstack([points3d_hom,np.ones((1,points3d_hom.shape[1]))])
        points3d_cam_current = np.linalg.inv(world_Tcurrent) @ points3d_hom

        # Computing the min distance and time for each point 3D against all robot points
        for i in range(c1.shape[1]):
            # computing the time to reach the minimum distance from the point to the line of trajectory
            time_to_dmin = -np.transpose(c1[:,i:i+1] - points3d_cam_current[:3,:]).dot(v)/(np.linalg.norm(v)**2)
            # computing the min distance where a collision can possibly occur
            numerator_cross_product = np.cross(np.transpose(v),np.transpose(c1[:,i:i+1] - points3d_cam_current[:3,:]))
            d_min_temp[:,i] = np.linalg.norm(numerator_cross_product,axis=1)/np.linalg.norm(v)
            t_min_temp[:,i] = np.transpose(((time_to_dmin)*delta_t)+offset)
            # normalizing the distance
            min_d_norm[:,i:i+1] = d_min_temp[:,i:i+1]/norm_factor
            # time constraint for small numbers
            t_min_temp[:,i:i+1][(t_min_temp[:,i:i+1]<min_time_clip_value)*(t_min_temp[:,i:i+1]>=0)] = min_time_clip_value
            t_min_temp[:,i:i+1][t_min_temp[:,i:i+1]<0.0001] = 100 # For negative values we assign a high time value so the risk becomes low

            # Calculate the risk by modelling it as a probability density function
            sigma = k * t_min_temp[:,i:i+1]
            min_sigma = k*min_time_clip_value
            risk_probability = (k/(sigma)) * np.exp(-(min_d_norm[:,i:i+1]**2)/(2*(sigma**2)))

            # Normalizing the risk. As all points will have a different risk function due to TTC= time_to_dmin*delta_t changes
            # in function to time_to_dmin, the risk needs a normalization by the max value that the function can reach, which 
            # happens when d_min = 0.
            max_risk_possible = (k/(min_sigma)) * np.exp(-np.divide(0**2,2*((min_sigma)**2)))
            normalized_risk_factor = risk_probability/max_risk_possible
            risk_factor_all_points[:,i:i+1] = normalized_risk_factor
  

        risk_factor = np.max(risk_factor_all_points,axis=1,keepdims=True) 
        risk_robot_pts_idxs = np.transpose(np.argmax(risk_factor_all_points,axis=1))

        return risk_factor, risk_robot_pts_idxs, risk_factor_all_points
    

    def risk_to_colormap(self, risk_factor, cam_points):
        """
        Function to convert the risk factor to a colormap. When the risk factor is 1, which is the maximum risk, the color is red. Instead,
        when the risk factor is 0, which is the minimum risk, the color is blue. The other colors are interpolated between red and blue depending
        on the risk factor value.
        """
        risk_colormap = cv2.applyColorMap((risk_factor*255).astype(np.uint8), cv2.COLORMAP_JET)
       
        points = []
        for i in range(cam_points.shape[0]):
            x = cam_points[i, 0] 
            y = cam_points[i, 1]
            z = cam_points[i, 2] 
            r = risk_colormap[i][0][2]
            g = risk_colormap[i][0][1]
            b = risk_colormap[i][0][0]
            a = 255
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
            pt = [x, y, z, rgb]
            points.append(pt)

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgba', 12, PointField.UINT32, 1),
                ]

        header = Header()
        header.frame_id = "map"
        color_pointcloud = pc2.create_cloud(header, fields, points)
        color_pointcloud.is_dense = True

        self.risktocolormap_pub.publish(color_pointcloud)