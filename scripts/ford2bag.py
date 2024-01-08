#!/usr/bin/python3
# SPDX-License-Identifier: BSD-2-Clause
import re
import os
import sys
import struct
import numpy
import scipy.io

import rosbag2_py
import progressbar

from rclpy.serialization import serialize_message
from sensor_msgs.msg import NavSatFix, NavSatStatus, PointCloud2
from geographic_msgs.msg import GeoPointStamped
from sensor_msgs_py import point_cloud2


def gps2navsat(filename, writer):
    with open(filename, 'rb') as file:
        try:
            while True:
                data = struct.unpack('qddd', file.read(8*4))
                time = data[0]
                local = data[1:]
                lat_lon_el_theta = struct.unpack('dddd', file.read(8*4))
                gps_cov = struct.unpack('dddddddddddddddd', file.read(8*16))

                if abs(lat_lon_el_theta[0]) < 1e-1:
                    continue

                navsat = NavSatFix()
                navsat.header.frame_id = 'gps'
                navsat.header.stamp.sec = time // 1000000
                navsat.header.stamp.nanosec = time % 1000000 * 1000
                navsat.status.status = NavSatStatus.STATUS_FIX
                navsat.status.service = NavSatStatus.SERVICE_GPS

                navsat.latitude = lat_lon_el_theta[0]
                navsat.longitude = lat_lon_el_theta[1]
                navsat.altitude = lat_lon_el_theta[2]

                navsat.position_covariance = numpy.array(gps_cov).reshape(4, 4)[:3, :3].flatten().tolist()
                navsat.position_covariance_type = NavSatFix.COVARIANCE_TYPE_KNOWN

                writer.write('/gps/fix', serialize_message(navsat), time * 1000)

                geopoint = GeoPointStamped()
                geopoint.header = navsat.header
                geopoint.position.latitude = lat_lon_el_theta[0]
                geopoint.position.longitude = lat_lon_el_theta[1]
                geopoint.position.altitude = lat_lon_el_theta[2]

                writer.write('/gps/geopoint', serialize_message(geopoint), time * 1000)
        except:
            pass


def mat2pointcloud(filename, writer):
    m = scipy.io.loadmat(filename)
    scan = numpy.transpose(m['SCAN']['XYZ'][0][0]).astype(numpy.float32)
    stamp = int(m['SCAN']['timestamp_laser'][0][0][0][0])

    cloud = PointCloud2()
    cloud.header.stamp.sec = stamp // 1000000
    cloud.header.stamp.nanosec = stamp % 1000000 * 1000
    cloud.header.frame_id = 'velodyne'
    cloud = point_cloud2.create_cloud_xyz32(cloud.header, scan)
    writer.write('/velodyne_points', serialize_message(cloud), stamp * 1000)


def main():
    if len(sys.argv) != 3:
        print('Usage: {} <IJRR Folder> <Bag Folder>'.format(sys.argv[0]))
        sys.exit(1)
    IJRR_FOLDER_PATH = sys.argv[1]
    BAG_FOLDER_PATH = sys.argv[2]

    writer = rosbag2_py.SequentialWriter()
    storage_options = rosbag2_py.StorageOptions(uri=BAG_FOLDER_PATH, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    writer.open(storage_options, converter_options)

    fix_topic = rosbag2_py.TopicMetadata(name='/gps/fix', type='sensor_msgs/msg/NavSatFix', serialization_format='cdr')
    geopoint_topic = rosbag2_py.TopicMetadata(name='/gps/geopoint', type='geographic_msgs/msg/GeoPointStamped', serialization_format='cdr')
    velodyne_topic = rosbag2_py.TopicMetadata(name='/velodyne_points', type='sensor_msgs/msg/PointCloud2', serialization_format='cdr')
    writer.create_topic(fix_topic)
    writer.create_topic(geopoint_topic)
    writer.create_topic(velodyne_topic)

    filenames = sorted([os.path.join(IJRR_FOLDER_PATH, 'SCANS', x) for x in os.listdir(os.path.join(IJRR_FOLDER_PATH, 'SCANS')) if re.match('Scan[0-9]*\.mat', x)])

    progress = progressbar.ProgressBar(maxval=len(filenames))
    progress.start()
    # pub = rclpy.Publisher('/velodyne_points', PointCloud2, queue_size=32)

    gps2navsat(os.path.join(IJRR_FOLDER_PATH, 'GPS.log'), writer)
    for i, filename in enumerate(filenames):
        progress.update(i)
        mat2pointcloud(filename, writer)
        # if pub.get_num_connections():
            # pub.publish(cloud)
    progress.finish()


if __name__ == '__main__':
    main()
