---
title: "Lab 2: Custom Messages and Services"
sidebar_position: 2
---

# Lab 2: Custom Messages and Services

## Overview

In this lab, you'll create custom message types and implement a service-based communication system. This builds upon the basic publisher-subscriber pattern by introducing request-response communication, which is essential for many robotics applications.

## Prerequisites

Before starting this lab, you should have:
- Completed Lab 1: Installation and Basic Publisher/Subscriber
- A working ROS 2 installation
- Basic understanding of ROS 2 packages and nodes

## Creating Custom Messages

First, create directories for your message definitions:

```bash
mkdir -p ~/ros2_ws/src/my_robot_package/msg
mkdir -p ~/ros2_ws/src/my_robot_package/srv
```

Create a custom message in `~/ros2_ws/src/my_robot_package/msg/RobotStatus.msg`:

```
# RobotStatus.msg
string robot_name
int32 battery_level
bool is_moving
float64[] position  # [x, y, theta]
```

Create a custom service in `~/ros2_ws/src/my_robot_package/srv/MoveRobot.srv`:

```
# MoveRobot.srv
float64 x
float64 y
float64 theta
---
bool success
string message
```

## Updating package.xml

Add the required dependencies to `package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.0</version>
  <description>Custom messages and services example</description>
  <maintainer email="your.email@example.com">Your Name</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>

  <buildtool_depend>ament_python</buildtool_depend>
  <buildtool_depend>rosidl_default_generators</buildtool_depend>

  <exec_depend>rosidl_default_runtime</exec_depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <member_of_group>rosidl_interface_packages</member_of_group>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

## Updating setup.py

Update `setup.py` to include message generation:

```python
from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'my_robot_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),
        (os.path.join('share', package_name, 'srv'), glob('srv/*.srv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Custom messages and services example',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = my_robot_package.publisher_member_function:main',
            'listener = my_robot_package.subscriber_member_function:main',
        ],
    },
    # Add the following for message generation
    package_data={
        'my_robot_package': ['msg/*.msg', 'srv/*.srv']
    },
)
```

## Service Server Implementation

Create a service server in `my_robot_package/my_robot_package/service_server.py`:

```python
import rclpy
from rclpy.node import Node
from my_robot_package.srv import MoveRobot


class MoveRobotService(Node):

    def __init__(self):
        super().__init__('move_robot_service')
        self.srv = self.create_service(
            MoveRobot,
            'move_robot',
            self.move_robot_callback
        )

    def move_robot_callback(self, request, response):
        self.get_logger().info(
            f'Received request to move robot to x: {request.x}, y: {request.y}, theta: {request.theta}'
        )

        # Simulate robot movement
        # In a real robot, this would send commands to the robot's motion controller
        success = True
        message = f'Robot successfully moved to position ({request.x}, {request.y}, {request.theta})'

        response.success = success
        response.message = message

        return response


def main(args=None):
    rclpy.init(args=args)
    move_robot_service = MoveRobotService()
    rclpy.spin(move_robot_service)
    move_robot_service.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Service Client Implementation

Create a service client in `my_robot_package/my_robot_package/service_client.py`:

```python
import sys
import rclpy
from rclpy.node import Node
from my_robot_package.srv import MoveRobot


class MoveRobotClient(Node):

    def __init__(self):
        super().__init__('move_robot_client')
        self.cli = self.create_client(MoveRobot, 'move_robot')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = MoveRobot.Request()

    def send_request(self, x, y, theta):
        self.req.x = x
        self.req.y = y
        self.req.theta = theta
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def main(args=None):
    rclpy.init(args=args)

    if len(sys.argv) != 4:
        print('Usage: ros2 run my_robot_package move_robot_client <x> <y> <theta>')
        return

    x = float(sys.argv[1])
    y = float(sys.argv[2])
    theta = float(sys.argv[3])

    move_robot_client = MoveRobotClient()
    response = move_robot_client.send_request(x, y, theta)

    if response:
        move_robot_client.get_logger().info(
            f'Result: success={response.success}, message={response.message}'
        )
    else:
        move_robot_client.get_logger().info('Service call failed')

    move_robot_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Updating setup.py for New Executables

Add the new executables to `setup.py`:

```python
entry_points={
    'console_scripts': [
        'talker = my_robot_package.publisher_member_function:main',
        'listener = my_robot_package.subscriber_member_function:main',
        'service_server = my_robot_package.service_server:main',
        'service_client = my_robot_package.service_client:main',
    ],
},
```

## Building and Testing

Build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_package --cmake-clean-cache
source install/setup.bash
```

Run the service server in one terminal:

```bash
ros2 run my_robot_package service_server
```

In another terminal, call the service:

```bash
ros2 run my_robot_package service_client 1.0 2.0 1.57
```

## Lab Exercise

1. Create additional custom message types for different robot states (e.g., RobotTelemetry, SensorData)
2. Implement a service that returns robot status information
3. Create a client that calls multiple services in sequence
4. Add error handling to your service implementation

## Summary

In this lab, you've learned to create custom message and service types, implement service servers and clients, and understand the request-response communication pattern in ROS 2. This pattern is essential for operations that require acknowledgment or specific results, such as robot control commands or configuration requests.