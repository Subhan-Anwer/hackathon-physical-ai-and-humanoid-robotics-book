---
title: "Lab 3: Multi-Node Robot System"
sidebar_position: 3
---

# Lab 3: Multi-Node Robot System

## Overview

In this lab, you'll create a more complex system with multiple nodes that work together to simulate a robot with sensors and control systems. This lab demonstrates how to build complete robotic applications using multiple interconnected nodes.

## Prerequisites

Before starting this lab, you should have:
- Completed Lab 1 and Lab 2
- A working ROS 2 installation with your my_robot_package
- Understanding of topics, services, and custom message types

## Robot Control Node

Create `my_robot_package/my_robot_package/robot_controller.py`:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import math


class RobotController(Node):

    def __init__(self):
        super().__init__('robot_controller')

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.battery_publisher = self.create_publisher(Float32, 'battery_level', 10)

        # Subscribers
        self.scan_subscriber = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10
        )

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        # Robot state
        self.laser_data = None
        self.battery_level = 100.0
        self.obstacle_distance = float('inf')

        # Initialize battery level
        self.publish_battery()

    def scan_callback(self, msg):
        # Process laser scan to find closest obstacle
        if len(msg.ranges) > 0:
            valid_ranges = [r for r in msg.ranges if r > 0 and not math.isinf(r)]
            if valid_ranges:
                self.obstacle_distance = min(valid_ranges)
            else:
                self.obstacle_distance = float('inf')

    def control_loop(self):
        # Simple obstacle avoidance behavior
        msg = Twist()

        if self.obstacle_distance < 1.0:  # Obstacle within 1 meter
            # Stop and turn
            msg.linear.x = 0.0
            msg.angular.z = 0.5  # Turn right
        else:
            # Move forward
            msg.linear.x = 0.5
            msg.angular.z = 0.0

        # Simulate battery drain
        self.battery_level -= 0.01
        if self.battery_level < 0:
            self.battery_level = 0

        self.cmd_vel_publisher.publish(msg)
        self.publish_battery()

    def publish_battery(self):
        battery_msg = Float32()
        battery_msg.data = self.battery_level
        self.battery_publisher.publish(battery_msg)


def main(args=None):
    rclpy.init(args=args)
    robot_controller = RobotController()
    rclpy.spin(robot_controller)
    robot_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Sensor Simulator Node

Create `my_robot_package/my_robot_package/sensor_simulator.py`:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import math
import random


class SensorSimulator(Node):

    def __init__(self):
        super().__init__('sensor_simulator')

        self.publisher = self.create_publisher(LaserScan, 'scan', 10)
        self.timer = self.create_timer(0.1, self.publish_scan)

        # Laser scan parameters
        self.angle_min = -math.pi / 2
        self.angle_max = math.pi / 2
        self.angle_increment = math.pi / 180  # 1 degree increments
        self.scan_time = 0.1
        self.range_min = 0.1
        self.range_max = 10.0

    def publish_scan(self):
        msg = LaserScan()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'laser_frame'

        msg.angle_min = self.angle_min
        msg.angle_max = self.angle_max
        msg.angle_increment = self.angle_increment
        msg.time_increment = 0.0
        msg.scan_time = self.scan_time
        msg.range_min = self.range_min
        msg.range_max = self.range_max

        # Generate simulated laser ranges
        num_ranges = int((self.angle_max - self.angle_min) / self.angle_increment) + 1
        ranges = []

        for i in range(num_ranges):
            angle = self.angle_min + i * self.angle_increment

            # Simulate some obstacles
            distance = self.range_max

            # Place an obstacle in front of the robot
            if -0.5 < angle < 0.5:
                distance = 1.5 + random.uniform(-0.2, 0.2)

            # Add some random noise
            distance += random.uniform(-0.05, 0.05)

            ranges.append(max(self.range_min, min(distance, self.range_max)))

        msg.ranges = ranges
        msg.intensities = []  # No intensity data

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    sensor_simulator = SensorSimulator()
    rclpy.spin(sensor_simulator)
    sensor_simulator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Dashboard Node

Create `my_robot_package/my_robot_package/dashboard.py`:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32


class Dashboard(Node):

    def __init__(self):
        super().__init__('dashboard')

        self.subscription_cmd_vel = self.create_subscription(
            Twist, 'cmd_vel', self.cmd_vel_callback, 10
        )
        self.subscription_scan = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10
        )
        self.subscription_battery = self.create_subscription(
            Float32, 'battery_level', self.battery_callback, 10
        )

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.battery_level = 100.0
        self.closest_obstacle = float('inf')

    def cmd_vel_callback(self, msg):
        self.linear_velocity = msg.linear.x
        self.angular_velocity = msg.angular.z
        self.get_logger().info(
            f'Velocity: linear={self.linear_velocity:.2f}, angular={self.angular_velocity:.2f}'
        )

    def scan_callback(self, msg):
        if len(msg.ranges) > 0:
            valid_ranges = [r for r in msg.ranges if r > 0 and not float('inf')]
            if valid_ranges:
                self.closest_obstacle = min(valid_ranges)
            else:
                self.closest_obstacle = float('inf')

    def battery_callback(self, msg):
        self.battery_level = msg.data
        if self.battery_level < 20:
            self.get_logger().warn(f'Low battery: {self.battery_level:.1f}%')


def main(args=None):
    rclpy.init(args=args)
    dashboard = Dashboard()

    try:
        rclpy.spin(dashboard)
    except KeyboardInterrupt:
        pass

    dashboard.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Launch File

Create a launch file to run all nodes together. First, create a launch directory:

```bash
mkdir -p ~/ros2_ws/src/my_robot_package/launch
```

Create `~/ros2_ws/src/my_robot_package/launch/multi_robot_system.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='sensor_simulator',
            name='sensor_simulator',
            output='screen'
        ),
        Node(
            package='my_robot_package',
            executable='robot_controller',
            name='robot_controller',
            output='screen'
        ),
        Node(
            package='my_robot_package',
            executable='dashboard',
            name='dashboard',
            output='screen'
        )
    ])
```

## Updating setup.py for New Nodes

Add the new executables to `setup.py`:

```python
entry_points={
    'console_scripts': [
        'talker = my_robot_package.publisher_member_function:main',
        'listener = my_robot_package.subscriber_member_function:main',
        'service_server = my_robot_package.service_server:main',
        'service_client = my_robot_package.service_client:main',
        'robot_controller = my_robot_package.robot_controller:main',
        'sensor_simulator = my_robot_package.sensor_simulator:main',
        'dashboard = my_robot_package.dashboard:main',
    ],
},
```

## Building and Running the Multi-Node System

Build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_package
source install/setup.bash
```

Run all nodes together using the launch file:

```bash
ros2 launch my_robot_package multi_robot_system.py
```

## Lab Exercise

1. Add more sensor types to the system (e.g., IMU, camera)
2. Implement a more sophisticated control algorithm (e.g., PID controller)
3. Add a navigation node that plans paths to specific goals
4. Create a parameter server to configure robot behavior at runtime
5. Add logging and visualization of robot state

## Understanding the System Architecture

This multi-node system demonstrates several important ROS 2 concepts:

- **Modularity**: Each node has a specific responsibility
- **Decoupling**: Nodes communicate through topics rather than direct function calls
- **Scalability**: New nodes can be added without modifying existing ones
- **Robustness**: If one node fails, others can continue operating

The system architecture includes:
- **Sensor Simulator**: Provides simulated sensor data
- **Robot Controller**: Processes sensor data and generates control commands
- **Dashboard**: Monitors system state and provides feedback

## Summary

In this lab, you've built a complete multi-node robotic system that demonstrates how different components work together. You've learned to design modular systems, use launch files to coordinate multiple nodes, and implement a complete robot control architecture with sensors, control, and monitoring.