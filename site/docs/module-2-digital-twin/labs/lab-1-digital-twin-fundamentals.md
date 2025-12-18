---
title: "Lab 1: Digital Twin Fundamentals"
sidebar_position: 1
---

# Lab 1: Digital Twin Fundamentals

## Overview
This lab introduces the fundamental concepts of digital twins by creating a basic simulation with sensor integration and establishing a bridge with ROS 2.

## Objectives
- Create a basic digital twin simulation using simple sensors
- Implement a physical-digital bridge with ROS 2
- Visualize real-time data synchronization
- Compare different fidelity levels in simulation

## Prerequisites
- Basic understanding of ROS 2 concepts
- Installed ROS 2 environment (Humble Hawksbill or later)
- Basic knowledge of simulation environments

## Lab Setup
1. Create a new ROS 2 workspace for the digital twin project
2. Set up a basic publisher node that simulates sensor data
3. Create a subscriber node that represents the digital twin
4. Establish communication between physical and digital systems

## Implementation Steps

### Step 1: Create the Physical System Simulator
Create a ROS 2 node that simulates a physical system with sensors:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
import math
import random

class PhysicalSystemSimulator(Node):
    def __init__(self):
        super().__init__('physical_system_simulator')
        self.publisher_ = self.create_publisher(JointState, 'physical_joint_states', 10)
        self.timer = self.create_timer(0.1, self.publish_joint_states)  # 10Hz
        self.time = 0.0

    def publish_joint_states(self):
        msg = JointState()
        msg.name = ['joint1', 'joint2']
        self.time += 0.1
        msg.position = [math.sin(self.time), math.cos(self.time)]
        msg.velocity = [math.cos(self.time), -math.sin(self.time)]
        msg.effort = [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PhysicalSystemSimulator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 2: Create the Digital Twin Node
Create a ROS 2 node that receives the sensor data and maintains the digital twin:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
import math

class DigitalTwin(Node):
    def __init__(self):
        super().__init__('digital_twin')
        self.subscription = self.create_subscription(
            JointState,
            'physical_joint_states',
            self.listener_callback,
            10)
        self.marker_publisher = self.create_publisher(Marker, 'digital_twin_marker', 10)
        self.current_joint_states = None

    def listener_callback(self, msg):
        self.current_joint_states = msg
        self.get_logger().info(f'Received joint states: pos={msg.position}, vel={msg.velocity}')
        self.publish_visualization()

    def publish_visualization(self):
        if self.current_joint_states is not None:
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "digital_twin"
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Position based on joint values
            marker.pose.position.x = self.current_joint_states.position[0] * 0.5
            marker.pose.position.y = self.current_joint_states.position[1] * 0.5
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            marker.color.a = 1.0  # Don't forget to set the alpha!
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0

            self.marker_publisher.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = DigitalTwin()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 3: Create Launch File
Create a launch file to start both nodes:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='digital_twin_lab',
            executable='physical_system_simulator',
            name='physical_system_simulator',
            output='screen'
        ),
        Node(
            package='digital_twin_lab',
            executable='digital_twin',
            name='digital_twin',
            output='screen'
        )
    ])
```

### Step 4: Compare Fidelity Levels
Implement different fidelity models to compare performance:

1. **Low Fidelity Model**: Simple kinematic representation
2. **Medium Fidelity Model**: Includes basic dynamics
3. **High Fidelity Model**: Full dynamic simulation with noise modeling

## Visualization and Analysis
1. Use RViz2 to visualize the digital twin representation
2. Plot the synchronization between physical and digital systems
3. Analyze the delay and accuracy of data transmission

## Assessment Questions
1. How does the synchronization delay affect the digital twin's accuracy?
2. What are the trade-offs between different fidelity levels?
3. How would you implement bidirectional communication in this system?

## What You Learned
In this lab, you implemented a basic digital twin system with real-time data synchronization. You learned how to create both physical simulators and their digital counterparts, and how to visualize and analyze the synchronization between them.