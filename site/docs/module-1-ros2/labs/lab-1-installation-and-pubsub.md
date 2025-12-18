---
title: "Lab 1: Installation and Basic Publisher/Subscriber"
sidebar_position: 1
---

# Lab 1: Installation and Basic Publisher/Subscriber

## Overview

In this lab, you'll install ROS 2, create your first ROS 2 package, and implement a basic publisher-subscriber system. This foundational lab will help you understand the basic communication patterns in ROS 2.

## Prerequisites

Before starting this lab, ensure you have:
- Ubuntu 22.04 (Jammy) or equivalent Linux distribution
- Administrative access to install software
- Basic Python programming knowledge

## Installing ROS 2

First, add the ROS 2 repository and install the desktop version:

```bash
# Add the ROS 2 GPG key
sudo apt update && sudo apt install curl gnupg lsb-release
curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add the repository to your sources list
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Update package lists and install ROS 2
sudo apt update
sudo apt install ros-humble-desktop
```

Set up your environment:

```bash
# Source the ROS 2 setup script
source /opt/ros/humble/setup.bash

# Add to your bashrc to make it permanent
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

## Creating Your First Package

Create a workspace and your first package:

```bash
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Create a new package
ros2 pkg create --build-type ament_python my_robot_package
cd ~/ros2_ws/src/my_robot_package
```

## Basic Publisher Implementation

Create the publisher node in `my_robot_package/my_robot_package/publisher_member_function.py`:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Basic Subscriber Implementation

Create the subscriber node in `my_robot_package/my_robot_package/subscriber_member_function.py`:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Package Configuration

Update `setup.py` to make your nodes executable:

```python
from setuptools import find_packages, setup

package_name = 'my_robot_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Basic publisher and subscriber example',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = my_robot_package.publisher_member_function:main',
            'listener = my_robot_package.subscriber_member_function:main',
        ],
    },
)
```

## Building and Running

Build your package:

```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_package
source install/setup.bash
```

Run the publisher and subscriber in separate terminals:

Terminal 1:
```bash
ros2 run my_robot_package talker
```

Terminal 2:
```bash
ros2 run my_robot_package listener
```

You should see the publisher sending messages and the subscriber receiving them.

## Lab Exercise

1. Modify the publisher to send different types of messages (e.g., include timestamps or robot status)
2. Create multiple subscribers to the same topic and observe the behavior
3. Experiment with different QoS settings (reliability, history) and observe the effects

## Summary

In this lab, you've successfully installed ROS 2, created your first package, and implemented a basic publisher-subscriber system. You now understand the fundamental communication pattern in ROS 2 and have a working development environment.