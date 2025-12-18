---
title: "Lab 4: Actions and Navigation"
sidebar_position: 4
---

# Lab 4: Actions and Navigation

## Overview

In this lab, you'll implement a navigation system using ROS 2 actions, which are perfect for long-running tasks that provide feedback. Actions are ideal for navigation, manipulation, and other tasks that take time to complete and need to report progress.

## Prerequisites

Before starting this lab, you should have:
- Completed the previous labs
- A working ROS 2 installation with your my_robot_package
- Understanding of topics, services, and multi-node systems

## Creating Custom Action

First, create an action definition in `~/ros2_ws/src/my_robot_package/action/NavigateToPose.action`:

```
# NavigateToPose.action
# Goal: target pose
float64 x
float64 y
float64 theta
---
# Result: success information
bool success
string message
---
# Feedback: progress information
float64 distance_to_goal
string status
```

## Updating package.xml for Actions

Add action generation to `package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.0</version>
  <description>Complete ROS 2 system with actions</description>
  <maintainer email="your.email@example.com">Your Name</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>sensor_msgs</depend>

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

## Navigation Action Server

Create `my_robot_package/my_robot_package/navigation_server.py`:

```python
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from my_robot_package.action import NavigateToPose
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math
import time


class NavigationServer(Node):

    def __init__(self):
        super().__init__('navigation_server')

        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_subscriber = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10
        )

        self.current_position = [0.0, 0.0, 0.0]  # x, y, theta
        self.laser_data = None
        self.is_navigating = False

    def scan_callback(self, msg):
        self.laser_data = msg

    def goal_callback(self, goal_request):
        # Accept all goals
        self.get_logger().info('Received navigation goal')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        # Accept all cancel requests
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing navigation goal...')

        # Get goal parameters
        target_x = goal_handle.request.x
        target_y = goal_handle.request.y
        target_theta = goal_handle.request.theta

        # Navigation parameters
        linear_speed = 0.3
        angular_speed = 0.3
        tolerance = 0.1

        self.is_navigating = True

        feedback_msg = NavigateToPose.Feedback()
        result = NavigateToPose.Result()

        while self.is_navigating and not goal_handle.is_cancel_requested:
            # Calculate distance to goal
            dx = target_x - self.current_position[0]
            dy = target_y - self.current_position[1]
            distance_to_goal = math.sqrt(dx*dx + dy*dy)

            # Update feedback
            feedback_msg.distance_to_goal = distance_to_goal
            feedback_msg.status = f'Navigating: {distance_to_goal:.2f}m to goal'
            goal_handle.publish_feedback(feedback_msg)

            # Check if we're close enough
            if distance_to_goal < tolerance:
                self.get_logger().info('Reached goal position!')
                break

            # Simple proportional controller
            cmd_vel = Twist()

            # Angular control (turn toward target)
            target_angle = math.atan2(dy, dx)
            angle_diff = target_angle - self.current_position[2]

            # Normalize angle to [-pi, pi]
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi

            cmd_vel.angular.z = angular_speed * angle_diff
            if abs(cmd_vel.angular.z) > angular_speed:
                cmd_vel.angular.z = angular_speed if cmd_vel.angular.z > 0 else -angular_speed

            # Linear control (move forward if roughly aligned)
            if abs(angle_diff) < 0.2:  # If roughly facing the target
                cmd_vel.linear.x = min(linear_speed, distance_to_goal * 0.5)

            # Safety: stop if obstacle is detected
            if self.laser_data and min(self.laser_data.ranges) < 0.5:
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.0
                self.get_logger().warn('Obstacle detected, stopping navigation!')

            # Publish command
            self.cmd_vel_publisher.publish(cmd_vel)

            # Update current position (simulation)
            self.current_position[0] += cmd_vel.linear.x * 0.1 * math.cos(self.current_position[2])
            self.current_position[1] += cmd_vel.linear.x * 0.1 * math.sin(self.current_position[2])
            self.current_position[2] += cmd_vel.angular.z * 0.1

            # Sleep briefly
            time.sleep(0.1)

        # Stop the robot
        stop_msg = Twist()
        self.cmd_vel_publisher.publish(stop_msg)

        # Check if goal was canceled
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.success = False
            result.message = 'Goal was canceled'
            self.get_logger().info('Goal canceled')
        else:
            result.success = True
            result.message = 'Successfully reached goal position'
            self.get_logger().info('Goal succeeded')

        self.is_navigating = False
        return result


def main(args=None):
    rclpy.init(args=args)
    navigation_server = NavigationServer()
    rclpy.spin(navigation_server)
    navigation_server.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Navigation Action Client

Create `my_robot_package/my_robot_package/navigation_client.py`:

```python
import sys
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from my_robot_package.action import NavigateToPose


class NavigationClient(Node):

    def __init__(self):
        super().__init__('navigation_client')
        self._action_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose')

    def send_goal(self, x, y, theta):
        goal_msg = NavigateToPose.Goal()
        goal_msg.x = x
        goal_msg.y = y
        goal_msg.theta = theta

        self.get_logger().info(f'Sending navigation goal: ({x}, {y}, {theta})')

        # Wait for the action server to be available
        self._action_client.wait_for_server()

        # Send the goal
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Feedback: {feedback.status}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.success}, {result.message}')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    if len(sys.argv) != 4:
        print('Usage: ros2 run my_robot_package navigation_client <x> <y> <theta>')
        return

    x = float(sys.argv[1])
    y = float(sys.argv[2])
    theta = float(sys.argv[3])

    action_client = NavigationClient()

    action_client.send_goal(x, y, theta)

    try:
        rclpy.spin(action_client)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
```

## Updating setup.py for Actions

Update `setup.py` to include actions:

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
        (os.path.join('share', package_name, 'action'), glob('action/*.action')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Complete ROS 2 system with actions',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = my_robot_package.publisher_member_function:main',
            'listener = my_robot_package.subscriber_member_function:main',
            'service_server = my_robot_package.service_server:main',
            'service_client = my_robot_package.service_client:main',
            'robot_controller = my_robot_package.robot_controller:main',
            'sensor_simulator = my_robot_package.sensor_simulator:main',
            'dashboard = my_robot_package.dashboard:main',
            'navigation_server = my_robot_package.navigation_server:main',
            'navigation_client = my_robot_package.navigation_client:main',
        ],
    },
    package_data={
        'my_robot_package': ['msg/*.msg', 'srv/*.srv', 'action/*.action']
    },
)
```

## Building and Testing the Navigation System

Build the complete package:

```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_package --cmake-clean-cache
source install/setup.bash
```

Run the navigation server:

```bash
ros2 run my_robot_package navigation_server
```

In another terminal, send a navigation goal:

```bash
ros2 run my_robot_package navigation_client 2.0 2.0 0.0
```

## Lab Exercise

1. Enhance the navigation system with a more sophisticated path planning algorithm
2. Add support for multiple simultaneous navigation goals
3. Implement obstacle avoidance with dynamic replanning
4. Create a simple GUI client for the navigation action
5. Add recovery behaviors when navigation fails

## Understanding Actions vs Services vs Topics

Actions provide several advantages over services for long-running tasks:

- **Feedback**: Continuous updates on progress
- **Cancelation**: Ability to cancel long-running tasks
- **Preemption**: Ability to replace current goal with new one
- **State Management**: Built-in state machine for complex operations

## Summary

In this lab, you've implemented a complete navigation system using ROS 2 actions. You've learned how actions differ from services and topics, and when to use each communication pattern. Actions are particularly useful for tasks that take time to complete and need to provide feedback, making them ideal for navigation, manipulation, and other robotics applications.