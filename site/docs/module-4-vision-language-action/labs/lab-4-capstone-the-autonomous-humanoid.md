---
title: "Lab 4: Capstone - The Autonomous Humanoid"
sidebar_position: 4
---

# Lab 4: Capstone - The Autonomous Humanoid

## Overview
This capstone lab integrates all components learned in Module 4 to create an autonomous humanoid robot system. The robot will receive voice commands, plan a path using cognitive planning, navigate obstacles, identify objects using vision-language perception, and manipulate them. This comprehensive project demonstrates the complete Vision-Language-Action (VLA) pipeline in a realistic humanoid robotics scenario.

## Objectives
- Design an end-to-end humanoid robot system
- Integrate voice commands, cognitive planning, and perception
- Implement complete task execution pipeline
- Test system performance with complex multi-step commands

## Prerequisites
- Completed all previous labs in Module 4
- Working knowledge of ROS 2, Nav2, and MoveIt2
- Understanding of speech recognition, LLM integration, and computer vision
- Access to simulation environment (Gazebo/Isaac Sim) or physical robot

## Lab Setup

### 1. Install Additional Dependencies
Install packages required for the capstone project:

```bash
# Navigation and manipulation dependencies
sudo apt update
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
sudo apt install ros-humble-moveit ros-humble-moveit-ros ros-humble-moveit-ros-planners
sudo apt install ros-humble-moveit-ros-perception

# Additional Python packages
pip install transforms3d
pip install pyquaternion
pip install control
```

### 2. Create Capstone Package
Create a new ROS 2 package for the capstone project:

```bash
cd ~/voice_command_ws/src
ros2 pkg create --build-type ament_python humanoid_capstone
cd humanoid_capstone
```

### 3. Set Up Simulation Environment (if using simulation)
For simulation, ensure you have Gazebo or Isaac Sim configured with a humanoid robot model.

## Implementation Steps

### Step 1: Create the Humanoid Orchestrator Node
Create the main orchestrator that coordinates all VLA components:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from builtin_interfaces.msg import Duration
import threading
import time
import json
from enum import Enum
from typing import Dict, Any, Optional

class RobotState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    PLANNING = "planning"
    EXECUTING = "executing"
    ERROR = "error"
    COMPLETED = "completed"

class HumanoidOrchestratorNode(Node):
    def __init__(self):
        super().__init__('humanoid_orchestrator_node')

        # Robot state management
        self.current_state = RobotState.IDLE
        self.current_task = None
        self.robot_pose = None
        self.navigation_active = False
        self.manipulation_active = False

        # Create subscribers
        self.voice_command_sub = self.create_subscription(
            String,
            'vl_command',
            self.voice_command_callback,
            10
        )

        self.planning_result_sub = self.create_subscription(
            String,
            'generated_plan',
            self.planning_result_callback,
            10
        )

        self.execution_status_sub = self.create_subscription(
            String,
            'executor_status',
            self.execution_status_callback,
            10
        )

        self.odometry_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odometry_callback,
            10
        )

        # Create publishers
        self.state_pub = self.create_publisher(
            String,
            'robot_state',
            10
        )

        self.plan_request_pub = self.create_publisher(
            String,
            'natural_language_command',
            10
        )

        self.status_pub = self.create_publisher(
            String,
            'capstone_status',
            10
        )

        self.task_complete_pub = self.create_publisher(
            Bool,
            'task_complete',
            10
        )

        # Task queue for handling multiple commands
        self.task_queue = []
        self.task_queue_lock = threading.Lock()

        # Timer for state monitoring
        self.state_timer = self.create_timer(0.1, self.state_monitor)

        self.get_logger().info("Humanoid Orchestrator Node initialized")

    def voice_command_callback(self, msg: String):
        """Handle incoming voice commands"""
        command = msg.data
        self.get_logger().info(f"Received voice command: {command}")

        # Add command to task queue
        with self.task_queue_lock:
            self.task_queue.append({
                'command': command,
                'timestamp': time.time(),
                'status': 'pending'
            })

        # Update state and trigger processing
        self.current_state = RobotState.PROCESSING
        self.publish_state()

        # Request plan generation
        plan_request = String()
        plan_request.data = command
        self.plan_request_pub.publish(plan_request)

    def planning_result_callback(self, msg: String):
        """Handle planning results"""
        try:
            plan_data = json.loads(msg.data)
            self.get_logger().info(f"Received plan with {len(plan_data.get('steps', []))} steps")

            # Transition to execution state
            self.current_state = RobotState.EXECUTING
            self.publish_state()

            # Publish status
            status_msg = String()
            status_msg.data = f"Starting execution of plan with {len(plan_data.get('steps', []))} steps"
            self.status_pub.publish(status_msg)

        except json.JSONDecodeError:
            self.get_logger().error("Invalid JSON in planning result")
            self.current_state = RobotState.ERROR
            self.publish_state()

    def execution_status_callback(self, msg: String):
        """Handle execution status updates"""
        status_text = msg.data
        self.get_logger().info(f"Execution status: {status_text}")

        # Check if execution is complete
        if "Plan execution completed" in status_text:
            self.current_state = RobotState.COMPLETED
            self.publish_state()

            # Publish task completion
            complete_msg = Bool()
            complete_msg.data = True
            self.task_complete_pub.publish(complete_msg)

            # Return to idle after a short delay
            self.create_timer(2.0, self.return_to_idle)

    def odometry_callback(self, msg: Odometry):
        """Update robot pose from odometry"""
        self.robot_pose = msg.pose.pose

    def state_monitor(self):
        """Monitor and update robot state"""
        # Publish current state
        state_msg = String()
        state_msg.data = self.current_state.value
        self.state_pub.publish(state_msg)

        # Process task queue if idle and tasks available
        if self.current_state == RobotState.IDLE:
            with self.task_queue_lock:
                if self.task_queue:
                    task = self.task_queue.pop(0)
                    self.get_logger().info(f"Processing queued task: {task['command']}")
                    self.current_state = RobotState.PROCESSING
                    self.publish_state()

                    # Request plan for the queued task
                    plan_request = String()
                    plan_request.data = task['command']
                    self.plan_request_pub.publish(plan_request)

    def return_to_idle(self):
        """Return to idle state after task completion"""
        self.current_state = RobotState.IDLE
        self.publish_state()

        # Publish completion status
        status_msg = String()
        status_msg.data = "Task completed, returned to idle state"
        self.status_pub.publish(status_msg)

        self.get_logger().info("Robot returned to idle state")

    def publish_state(self):
        """Publish current robot state"""
        state_msg = String()
        state_msg.data = self.current_state.value
        self.state_pub.publish(state_msg)

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidOrchestratorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Humanoid Orchestrator Node")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 2: Create the Humanoid Navigation Node
Create a specialized navigation node for humanoid-specific navigation:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import String
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs
import numpy as np
import math
from typing import List, Tuple

class HumanoidNavigationNode(Node):
    def __init__(self):
        super().__init__('humanoid_navigation_node')

        # Create action client for navigation
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Create TF buffer and listener for coordinate transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Create subscribers
        self.navigation_goal_sub = self.create_subscription(
            PoseStamped,
            'navigation_goal',
            self.navigation_goal_callback,
            10
        )

        self.laser_scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.laser_scan_callback,
            10
        )

        # Create publishers
        self.navigation_status_pub = self.create_publisher(
            String,
            'navigation_status',
            10
        )

        # Navigation parameters for humanoid
        self.humanoid_navigation_params = {
            'max_linear_speed': 0.5,      # m/s
            'max_angular_speed': 0.5,     # rad/s
            'min_distance_to_obstacle': 0.5,  # m
            'inflation_radius': 0.8,      # m
            'footprint_padding': 0.3      # m
        }

        # Store current navigation state
        self.current_goal = None
        self.navigation_active = False
        self.safe_to_navigate = True

        self.get_logger().info("Humanoid Navigation Node initialized")

    def navigation_goal_callback(self, msg: PoseStamped):
        """Handle navigation goals for humanoid"""
        self.get_logger().info(f"Received navigation goal: ({msg.pose.position.x}, {msg.pose.position.y})")

        # Check if navigation server is available
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Navigation server not available")
            status_msg = String()
            status_msg.data = "Navigation server not available"
            self.navigation_status_pub.publish(status_msg)
            return

        # Create navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = msg  # Use the received pose as the goal

        # Add humanoid-specific navigation parameters
        goal_msg.behavior_tree = "navigate_w_replanning_and_recovery"  # Use appropriate BT

        # Send navigation goal
        self.current_goal = goal_msg
        self.navigation_active = True

        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.navigation_done_callback)

        status_msg = String()
        status_msg.data = f"Navigating to ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})"
        self.navigation_status_pub.publish(status_msg)

    def navigation_done_callback(self, future):
        """Handle navigation completion"""
        try:
            goal_handle = future.result()
            if goal_handle.accepted:
                self.get_logger().info("Navigation goal accepted")
                result_future = goal_handle.get_result_async()
                result_future.add_done_callback(self.navigation_result_callback)
            else:
                self.get_logger().error("Navigation goal rejected")
                self.navigation_active = False
                status_msg = String()
                status_msg.data = "Navigation goal rejected"
                self.navigation_status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f"Navigation failed: {str(e)}")
            self.navigation_active = False
            status_msg = String()
            status_msg.data = f"Navigation failed: {str(e)}"
            self.navigation_status_pub.publish(status_msg)

    def navigation_result_callback(self, future):
        """Handle navigation result"""
        try:
            result = future.result().result
            self.navigation_active = False

            status_msg = String()
            if result:
                status_msg.data = "Navigation completed successfully"
                self.get_logger().info("Navigation completed successfully")
            else:
                status_msg.data = "Navigation failed"
                self.get_logger().error("Navigation failed")

            self.navigation_status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f"Navigation result error: {str(e)}")
            self.navigation_active = False
            status_msg = String()
            status_msg.data = f"Navigation result error: {str(e)}"
            self.navigation_status_pub.publish(status_msg)

    def laser_scan_callback(self, msg: LaserScan):
        """Process laser scan data for obstacle detection"""
        # Convert laser scan to obstacle distances
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        valid_ranges = [r for r in msg.ranges if msg.range_min < r < msg.range_max]

        if not valid_ranges:
            return

        # Check for obstacles in humanoid's path
        min_distance = min(valid_ranges) if valid_ranges else float('inf')

        if min_distance < self.humanoid_navigation_params['min_distance_to_obstacle']:
            self.safe_to_navigate = False
            self.get_logger().warn(f"Obstacle detected at {min_distance:.2f}m, stopping navigation")

            # If currently navigating, consider pausing or replanning
            if self.navigation_active:
                status_msg = String()
                status_msg.data = f"Obstacle detected at {min_distance:.2f}m, replanning route"
                self.navigation_status_pub.publish(status_msg)
        else:
            self.safe_to_navigate = True

    def transform_pose(self, pose: PoseStamped, target_frame: str) -> Optional[PoseStamped]:
        """Transform pose to target frame"""
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                pose.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            transformed_pose = tf2_geometry_msgs.do_transform_pose(pose, transform)
            transformed_pose.header.frame_id = target_frame
            return transformed_pose
        except TransformException as e:
            self.get_logger().error(f"Transform failed: {str(e)}")
            return None

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidNavigationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Humanoid Navigation Node")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 3: Create the Humanoid Manipulation Node
Create a node for humanoid-specific manipulation tasks:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import JointState
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK, GetPositionFK
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
import numpy as np
import math
from typing import List, Dict, Any

class HumanoidManipulationNode(Node):
    def __init__(self):
        super().__init__('humanoid_manipulation_node')

        # Create action client for MoveIt
        self.move_group_client = ActionClient(self, MoveGroup, 'move_group')
        self.trajectory_client = ActionClient(self, FollowJointTrajectory, 'joint_trajectory_controller/follow_joint_trajectory')

        # Create subscribers
        self.manipulation_command_sub = self.create_subscription(
            String,
            'manipulation_command',
            self.manipulation_command_callback,
            10
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        # Create publishers
        self.manipulation_status_pub = self.create_publisher(
            String,
            'manipulation_status',
            10
        )

        # Service clients
        self.ik_client = self.create_client(GetPositionIK, 'compute_ik')
        self.fk_client = self.create_client(GetPositionFK, 'compute_fk')

        # Store current joint states
        self.current_joint_states = {}
        self.manipulation_active = False

        # Humanoid-specific parameters
        self.humanoid_params = {
            'arm_joints': ['left_shoulder_joint', 'left_elbow_joint', 'left_wrist_joint',
                          'right_shoulder_joint', 'right_elbow_joint', 'right_wrist_joint'],
            'gripper_joints': ['left_gripper_joint', 'right_gripper_joint'],
            'arm_links': ['left_upper_arm', 'left_forearm', 'left_hand',
                         'right_upper_arm', 'right_forearm', 'right_hand'],
            'max_reach': 1.0,  # meters
            'min_grasp_distance': 0.1,  # meters
            'grasp_tolerance': 0.05  # meters
        }

        self.get_logger().info("Humanoid Manipulation Node initialized")

    def manipulation_command_callback(self, msg: String):
        """Handle manipulation commands"""
        try:
            command_data = json.loads(msg.data)
            command_type = command_data.get('command_type')

            self.get_logger().info(f"Received manipulation command: {command_type}")

            if command_type == 'grasp_object':
                self.execute_grasp_command(command_data)
            elif command_type == 'place_object':
                self.execute_place_command(command_data)
            elif command_type == 'move_arm':
                self.execute_move_arm_command(command_data)
            elif command_type == 'open_gripper':
                self.execute_gripper_command(command_data, 'open')
            elif command_type == 'close_gripper':
                self.execute_gripper_command(command_data, 'close')
            else:
                self.get_logger().error(f"Unknown manipulation command: {command_type}")

        except json.JSONDecodeError:
            self.get_logger().error("Invalid JSON in manipulation command")

    def joint_state_callback(self, msg: JointState):
        """Update current joint states"""
        for i, name in enumerate(msg.name):
            if name in self.humanoid_params['arm_joints'] or name in self.humanoid_params['gripper_joints']:
                self.current_joint_states[name] = msg.position[i]

    def execute_grasp_command(self, command_data: Dict[str, Any]):
        """Execute grasp object command"""
        self.manipulation_active = True
        object_pose = command_data.get('object_pose')
        arm = command_data.get('arm', 'right')  # default to right arm

        if not object_pose:
            self.get_logger().error("No object pose provided for grasp command")
            self.manipulation_active = False
            return

        self.get_logger().info(f"Attempting to grasp object at ({object_pose['x']}, {object_pose['y']}, {object_pose['z']}) with {arm} arm")

        # Plan approach trajectory
        approach_pose = self.calculate_approach_pose(object_pose, arm)

        # Move to approach pose
        if self.move_to_pose(approach_pose, arm):
            # Move to grasp pose
            grasp_pose = self.calculate_grasp_pose(object_pose, arm)
            if self.move_to_pose(grasp_pose, arm):
                # Close gripper
                self.close_gripper(arm)

                # Lift object slightly
                lift_pose = self.calculate_lift_pose(grasp_pose, arm)
                self.move_to_pose(lift_pose, arm)

                status_msg = String()
                status_msg.data = f"Successfully grasped object with {arm} arm"
                self.manipulation_status_pub.publish(status_msg)
            else:
                self.get_logger().error("Failed to reach grasp pose")
        else:
            self.get_logger().error("Failed to reach approach pose")

        self.manipulation_active = False

    def execute_place_command(self, command_data: Dict[str, Any]):
        """Execute place object command"""
        self.manipulation_active = True
        target_pose = command_data.get('target_pose')
        arm = command_data.get('arm', 'right')

        if not target_pose:
            self.get_logger().error("No target pose provided for place command")
            self.manipulation_active = False
            return

        self.get_logger().info(f"Attempting to place object at ({target_pose['x']}, {target_pose['y']}, {target_pose['z']}) with {arm} arm")

        # Plan approach trajectory to placement location
        approach_pose = self.calculate_approach_pose(target_pose, arm)

        if self.move_to_pose(approach_pose, arm):
            # Move to placement pose
            place_pose = self.calculate_place_pose(target_pose, arm)
            if self.move_to_pose(place_pose, arm):
                # Open gripper to release object
                self.open_gripper(arm)

                # Retract arm
                retract_pose = self.calculate_retract_pose(place_pose, arm)
                self.move_to_pose(retract_pose, arm)

                status_msg = String()
                status_msg.data = f"Successfully placed object with {arm} arm"
                self.manipulation_status_pub.publish(status_msg)
            else:
                self.get_logger().error("Failed to reach place pose")
        else:
            self.get_logger().error("Failed to reach approach pose")

        self.manipulation_active = False

    def calculate_approach_pose(self, object_pose: Dict, arm: str) -> Pose:
        """Calculate approach pose for grasping"""
        approach_offset = 0.15  # 15cm from object
        approach_pose = Pose()

        # Calculate approach direction based on object orientation and preferred direction
        approach_pose.position.x = object_pose['x'] - approach_offset
        approach_pose.position.y = object_pose['y']
        approach_pose.position.z = object_pose['z'] + 0.05  # slightly above object

        # Set orientation to face the object
        approach_pose.orientation = self.calculate_orientation_to_object(object_pose, arm)

        return approach_pose

    def calculate_grasp_pose(self, object_pose: Dict, arm: str) -> Pose:
        """Calculate final grasp pose"""
        grasp_pose = Pose()
        grasp_pose.position.x = object_pose['x']
        grasp_pose.position.y = object_pose['y']
        grasp_pose.position.z = object_pose['z']

        # Use same orientation as approach
        grasp_pose.orientation = self.calculate_orientation_to_object(object_pose, arm)

        return grasp_pose

    def calculate_place_pose(self, target_pose: Dict, arm: str) -> Pose:
        """Calculate placement pose"""
        place_pose = Pose()
        place_pose.position.x = target_pose['x']
        place_pose.position.y = target_pose['y']
        place_pose.position.z = target_pose['z']

        place_pose.orientation = self.calculate_orientation_for_placement(target_pose, arm)

        return place_pose

    def calculate_orientation_to_object(self, object_pose: Dict, arm: str) -> Quaternion:
        """Calculate orientation to face an object"""
        # Simple implementation - in practice, this would be more sophisticated
        # For now, return a default orientation
        return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

    def calculate_orientation_for_placement(self, target_pose: Dict, arm: str) -> Quaternion:
        """Calculate orientation for object placement"""
        # Simple implementation - in practice, this would consider the placement surface
        return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

    def calculate_lift_pose(self, grasp_pose: Pose, arm: str) -> Pose:
        """Calculate pose after grasping to lift object"""
        lift_pose = Pose()
        lift_pose.position.x = grasp_pose.position.x
        lift_pose.position.y = grasp_pose.position.y
        lift_pose.position.z = grasp_pose.position.z + 0.1  # lift 10cm
        lift_pose.orientation = grasp_pose.orientation
        return lift_pose

    def calculate_retract_pose(self, place_pose: Pose, arm: str) -> Pose:
        """Calculate retraction pose after placing"""
        retract_pose = Pose()
        retract_pose.position.x = place_pose.position.x - 0.1  # move back 10cm
        retract_pose.position.y = place_pose.position.y
        retract_pose.position.z = place_pose.position.z + 0.05  # lift slightly
        retract_pose.orientation = place_pose.orientation
        return retract_pose

    def move_to_pose(self, target_pose: Pose, arm: str) -> bool:
        """Move arm to target pose using MoveIt"""
        # In a real implementation, this would use MoveIt's planning and execution
        # For this example, we'll simulate the movement
        self.get_logger().info(f"Moving {arm} arm to target pose")

        # Simulate movement time
        import time
        time.sleep(2.0)

        return True  # Simulate success

    def close_gripper(self, arm: str):
        """Close the gripper"""
        self.get_logger().info(f"Closing {arm} gripper")
        # In a real implementation, this would send commands to the gripper controller

    def open_gripper(self, arm: str):
        """Open the gripper"""
        self.get_logger().info(f"Opening {arm} gripper")
        # In a real implementation, this would send commands to the gripper controller

    def execute_move_arm_command(self, command_data: Dict[str, Any]):
        """Execute move arm command"""
        arm = command_data.get('arm', 'right')
        target_pose = command_data.get('target_pose')

        if target_pose:
            success = self.move_to_pose(self.dict_to_pose(target_pose), arm)
            status_msg = String()
            status_msg.data = f"Move arm command {'succeeded' if success else 'failed'}"
            self.manipulation_status_pub.publish(status_msg)

    def execute_gripper_command(self, command_data: Dict[str, Any], action: str):
        """Execute gripper command"""
        arm = command_data.get('arm', 'right')

        if action == 'close':
            self.close_gripper(arm)
        else:
            self.open_gripper(arm)

        status_msg = String()
        status_msg.data = f"Gripper command ({action}) executed for {arm} arm"
        self.manipulation_status_pub.publish(status_msg)

    def dict_to_pose(self, pose_dict: Dict) -> Pose:
        """Convert dictionary to Pose message"""
        pose = Pose()
        pose.position.x = pose_dict.get('x', 0.0)
        pose.position.y = pose_dict.get('y', 0.0)
        pose.position.z = pose_dict.get('z', 0.0)
        pose.orientation.x = pose_dict.get('qx', 0.0)
        pose.orientation.y = pose_dict.get('qy', 0.0)
        pose.orientation.z = pose_dict.get('qz', 0.0)
        pose.orientation.w = pose_dict.get('qw', 1.0)
        return pose

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidManipulationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Humanoid Manipulation Node")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 4: Create the Task Manager Node
Create a node to manage complex multi-step tasks:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from cognitive_planning_interfaces.msg import Plan, PlanStep
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Time
import json
import time
from typing import List, Dict, Any
from enum import Enum

class TaskState(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskManagerNode(Node):
    def __init__(self):
        super().__init__('task_manager_node')

        # Create subscribers
        self.task_request_sub = self.create_subscription(
            String,
            'task_request',
            self.task_request_callback,
            10
        )

        self.vision_detections_sub = self.create_subscription(
            Detection2DArray,
            'vision_language_detections',
            self.vision_detections_callback,
            10
        )

        self.plan_sub = self.create_subscription(
            Plan,
            'generated_plan',
            self.plan_callback,
            10
        )

        self.execution_status_sub = self.create_subscription(
            String,
            'executor_status',
            self.execution_status_callback,
            10
        )

        # Create publishers
        self.task_status_pub = self.create_publisher(
            String,
            'task_status',
            10
        )

        self.navigation_goal_pub = self.create_publisher(
            PoseStamped,
            'navigation_goal',
            10
        )

        self.manipulation_command_pub = self.create_publisher(
            String,
            'manipulation_command',
            10
        )

        self.task_complete_pub = self.create_publisher(
            Bool,
            'task_complete',
            10
        )

        # Task management
        self.active_tasks = {}
        self.object_detections = {}
        self.current_plan = None
        self.current_plan_step = 0

        self.get_logger().info("Task Manager Node initialized")

    def task_request_callback(self, msg: String):
        """Handle new task requests"""
        try:
            task_data = json.loads(msg.data)
            task_id = task_data.get('task_id', str(int(time.time())))
            task_description = task_data.get('description', '')

            self.get_logger().info(f"Received task request: {task_description} (ID: {task_id})")

            # Create new task
            new_task = {
                'id': task_id,
                'description': task_description,
                'state': TaskState.PENDING,
                'created_time': self.get_clock().now().to_msg(),
                'steps_completed': 0,
                'total_steps': 0,
                'requirements': task_data.get('requirements', {}),
                'results': {}
            }

            self.active_tasks[task_id] = new_task
            self.publish_task_status(task_id, f"Task {task_id} created and pending")

            # Process the task
            self.process_task(task_id, task_data)

        except json.JSONDecodeError:
            self.get_logger().error("Invalid JSON in task request")

    def vision_detections_callback(self, msg: Detection2DArray):
        """Handle vision detections"""
        detections = []
        for detection in msg.detections:
            if detection.results:
                # Get the best result
                best_result = max(detection.results, key=lambda r: r.score)
                detections.append({
                    'label': best_result.id,
                    'confidence': best_result.score,
                    'bbox_center_x': detection.bbox.center.x,
                    'bbox_center_y': detection.bbox.center.y,
                    'bbox_size_x': detection.bbox.size_x,
                    'bbox_size_y': detection.bbox.size_y
                })

        # Store detections for later use
        self.object_detections = {
            'timestamp': self.get_clock().now().to_msg(),
            'detections': detections
        }

        self.get_logger().info(f"Stored {len(detections)} object detections")

    def plan_callback(self, msg: Plan):
        """Handle received plans"""
        self.current_plan = msg
        self.current_plan_step = 0

        self.get_logger().info(f"Received plan with {len(msg.steps)} steps")

    def execution_status_callback(self, msg: String):
        """Handle execution status updates"""
        status_text = msg.data
        self.get_logger().info(f"Execution status: {status_text}")

        # Check if current plan step is complete
        if "Executing step" in status_text and "completed" in status_text:
            self.current_plan_step += 1

            if self.current_plan and self.current_plan_step >= len(self.current_plan.steps):
                # Plan completed
                self.get_logger().info("Task plan completed")

                # Publish task completion
                complete_msg = Bool()
                complete_msg.data = True
                self.task_complete_pub.publish(complete_msg)

    def process_task(self, task_id: str, task_data: Dict[str, Any]):
        """Process a task by generating and executing a plan"""
        # Update task state
        self.active_tasks[task_id]['state'] = TaskState.IN_PROGRESS
        self.publish_task_status(task_id, f"Processing task: {task_data.get('description', '')}")

        # Request plan generation
        plan_request = String()
        plan_request.data = task_data.get('description', '')

        plan_request_publisher = self.create_publisher(String, 'natural_language_command', 10)
        plan_request_publisher.publish(plan_request)

        self.get_logger().info(f"Requested plan generation for task {task_id}")

    def publish_task_status(self, task_id: str, status: str):
        """Publish task status"""
        status_msg = String()
        status_msg.data = json.dumps({
            'task_id': task_id,
            'status': status,
            'timestamp': self.get_clock().now().nanoseconds
        })
        self.task_status_pub.publish(status_msg)

    def find_object_by_description(self, description: str) -> Dict[str, Any]:
        """Find an object in recent detections by description"""
        for detection in self.object_detections.get('detections', []):
            if description.lower() in detection['label'].lower():
                return detection

        return None

def main(args=None):
    rclpy.init(args=args)
    node = TaskManagerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Task Manager Node")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 5: Create the System Monitor Node
Create a node to monitor the entire system:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, LaserScan
from builtin_interfaces.msg import Time
import time
from collections import deque
import json

class SystemMonitorNode(Node):
    def __init__(self):
        super().__init__('system_monitor_node')

        # Create subscribers
        self.state_sub = self.create_subscription(
            String,
            'robot_state',
            self.state_callback,
            10
        )

        self.status_sub = self.create_subscription(
            String,
            'capstone_status',
            self.status_callback,
            10
        )

        self.task_status_sub = self.create_subscription(
            String,
            'task_status',
            self.task_status_callback,
            10
        )

        self.performance_sub = self.create_subscription(
            String,
            'performance_metrics',
            self.performance_callback,
            10
        )

        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )

        self.laser_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.laser_callback,
            10
        )

        self.task_complete_sub = self.create_subscription(
            Bool,
            'task_complete',
            self.task_complete_callback,
            10
        )

        # Create publishers
        self.system_status_pub = self.create_publisher(
            String,
            'system_status',
            10
        )

        # System metrics
        self.system_metrics = {
            'robot_state': 'idle',
            'last_task_completion': None,
            'total_tasks_completed': 0,
            'average_task_time': 0.0,
            'task_times': deque(maxlen=100),
            'last_image_time': None,
            'last_laser_time': None,
            'performance_metrics': {}
        }

        # Timer for system status updates
        self.status_timer = self.create_timer(2.0, self.publish_system_status)

        self.get_logger().info("System Monitor Node initialized")

    def state_callback(self, msg: String):
        """Update robot state"""
        self.system_metrics['robot_state'] = msg.data

    def status_callback(self, msg: String):
        """Handle status updates"""
        self.get_logger().info(f"Status update: {msg.data}")

    def task_status_callback(self, msg: String):
        """Handle task status updates"""
        try:
            status_data = json.loads(msg.data)
            self.get_logger().info(f"Task status: {status_data}")
        except json.JSONDecodeError:
            self.get_logger().error("Invalid JSON in task status")

    def performance_callback(self, msg: String):
        """Handle performance metrics"""
        try:
            perf_data = json.loads(msg.data)
            self.system_metrics['performance_metrics'] = perf_data
        except json.JSONDecodeError:
            self.get_logger().error("Invalid JSON in performance metrics")

    def image_callback(self, msg: Image):
        """Track image reception"""
        current_time = time.time()
        if self.system_metrics['last_image_time']:
            # Calculate frame rate
            frame_interval = current_time - self.system_metrics['last_image_time']
            fps = 1.0 / frame_interval if frame_interval > 0 else 0
            self.get_logger().debug(f"Camera FPS: {fps:.2f}")

        self.system_metrics['last_image_time'] = current_time

    def laser_callback(self, msg: LaserScan):
        """Track laser scan reception"""
        self.system_metrics['last_laser_time'] = time.time()

    def task_complete_callback(self, msg: Bool):
        """Handle task completion"""
        if msg.data:
            self.system_metrics['total_tasks_completed'] += 1
            self.system_metrics['last_task_completion'] = time.time()
            self.get_logger().info(f"Task completed. Total completed: {self.system_metrics['total_tasks_completed']}")

    def publish_system_status(self):
        """Publish overall system status"""
        system_status = {
            'robot_state': self.system_metrics['robot_state'],
            'total_tasks_completed': self.system_metrics['total_tasks_completed'],
            'last_task_completion': self.system_metrics['last_task_completion'],
            'performance': self.system_metrics['performance_metrics'],
            'system_uptime': self.get_clock().now().nanoseconds,
            'timestamp': time.time()
        }

        status_msg = String()
        status_msg.data = json.dumps(system_status, indent=2)
        self.system_status_pub.publish(status_msg)

        # Log system summary
        self.get_logger().info(
            f"System Status - State: {self.system_metrics['robot_state']}, "
            f"Tasks completed: {self.system_metrics['total_tasks_completed']}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = SystemMonitorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down System Monitor Node")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 6: Create Launch File
Create a comprehensive launch file for the capstone project:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    # Launch configuration
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation (Gazebo) clock if true'
        ),

        # Humanoid Orchestrator Node
        Node(
            package='humanoid_capstone',
            executable='humanoid_orchestrator_node',
            name='humanoid_orchestrator_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # Humanoid Navigation Node
        Node(
            package='humanoid_capstone',
            executable='humanoid_navigation_node',
            name='humanoid_navigation_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # Humanoid Manipulation Node
        Node(
            package='humanoid_capstone',
            executable='humanoid_manipulation_node',
            name='humanoid_manipulation_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # Task Manager Node
        Node(
            package='humanoid_capstone',
            executable='task_manager_node',
            name='task_manager_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # System Monitor Node
        Node(
            package='humanoid_capstone',
            executable='system_monitor_node',
            name='system_monitor_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # Include other necessary nodes from previous labs
        # Voice Command System
        Node(
            package='voice_command_system',
            executable='audio_capture_node',
            name='audio_capture_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        Node(
            package='voice_command_system',
            executable='whisper_node',
            name='whisper_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        Node(
            package='voice_command_system',
            executable='voice_command_parser_node',
            name='voice_command_parser_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # Cognitive Planning System
        Node(
            package='cognitive_planning',
            executable='llm_planner_node',
            name='llm_planner_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        Node(
            package='cognitive_planning',
            executable='plan_executor_node',
            name='plan_executor_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        # Vision-Language Perception System
        Node(
            package='vision_language_perception',
            executable='vision_language_perception_node',
            name='vision_language_perception_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),

        Node(
            package='vision_language_perception',
            executable='object_identification_node',
            name='object_identification_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        )
    ])
```

## Testing and Validation

### 1. Basic Functionality Test
Test the integrated system with simple commands:
- "Move to the kitchen"
- "Pick up the red cup"
- "Go to the table and place the cup there"

### 2. Complex Multi-Step Task Test
Test with complex multi-step commands:
- "Go to the kitchen, find the red cup, pick it up, then go to the living room and place it on the table"
- "Find the keys in the bedroom, then navigate to the office and put them on the desk"

### 3. System Integration Test
Test the complete VLA pipeline:
- Voice command recognition and parsing
- Cognitive planning with LLMs
- Vision-language object identification
- Navigation and manipulation execution
- System monitoring and feedback

### 4. Performance Evaluation
Evaluate the complete system:
- End-to-end task completion time
- Success rate for different task types
- System resource utilization
- Robustness to environmental changes

## Optional Extensions

### 1. Advanced Humanoid Behaviors
Implement more sophisticated humanoid behaviors:
- Human-like motion planning
- Social interaction capabilities
- Adaptive learning from user feedback

### 2. Enhanced Perception
Add advanced perception capabilities:
- 3D object reconstruction
- Semantic scene understanding
- Human activity recognition

### 3. Improved Planning
Enhance planning with:
- Dynamic replanning capabilities
- Multi-objective optimization
- Risk assessment and mitigation

## Assessment Questions
1. How well does the integrated VLA system handle ambiguous or complex natural language commands?
2. What are the main bottlenecks in the end-to-end autonomous humanoid system?
3. How could you improve the system's robustness to environmental uncertainties?
4. What safety considerations are critical for autonomous humanoid robots in human environments?

## What You Learned
In this capstone lab, you implemented a complete autonomous humanoid robot system that integrates all components of the Vision-Language-Action pipeline. You learned how to orchestrate complex multi-step tasks, coordinate navigation and manipulation systems, integrate speech recognition with cognitive planning, and create a robust system architecture for humanoid robotics. This project demonstrates the practical application of VLA systems in creating truly autonomous humanoid robots capable of understanding and executing complex natural language commands in real-world environments.