---
title: "Lab 2: Cognitive Planning Pipeline"
sidebar_position: 2
---

# Lab 2: Cognitive Planning Pipeline

## Overview
This lab focuses on implementing a cognitive planning system that translates natural language commands into sequences of robotic actions using Large Language Models (LLMs). You will configure LLM integration for task planning, implement natural language to action sequence translation, and create planning interfaces with ROS 2 navigation stack.

## Objectives
- Configure LLM integration for task planning
- Implement natural language to action sequence translation
- Create planning interfaces with ROS 2 navigation stack
- Validate planning accuracy and execution reliability

## Prerequisites
- Completed Lab 1 (Voice Command Recognition System)
- Basic understanding of ROS 2 navigation (Nav2)
- OpenAI API account or access to LLM API
- Python programming experience
- Basic knowledge of task planning concepts

## Lab Setup

### 1. Install Required Dependencies
Ensure you have the necessary packages installed:

```bash
pip install openai
pip install langchain
pip install langchain-openai
pip install numpy
pip install networkx
```

### 2. Set Up LLM Access
Set up your OpenAI API key or other LLM access:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Create Planning Package
Create a new ROS 2 package for the cognitive planning system:

```bash
cd ~/voice_command_ws/src
ros2 pkg create --build-type ament_python cognitive_planning
cd cognitive_planning
```

## Implementation Steps

### Step 1: Create the LLM Planner Node
Create a ROS 2 node that uses LLMs for task decomposition and planning:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from cognitive_planning_interfaces.msg import PlanStep, Plan
import json
import openai
from typing import List, Dict, Any
import time

class LLMPlannerNode(Node):
    def __init__(self):
        super().__init__('llm_planner_node')

        # Create subscribers and publishers
        self.command_sub = self.create_subscription(
            String,
            'natural_language_command',
            self.command_callback,
            10
        )

        self.plan_pub = self.create_publisher(
            Plan,
            'generated_plan',
            10
        )

        self.status_pub = self.create_publisher(
            String,
            'planner_status',
            10
        )

        # Robot capabilities for planning context
        self.robot_capabilities = [
            "navigate to location",
            "pick up object",
            "place object",
            "detect object",
            "open gripper",
            "close gripper",
            "move arm to position",
            "take photo",
            "stop robot"
        ]

        # Environment knowledge
        self.environment_knowledge = {
            "locations": ["kitchen", "living room", "bedroom", "office", "dining room", "hallway"],
            "objects": ["cup", "book", "pen", "bottle", "phone", "keys", "laptop", "chair", "table"],
            "navigation_constraints": {
                "kitchen_to_bedroom": "through hallway",
                "office_to_kitchen": "through hallway and dining room"
            }
        }

        self.get_logger().info("LLM Planner Node initialized")

    def command_callback(self, msg):
        """Handle incoming natural language commands"""
        command = msg.data
        self.get_logger().info(f"Received command: {command}")

        try:
            # Generate plan using LLM
            plan_steps = self.generate_plan_with_llm(command)

            if plan_steps:
                # Create and publish plan message
                plan_msg = self.create_plan_message(plan_steps, command)
                self.plan_pub.publish(plan_msg)

                self.get_logger().info(f"Published plan with {len(plan_steps)} steps")
            else:
                self.get_logger().error("Failed to generate plan for command")

        except Exception as e:
            self.get_logger().error(f"Error generating plan: {str(e)}")
            status_msg = String()
            status_msg.data = f"Planning failed: {str(e)}"
            self.status_pub.publish(status_msg)

    def generate_plan_with_llm(self, command: str) -> List[Dict[str, Any]]:
        """Generate a plan using LLM"""
        # Create a structured prompt for the LLM
        prompt = f"""
        You are a robot task planner. Given the following command, decompose it into a sequence of executable steps.

        Command: {command}

        Robot capabilities: {', '.join(self.robot_capabilities)}

        Environment: {json.dumps(self.environment_knowledge, indent=2)}

        Please provide the plan as a JSON array of steps, where each step has:
        - id: integer step identifier
        - action: the specific action to perform
        - parameters: any required parameters for the action
        - description: human-readable description of the step

        Example format:
        [
            {{
                "id": 1,
                "action": "navigate_to",
                "parameters": {{"location": "kitchen"}},
                "description": "Move to the kitchen"
            }},
            {{
                "id": 2,
                "action": "detect_object",
                "parameters": {{"object_type": "cup"}},
                "description": "Look for the cup"
            }}
        ]

        Only return the JSON array, nothing else.
        """

        try:
            # Call the LLM to generate the plan
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a robot task planner. Return only valid JSON as specified."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            # Extract the plan from the response
            plan_text = response.choices[0].message.content.strip()

            # Clean up the response if it contains markdown code block markers
            if plan_text.startswith("```json"):
                plan_text = plan_text[7:]  # Remove ```json
            if plan_text.endswith("```"):
                plan_text = plan_text[:-3]  # Remove ```

            # Parse the JSON response
            plan_steps = json.loads(plan_text)

            self.get_logger().info(f"Generated plan with {len(plan_steps)} steps")
            return plan_steps

        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse LLM response as JSON: {str(e)}")
            self.get_logger().info(f"LLM response: {plan_text}")
            return []
        except Exception as e:
            self.get_logger().error(f"Error calling LLM: {str(e)}")
            return []

    def create_plan_message(self, plan_steps: List[Dict], original_command: str) -> Plan:
        """Create a Plan message from plan steps"""
        plan_msg = Plan()
        plan_msg.header.stamp = self.get_clock().now().to_msg()
        plan_msg.header.frame_id = "map"
        plan_msg.original_command = original_command

        for step_dict in plan_steps:
            step_msg = PlanStep()
            step_msg.id = step_dict.get('id', 0)
            step_msg.action = step_dict.get('action', '')
            step_msg.description = step_dict.get('description', '')

            # Convert parameters to JSON string
            params = step_dict.get('parameters', {})
            step_msg.parameters = json.dumps(params)

            plan_msg.steps.append(step_msg)

        return plan_msg

def main(args=None):
    rclpy.init(args=args)
    node = LLMPlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 2: Create the Plan Validator Node
Create a node that validates and refines plans generated by the LLM:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from cognitive_planning_interfaces.msg import Plan, PlanStep
from std_msgs.msg import String
import json
from typing import List, Dict, Any

class PlanValidatorNode(Node):
    def __init__(self):
        super().__init__('plan_validator_node')

        # Create subscribers and publishers
        self.plan_sub = self.create_subscription(
            Plan,
            'generated_plan',
            self.plan_callback,
            10
        )

        self.validated_plan_pub = self.create_publisher(
            Plan,
            'validated_plan',
            10
        )

        self.status_pub = self.create_publisher(
            String,
            'validator_status',
            10
        )

        # Known valid actions for the robot
        self.valid_actions = {
            'navigate_to', 'detect_object', 'grasp_object', 'place_object',
            'move_arm', 'take_photo', 'stop_robot', 'open_gripper', 'close_gripper'
        }

        # Known locations in the environment
        self.known_locations = {
            'kitchen', 'living room', 'bedroom', 'office', 'dining room', 'hallway'
        }

        self.get_logger().info("Plan Validator Node initialized")

    def plan_callback(self, msg: Plan):
        """Validate incoming plan"""
        self.get_logger().info(f"Validating plan with {len(msg.steps)} steps")

        try:
            # Validate each step in the plan
            validated_plan = self.validate_plan(msg)

            if validated_plan:
                # Publish validated plan
                self.validated_plan_pub.publish(validated_plan)
                self.get_logger().info("Plan validation successful")

                status_msg = String()
                status_msg.data = f"Plan validated successfully: {len(validated_plan.steps)} steps"
                self.status_pub.publish(status_msg)
            else:
                self.get_logger().error("Plan validation failed")

                status_msg = String()
                status_msg.data = "Plan validation failed"
                self.status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f"Error validating plan: {str(e)}")

    def validate_plan(self, plan_msg: Plan) -> Plan:
        """Validate a plan and return a validated version"""
        validated_plan = Plan()
        validated_plan.header = plan_msg.header
        validated_plan.original_command = plan_msg.original_command
        validated_plan.steps = []

        for step in plan_msg.steps:
            # Validate action
            if step.action not in self.valid_actions:
                self.get_logger().warn(f"Invalid action in plan: {step.action}")
                continue

            # Validate parameters
            try:
                params = json.loads(step.parameters)

                # Validate location if present
                if 'location' in params and params['location'] not in self.known_locations:
                    self.get_logger().warn(f"Unknown location: {params['location']}")
                    continue

                # Validate object type if present
                if 'object_type' in params:
                    # Object validation can be more complex, for now just accept
                    pass

                # If all validations pass, add to validated plan
                validated_plan.steps.append(step)

            except json.JSONDecodeError:
                self.get_logger().error(f"Invalid JSON parameters in step {step.id}")
                continue

        # If no valid steps remain, return None
        if len(validated_plan.steps) == 0:
            self.get_logger().error("No valid steps in plan after validation")
            return None

        return validated_plan

def main(args=None):
    rclpy.init(args=args)
    node = PlanValidatorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 3: Create the Plan Executor Node
Create a node that executes validated plans by interfacing with ROS 2 navigation and other systems:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from cognitive_planning_interfaces.msg import Plan, PlanStep
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import json
import threading
import time

class PlanExecutorNode(Node):
    def __init__(self):
        super().__init__('plan_executor_node')

        # Create subscribers
        self.plan_sub = self.create_subscription(
            Plan,
            'validated_plan',
            self.plan_callback,
            10
        )

        self.status_pub = self.create_publisher(
            String,
            'executor_status',
            10
        )

        # Create action client for navigation
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # State management
        self.current_plan = None
        self.is_executing = False
        self.execution_thread = None

        # CV bridge for image processing
        self.cv_bridge = CvBridge()

        # Location coordinates (in a real system, these would come from a map)
        self.location_coordinates = {
            'kitchen': {'x': 2.0, 'y': 1.0, 'theta': 0.0},
            'living room': {'x': 0.0, 'y': 0.0, 'theta': 0.0},
            'bedroom': {'x': -2.0, 'y': 1.0, 'theta': 3.14},
            'office': {'x': 0.0, 'y': 2.0, 'theta': 1.57},
            'dining room': {'x': 1.0, 'y': -1.0, 'theta': -1.57},
            'hallway': {'x': 0.0, 'y': 1.0, 'theta': 0.0}
        }

        self.get_logger().info("Plan Executor Node initialized")

    def plan_callback(self, msg: Plan):
        """Handle incoming validated plans"""
        if self.is_executing:
            self.get_logger().warn("Plan execution already in progress, skipping new plan")
            return

        self.get_logger().info(f"Received plan with {len(msg.steps)} steps")
        self.current_plan = msg
        self.is_executing = True

        # Start execution in a separate thread
        self.execution_thread = threading.Thread(target=self.execute_plan, daemon=True)
        self.execution_thread.start()

    def execute_plan(self):
        """Execute the current plan step by step"""
        if not self.current_plan:
            return

        for i, step_msg in enumerate(self.current_plan.steps):
            if not self.is_executing:
                self.get_logger().info("Plan execution stopped")
                break

            self.get_logger().info(f"Executing step {i+1}/{len(self.current_plan.steps)}: {step_msg.description}")

            # Update status
            status_msg = String()
            status_msg.data = f"Executing step {i+1}: {step_msg.description}"
            self.status_pub.publish(status_msg)

            # Execute the step based on its action
            success = self.execute_step(step_msg)

            if not success:
                self.get_logger().error(f"Failed to execute step {i+1}")
                status_msg = String()
                status_msg.data = f"Failed to execute step {i+1}: {step_msg.description}"
                self.status_pub.publish(status_msg)
                break

        self.is_executing = False
        self.get_logger().info("Plan execution completed")

        # Final status
        status_msg = String()
        status_msg.data = "Plan execution completed"
        self.status_pub.publish(status_msg)

    def execute_step(self, step_msg: PlanStep) -> bool:
        """Execute a single plan step"""
        try:
            params = json.loads(step_msg.parameters)

            if step_msg.action == 'navigate_to':
                return self.execute_navigation_step(params)
            elif step_msg.action == 'detect_object':
                return self.execute_detection_step(params)
            elif step_msg.action == 'grasp_object':
                return self.execute_grasp_step(params)
            elif step_msg.action == 'place_object':
                return self.execute_place_step(params)
            elif step_msg.action == 'take_photo':
                return self.execute_photo_step(params)
            elif step_msg.action == 'stop_robot':
                return self.execute_stop_step(params)
            else:
                self.get_logger().warn(f"Unknown action: {step_msg.action}")
                return False

        except json.JSONDecodeError:
            self.get_logger().error(f"Invalid JSON parameters in step: {step_msg.parameters}")
            return False
        except Exception as e:
            self.get_logger().error(f"Error executing step: {str(e)}")
            return False

    def execute_navigation_step(self, params: Dict) -> bool:
        """Execute a navigation step"""
        location = params.get('location')
        if not location:
            self.get_logger().error("No location specified for navigation")
            return False

        if location not in self.location_coordinates:
            self.get_logger().error(f"Unknown location: {location}")
            return False

        # Wait for navigation server
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Navigation server not available")
            return False

        # Create navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'

        coords = self.location_coordinates[location]
        goal_msg.pose.pose.position.x = coords['x']
        goal_msg.pose.pose.position.y = coords['y']
        goal_msg.pose.pose.position.z = 0.0

        # Convert theta to quaternion (simplified for this example)
        import math
        theta = coords['theta']
        goal_msg.pose.pose.orientation.z = math.sin(theta / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(theta / 2.0)

        # Send navigation goal
        future = self.nav_client.send_goal_async(goal_msg)

        # Wait for result
        rclpy.spin_until_future_complete(self, future, timeout_sec=30.0)

        if future.result() is not None:
            goal_handle = future.result()
            if goal_handle.accepted:
                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)

                if result_future.result() is not None:
                    result = result_future.result().result
                    self.get_logger().info(f"Navigation completed: {result}")
                    return True
                else:
                    self.get_logger().error("Navigation result future was None")
                    return False
            else:
                self.get_logger().error("Navigation goal was rejected")
                return False
        else:
            self.get_logger().error("Navigation goal future was None")
            return False

    def execute_detection_step(self, params: Dict) -> bool:
        """Execute a detection step"""
        object_type = params.get('object_type', 'object')
        self.get_logger().info(f"Detecting {object_type}")

        # In a real system, this would interface with perception system
        # For now, simulate detection
        time.sleep(2.0)
        self.get_logger().info(f"Detection of {object_type} completed")
        return True

    def execute_grasp_step(self, params: Dict) -> bool:
        """Execute a grasp step"""
        object_type = params.get('object_type', 'object')
        self.get_logger().info(f"Grasping {object_type}")

        # In a real system, this would interface with manipulation system
        # For now, simulate grasping
        time.sleep(3.0)
        self.get_logger().info(f"Grasping of {object_type} completed")
        return True

    def execute_place_step(self, params: Dict) -> bool:
        """Execute a place step"""
        location = params.get('location', 'default')
        self.get_logger().info(f"Placing object at {location}")

        # In a real system, this would interface with manipulation system
        # For now, simulate placing
        time.sleep(2.5)
        self.get_logger().info(f"Placing object at {location} completed")
        return True

    def execute_photo_step(self, params: Dict) -> bool:
        """Execute a photo step"""
        self.get_logger().info("Taking photo")

        # In a real system, this would interface with camera system
        # For now, simulate photo capture
        time.sleep(1.0)
        self.get_logger().info("Photo taken")
        return True

    def execute_stop_step(self, params: Dict) -> bool:
        """Execute a stop step"""
        self.get_logger().info("Stopping robot")
        # In a real system, this would send stop commands to robot
        return True

def main(args=None):
    rclpy.init(args=args)
    node = PlanExecutorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Plan Executor Node")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 4: Create the Plan Monitor Node
Create a node that monitors plan execution and provides feedback:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from cognitive_planning_interfaces.msg import Plan, PlanStep
from std_msgs.msg import String, Bool
from std_msgs.msg import Empty
import time
from typing import Dict, Any

class PlanMonitorNode(Node):
    def __init__(self):
        super().__init__('plan_monitor_node')

        # Create subscribers
        self.plan_sub = self.create_subscription(
            Plan,
            'generated_plan',
            self.plan_received_callback,
            10
        )

        self.validated_plan_sub = self.create_subscription(
            Plan,
            'validated_plan',
            self.validated_plan_callback,
            10
        )

        self.executor_status_sub = self.create_subscription(
            String,
            'executor_status',
            self.executor_status_callback,
            10
        )

        # Create publishers
        self.monitor_status_pub = self.create_publisher(
            String,
            'monitor_status',
            10
        )

        self.plan_success_pub = self.create_publisher(
            Bool,
            'plan_success',
            10
        )

        # State tracking
        self.current_plan = None
        self.plan_start_time = None
        self.steps_completed = 0
        self.total_steps = 0

        self.get_logger().info("Plan Monitor Node initialized")

    def plan_received_callback(self, msg: Plan):
        """Handle when a plan is received"""
        self.current_plan = msg
        self.plan_start_time = time.time()
        self.steps_completed = 0
        self.total_steps = len(msg.steps)

        self.get_logger().info(f"Monitoring plan with {self.total_steps} steps")

        status_msg = String()
        status_msg.data = f"Plan received with {self.total_steps} steps"
        self.monitor_status_pub.publish(status_msg)

    def validated_plan_callback(self, msg: Plan):
        """Handle when a plan is validated"""
        self.total_steps = len(msg.steps)
        self.get_logger().info(f"Plan validated with {self.total_steps} steps")

        status_msg = String()
        status_msg.data = f"Plan validated with {self.total_steps} steps"
        self.monitor_status_pub.publish(status_msg)

    def executor_status_callback(self, msg: String):
        """Handle executor status updates"""
        status_text = msg.data

        # Check if this is a step completion message
        if "Executing step" in status_text:
            # Extract step number from message
            try:
                # Simple parsing: "Executing step X: description"
                parts = status_text.split(':')
                if len(parts) > 0:
                    step_part = parts[0]
                    if 'step' in step_part:
                        step_num = int(step_part.split()[2])  # Get the step number
                        self.steps_completed = step_num - 1  # Update completed steps
            except (ValueError, IndexError):
                pass  # Could not parse step number

        # Check if execution completed
        if "Plan execution completed" in status_text:
            success_msg = Bool()
            success_msg.data = True
            self.plan_success_pub.publish(success_msg)

            elapsed_time = time.time() - self.plan_start_time if self.plan_start_time else 0
            self.get_logger().info(f"Plan completed in {elapsed_time:.2f} seconds")

        # Update monitor status
        progress = f"{self.steps_completed}/{self.total_steps}" if self.total_steps > 0 else "0/0"
        status_msg = String()
        status_msg.data = f"Progress: {progress}. Status: {status_text}"
        self.monitor_status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PlanMonitorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 5: Create Custom Message Definitions
Create the custom message definitions needed for the planning system. First, create the package structure:

```bash
mkdir -p ~/voice_command_ws/src/cognitive_planning/cognitive_planning_interfaces/msg
```

Create `PlanStep.msg`:
```text
# PlanStep.msg
int32 id
string action
string parameters
string description
```

Create `Plan.msg`:
```text
# Plan.msg
std_msgs/Header header
string original_command
PlanStep[] steps
```

Update the `package.xml`:
```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>cognitive_planning_interfaces</name>
  <version>0.0.0</version>
  <description>Custom message definitions for cognitive planning</description>
  <maintainer email="user@example.com">user</maintainer>
  <license>Apache-2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>
  <buildtool_depend>rosidl_default_generators</buildtool_depend>

  <depend>std_msgs</depend>

  <exec_depend>rosidl_default_runtime</exec_depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <member_of_group>rosidl_interface_packages</member_of_group>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

Update the `setup.py`:
```python
from setuptools import find_packages, setup

package_name = 'cognitive_planning_interfaces'

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
    maintainer='user',
    maintainer_email='user@example.com',
    description='Custom message definitions for cognitive planning',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
```

### Step 6: Create Launch File
Create a launch file to start all cognitive planning nodes:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    return LaunchDescription([
        # LLM Planner node
        Node(
            package='cognitive_planning',
            executable='llm_planner_node',
            name='llm_planner_node',
            output='screen'
        ),

        # Plan Validator node
        Node(
            package='cognitive_planning',
            executable='plan_validator_node',
            name='plan_validator_node',
            output='screen'
        ),

        # Plan Executor node
        Node(
            package='cognitive_planning',
            executable='plan_executor_node',
            name='plan_executor_node',
            output='screen'
        ),

        # Plan Monitor node
        Node(
            package='cognitive_planning',
            executable='plan_monitor_node',
            name='plan_monitor_node',
            output='screen'
        )
    ])
```

## Testing and Validation

### 1. Basic Functionality Test
Test the cognitive planning system with simple commands:
- "Go to the kitchen and find the red cup"
- "Navigate to the office and take a photo"
- "Move to the living room and stop"

### 2. Complex Task Test
Test with more complex multi-step commands:
- "Go to the kitchen, pick up the cup, then go to the living room and place it on the table"
- "Find the keys in the bedroom, then navigate to the office"

### 3. Plan Validation Test
Test the system's ability to validate and reject invalid plans:
- Commands with unknown locations
- Commands with invalid actions
- Commands that conflict with robot capabilities

### 4. Performance Evaluation
Record and analyze:
- Planning time (from command to plan generation)
- Plan validation accuracy
- Execution success rate
- Error recovery capabilities

## Optional Extensions

### 1. Context-Aware Planning
Enhance the system to consider context such as:
- Robot's current location and state
- Environmental conditions
- Previous commands and outcomes

### 2. Plan Refinement
Implement a plan refinement system that:
- Monitors execution in real-time
- Adjusts plans based on environmental changes
- Handles partial plan failures gracefully

### 3. Learning from Execution
Add a learning component that:
- Records successful plan executions
- Improves future planning based on execution outcomes
- Adapts to user preferences over time

## Assessment Questions
1. How does the LLM-based planning approach compare to traditional symbolic planning methods?
2. What are the main challenges in translating natural language to executable robotic actions?
3. How could you improve the system's ability to handle ambiguous or incomplete commands?
4. What safety considerations should be addressed in LLM-driven robotic planning?

## What You Learned
In this lab, you implemented a comprehensive cognitive planning system that translates natural language commands into executable robotic actions using Large Language Models. You learned how to integrate LLMs with ROS 2 navigation systems, validate and execute complex multi-step plans, and monitor plan execution for reliability. You also explored the challenges and opportunities in LLM-driven robotic planning and considered potential improvements for real-world applications.