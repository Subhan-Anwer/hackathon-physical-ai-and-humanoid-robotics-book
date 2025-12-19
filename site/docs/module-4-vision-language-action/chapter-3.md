---
title: "Chapter 3: Cognitive Planning with LLMs"
sidebar_position: 3
---

# Chapter 3: Cognitive Planning with LLMs

## Learning Objectives

By the end of this chapter, you will be able to:
- Implement LLM-based task decomposition and planning for robotic systems
- Design natural language to action mapping systems using Large Language Models
- Integrate LLM planning with ROS 2 navigation and manipulation systems
- Apply planning strategies for uncertain and dynamic environments
- Monitor and execute multi-step tasks with LLM-driven planning

## LLM-Based Task Decomposition and Planning

### The Role of LLMs in Robotic Planning

Large Language Models have revolutionized robotic planning by providing the ability to understand high-level, abstract commands and decompose them into executable action sequences. Unlike traditional planning systems that require explicit, low-level instructions, LLMs can leverage their vast knowledge base to infer appropriate action sequences from natural language descriptions.

The planning process with LLMs involves several key steps:

1. **Command Interpretation**: Understanding the high-level goal expressed in natural language
2. **Task Decomposition**: Breaking down complex goals into manageable subtasks
3. **Action Sequencing**: Arranging subtasks in the correct order with appropriate preconditions
4. **Context Integration**: Incorporating environmental and robot state information
5. **Plan Validation**: Ensuring the generated plan is feasible and safe

### Task Decomposition Strategies

LLMs excel at decomposing complex tasks into simpler, executable components. Consider the command "Clean the room." A human would naturally break this down into subtasks like:

- Identify dirty objects
- Collect trash items
- Organize misplaced items
- Vacuum or mop the floor

An LLM-based planning system can learn to perform similar decompositions by training on examples or using in-context learning with appropriate prompts.

### Planning Architecture with LLM Integration

```python
import openai
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class TaskType(Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PERCEPTION = "perception"
    INTERACTION = "interaction"

@dataclass
class PlanStep:
    """Represents a single step in the execution plan"""
    id: int
    action: str
    parameters: Dict[str, Any]
    task_type: TaskType
    preconditions: List[str]
    postconditions: List[str]
    description: str

class LLMPlanner:
    """LLM-based task planner for robotic systems"""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.planning_history = []

    def decompose_task(self, natural_language_command: str, robot_capabilities: List[str],
                      environment_state: Dict[str, Any]) -> List[PlanStep]:
        """Decompose a natural language command into executable steps"""

        # Create a structured prompt for the LLM
        prompt = self._create_planning_prompt(natural_language_command, robot_capabilities, environment_state)

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                functions=[
                    {
                        "name": "create_plan",
                        "description": "Create a step-by-step plan for the robot to execute",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "steps": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "integer"},
                                            "action": {"type": "string"},
                                            "parameters": {"type": "object"},
                                            "task_type": {"type": "string", "enum": ["navigation", "manipulation", "perception", "interaction"]},
                                            "description": {"type": "string"}
                                        },
                                        "required": ["id", "action", "parameters", "task_type", "description"]
                                    }
                                }
                            },
                            "required": ["steps"]
                        }
                    }
                ],
                function_call={"name": "create_plan"}
            )

            # Parse the response
            plan_data = json.loads(response.choices[0].message.function_call.arguments)
            plan_steps = []

            for step_data in plan_data['steps']:
                step = PlanStep(
                    id=step_data['id'],
                    action=step_data['action'],
                    parameters=step_data['parameters'],
                    task_type=TaskType(step_data['task_type']),
                    preconditions=[],
                    postconditions=[],
                    description=step_data['description']
                )
                plan_steps.append(step)

            return plan_steps

        except Exception as e:
            print(f"Error in LLM planning: {e}")
            return self._fallback_planning(natural_language_command)

    def _create_planning_prompt(self, command: str, capabilities: List[str],
                               environment: Dict[str, Any]) -> str:
        """Create a structured prompt for the LLM planner"""
        return f"""
        Natural language command: {command}

        Robot capabilities: {', '.join(capabilities)}

        Current environment state: {json.dumps(environment, indent=2)}

        Please decompose this command into specific, executable steps that the robot can perform.
        Each step should be a discrete action with clear parameters.
        Return the steps in a structured format with action type (navigation, manipulation, perception, interaction).
        """

    def _get_system_prompt(self) -> str:
        """System prompt to guide the LLM's planning behavior"""
        return """
        You are an expert robotic task planner. Your role is to decompose high-level natural language commands
        into specific, executable steps for a robot. Each step should be:
        1. Feasible given the robot's capabilities
        2. Specific enough to be executed by the robot
        3. In the correct sequence to achieve the overall goal
        4. Include necessary parameters for execution

        Consider the environment state when creating the plan to ensure feasibility.
        """

    def _fallback_planning(self, command: str) -> List[PlanStep]:
        """Fallback planning in case the LLM fails"""
        # Simple fallback for common commands
        if "move" in command.lower() or "go" in command.lower():
            return [PlanStep(
                id=1,
                action="navigate_to",
                parameters={"target_location": "default"},
                task_type=TaskType.NAVIGATION,
                preconditions=[],
                postconditions=[],
                description="Move to target location"
            )]
        else:
            return [PlanStep(
                id=1,
                action="unknown_command",
                parameters={},
                task_type=TaskType.PERCEPTION,
                preconditions=[],
                postconditions=[],
                description="Unable to parse command"
            )]

# Example usage
planner = LLMPlanner()
capabilities = ["navigate", "grasp", "place", "detect_objects", "open_gripper", "close_gripper"]
environment = {"objects": ["cup", "book", "pen"], "locations": ["table", "kitchen", "desk"]}

plan = planner.decompose_task("Pick up the red cup and place it on the table", capabilities, environment)
for step in plan:
    print(f"Step {step.id}: {step.description} ({step.task_type.value})")
```

## Natural Language to Action Mapping

### Semantic Action Mapping

The conversion from natural language to robotic actions requires sophisticated semantic understanding and mapping. This process involves:

**Action Grounding**: Mapping abstract language concepts to concrete physical actions the robot can perform.

**Parameter Extraction**: Identifying specific parameters needed for action execution (locations, objects, gripper positions, etc.).

**Constraint Handling**: Understanding constraints and conditions that must be satisfied before or during action execution.

### Action Schema Framework

```python
from typing import Dict, List, Optional, Callable
import inspect

class ActionSchema:
    """Defines the structure and execution of robot actions"""

    def __init__(self, name: str, description: str, parameters: Dict[str, Dict],
                 executor: Callable, preconditions: List[str] = None):
        self.name = name
        self.description = description
        self.parameters = parameters  # {param_name: {"type": type, "required": bool, "description": str}}
        self.executor = executor
        self.preconditions = preconditions or []

    def validate_parameters(self, params: Dict) -> bool:
        """Validate that required parameters are provided"""
        for param_name, param_info in self.parameters.items():
            if param_info.get("required", False):
                if param_name not in params:
                    return False
        return True

    def execute(self, **kwargs):
        """Execute the action with provided parameters"""
        if not self.validate_parameters(kwargs):
            raise ValueError(f"Missing required parameters for action {self.name}")

        return self.executor(**kwargs)

class ActionMapper:
    """Maps natural language commands to executable actions"""

    def __init__(self):
        self.action_schemas = {}
        self._register_default_actions()

    def _register_default_actions(self):
        """Register common robot actions"""
        # Navigation action
        def navigate_to(location: str, speed: float = 0.5):
            """Navigate to a specific location"""
            print(f"Navigating to {location} at speed {speed}")
            # In a real system, this would interface with navigation stack
            return {"status": "success", "location": location}

        self.action_schemas["navigate_to"] = ActionSchema(
            name="navigate_to",
            description="Move the robot to a specified location",
            parameters={
                "location": {"type": str, "required": True, "description": "Target location"},
                "speed": {"type": float, "required": False, "description": "Movement speed (0.0-1.0)", "default": 0.5}
            },
            executor=navigate_to
        )

        # Manipulation action
        def grasp_object(object_name: str, location: str = None):
            """Grasp an object"""
            print(f"Grasping {object_name} at {location or 'current location'}")
            return {"status": "success", "object": object_name}

        self.action_schemas["grasp_object"] = ActionSchema(
            name="grasp_object",
            description="Grasp a specific object",
            parameters={
                "object_name": {"type": str, "required": True, "description": "Name of object to grasp"},
                "location": {"type": str, "required": False, "description": "Location of object"}
            },
            executor=grasp_object
        )

        # Object detection action
        def detect_objects(target_object: str = None, location: str = None):
            """Detect objects in the environment"""
            print(f"Detecting objects{' of type ' + target_object if target_object else ''} at {location or 'current location'}")
            # In a real system, this would interface with perception system
            return {"status": "success", "objects": ["cup", "book"]}

        self.action_schemas["detect_objects"] = ActionSchema(
            name="detect_objects",
            description="Detect objects in the environment",
            parameters={
                "target_object": {"type": str, "required": False, "description": "Specific object to detect"},
                "location": {"type": str, "required": False, "description": "Location to search"}
            },
            executor=detect_objects
        )

    def map_command_to_action(self, parsed_command: Dict) -> Optional[ActionSchema]:
        """Map a parsed command to an appropriate action schema"""
        action_type = parsed_command.get('action_type')
        target_object = parsed_command.get('target_object')
        target_location = parsed_command.get('target_location')

        # Simple mapping logic - in practice, this would be more sophisticated
        if action_type == 'navigation':
            return self.action_schemas.get('navigate_to')
        elif action_type == 'manipulation':
            if target_object:
                return self.action_schemas.get('grasp_object')
        elif action_type == 'observation':
            return self.action_schemas.get('detect_objects')

        return None

    def execute_plan_step(self, plan_step: PlanStep, context: Dict) -> Dict:
        """Execute a single step of a plan"""
        action_schema = self.action_schemas.get(plan_step.action)
        if not action_schema:
            return {"status": "error", "message": f"Unknown action: {plan_step.action}"}

        # Prepare parameters based on plan step and context
        params = plan_step.parameters.copy()

        # Add context information if needed
        if 'target_location' in params and params['target_location'] == 'default':
            params['target_location'] = context.get('default_location', 'unknown')

        try:
            result = action_schema.execute(**params)
            return result
        except Exception as e:
            return {"status": "error", "message": str(e)}

# Example usage
mapper = ActionMapper()
step = PlanStep(
    id=1,
    action="navigate_to",
    parameters={"location": "kitchen", "speed": 0.7},
    task_type=TaskType.NAVIGATION,
    preconditions=[],
    postconditions=[],
    description="Move to kitchen"
)

context = {"default_location": "living_room"}
result = mapper.execute_plan_step(step, context)
print(f"Execution result: {result}")
```

## Integration with ROS 2 Navigation and Manipulation Systems

### ROS 2 Action Interface for Planning

Integrating LLM-based planning with ROS 2 requires careful coordination between the high-level planning system and the low-level control systems. The typical architecture involves:

1. **LLM Planner Node**: Generates high-level plans from natural language commands
2. **Plan Execution Node**: Translates plan steps into ROS 2 actions and monitors execution
3. **Navigation System**: Executes navigation actions using Nav2
4. **Manipulation System**: Executes manipulation actions using MoveIt2 and controllers

### Plan Execution with ROS 2 Actions

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import threading
import time

class PlanExecutorNode(Node):
    """Executes plans generated by LLM planner using ROS 2 actions"""

    def __init__(self):
        super().__init__('plan_executor_node')

        # Create action clients for navigation
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Create subscribers
        self.plan_sub = self.create_subscription(
            String,
            'high_level_plan',
            self.plan_callback,
            10
        )

        self.status_pub = self.create_publisher(
            String,
            'execution_status',
            10
        )

        # Plan execution state
        self.current_plan = None
        self.execution_thread = None
        self.is_executing = False

        self.get_logger().info("Plan Executor Node initialized")

    def plan_callback(self, msg):
        """Handle incoming plan from LLM planner"""
        try:
            plan_data = json.loads(msg.data)
            self.current_plan = plan_data['steps']
            self.get_logger().info(f"Received plan with {len(self.current_plan)} steps")

            # Start execution in a separate thread
            if self.execution_thread is None or not self.execution_thread.is_alive():
                self.execution_thread = threading.Thread(target=self.execute_plan)
                self.execution_thread.start()
            else:
                self.get_logger().warn("Plan execution already in progress, skipping new plan")

        except json.JSONDecodeError:
            self.get_logger().error("Invalid JSON in plan message")
        except Exception as e:
            self.get_logger().error(f"Error processing plan: {str(e)}")

    def execute_plan(self):
        """Execute the current plan step by step"""
        if not self.current_plan:
            return

        self.is_executing = True

        for step_idx, step in enumerate(self.current_plan):
            if not self.is_executing:
                self.get_logger().info("Plan execution stopped by user")
                break

            self.get_logger().info(f"Executing step {step_idx + 1}: {step['description']}")

            # Update status
            status_msg = String()
            status_msg.data = f"Executing step {step_idx + 1}: {step['description']}"
            self.status_pub.publish(status_msg)

            # Execute the step based on its type
            success = self.execute_step(step)

            if not success:
                self.get_logger().error(f"Failed to execute step {step_idx + 1}")
                status_msg.data = f"Failed executing step {step_idx + 1}: {step['description']}"
                self.status_pub.publish(status_msg)
                break

        self.is_executing = False
        self.get_logger().info("Plan execution completed")

        # Publish completion status
        status_msg = String()
        status_msg.data = "Plan execution completed"
        self.status_pub.publish(status_msg)

    def execute_step(self, step: Dict) -> bool:
        """Execute a single plan step"""
        try:
            if step['task_type'] == 'navigation':
                return self.execute_navigation_step(step)
            elif step['task_type'] == 'manipulation':
                return self.execute_manipulation_step(step)
            elif step['task_type'] == 'perception':
                return self.execute_perception_step(step)
            else:
                self.get_logger().warn(f"Unknown step type: {step['task_type']}")
                return False
        except Exception as e:
            self.get_logger().error(f"Error executing step: {str(e)}")
            return False

    def execute_navigation_step(self, step: Dict) -> bool:
        """Execute a navigation step using Nav2"""
        # Wait for action server
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Navigation action server not available")
            return False

        # Create goal message
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'

        # Set target pose (simplified - in practice, you'd get this from step parameters)
        target_location = step['parameters'].get('location', 'default')

        # This is a simplified example - in practice, you'd look up coordinates
        # for the named location from a map or knowledge base
        if target_location == 'kitchen':
            goal_msg.pose.pose.position.x = 2.0
            goal_msg.pose.pose.position.y = 1.0
        elif target_location == 'living_room':
            goal_msg.pose.pose.position.x = 0.0
            goal_msg.pose.pose.position.y = 0.0
        else:
            # Default coordinates
            goal_msg.pose.pose.position.x = 1.0
            goal_msg.pose.pose.position.y = 1.0

        goal_msg.pose.pose.orientation.w = 1.0

        # Send goal
        future = self.nav_client.send_goal_async(goal_msg)

        # Wait for result (with timeout)
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

    def execute_manipulation_step(self, step: Dict) -> bool:
        """Execute a manipulation step"""
        # This would interface with MoveIt2 or other manipulation systems
        # For now, we'll simulate the execution
        self.get_logger().info(f"Executing manipulation: {step['parameters']}")
        time.sleep(2.0)  # Simulate execution time
        return True

    def execute_perception_step(self, step: Dict) -> bool:
        """Execute a perception step"""
        # This would interface with perception systems
        # For now, we'll simulate the execution
        self.get_logger().info(f"Executing perception: {step['parameters']}")
        time.sleep(1.0)  # Simulate execution time
        return True

def main(args=None):
    rclpy.init(args=args)
    executor_node = PlanExecutorNode()

    try:
        rclpy.spin(executor_node)
    except KeyboardInterrupt:
        executor_node.get_logger().info("Shutting down plan executor")
    finally:
        executor_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Planning Under Uncertainty and Dynamic Environments

### Handling Uncertainty in LLM-Driven Planning

Robotic environments are inherently uncertain and dynamic, which presents challenges for LLM-driven planning. Key considerations include:

**Perception Uncertainty**: The robot's understanding of the environment may be incomplete or incorrect.

**Action Execution Uncertainty**: Actions may not execute as expected due to environmental factors or robot limitations.

**Temporal Uncertainty**: Environmental conditions may change between planning and execution.

### Adaptive Planning Framework

```python
from typing import Dict, List, Tuple, Optional
import random

class AdaptivePlanner:
    """Handles planning in uncertain and dynamic environments"""

    def __init__(self, llm_planner: LLMPlanner):
        self.llm_planner = llm_planner
        self.execution_history = []
        self.uncertainty_model = UncertaintyModel()

    def create_adaptive_plan(self, command: str, robot_capabilities: List[str],
                           environment_state: Dict[str, Any]) -> List[PlanStep]:
        """Create a plan that accounts for uncertainty"""
        # Generate initial plan
        initial_plan = self.llm_planner.decompose_task(command, robot_capabilities, environment_state)

        # Add uncertainty handling to the plan
        adaptive_plan = self._add_uncertainty_handling(initial_plan, environment_state)

        return adaptive_plan

    def _add_uncertainty_handling(self, plan: List[PlanStep], env_state: Dict[str, Any]) -> List[PlanStep]:
        """Add uncertainty handling steps to the plan"""
        adaptive_plan = []

        for step in plan:
            # Add perception steps before critical actions
            if step.task_type in [TaskType.NAVIGATION, TaskType.MANIPULATION]:
                # Add environment verification before critical steps
                verification_step = PlanStep(
                    id=len(adaptive_plan) + 1,
                    action="verify_environment",
                    parameters={"target": self._get_verification_target(step)},
                    task_type=TaskType.PERCEPTION,
                    preconditions=[],
                    postconditions=[],
                    description=f"Verify environment for {step.description}"
                )
                adaptive_plan.append(verification_step)

            # Add the original step
            step.id = len(adaptive_plan) + 1
            adaptive_plan.append(step)

            # Add post-execution verification
            verification_step = PlanStep(
                id=len(adaptive_plan) + 1,
                action="verify_execution",
                parameters={"expected_result": self._get_expected_result(step)},
                task_type=TaskType.PERCEPTION,
                preconditions=[],
                postconditions=[],
                description=f"Verify successful execution of {step.description}"
            )
            adaptive_plan.append(verification_step)

        return adaptive_plan

    def _get_verification_target(self, step: PlanStep) -> str:
        """Get the target for environment verification"""
        if step.task_type == TaskType.NAVIGATION:
            return step.parameters.get('location', 'unknown')
        elif step.task_type == TaskType.MANIPULATION:
            return step.parameters.get('object_name', 'unknown')
        return 'environment'

    def _get_expected_result(self, step: PlanStep) -> Dict[str, Any]:
        """Get the expected result of executing a step"""
        return {
            'action': step.action,
            'parameters': step.parameters,
            'completed': True
        }

    def handle_execution_failure(self, failed_step: PlanStep, error: Exception,
                               current_env: Dict[str, Any]) -> List[PlanStep]:
        """Handle failure of a plan step and generate recovery actions"""
        self.get_logger().warn(f"Step failed: {failed_step.description}, error: {str(error)}")

        # Generate recovery plan based on the type of failure
        if "navigation" in failed_step.action.lower():
            return self._handle_navigation_failure(failed_step, current_env)
        elif "manipulation" in failed_step.action.lower():
            return self._handle_manipulation_failure(failed_step, current_env)
        else:
            return self._handle_general_failure(failed_step, current_env)

    def _handle_navigation_failure(self, step: PlanStep, env: Dict[str, Any]) -> List[PlanStep]:
        """Handle navigation failure"""
        # Try alternative route or ask for help
        recovery_steps = [
            PlanStep(
                id=1,
                action="detect_obstacles",
                parameters={"location": step.parameters.get('location')},
                task_type=TaskType.PERCEPTION,
                preconditions=[],
                postconditions=[],
                description="Detect obstacles in path"
            ),
            PlanStep(
                id=2,
                action="request_assistance",
                parameters={"reason": "navigation_failed", "location": step.parameters.get('location')},
                task_type=TaskType.INTERACTION,
                preconditions=[],
                postconditions=[],
                description="Request human assistance for navigation"
            )
        ]
        return recovery_steps

    def _handle_manipulation_failure(self, step: PlanStep, env: Dict[str, Any]) -> List[PlanStep]:
        """Handle manipulation failure"""
        recovery_steps = [
            PlanStep(
                id=1,
                action="relocate_object",
                parameters={"object": step.parameters.get('object_name')},
                task_type=TaskType.NAVIGATION,
                preconditions=[],
                postconditions=[],
                description="Move closer to object for better manipulation"
            ),
            PlanStep(
                id=2,
                action="verify_object_properties",
                parameters={"object": step.parameters.get('object_name')},
                task_type=TaskType.PERCEPTION,
                preconditions=[],
                postconditions=[],
                description="Verify object properties for manipulation"
            )
        ]
        return recovery_steps

    def _handle_general_failure(self, step: PlanStep, env: Dict[str, Any]) -> List[PlanStep]:
        """Handle general failure"""
        recovery_steps = [
            PlanStep(
                id=1,
                action="request_clarification",
                parameters={"step": step.description},
                task_type=TaskType.INTERACTION,
                preconditions=[],
                postconditions=[],
                description="Request clarification about the task"
            )
        ]
        return recovery_steps

class UncertaintyModel:
    """Models uncertainty in the planning process"""

    def __init__(self):
        self.uncertainty_factors = {
            'navigation': 0.1,  # 10% failure rate for navigation
            'manipulation': 0.15,  # 15% failure rate for manipulation
            'perception': 0.05,  # 5% failure rate for perception
            'interaction': 0.08   # 8% failure rate for interaction
        }

    def calculate_success_probability(self, step: PlanStep, env_state: Dict[str, Any]) -> float:
        """Calculate the probability of success for a step"""
        base_prob = 1.0 - self.uncertainty_factors.get(step.task_type.value, 0.1)

        # Adjust based on environmental factors
        if env_state.get('obstacles', 0) > 5 and step.task_type == TaskType.NAVIGATION:
            base_prob *= 0.7  # Reduce success probability in cluttered environment

        if env_state.get('lighting', 'good') == 'poor' and step.task_type == TaskType.MANIPULATION:
            base_prob *= 0.8  # Reduce success probability in poor lighting

        return max(0.0, min(1.0, base_prob))
```

## Multi-Step Task Execution and Monitoring

### Execution Monitoring System

Executing multi-step tasks requires continuous monitoring and the ability to adapt to changing conditions. The monitoring system tracks:

- **Step Completion**: Whether each step has been successfully completed
- **Plan Deviation**: Detection of deviations from the expected plan
- **Recovery Actions**: Execution of recovery steps when failures occur
- **Progress Tracking**: Monitoring overall progress toward the goal

### Task Execution Monitor

```python
from datetime import datetime, timedelta
from enum import Enum

class ExecutionStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    RECOVERING = "recovering"

class TaskExecutionMonitor:
    """Monitors execution of multi-step tasks and handles deviations"""

    def __init__(self, adaptive_planner: AdaptivePlanner):
        self.adaptive_planner = adaptive_planner
        self.current_task = None
        self.current_plan = []
        self.step_status = {}  # {step_id: ExecutionStatus}
        self.execution_start_time = None
        self.last_update_time = None
        self.max_execution_time = 300  # 5 minutes max execution time

    def start_task_execution(self, plan: List[PlanStep], task_description: str):
        """Start execution of a multi-step task"""
        self.current_task = task_description
        self.current_plan = plan
        self.step_status = {step.id: ExecutionStatus.PENDING for step in plan}
        self.execution_start_time = datetime.now()
        self.last_update_time = datetime.now()

        self.get_logger().info(f"Starting execution of task: {task_description}")
        self.get_logger().info(f"Plan contains {len(plan)} steps")

    def update_step_status(self, step_id: int, status: ExecutionStatus, details: str = ""):
        """Update the status of a specific step"""
        if step_id in self.step_status:
            self.step_status[step_id] = status
            self.last_update_time = datetime.now()

            if status == ExecutionStatus.SUCCESS:
                self.get_logger().info(f"Step {step_id} completed successfully")
            elif status == ExecutionStatus.FAILED:
                self.get_logger().error(f"Step {step_id} failed: {details}")
            elif status == ExecutionStatus.EXECUTING:
                self.get_logger().info(f"Step {step_id} started execution")

    def check_execution_timeout(self) -> bool:
        """Check if the current execution has timed out"""
        if self.execution_start_time:
            elapsed = datetime.now() - self.execution_start_time
            return elapsed.total_seconds() > self.max_execution_time
        return False

    def get_overall_progress(self) -> Dict[str, Any]:
        """Get overall progress of the current task"""
        if not self.current_plan:
            return {"status": "no_task", "progress": 0.0}

        completed_steps = sum(1 for status in self.step_status.values()
                             if status in [ExecutionStatus.SUCCESS, ExecutionStatus.FAILED])
        total_steps = len(self.current_plan)
        progress = completed_steps / total_steps if total_steps > 0 else 0.0

        # Determine overall status
        if any(status == ExecutionStatus.FAILED for status in self.step_status.values()):
            overall_status = "failed"
        elif all(status == ExecutionStatus.SUCCESS for status in self.step_status.values()):
            overall_status = "completed"
        elif any(status == ExecutionStatus.EXECUTING for status in self.step_status.values()):
            overall_status = "executing"
        else:
            overall_status = "pending"

        return {
            "status": overall_status,
            "progress": progress,
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "elapsed_time": (datetime.now() - self.execution_start_time).total_seconds() if self.execution_start_time else 0
        }

    def handle_step_failure(self, failed_step: PlanStep, error: Exception,
                          current_env: Dict[str, Any]) -> List[PlanStep]:
        """Handle failure of a step and return recovery plan"""
        self.update_step_status(failed_step.id, ExecutionStatus.FAILED, str(error))

        # Generate recovery plan
        recovery_plan = self.adaptive_planner.handle_execution_failure(
            failed_step, error, current_env
        )

        # Update status to recovering
        for recovery_step in recovery_plan:
            self.step_status[recovery_step.id] = ExecutionStatus.PENDING

        return recovery_plan

    def is_task_complete(self) -> bool:
        """Check if the current task is complete"""
        if not self.current_plan:
            return True

        return all(status == ExecutionStatus.SUCCESS for status in self.step_status.values())

    def get_logger(self):
        """Simple logger for the monitor"""
        class SimpleLogger:
            def info(self, msg):
                print(f"INFO: {msg}")
            def error(self, msg):
                print(f"ERROR: {msg}")
            def warn(self, msg):
                print(f"WARN: {msg}")
        return SimpleLogger()

# Example usage
def example_task_execution():
    # Create components
    llm_planner = LLMPlanner()
    adaptive_planner = AdaptivePlanner(llm_planner)
    monitor = TaskExecutionMonitor(adaptive_planner)

    # Create a sample plan
    sample_plan = [
        PlanStep(1, "navigate_to", {"location": "kitchen"}, TaskType.NAVIGATION, [], [], "Go to kitchen"),
        PlanStep(2, "detect_objects", {"target_object": "cup"}, TaskType.PERCEPTION, [], [], "Find the cup"),
        PlanStep(3, "grasp_object", {"object_name": "cup"}, TaskType.MANIPULATION, [], [], "Pick up the cup"),
        PlanStep(4, "navigate_to", {"location": "table"}, TaskType.NAVIGATION, [], [], "Go to table"),
        PlanStep(5, "place_object", {"location": "table"}, TaskType.MANIPULATION, [], [], "Place cup on table")
    ]

    # Start execution
    monitor.start_task_execution(sample_plan, "Bring cup from kitchen to table")

    # Simulate execution progress
    for i, step in enumerate(sample_plan):
        print(f"Executing step {i+1}: {step.description}")
        monitor.update_step_status(step.id, ExecutionStatus.EXECUTING)

        # Simulate some processing time
        import time
        time.sleep(0.5)

        # Simulate success for most steps, failure for step 3
        if step.id == 3:  # The grasp step
            monitor.update_step_status(step.id, ExecutionStatus.FAILED, "Object not graspable")

            # Handle the failure
            recovery_plan = monitor.handle_step_failure(
                step, Exception("Object not graspable"), {"objects": ["cup"]}
            )
            print(f"Generated recovery plan with {len(recovery_plan)} steps")
        else:
            monitor.update_step_status(step.id, ExecutionStatus.SUCCESS)

    # Get final status
    progress = monitor.get_overall_progress()
    print(f"Final progress: {progress}")

if __name__ == "__main__":
    example_task_execution()
```

## What You Learned

In this chapter, you've learned how to implement sophisticated cognitive planning systems using Large Language Models for robotics applications. You now understand how to decompose complex tasks into executable steps, map natural language commands to robotic actions, integrate planning with ROS 2 navigation and manipulation systems, handle uncertainty and dynamic environments, and monitor multi-step task execution. These capabilities form the cognitive core of VLA systems, enabling robots to understand high-level commands and execute them through complex, adaptive behaviors.