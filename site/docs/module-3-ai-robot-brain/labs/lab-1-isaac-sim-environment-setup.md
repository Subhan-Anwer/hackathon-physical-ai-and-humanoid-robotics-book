---
title: "Lab 1: Isaac Sim Environment Setup"
sidebar_position: 1
---

# Lab 1: Isaac Sim Environment Setup

## Objective

In this lab, you will install and configure NVIDIA Isaac Sim, create a basic simulation environment with obstacles, integrate a robot model with realistic sensors, and implement basic perception and navigation tasks. This foundational lab establishes the simulation environment that will be used throughout the module.

## Prerequisites

- NVIDIA GPU with CUDA support (RTX series recommended)
- Ubuntu 20.04 or 22.04 LTS
- NVIDIA GPU drivers (version 470 or higher)
- Docker and Docker Compose
- Basic knowledge of ROS 2
- Understanding of robotics simulation concepts

## Step-by-Step Instructions

### Step 1: Install NVIDIA Isaac Sim

1. **Verify System Requirements**
   ```bash
   nvidia-smi
   nvcc --version
   ```
   Ensure your system has a compatible NVIDIA GPU with the latest drivers installed.

2. **Download Isaac Sim**
   - Visit the NVIDIA Developer website and download Isaac Sim
   - Create a free NVIDIA Developer account if you don't have one
   - Download the appropriate version for your operating system

3. **Install Isaac Sim**
   ```bash
   # Extract the downloaded package
   tar -xzf isaac_sim-2023.1.1.tar.gz
   cd isaac_sim-2023.1.1

   # Run the setup script
   ./isaac-sim.focal-x86_64.sh
   ```

4. **Launch Isaac Sim**
   ```bash
   # Source the setup script
   source setup_bash.sh

   # Launch Isaac Sim
   ./isaac-sim.sh
   ```

### Step 2: Configure Isaac Sim Environment

1. **Set Up Environment Variables**
   ```bash
   # Add to your ~/.bashrc
   export ISAACSIM_PATH=/path/to/your/isaac_sim
   export ISAACSIM_PYTHON_EXE=/path/to/your/isaac_sim/python.sh
   ```

2. **Verify Installation**
   - Launch Isaac Sim and check that the UI loads correctly
   - Verify that the Omniverse components are functioning
   - Test basic scene loading capabilities

### Step 3: Create a Basic Simulation Environment

1. **Create a New Scene**
   - In Isaac Sim, go to File → New Scene
   - Save the scene as `basic_environment.usd`
   - Set up basic lighting and ground plane

2. **Add Static Obstacles**
   ```python
   # Using the Isaac Sim Python API
   from omni.isaac.core import World
   from omni.isaac.core.utils.prims import create_prim
   from omni.isaac.core.utils.stage import add_reference_to_stage

   # Create various obstacle shapes
   create_prim("/World/Box", "Cube", position=[2.0, 0.0, 0.5], size=1.0)
   create_prim("/World/Cylinder", "Cylinder", position=[-2.0, 1.0, 0.5], radius=0.5, height=1.0)
   create_prim("/World/Sphere", "Sphere", position=[0.0, -2.0, 0.5], radius=0.7)
   ```

3. **Configure Environmental Properties**
   - Adjust lighting conditions to simulate different scenarios
   - Set up collision properties for obstacles
   - Configure friction and material properties

### Step 4: Integrate a Robot Model

1. **Import a Robot Model**
   - Download a sample robot model (e.g., TurtleBot3 or similar)
   - Import the URDF file into Isaac Sim
   - Verify that the robot model loads correctly with proper articulation

2. **Configure Robot Properties**
   ```python
   # Example robot configuration
   from omni.isaac.core.robots import Robot
   from omni.isaac.core.utils.nucleus import get_assets_root_path

   # Load robot from USD
   add_reference_to_stage(
       usd_path="/Isaac/Robots/TurtleBot3/turtlebot3.usd",
       prim_path="/World/Robot"
   )
   ```

3. **Set Up Robot Controllers**
   - Configure joint controllers for robot movement
   - Set up ROS 2 bridge for communication
   - Verify that robot responds to basic commands

### Step 5: Add Realistic Sensors

1. **Configure Camera Sensors**
   ```python
   from omni.isaac.sensor import Camera

   # Add RGB camera to the robot
   camera = Camera(
       prim_path="/World/Robot/chassis/camera",
       frequency=30,
       resolution=(640, 480)
   )
   camera.initialize()
   ```

2. **Add LIDAR Sensor**
   ```python
   from omni.isaac.range_sensor import _range_sensor
   lidar_interface = _range_sensor.acquire_lidar_sensor_interface()

   # Configure LIDAR parameters
   lidar_config = {
       "rotation_frequency": 10,
       "points_per_second": 500000,
       "laser_as_line": False,
       "enable_strong_directional_laser": False
   }
   ```

3. **Add IMU Sensor**
   - Configure IMU with appropriate noise models
   - Set up proper mounting position on the robot
   - Verify sensor data publication

### Step 6: Implement Basic Perception Tasks

1. **Set Up ROS 2 Bridge**
   ```bash
   # Install Isaac ROS bridge
   sudo apt update
   sudo apt install ros-humble-isaac-ros-common
   ```

2. **Create Perception Node**
   ```python
   # perception_node.py
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import Image, LaserScan
   from cv_bridge import CvBridge

   class PerceptionNode(Node):
       def __init__(self):
           super().__init__('perception_node')
           self.bridge = CvBridge()

           # Subscribe to camera and LIDAR data
           self.image_sub = self.create_subscription(
               Image, '/camera/image_raw', self.image_callback, 10)
           self.lidar_sub = self.create_subscription(
               LaserScan, '/scan', self.lidar_callback, 10)

       def image_callback(self, msg):
           cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
           # Process image data
           self.get_logger().info(f"Received image: {cv_image.shape}")

       def lidar_callback(self, msg):
           # Process LIDAR data
           self.get_logger().info(f"LIDAR range count: {len(msg.ranges)}")

   def main(args=None):
       rclpy.init(args=args)
       perception_node = PerceptionNode()
       rclpy.spin(perception_node)
       perception_node.destroy_node()
       rclpy.shutdown()
   ```

### Step 7: Implement Basic Navigation Tasks

1. **Configure Navigation Stack**
   - Set up Navigation2 configuration files
   - Configure costmap parameters for your environment
   - Test basic navigation with simple goals

2. **Create Navigation Test Script**
   ```python
   # navigation_test.py
   import rclpy
   from rclpy.action import ActionClient
   from nav2_msgs.action import NavigateToPose
   from geometry_msgs.msg import PoseStamped
   import math

   class NavigationTest(Node):
       def __init__(self):
           super().__init__('navigation_test')
           self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

       def navigate_to_goal(self, x, y, theta):
           # Wait for navigation server
           self.nav_client.wait_for_server()

           # Create goal pose
           goal_msg = NavigateToPose.Goal()
           goal_msg.pose.header.frame_id = 'map'
           goal_msg.pose.pose.position.x = x
           goal_msg.pose.pose.position.y = y
           goal_msg.pose.pose.orientation.z = math.sin(theta / 2.0)
           goal_msg.pose.pose.orientation.w = math.cos(theta / 2.0)

           # Send goal
           self.nav_client.send_goal_async(goal_msg)
   ```

## Expected Outcome

Upon completion of this lab, you should have:

- A fully configured Isaac Sim environment running on your system
- A basic simulation scene with obstacles and a robot model
- Properly configured sensors (camera, LIDAR, IMU) on the robot
- Working ROS 2 bridge connecting simulation to external nodes
- Basic perception and navigation capabilities demonstrated

The robot should be able to:
- Navigate to specified goals while avoiding obstacles
- Publish sensor data through ROS 2 topics
- Respond to external commands from perception nodes

## Troubleshooting

- **Isaac Sim fails to launch**: Verify GPU drivers and CUDA installation
- **Robot model not loading**: Check URDF file paths and dependencies
- **ROS 2 bridge not connecting**: Ensure correct network configuration and ROS_DOMAIN_ID
- **Sensors not publishing**: Verify sensor configuration and frame IDs

## Optional Extension Tasks

1. **Advanced Environment Creation**: Create a more complex environment with multiple rooms, furniture, and dynamic obstacles.

2. **Multi-Robot Setup**: Configure multiple robots in the same environment with coordinated navigation.

3. **Realistic Lighting**: Implement dynamic lighting conditions to simulate day/night cycles and varying illumination.

4. **Physics Property Tuning**: Fine-tune friction, mass, and other physical properties to match real-world robot characteristics.

## Summary

This lab established the foundational Isaac Sim environment with a robot model and sensors. You've learned to configure the simulation environment, integrate robot models, and set up basic perception and navigation capabilities. This setup will serve as the platform for more advanced robotics development in subsequent labs.
