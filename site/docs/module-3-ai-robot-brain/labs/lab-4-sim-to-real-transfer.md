---
title: "Lab 4: Sim-to-Real Transfer"
sidebar_position: 4
---

# Lab 4: Sim-to-Real Transfer

## Objective

In this lab, you will apply domain randomization techniques to improve sim-to-real transfer, train robot behaviors in simulation, transfer learned behaviors to real hardware, and evaluate performance and adaptation strategies. This lab demonstrates the complete pipeline from simulation-based training to real-world deployment and validation.

## Prerequisites

- Completed all previous labs (Lab 3.1 through Lab 3.3)
- Access to a physical robot platform (TurtleBot3, Husky, or similar)
- Simulation environment with trained behaviors from previous labs
- Understanding of reinforcement learning concepts
- Basic knowledge of robot calibration procedures

## Step-by-Step Instructions

### Step 1: Apply Domain Randomization Techniques

1. **Configure Domain Randomization in Isaac Sim**
   ```python
   # domain_randomization_config.py
   from omni.isaac.core.utils.prims import get_prim_at_path
   from omni.isaac.core.utils.stage import get_current_stage
   import carb
   import numpy as np
   import random

   class DomainRandomization:
       def __init__(self):
           self.stage = get_current_stage()
           self.randomization_params = {
               'lighting': {
                   'intensity_range': (0.5, 2.0),
                   'color_temperature_range': (3000, 8000),
                   'direction_range': (0, 360)
               },
               'textures': {
                   'roughness_range': (0.0, 1.0),
                   'metallic_range': (0.0, 1.0),
                   'albedo_range': (0.0, 1.0)
               },
               'physics': {
                   'friction_range': (0.1, 1.0),
                   'restitution_range': (0.0, 0.5),
                   'mass_variance': 0.1
               },
               'sensor_noise': {
                   'camera_noise': 0.01,
                   'lidar_noise': 0.02,
                   'imu_drift': 0.001
               }
           }

       def randomize_lighting(self):
           """Randomize lighting conditions in the scene"""
           lights = self.get_all_lights()
           for light in lights:
               # Randomize intensity
               intensity = random.uniform(
                   self.randomization_params['lighting']['intensity_range'][0],
                   self.randomization_params['lighting']['intensity_range'][1]
               )
               light.GetAttribute("intensity").Set(intensity)

               # Randomize color temperature
               color_temp = random.uniform(
                   self.randomization_params['lighting']['color_temperature_range'][0],
                   self.randomization_params['lighting']['color_temperature_range'][1]
               )
               # Convert color temperature to RGB (simplified)
               rgb = self.color_temperature_to_rgb(color_temp)
               light.GetAttribute("color").Set(carb.Float3(rgb[0], rgb[1], rgb[2]))

       def randomize_textures(self):
           """Randomize surface textures and materials"""
           # Get all objects in the scene
           objects = self.get_all_objects()
           for obj in objects:
               # Randomize material properties
               roughness = random.uniform(
                   self.randomization_params['textures']['roughness_range'][0],
                   self.randomization_params['textures']['roughness_range'][1]
               )
               metallic = random.uniform(
                   self.randomization_params['textures']['metallic_range'][0],
                   self.randomization_params['textures']['metallic_range'][1]
               )
               albedo = random.uniform(
                   self.randomization_params['textures']['albedo_range'][0],
                   self.randomization_params['textures']['albedo_range'][1]
               )

               # Apply to material
               self.apply_material_properties(obj, roughness, metallic, albedo)

       def randomize_physics(self):
           """Randomize physics properties"""
           objects = self.get_all_objects()
           for obj in objects:
               # Randomize friction
               friction = random.uniform(
                   self.randomization_params['physics']['friction_range'][0],
                   self.randomization_params['physics']['friction_range'][1]
               )
               obj.GetAttribute("physics:staticFriction").Set(friction)
               obj.GetAttribute("physics:dynamicFriction").Set(friction)

               # Randomize mass with variance
               base_mass = obj.GetAttribute("physics:mass").Get()
               variance = self.randomization_params['physics']['mass_variance']
               new_mass = base_mass * random.uniform(1-variance, 1+variance)
               obj.GetAttribute("physics:mass").Set(new_mass)

       def add_sensor_noise(self, sensor_data):
           """Add realistic noise to sensor data"""
           # Add noise to camera data
           if 'camera' in sensor_data:
               noise = np.random.normal(0, self.randomization_params['sensor_noise']['camera_noise'], sensor_data['camera'].shape)
               sensor_data['camera'] = sensor_data['camera'] + noise
               sensor_data['camera'] = np.clip(sensor_data['camera'], 0, 255)

           # Add noise to LIDAR data
           if 'lidar' in sensor_data:
               noise = np.random.normal(0, self.randomization_params['sensor_noise']['lidar_noise'], len(sensor_data['lidar']))
               sensor_data['lidar'] = sensor_data['lidar'] + noise
               sensor_data['lidar'] = np.maximum(sensor_data['lidar'], 0)

           # Simulate IMU drift
           if 'imu' in sensor_data:
               drift = np.random.normal(0, self.randomization_params['sensor_noise']['imu_drift'], 6)
               sensor_data['imu'] = sensor_data['imu'] + drift

           return sensor_data

       def color_temperature_to_rgb(self, color_temp):
           """Convert color temperature to RGB values (simplified approximation)"""
           temp = color_temp / 100
           if temp <= 66:
               red = 255
               green = temp
               green = 99.4708025861 * math.log(green) - 161.1195681661
           else:
               red = temp - 60
               red = 329.698727446 * (red ** -0.1332047592)
               green = temp - 60
               green = 288.1221695283 * (green ** -0.0755148492)

           if temp >= 66:
               blue = 255
           elif temp <= 19:
               blue = 0
           else:
               blue = temp - 10
               blue = 138.5177312231 * math.log(blue) - 305.0447927307

           return [max(0, min(255, red))/255, max(0, min(255, green))/255, max(0, min(255, blue))/255]

       def get_all_lights(self):
           """Get all light prims in the scene"""
           from pxr import UsdLux
           lights = []
           for prim in self.stage.TraverseAll():
               if prim.IsA(UsdLux.DistantLight) or prim.IsA(UsdLux.DiskLight) or prim.IsA(UsdLux.SphereLight):
                   lights.append(prim)
           return lights

       def get_all_objects(self):
           """Get all object prims in the scene"""
           objects = []
           for prim in self.stage.TraverseAll():
               if prim.GetTypeName() in ['Cube', 'Sphere', 'Cylinder', 'Mesh']:
                   objects.append(prim)
           return objects

       def apply_material_properties(self, obj, roughness, metallic, albedo):
           """Apply material properties to an object"""
           # This is a simplified implementation
           # In practice, you'd use Isaac Sim's material system
           pass

   # Usage in simulation training loop
   dr = DomainRandomization()
   ```

2. **Implement Randomization During Training**
   ```python
   # training_with_randomization.py
   import rclpy
   from rclpy.node import Node
   import numpy as np
   import torch
   import torch.nn as nn
   import torch.optim as optim

   class TrainingWithRandomization(Node):
       def __init__(self):
           super().__init__('training_with_randomization')

           # Initialize domain randomization
           self.domain_randomizer = DomainRandomization()

           # Neural network for policy
           self.policy_network = self.create_policy_network()
           self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)

           # Training parameters
           self.episode_count = 0
           self.max_episodes = 1000

           # Apply randomization every N steps
           self.randomization_frequency = 10

       def train_episode(self):
           """Train one episode with domain randomization"""
           # Apply domain randomization
           if self.episode_count % self.randomization_frequency == 0:
               self.domain_randomizer.randomize_lighting()
               self.domain_randomizer.randomize_textures()
               self.domain_randomizer.randomize_physics()
               self.get_logger().info(f"Applied domain randomization at episode {self.episode_count}")

           # Reset environment
           obs = self.reset_environment()
           done = False
           episode_reward = 0

           while not done:
               # Get action from policy
               action = self.get_action(obs)

               # Execute action
               next_obs, reward, done = self.step_environment(action)

               # Add sensor noise
               next_obs = self.domain_randomizer.add_sensor_noise(next_obs)

               # Store transition
               self.store_transition(obs, action, reward, next_obs, done)

               # Update policy
               self.update_policy()

               obs = next_obs
               episode_reward += reward

           self.episode_count += 1
           self.get_logger().info(f"Episode {self.episode_count}, Reward: {episode_reward}")

       def create_policy_network(self):
           """Create neural network for policy"""
           class PolicyNetwork(nn.Module):
               def __init__(self, input_size, output_size):
                   super(PolicyNetwork, self).__init__()
                   self.fc1 = nn.Linear(input_size, 256)
                   self.fc2 = nn.Linear(256, 256)
                   self.fc3 = nn.Linear(256, output_size)
                   self.relu = nn.ReLU()
                   self.softmax = nn.Softmax(dim=-1)

               def forward(self, x):
                   x = self.relu(self.fc1(x))
                   x = self.relu(self.fc2(x))
                   x = self.fc3(x)
                   return self.softmax(x)

           return PolicyNetwork(input_size=24, output_size=4)  # Adjust sizes as needed

       def get_action(self, obs):
           """Get action from policy network"""
           obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
           action_probs = self.policy_network(obs_tensor)
           action = torch.multinomial(action_probs, 1).item()
           return action

       def update_policy(self):
           """Update policy using stored transitions"""
           # Implement policy update (e.g., PPO, DQN, etc.)
           pass

       def store_transition(self, obs, action, reward, next_obs, done):
           """Store transition in replay buffer"""
           pass

       def reset_environment(self):
           """Reset simulation environment"""
           pass

       def step_environment(self, action):
           """Execute action in environment"""
           pass

   def main(args=None):
       rclpy.init(args=args)
       trainer = TrainingWithRandomization()

       # Run training episodes
       while trainer.episode_count < trainer.max_episodes:
           trainer.train_episode()

       # Save trained model
       torch.save(trainer.policy_network.state_dict(), "trained_policy.pth")

       trainer.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

### Step 2: Train Robot Behaviors in Simulation

1. **Set Up Reinforcement Learning Environment**
   ```python
   # rl_environment.py
   import gym
   from gym import spaces
   import numpy as np
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import LaserScan, Image
   from geometry_msgs.msg import Twist, Pose
   from nav_msgs.msg import Odometry
   from cv_bridge import CvBridge

   class RobotRLEnvironment(gym.Env):
       def __init__(self):
           super(RobotRLEnvironment, self).__init__()

           # Define action and observation spaces
           self.action_space = spaces.Discrete(4)  # Forward, backward, left, right
           self.observation_space = spaces.Box(
               low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32
           )  # 20 LIDAR readings + 3 pose values + 1 velocity

           # ROS setup
           self.node = rclpy.create_node('rl_environment')
           self.bridge = CvBridge()

           # Publishers and subscribers
           self.cmd_vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
           self.laser_sub = self.node.create_subscription(
               LaserScan, '/scan', self.laser_callback, 10)
           self.odom_sub = self.node.create_subscription(
               Odometry, '/odom', self.odom_callback, 10)

           # State variables
           self.laser_data = np.zeros(20)
           self.pose = np.zeros(3)  # x, y, theta
           self.velocity = 0.0
           self.collision = False

           # Episode parameters
           self.max_steps = 1000
           self.current_step = 0
           self.goal_position = np.array([5.0, 5.0])
           self.start_position = np.array([0.0, 0.0])

       def laser_callback(self, msg):
           """Handle laser scan data"""
           # Take 20 readings evenly spaced around 360 degrees
           num_readings = 20
           step = len(msg.ranges) // num_readings
           self.laser_data = np.array([msg.ranges[i*step] for i in range(num_readings)])
           self.laser_data = np.nan_to_num(self.laser_data, nan=10.0, posinf=10.0, neginf=0.0)

       def odom_callback(self, msg):
           """Handle odometry data"""
           self.pose[0] = msg.pose.pose.position.x
           self.pose[1] = msg.pose.pose.position.y

           # Convert quaternion to euler
           orientation = msg.pose.pose.orientation
           self.pose[2] = self.quaternion_to_euler(orientation)

           self.velocity = np.sqrt(
               msg.twist.twist.linear.x**2 +
               msg.twist.twist.linear.y**2 +
               msg.twist.twist.linear.z**2
           )

       def quaternion_to_euler(self, quat):
           """Convert quaternion to euler angle (simplified for z-axis rotation)"""
           siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
           cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
           return np.arctan2(siny_cosp, cosy_cosp)

       def step(self, action):
           """Execute one step in the environment"""
           # Convert action to velocity command
           cmd_vel = Twist()
           if action == 0:  # Forward
               cmd_vel.linear.x = 0.5
               cmd_vel.angular.z = 0.0
           elif action == 1:  # Backward
               cmd_vel.linear.x = -0.5
               cmd_vel.angular.z = 0.0
           elif action == 2:  # Turn left
               cmd_vel.linear.x = 0.0
               cmd_vel.angular.z = 0.5
           elif action == 3:  # Turn right
               cmd_vel.linear.x = 0.0
               cmd_vel.angular.z = -0.5

           # Publish command
           self.cmd_vel_pub.publish(cmd_vel)

           # Wait for next observation
           rclpy.spin_once(self.node, timeout_sec=0.1)

           # Check for collision
           min_distance = np.min(self.laser_data)
           self.collision = min_distance < 0.3

           # Calculate reward
           reward = self.calculate_reward()

           # Check if episode is done
           current_pos = np.array([self.pose[0], self.pose[1]])
           distance_to_goal = np.linalg.norm(current_pos - self.goal_position)
           done = (self.collision or
                   distance_to_goal < 0.5 or
                   self.current_step >= self.max_steps)

           # Prepare observation
           obs = np.concatenate([
               self.laser_data,
               self.pose,
               [self.velocity]
           ]).astype(np.float32)

           self.current_step += 1

           return obs, reward, done, {}

       def calculate_reward(self):
           """Calculate reward based on current state"""
           current_pos = np.array([self.pose[0], self.pose[1]])
           distance_to_goal = np.linalg.norm(current_pos - self.goal_position)

           # Reward for getting closer to goal
           reward = -distance_to_goal * 0.1

           # Bonus for reaching goal
           if distance_to_goal < 0.5:
               reward += 100

           # Penalty for collision
           if self.collision:
               reward -= 100

           # Small penalty for each step to encourage efficiency
           reward -= 0.1

           return reward

       def reset(self):
           """Reset the environment"""
           self.current_step = 0

           # Reset robot position (in simulation this would reset the robot)
           # For this example, we'll just reset our tracking variables
           self.laser_data = np.zeros(20)
           self.pose = np.zeros(3)
           self.velocity = 0.0
           self.collision = False

           # In a real setup, you would reset the robot in the simulator
           # For now, return a random initial observation
           obs = np.concatenate([
               np.random.uniform(1, 10, 20),  # Simulated laser data
               np.random.uniform(-1, 1, 3),   # Simulated pose
               [0.0]                          # Simulated velocity
           ]).astype(np.float32)

           return obs

       def close(self):
           """Clean up resources"""
           self.node.destroy_node()
   ```

2. **Train Navigation Policy**
   ```python
   # train_navigation_policy.py
   import torch
   import torch.nn as nn
   import torch.optim as optim
   import numpy as np
   from collections import deque
   import random

   class DQNetwork(nn.Module):
       def __init__(self, input_size, output_size, hidden_size=128):
           super(DQNetwork, self).__init__()
           self.fc1 = nn.Linear(input_size, hidden_size)
           self.fc2 = nn.Linear(hidden_size, hidden_size)
           self.fc3 = nn.Linear(hidden_size, hidden_size)
           self.fc4 = nn.Linear(hidden_size, output_size)
           self.relu = nn.ReLU()
           self.dropout = nn.Dropout(0.2)

       def forward(self, x):
           x = self.relu(self.fc1(x))
           x = self.dropout(x)
           x = self.relu(self.fc2(x))
           x = self.dropout(x)
           x = self.relu(self.fc3(x))
           x = self.fc4(x)
           return x

   class DQNAgent:
       def __init__(self, state_size, action_size, lr=0.001):
           self.state_size = state_size
           self.action_size = action_size
           self.memory = deque(maxlen=10000)
           self.epsilon = 1.0  # exploration rate
           self.epsilon_min = 0.01
           self.epsilon_decay = 0.995
           self.learning_rate = lr
           self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

           self.q_network = DQNetwork(state_size, action_size).to(self.device)
           self.target_network = DQNetwork(state_size, action_size).to(self.device)
           self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

           # Update target network
           self.update_target_network()

       def update_target_network(self):
           self.target_network.load_state_dict(self.q_network.state_dict())

       def remember(self, state, action, reward, next_state, done):
           self.memory.append((state, action, reward, next_state, done))

       def act(self, state):
           if np.random.random() <= self.epsilon:
               return random.randrange(self.action_size)

           state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
           q_values = self.q_network(state_tensor)
           return np.argmax(q_values.cpu().data.numpy())

       def replay(self, batch_size=32):
           if len(self.memory) < batch_size:
               return

           batch = random.sample(self.memory, batch_size)
           states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
           actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
           rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
           next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
           dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)

           current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
           next_q_values = self.target_network(next_states).max(1)[0].detach()
           target_q_values = rewards + (0.99 * next_q_values * ~dones)

           loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

           self.optimizer.zero_grad()
           loss.backward()
           self.optimizer.step()

           if self.epsilon > self.epsilon_min:
               self.epsilon *= self.epsilon_decay

       def load(self, name):
           self.q_network.load_state_dict(torch.load(name))

       def save(self, name):
           torch.save(self.q_network.state_dict(), name)

   def train_agent():
       env = RobotRLEnvironment()
       state_size = env.observation_space.shape[0]
       action_size = env.action_space.n
       agent = DQNAgent(state_size, action_size)

       episodes = 1000
       batch_size = 32
       scores = deque(maxlen=100)

       for e in range(episodes):
           state = env.reset()
           total_reward = 0

           for time in range(1000):
               action = agent.act(state)
               next_state, reward, done, _ = env.step(action)
               agent.remember(state, action, reward, next_state, done)
               state = next_state
               total_reward += reward

               if done:
                   break

               if len(agent.memory) > batch_size:
                   agent.replay(batch_size)

           scores.append(total_reward)
           avg_score = np.mean(scores)

           print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}")

           # Update target network every 100 episodes
           if e % 100 == 0:
               agent.update_target_network()

       # Save the trained model
       agent.save("navigation_policy.pth")
       print("Model saved as navigation_policy.pth")

   if __name__ == "__main__":
       train_agent()
   ```

### Step 3: Transfer Learned Behaviors to Real Hardware

1. **Prepare for Real Robot Deployment**
   ```python
   # real_robot_interface.py
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import LaserScan, Image
   from geometry_msgs.msg import Twist, Pose
   from nav_msgs.msg import Odometry
   from std_msgs.msg import String
   import torch
   import numpy as np
   from cv_bridge import CvBridge

   class RealRobotInterface(Node):
       def __init__(self):
           super().__init__('real_robot_interface')
           self.bridge = CvBridge()

           # Load trained policy
           self.policy_network = self.load_policy_network()

           # Robot interface
           self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
           self.laser_sub = self.create_subscription(
               LaserScan, '/scan', self.laser_callback, 10)
           self.odom_sub = self.create_subscription(
               Odometry, '/odom', self.odom_callback, 10)

           # State variables
           self.laser_data = np.zeros(20)
           self.pose = np.zeros(3)
           self.velocity = 0.0
           self.is_running = False

           # Parameters
           self.action_frequency = 10  # Hz
           self.timer = self.create_timer(1.0/self.action_frequency, self.control_loop)

           # Goal position
           self.goal_position = np.array([5.0, 5.0])

           self.get_logger().info("Real Robot Interface initialized")

       def load_policy_network(self):
           """Load the trained policy network"""
           # Define the same network architecture as during training
           class PolicyNetwork(torch.nn.Module):
               def __init__(self, input_size, output_size, hidden_size=128):
                   super(PolicyNetwork, self).__init__()
                   self.fc1 = torch.nn.Linear(input_size, hidden_size)
                   self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
                   self.fc3 = torch.nn.Linear(hidden_size, hidden_size)
                   self.fc4 = torch.nn.Linear(hidden_size, output_size)
                   self.relu = torch.nn.ReLU()

               def forward(self, x):
                   x = self.relu(self.fc1(x))
                   x = self.relu(self.fc2(x))
                   x = self.relu(self.fc3(x))
                   x = self.fc4(x)
                   return x

           policy = PolicyNetwork(input_size=24, output_size=4)

           try:
               policy.load_state_dict(torch.load("navigation_policy.pth", map_location=torch.device('cpu')))
               policy.eval()
               self.get_logger().info("Policy model loaded successfully")
               return policy
           except Exception as e:
               self.get_logger().error(f"Failed to load policy: {e}")
               return None

       def laser_callback(self, msg):
           """Handle laser scan data from real robot"""
           # Process laser data similar to simulation
           num_readings = 20
           step = len(msg.ranges) // num_readings
           self.laser_data = np.array([msg.ranges[i*step] for i in range(num_readings)])
           self.laser_data = np.nan_to_num(self.laser_data, nan=10.0, posinf=10.0, neginf=0.0)

       def odom_callback(self, msg):
           """Handle odometry data from real robot"""
           self.pose[0] = msg.pose.pose.position.x
           self.pose[1] = msg.pose.pose.position.y

           # Convert quaternion to euler
           orientation = msg.pose.pose.orientation
           self.pose[2] = self.quaternion_to_euler(orientation)

           self.velocity = np.sqrt(
               msg.twist.twist.linear.x**2 +
               msg.twist.twist.linear.y**2 +
               msg.twist.twist.linear.z**2
           )

       def quaternion_to_euler(self, quat):
           """Convert quaternion to euler angle"""
           siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
           cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
           return np.arctan2(siny_cosp, cosy_cosp)

       def control_loop(self):
           """Main control loop - get action from policy and execute"""
           if not self.is_running or self.policy_network is None:
               return

           # Prepare observation
           obs = np.concatenate([
               self.laser_data,
               self.pose,
               [self.velocity]
           ]).astype(np.float32)

           # Get action from policy
           with torch.no_grad():
               obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
               action_probs = self.policy_network(obs_tensor)
               action = torch.argmax(action_probs, dim=1).item()

           # Convert action to velocity command
           cmd_vel = Twist()
           if action == 0:  # Forward
               cmd_vel.linear.x = 0.3
               cmd_vel.angular.z = 0.0
           elif action == 1:  # Backward
               cmd_vel.linear.x = -0.2
               cmd_vel.angular.z = 0.0
           elif action == 2:  # Turn left
               cmd_vel.linear.x = 0.0
               cmd_vel.angular.z = 0.4
           elif action == 3:  # Turn right
               cmd_vel.linear.x = 0.0
               cmd_vel.angular.z = -0.4

           # Publish command
           self.cmd_vel_pub.publish(cmd_vel)

       def start_navigation(self):
           """Start the navigation process"""
           self.is_running = True
           self.get_logger().info("Navigation started")

       def stop_navigation(self):
           """Stop the navigation process"""
           self.is_running = False
           # Stop the robot
           cmd_vel = Twist()
           self.cmd_vel_pub.publish(cmd_vel)
           self.get_logger().info("Navigation stopped")

   def main(args=None):
       rclpy.init(args=args)
       robot_interface = RealRobotInterface()

       # Start navigation
       robot_interface.start_navigation()

       try:
           rclpy.spin(robot_interface)
       except KeyboardInterrupt:
           robot_interface.stop_navigation()

       robot_interface.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

2. **Implement Sim-to-Real Adaptation**
   ```python
   # adaptation_module.py
   import numpy as np
   import torch
   import torch.nn as nn
   from collections import deque
   import statistics

   class AdaptationModule:
       def __init__(self, policy_network, adaptation_rate=0.01):
           self.policy_network = policy_network
           self.adaptation_rate = adaptation_rate

           # Track performance metrics
           self.performance_history = deque(maxlen=100)
           self.laser_stats = {
               'mean': deque(maxlen=50),
               'std': deque(maxlen=50)
           }

           # Adaptation parameters
           self.sim_laser_mean = 5.0  # Average from simulation
           self.sim_laser_std = 2.0   # Std from simulation

           # Real robot calibration
           self.real_laser_mean = 5.0
           self.real_laser_std = 2.0

           # Initialize normalization parameters
           self.update_calibration()

       def update_calibration(self):
           """Update calibration based on recent real-world data"""
           if len(self.laser_stats['mean']) > 10:
               self.real_laser_mean = statistics.mean(self.laser_stats['mean'])
               self.real_laser_std = statistics.mean(self.laser_stats['std'])

       def normalize_laser_data(self, laser_data):
           """Normalize laser data from real robot to match simulation distribution"""
           # Calculate current real-world statistics
           current_mean = np.mean(laser_data)
           current_std = np.std(laser_data)

           # Update statistics
           self.laser_stats['mean'].append(current_mean)
           self.laser_stats['std'].append(current_std)

           # Normalize to simulation distribution
           normalized = (laser_data - current_mean) / (current_std + 1e-8)
           normalized = normalized * self.sim_laser_std + self.sim_laser_mean

           # Clip to reasonable values
           normalized = np.clip(normalized, 0.1, 10.0)

           return normalized

       def adapt_policy(self, state, action, reward, next_state):
           """Adapt policy based on real-world experience"""
           # Add small adaptation based on performance
           performance = reward  # Simplified performance metric
           self.performance_history.append(performance)

           # If performance is consistently low, adjust exploration
           if len(self.performance_history) >= 10:
               avg_performance = statistics.mean(list(self.performance_history)[-10:])
               if avg_performance < -5:  # Threshold for poor performance
                   # Increase exploration temporarily
                   self.adaptation_rate *= 1.1
               else:
                   # Decrease adaptation rate back to normal
                   self.adaptation_rate = max(self.adaptation_rate * 0.99, 0.01)

       def process_observation(self, obs):
           """Process observation from real robot before policy evaluation"""
           # Extract laser data (first 20 elements)
           laser_data = obs[:20]
           other_data = obs[20:]

           # Normalize laser data
           normalized_laser = self.normalize_laser_data(laser_data)

           # Combine normalized data
           processed_obs = np.concatenate([normalized_laser, other_data])

           return processed_obs
   ```

### Step 4: Evaluate Performance and Adaptation Strategies

1. **Create Evaluation Framework**
   ```python
   # evaluation_framework.py
   import rclpy
   from rclpy.node import Node
   from geometry_msgs.msg import PoseStamped
   from nav_msgs.msg import Path
   from std_msgs.msg import Float32
   import numpy as np
   import time
   from collections import deque

   class EvaluationFramework(Node):
       def __init__(self):
           super().__init__('evaluation_framework')

           # Publishers for metrics
           self.success_rate_pub = self.create_publisher(Float32, '/eval/success_rate', 10)
           self.path_efficiency_pub = self.create_publisher(Float32, '/eval/path_efficiency', 10)
           self.collision_rate_pub = self.create_publisher(Float32, '/eval/collision_rate', 10)
           self.time_to_goal_pub = self.create_publisher(Float32, '/eval/time_to_goal', 10)

           # Subscribers
           self.pose_sub = self.create_subscription(
               PoseStamped, '/robot_pose', self.pose_callback, 10)
           self.path_sub = self.create_subscription(
               Path, '/robot_path', self.path_callback, 10)

           # Evaluation parameters
           self.goal_position = np.array([5.0, 5.0])
           self.start_time = None
           self.current_pose = np.array([0.0, 0.0])
           self.path_history = deque(maxlen=1000)
           self.evaluation_trials = 0
           self.successful_trials = 0
           self.collision_trials = 0
           self.total_path_length = 0.0

           # Evaluation timer
           self.eval_timer = self.create_timer(1.0, self.evaluate_performance)

       def pose_callback(self, msg):
           """Update current pose"""
           self.current_pose = np.array([msg.pose.position.x, msg.pose.position.y])

           # Record path if we have a start time
           if self.start_time is not None:
               self.path_history.append(self.current_pose.copy())

       def path_callback(self, msg):
           """Record path data"""
           pass  # Path is recorded via pose updates

       def start_evaluation_trial(self):
           """Start a new evaluation trial"""
           self.start_time = time.time()
           self.path_history.clear()
           self.current_pose = np.array([0.0, 0.0])  # Reset to start position
           self.evaluation_trials += 1
           self.get_logger().info(f"Started evaluation trial {self.evaluation_trials}")

       def end_evaluation_trial(self, success, collision=False):
           """End current evaluation trial"""
           if self.start_time is None:
               return

           trial_time = time.time() - self.start_time
           self.start_time = None

           if success:
               self.successful_trials += 1
               self.total_path_length += self.calculate_path_length()

               # Calculate metrics
               success_rate = self.successful_trials / self.evaluation_trials
               path_efficiency = self.calculate_path_efficiency()

               # Publish metrics
               self.publish_metrics(success_rate, path_efficiency, trial_time, collision)

               self.get_logger().info(f"Trial completed - Success: {success}, Time: {trial_time:.2f}s")
           else:
               if collision:
                   self.collision_trials += 1
               self.get_logger().info(f"Trial failed - Success: {success}")

       def calculate_path_length(self):
           """Calculate total path length"""
           if len(self.path_history) < 2:
               return 0.0

           total_length = 0.0
           for i in range(1, len(self.path_history)):
               dist = np.linalg.norm(self.path_history[i] - self.path_history[i-1])
               total_length += dist

           return total_length

       def calculate_path_efficiency(self):
           """Calculate path efficiency as optimal path length / actual path length"""
           if len(self.path_history) == 0:
               return 0.0

           actual_path_length = self.calculate_path_length()
           optimal_distance = np.linalg.norm(self.goal_position - np.array([0.0, 0.0]))

           if actual_path_length == 0:
               return 0.0

           efficiency = optimal_distance / actual_path_length
           return min(efficiency, 1.0)  # Cap at 1.0

       def evaluate_performance(self):
           """Periodically evaluate performance"""
           if self.start_time is not None:
               # Check if reached goal
               distance_to_goal = np.linalg.norm(self.current_pose - self.goal_position)
               if distance_to_goal < 0.5:  # Within goal tolerance
                   self.end_evaluation_trial(success=True)

               # Check if timed out
               elif time.time() - self.start_time > 120:  # 2 minutes timeout
                   self.end_evaluation_trial(success=False)

       def publish_metrics(self, success_rate, path_efficiency, time_to_goal, collision):
           """Publish evaluation metrics"""
           success_msg = Float32()
           success_msg.data = success_rate
           self.success_rate_pub.publish(success_msg)

           efficiency_msg = Float32()
           efficiency_msg.data = path_efficiency
           self.path_efficiency_pub.publish(efficiency_msg)

           collision_msg = Float32()
           collision_msg.data = 1.0 if collision else 0.0
           self.collision_rate_pub.publish(collision_msg)

           time_msg = Float32()
           time_msg.data = time_to_goal
           self.time_to_goal_pub.publish(time_msg)

       def get_current_metrics(self):
           """Get current evaluation metrics"""
           success_rate = self.successful_trials / max(self.evaluation_trials, 1)
           collision_rate = self.collision_trials / max(self.evaluation_trials, 1)

           return {
               'success_rate': success_rate,
               'collision_rate': collision_rate,
               'trials_completed': self.evaluation_trials,
               'successful_trials': self.successful_trials
           }

   def run_evaluation():
       """Run evaluation for specified number of trials"""
       rclpy.init()
       evaluator = EvaluationFramework()

       # Run multiple trials
       num_trials = 20
       for trial in range(num_trials):
           evaluator.get_logger().info(f"Starting trial {trial + 1}/{num_trials}")
           evaluator.start_evaluation_trial()

           # Wait for trial to complete (in practice, this would be event-driven)
           import time
           time.sleep(120)  # Max time per trial

           # Force end if still running
           if evaluator.start_time is not None:
               evaluator.end_evaluation_trial(success=False)

       # Print final metrics
       metrics = evaluator.get_current_metrics()
       evaluator.get_logger().info(f"Final metrics: {metrics}")

       evaluator.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       run_evaluation()
   ```

2. **Implement Performance Comparison Tools**
   ```python
   # performance_comparison.py
   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
   import seaborn as sns

   class PerformanceComparison:
       def __init__(self):
           self.simulation_data = []
           self.real_world_data = []
           self.metrics = ['success_rate', 'path_efficiency', 'collision_rate', 'time_to_goal']

       def add_simulation_data(self, data_point):
           """Add simulation performance data"""
           self.simulation_data.append(data_point)

       def add_real_world_data(self, data_point):
           """Add real world performance data"""
           self.real_world_data.append(data_point)

       def plot_comparison(self):
           """Plot comparison between simulation and real world"""
           fig, axes = plt.subplots(2, 2, figsize=(15, 10))
           fig.suptitle('Sim-to-Real Performance Comparison', fontsize=16)

           for i, metric in enumerate(self.metrics):
               ax = axes[i//2, i%2]

               # Extract data for the metric
               sim_values = [d[metric] for d in self.simulation_data if metric in d]
               real_values = [d[metric] for d in self.real_world_data if metric in d]

               # Create comparison plot
               x = np.arange(len(sim_values))
               width = 0.35

               ax.bar(x - width/2, sim_values, width, label='Simulation', alpha=0.8)
               ax.bar(x + width/2, real_values, width, label='Real World', alpha=0.8)

               ax.set_xlabel('Trial')
               ax.set_ylabel(metric.replace('_', ' ').title())
               ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
               ax.legend()
               ax.grid(True, alpha=0.3)

           plt.tight_layout()
           plt.savefig('sim_to_real_comparison.png', dpi=300, bbox_inches='tight')
           plt.show()

       def calculate_gap_metrics(self):
           """Calculate sim-to-real gap metrics"""
           gaps = {}

           for metric in self.metrics:
               sim_values = [d[metric] for d in self.simulation_data if metric in d]
               real_values = [d[metric] for d in self.real_world_data if metric in d]

               if sim_values and real_values:
                   sim_mean = np.mean(sim_values)
                   real_mean = np.mean(real_values)
                   gap = abs(sim_mean - real_mean)
                   relative_gap = gap / max(abs(sim_mean), abs(real_mean), 1e-8)

                   gaps[metric] = {
                       'absolute_gap': gap,
                       'relative_gap': relative_gap,
                       'sim_mean': sim_mean,
                       'real_mean': real_mean
                   }

           return gaps

       def print_gap_analysis(self):
           """Print detailed gap analysis"""
           gaps = self.calculate_gap_metrics()

           print("Sim-to-Real Gap Analysis:")
           print("=" * 50)

           for metric, values in gaps.items():
               print(f"\n{metric.replace('_', ' ').title()}:")
               print(f"  Simulation Mean: {values['sim_mean']:.3f}")
               print(f"  Real World Mean: {values['real_mean']:.3f}")
               print(f"  Absolute Gap: {values['absolute_gap']:.3f}")
               print(f"  Relative Gap: {values['relative_gap']:.1%}")

               if values['relative_gap'] > 0.2:
                   print(f"  ⚠️  Significant gap detected - may require domain randomization adjustment")
               else:
                   print(f"  ✅ Gap within acceptable range")

   # Example usage
   if __name__ == "__main__":
       # Create sample data
       comparison = PerformanceComparison()

       # Add sample simulation data
       for i in range(20):
           comparison.add_simulation_data({
               'success_rate': np.random.normal(0.9, 0.05),
               'path_efficiency': np.random.normal(0.8, 0.1),
               'collision_rate': np.random.normal(0.05, 0.02),
               'time_to_goal': np.random.normal(30, 5)
           })

       # Add sample real world data
       for i in range(20):
           comparison.add_real_world_data({
               'success_rate': np.random.normal(0.7, 0.1),
               'path_efficiency': np.random.normal(0.6, 0.15),
               'collision_rate': np.random.normal(0.15, 0.05),
               'time_to_goal': np.random.normal(45, 10)
           })

       # Plot comparison
       comparison.plot_comparison()

       # Print gap analysis
       comparison.print_gap_analysis()
   ```

## Expected Outcome

Upon completion of this lab, you should have:

- Successfully applied domain randomization techniques to improve sim-to-real transfer
- Trained robot behaviors in simulation that can operate effectively in the real world
- Implemented a complete pipeline for transferring learned behaviors to physical hardware
- Evaluated performance metrics comparing simulation and real-world results
- Developed adaptation strategies to handle sim-to-real discrepancies

The system should demonstrate:
- Successful navigation behaviors transferred from simulation to real robot
- Quantified performance gap between simulation and reality
- Adaptation mechanisms that improve real-world performance
- Evaluation framework for ongoing performance assessment

## Troubleshooting

- **Poor real-world performance**: Increase domain randomization range and retrain
- **Sensor data mismatch**: Implement better sensor calibration and normalization
- **Control instability**: Reduce control gains and add filtering to sensor data
- **Safety concerns**: Implement safety boundaries and emergency stops

## Optional Extension Tasks

1. **Online Adaptation**: Implement continuous adaptation during real-world operation.

2. **Multi-Task Transfer**: Transfer multiple behaviors simultaneously (navigation, manipulation, etc.).

3. **Cross-Robot Transfer**: Transfer behaviors between different robot platforms.

4. **Systematic Evaluation**: Create a comprehensive evaluation suite with various environmental conditions.

## Summary

This lab completed the full sim-to-real transfer pipeline, demonstrating how to apply domain randomization, train behaviors in simulation, transfer them to real hardware, and evaluate performance. You've learned to quantify and address the sim-to-real gap, implement adaptation strategies, and create evaluation frameworks for ongoing performance assessment. These skills are essential for deploying simulation-trained robotic systems in real-world applications.
