---
title: "Chapter 1: ROS 2 Fundamentals"
sidebar_position: 1
---

# Chapter 1: ROS 2 Fundamentals

Welcome to the world of ROS 2, the next-generation Robot Operating System that serves as the backbone for modern robotics development. As the "nervous system" of robotic applications, ROS 2 provides the essential communication infrastructure that enables different components of a robot to work together seamlessly. In this chapter, we'll explore the foundational concepts that make ROS 2 the standard framework for robotics development.

## ROS 2 vs ROS 1: Understanding the Evolution

ROS 2 represents a significant evolution from its predecessor, addressing many of the limitations that emerged as robotics applications became more complex and safety-critical. The primary motivation for this transition was the need for better real-time performance, improved security, and enhanced support for commercial and industrial applications.

### Key Differences

**Architecture Changes:**
- **ROS 1** relied on a centralized master-slave architecture with the ROS Master as the single point of failure
- **ROS 2** adopts a distributed architecture using DDS (Data Distribution Service) as the middleware, eliminating the need for a central master

**Communication Protocol:**
- **ROS 1** used TCPROS/UDPROS for communication
- **ROS 2** uses DDS, which provides better real-time performance and quality of service controls

**Security:**
- **ROS 1** had minimal security features
- **ROS 2** includes built-in security mechanisms including authentication, encryption, and access control

**Real-time Support:**
- **ROS 1** was not designed with real-time constraints in mind
- **ROS 2** provides better support for real-time systems with deterministic behavior

### Migration Considerations

When moving from ROS 1 to ROS 2, developers need to consider several changes:
- Package structure remains similar but some APIs have changed
- Launch files use a new Python-based format
- Message和服务 definitions remain compatible but client library APIs differ
- Testing frameworks have been updated

## Understanding DDS (Data Distribution Service)

DDS (Data Distribution Service) is the middleware that powers ROS 2's communication system. It's a standardized protocol designed for real-time, high-performance systems, making it ideal for robotics applications where timing and reliability are critical.

### DDS Architecture

DDS implements a publish-subscribe communication pattern where:
- **Publishers** send data to specific topics
- **Subscribers** receive data from topics they're interested in
- **DDS Implementation** handles the discovery, routing, and delivery of messages

The DDS specification defines several Quality of Service (QoS) policies that allow fine-tuning of communication behavior, which we'll explore in detail later in this module.

### DDS Implementations in ROS 2

ROS 2 supports multiple DDS implementations:
- **Fast DDS** (formerly FastRTPS) - Default implementation from eProsima
- **Cyclone DDS** - High-performance implementation from Eclipse
- **RTI Connext DDS** - Commercial implementation from RTI
- **OpenSplice DDS** - Open-source implementation

Each implementation has its own strengths and is suitable for different use cases, from embedded systems to high-performance computing clusters.

## Core ROS 2 Concepts: Nodes, Topics, Services, and Actions

### Nodes: The Building Blocks of ROS 2

A **node** is an executable process that works as part of a ROS 2 system. It's the fundamental unit of computation in ROS 2. Nodes contain the application logic and communicate with other nodes through topics, services, and actions.

Here's a simple example of creating a node in Python:

```python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
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
    minimal_publisher = MinimalNode()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Topics: Publish-Subscribe Communication

**Topics** enable asynchronous, many-to-many communication between nodes using a publish-subscribe pattern. Publishers send messages to topics, and subscribers receive messages from topics they're listening to.

Key characteristics of topics:
- **Asynchronous**: Publishers and subscribers don't need to be synchronized
- **Many-to-many**: Multiple publishers can send to the same topic, and multiple subscribers can listen to the same topic
- **Unidirectional**: Communication flows in one direction from publisher to subscriber
- **Typed**: Each topic has a specific message type that defines its structure

### Services: Request-Response Communication

**Services** provide synchronous, request-response communication between nodes. A client sends a request to a service, and the service processes the request and returns a response.

Service communication is:
- **Synchronous**: The client waits for the service to respond
- **One-to-one**: One client communicates with one service server
- **Bidirectional**: Request goes one way, response goes the other way
- **Typed**: Both request and response have defined message types

### Actions: Goal-Based Communication

**Actions** are used for long-running tasks that provide feedback and can be canceled. They combine the features of services and topics, providing:
- **Goal**: Request sent to start a task
- **Feedback**: Continuous updates on task progress
- **Result**: Final outcome when task completes

Actions are ideal for tasks like navigation, where you want to send a goal (navigate to position X), receive feedback (current progress), and get a result (success/failure).

## Parameters: Configuration Management

ROS 2 **parameters** provide a way to configure nodes at runtime. Parameters are key-value pairs that can be set when launching nodes or changed during execution.

Parameters support various data types:
- Integers and floating-point numbers
- Strings
- Booleans
- Lists and arrays
- Dictionaries

Example of parameter usage in a node:

```python
class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'turtlebot')
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('sensors_enabled', True)

        # Access parameter values
        robot_name = self.get_parameter('robot_name').value
        max_speed = self.get_parameter('max_speed').value
        sensors_enabled = self.get_parameter('sensors_enabled').value
```

## Launch Files: System Configuration

ROS 2 uses **launch files** to start multiple nodes with specific configurations. Unlike ROS 1's XML-based launch files, ROS 2 uses Python-based launch files, providing more flexibility and programmability.

Example launch file:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='turtlesim',
            executable='turtlesim_node',
            name='sim'
        ),
        Node(
            package='turtlesim',
            executable='turtle_teleop_key',
            name='teleop',
            remappings=[
                ('/turtle1/cmd_vel', '/cmd_vel'),
            ]
        ),
        Node(
            package='demo_nodes_py',
            executable='listener',
            name='listener'
        )
    ])
```

Launch files can include:
- Node definitions with parameters and remappings
- Conditional logic for different configurations
- Launch file composition (including other launch files)
- Parameter file loading

## What You Learned

In this chapter, you've gained a foundational understanding of ROS 2's core concepts and architecture. You now understand how ROS 2 differs from ROS 1, the role of DDS as the middleware, and the fundamental communication patterns: nodes, topics, services, and actions. You've also learned about parameter management and launch file configuration, which are essential for building complex robotic systems. These concepts form the basis for all ROS 2 applications and will be expanded upon in the following chapters.