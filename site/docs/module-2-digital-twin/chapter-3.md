---
title: "Chapter 3: Robot Modeling with URDF & Sensors"
sidebar_position: 1
---

# Chapter 3: Robot Modeling with URDF & Sensors

## Introduction

Unified Robot Description Format (URDF) is the standard XML-based format for describing robot models in ROS. It defines the kinematic and dynamic properties of robots, including their links, joints, and associated sensors. This chapter explores the structure of URDF files, the process of creating robot models, and the integration of sensors into robot descriptions for simulation and real-world applications.

## URDF XML Structure and Element Hierarchy

URDF follows a hierarchical structure where a robot is composed of links connected by joints. The fundamental elements include:

### Root Structure
```xml
<robot name="robot_name">
  <!-- Links, joints, and other elements -->
</robot>
```

### Link Elements
Links represent rigid bodies in the robot model and contain three main sub-elements:

```xml
<link name="link_name">
  <!-- Visual properties for display -->
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="1 1 1"/>
    </geometry>
    <material name="color">
      <color rgba="0.8 0.2 0.2 1.0"/>
    </material>
  </visual>

  <!-- Collision properties for physics simulation -->
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="1 1 1"/>
    </geometry>
  </collision>

  <!-- Inertial properties for dynamics -->
  <inertial>
    <mass value="1.0"/>
    <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
  </inertial>
</link>
```

### Joint Elements
Joints define the kinematic and dynamic relationships between links:

```xml
<joint name="joint_name" type="joint_type">
  <parent link="parent_link_name"/>
  <child link="child_link_name"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
</joint>
```

## Links, Joints, and Transmissions in Robot Modeling

### Links
Links represent the rigid bodies of a robot. Each link can have:
- **Visual**: Defines how the link appears in simulation and visualization
- **Collision**: Defines the collision geometry for physics simulation
- **Inertial**: Defines mass, center of mass, and inertia tensor for dynamics

### Joint Types
URDF supports several joint types:
- **Revolute**: Rotational joint with limited range
- **Continuous**: Rotational joint without limits
- **Prismatic**: Linear sliding joint with limits
- **Fixed**: No movement between links
- **Floating**: 6 DOF movement
- **Planar**: Movement in a plane

### Transmissions
Transmissions define how actuators connect to joints:

```xml
<transmission name="tran1">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="joint1">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
  </joint>
  <actuator name="motor1">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

## Kinematic and Dynamic Properties Definition

### Kinematic Properties
Kinematic properties define the geometric relationships and movement constraints:

```xml
<joint name="shoulder_joint" type="revolute">
  <parent link="torso"/>
  <child link="upper_arm"/>
  <origin xyz="0.0 0.2 0.3" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

### Dynamic Properties
Dynamic properties define mass, inertia, and other physical characteristics:

```xml
<inertial>
  <mass value="2.0"/>
  <origin xyz="0.0 0.0 0.1" rpy="0 0 0"/>
  <inertia
    ixx="0.02" ixy="0.0" ixz="0.0"
    iyy="0.03" iyz="0.0"
    izz="0.01"/>
</inertial>
```

For complex shapes, the inertia can be calculated using:
- `ixx`, `iyy`, `izz`: Moments of inertia about the x, y, z axes
- `ixy`, `ixz`, `iyz`: Products of inertia

## Visual and Collision Properties

### Visual Properties
Visual elements define how the robot appears in simulation and visualization tools:

```xml
<visual>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <!-- Box geometry -->
    <box size="0.1 0.1 0.2"/>

    <!-- Cylinder geometry -->
    <cylinder radius="0.05" length="0.1"/>

    <!-- Sphere geometry -->
    <sphere radius="0.05"/>

    <!-- Mesh geometry -->
    <mesh filename="package://robot_description/meshes/link1.dae" scale="1 1 1"/>
  </geometry>
  <material name="red">
    <color rgba="0.8 0.2 0.2 1.0"/>
  </material>
</visual>
```

### Collision Properties
Collision elements define the geometry used for collision detection:

```xml
<collision>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <geometry>
    <box size="0.1 0.1 0.2"/>
  </geometry>
</collision>
```

Collision geometry is often simplified compared to visual geometry for computational efficiency.

## Sensor Integration within Robot Models

Sensors can be integrated into URDF models to define their mounting positions and properties:

### Camera Sensor Integration
```xml
<link name="camera_link">
  <visual>
    <geometry>
      <box size="0.02 0.08 0.04"/>
    </geometry>
    <material name="black"/>
  </visual>
  <collision>
    <geometry>
      <box size="0.02 0.08 0.04"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.1"/>
    <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
  </inertial>
</link>

<joint name="camera_joint" type="fixed">
  <parent link="base_link"/>
  <child link="camera_link"/>
  <origin xyz="0.1 0 0.1" rpy="0 0 0"/>
</joint>

<gazebo reference="camera_link">
  <sensor type="camera" name="camera1">
    <update_rate>30.0</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>800</width>
        <height>600</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_optical_frame</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

### IMU Sensor Integration
```xml
<link name="imu_link">
  <inertial>
    <mass value="0.01"/>
    <inertia ixx="0.000001" ixy="0" ixz="0" iyy="0.000001" iyz="0" izz="0.000001"/>
  </inertial>
</link>

<joint name="imu_joint" type="fixed">
  <parent link="base_link"/>
  <child link="imu_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
</joint>

<gazebo reference="imu_link">
  <sensor type="imu" name="imu_sensor">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>false</visualize>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
  </sensor>
</gazebo>
```

### LIDAR Sensor Integration
```xml
<link name="lidar_link">
  <visual>
    <geometry>
      <cylinder radius="0.05" length="0.04"/>
    </geometry>
    <material name="black"/>
  </visual>
  <collision>
    <geometry>
      <cylinder radius="0.05" length="0.04"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.2"/>
    <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
  </inertial>
</link>

<joint name="lidar_joint" type="fixed">
  <parent link="base_link"/>
  <child link="lidar_link"/>
  <origin xyz="0 0 0.2" rpy="0 0 0"/>
</joint>

<gazebo reference="lidar_link">
  <sensor type="ray" name="lidar_sensor">
    <always_on>true</always_on>
    <visualize>true</visualize>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="lidar_controller" filename="libgazebo_ros_laser.so">
      <topic_name>scan</topic_name>
      <frame_name>lidar_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

## Xacro Macros for Complex Model Generation

Xacro (XML Macros) extends URDF with features like constants, properties, and macros:

### Basic Xacro Structure
```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="robot_name">

  <!-- Constants -->
  <xacro:property name="M_PI" value="3.1415926535897931"/>
  <xacro:property name="wheel_radius" value="0.1"/>
  <xacro:property name="wheel_width" value="0.05"/>

  <!-- Macros -->
  <xacro:macro name="wheel" params="prefix parent xyz rpy">
    <link name="${prefix}_wheel">
      <visual>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </visual>
      <collision>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.5"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.02"/>
      </inertial>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <axis xyz="0 1 0"/>
    </joint>
  </xacro:macro>

  <!-- Using the macro -->
  <xacro:wheel prefix="front_left" parent="base_link" xyz="0.2 0.2 0" rpy="0 0 0"/>
  <xacro:wheel prefix="front_right" parent="base_link" xyz="0.2 -0.2 0" rpy="0 0 0"/>
  <xacro:wheel prefix="rear_left" parent="base_link" xyz="-0.2 0.2 0" rpy="0 0 0"/>
  <xacro:wheel prefix="rear_right" parent="base_link" xyz="-0.2 -0.2 0" rpy="0 0 0"/>

</robot>
```

### Conditional Statements
Xacro supports conditional statements for flexible model generation:

```xml
<xacro:property name="has_camera" value="true"/>

<xacro:if value="${has_camera}">
  <link name="camera_mount">
    <visual>
      <geometry>
        <box size="0.02 0.08 0.04"/>
      </geometry>
    </visual>
  </link>
</xacro:if>
```

## What You Learned

In this chapter, you learned about the structure and elements of URDF robot descriptions, including links, joints, and transmissions. You explored how to define kinematic and dynamic properties, configure visual and collision properties, and integrate various sensor types into robot models. You also discovered how to use Xacro macros to create complex, parameterized robot models efficiently. This knowledge is essential for creating accurate robot models for simulation and real-world applications.