---
title: "Lab 4: Unity High-Fidelity Simulation"
sidebar_position: 4
---

# Lab 4: Unity High-Fidelity Simulation

## Overview
This lab focuses on creating photorealistic simulation environments in Unity with advanced sensor simulation and user interfaces for robot teleoperation. You will set up Unity with ROS 2 integration and implement high-fidelity sensor systems.

## Objectives
- Set up Unity with ROS 2 integration
- Create photorealistic indoor/outdoor environments
- Implement advanced sensor simulation in Unity
- Build user interfaces for robot teleoperation

## Prerequisites
- Unity 2021.3 LTS or later installed
- ROS 2 installation (Humble Hawksbill or later)
- Unity Robotics Hub package
- Basic C# programming knowledge

## Lab Setup
1. Install Unity 2021.3 LTS or later
2. Import Unity Robotics Hub packages
3. Set up ROS 2 bridge for communication
4. Create a new Unity project for robotics simulation

## Implementation Steps

### Step 1: Install Unity Robotics Hub
1. Open Unity Hub and create a new 3D project
2. Open the Package Manager (Window > Package Manager)
3. Add the Unity Robotics packages:
   - Unity ROS TCP Connector
   - Unity URDF Importer (optional)
   - Unity Robotics Tools

### Step 2: Basic ROS 2 Connection Setup
Create a basic connection script to establish communication with ROS 2:

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;
using RosMessageTypes.Std;

public class ROSConnectionTest : MonoBehaviour
{
    ROSConnection ros;
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;

    void Start()
    {
        // Get the ROS connection object
        ros = ROSConnection.GetOrCreateInstance();
        ros.Initialize(rosIPAddress, rosPort);

        // Start sending messages on a regular interval
        InvokeRepeating("SendTwistMessage", 0.0f, 0.5f);
    }

    void SendTwistMessage()
    {
        // Create a Twist message
        var twist = new TwistMsg();
        twist.linear = new Vector3Msg(0.1f, 0, 0);  // Move forward
        twist.angular = new Vector3Msg(0, 0, 0.2f); // Rotate

        // Send the message to the 'cmd_vel' topic
        ros.Send("cmd_vel", twist);
    }

    void OnApplicationQuit()
    {
        ros.Disconnect();
    }
}
```

### Step 3: Create Photorealistic Environments
1. **Using Built-in Render Pipeline:**
   - Create a new scene
   - Add a Terrain object (GameObject > 3D Object > Terrain)
   - Sculpt terrain using terrain tools
   - Add textures using the "Paint Texture" tool
   - Add trees and details using the "Paint Trees" and "Paint Details" tools

2. **Enhanced Lighting Setup:**
```csharp
using UnityEngine;

public class EnvironmentLighting : MonoBehaviour
{
    public Light sunLight;
    public float dayNightCycleSpeed = 0.5f;
    private float timeOfDay = 0.5f; // 0.5 = noon, 0 = midnight

    void Start()
    {
        if (sunLight == null)
        {
            sunLight = FindObjectOfType<Light>();
            if (sunLight != null && sunLight.type == LightType.Directional)
            {
                // Found the directional light (sun)
            }
            else
            {
                // Create a sun light if none exists
                GameObject sunObj = new GameObject("Sun");
                sunLight = sunObj.AddComponent<Light>();
                sunLight.type = LightType.Directional;
                sunLight.color = Color.white;
                sunLight.intensity = 1.0f;
            }
        }
    }

    void Update()
    {
        // Update time of day
        timeOfDay += Time.deltaTime * dayNightCycleSpeed * 0.01f;
        if (timeOfDay >= 1.0f) timeOfDay = 0.0f;

        // Update sun position based on time of day
        float sunAngle = timeOfDay * 360f - 90f; // Start at dawn (sunrise)
        sunLight.transform.rotation = Quaternion.Euler(sunAngle, 10f, 0f);

        // Adjust light intensity and color based on sun position
        float intensityMultiplier = Mathf.Clamp01(Mathf.Sin(timeOfDay * Mathf.PI));
        sunLight.intensity = 1.0f * intensityMultiplier;

        // Change color from dawn/dusk (orange) to midday (white)
        float colorBlend = Mathf.Clamp01(Mathf.Abs(timeOfDay - 0.5f) * 2f);
        sunLight.color = Color.Lerp(new Color(1f, 0.7f, 0.4f), Color.white, 1f - colorBlend);
    }
}
```

### Step 4: Implement Advanced Sensor Simulation
Create scripts for different sensor types:

1. **RGB Camera Sensor:**
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using System.Collections.Generic;
using System.Threading.Tasks;

public class RGBCameraSensor : MonoBehaviour
{
    ROSConnection ros;
    public string topicName = "camera/image_raw";
    public int width = 640;
    public int height = 480;
    public float fieldOfView = 60f;

    Camera cam;
    RenderTexture renderTexture;
    Texture2D texture2D;
    byte[] rawImage;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        // Setup camera
        cam = GetComponent<Camera>();
        cam.fieldOfView = fieldOfView;

        // Create render texture
        renderTexture = new RenderTexture(width, height, 24);
        cam.targetTexture = renderTexture;

        // Create texture2D for reading pixels
        texture2D = new Texture2D(width, height, TextureFormat.RGB24, false);

        // Start capturing at a regular interval
        InvokeRepeating("CaptureAndSendImage", 0.0f, 0.1f); // 10 FPS
    }

    void CaptureAndSendImage()
    {
        // Set the active render texture
        RenderTexture.active = renderTexture;

        // Read pixels from the render texture
        texture2D.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        texture2D.Apply();

        // Get raw image bytes
        rawImage = texture2D.EncodeToJPG();

        // Create ROS Image message
        ImageMsg imageMsg = new ImageMsg();
        imageMsg.header = new std_msgs.HeaderMsg(0, new builtin_interfaces.TimeMsg(), topicName);
        imageMsg.height = (uint)height;
        imageMsg.width = (uint)width;
        imageMsg.encoding = "rgb8";
        imageMsg.is_bigendian = 0;
        imageMsg.step = (uint)(width * 3); // 3 bytes per pixel (RGB)
        imageMsg.data = rawImage;

        // Send the image message
        ros.Send(topicName, imageMsg);
    }
}
```

2. **Depth Camera Sensor:**
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class DepthCameraSensor : MonoBehaviour
{
    ROSConnection ros;
    public string topicName = "depth_camera/image_raw";
    public int width = 320;
    public int height = 240;
    public float fieldOfView = 60f;
    public float maxRange = 10.0f;

    Camera cam;
    RenderTexture depthTexture;
    Texture2D texture2D;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        // Setup camera for depth rendering
        cam = GetComponent<Camera>();
        cam.fieldOfView = fieldOfView;
        cam.depthTextureMode = DepthTextureMode.Depth;

        // Create render texture for depth
        depthTexture = new RenderTexture(width, height, 24, RenderTextureFormat.RFloat);
        cam.targetTexture = depthTexture;

        // Create texture2D for reading depth data
        texture2D = new Texture2D(width, height, TextureFormat.RFloat, false);

        // Start capturing at a regular interval
        InvokeRepeating("CaptureAndSendDepth", 0.0f, 0.1f); // 10 FPS
    }

    void CaptureAndSendDepth()
    {
        // Set the active render texture
        RenderTexture.active = depthTexture;

        // Read pixels from the depth texture
        texture2D.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        texture2D.Apply();

        // Get raw depth data
        Color[] depthColors = texture2D.GetPixels();
        float[] depthValues = new float[width * height];

        for (int i = 0; i < depthValues.Length; i++)
        {
            // Convert from 0-1 range to actual distance
            depthValues[i] = depthColors[i].r * maxRange;
        }

        // Create ROS Image message for depth
        ImageMsg depthMsg = new ImageMsg();
        depthMsg.header = new std_msgs.HeaderMsg(0, new builtin_interfaces.TimeMsg(), topicName);
        depthMsg.height = (uint)height;
        depthMsg.width = (uint)width;
        depthMsg.encoding = "32FC1"; // 32-bit float, single channel
        depthMsg.is_bigendian = 0;
        depthMsg.step = (uint)(width * sizeof(float)); // 4 bytes per float

        // Convert float array to byte array
        byte[] depthBytes = new byte[depthValues.Length * sizeof(float)];
        System.Buffer.BlockCopy(depthValues, 0, depthBytes, 0, depthBytes.Length);
        depthMsg.data = depthBytes;

        // Send the depth message
        ros.Send(topicName, depthMsg);
    }
}
```

3. **LIDAR Sensor Simulation:**
```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class LIDARSensor : MonoBehaviour
{
    ROSConnection ros;
    public string topicName = "scan";
    public int numberOfRays = 360;
    public float minAngle = -Mathf.PI;
    public float maxAngle = Mathf.PI;
    public float maxRange = 10.0f;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        InvokeRepeating("SendLIDARData", 0.0f, 0.1f); // 10 FPS
    }

    void SendLIDARData()
    {
        // Perform raycasts to simulate LIDAR
        float[] ranges = new float[numberOfRays];
        float angleStep = (maxAngle - minAngle) / numberOfRays;

        for (int i = 0; i < numberOfRays; i++)
        {
            float angle = minAngle + (i * angleStep);
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
            direction = transform.TransformDirection(direction);

            RaycastHit hit;
            if (Physics.Raycast(transform.position, direction, out hit, maxRange))
            {
                ranges[i] = hit.distance;
            }
            else
            {
                ranges[i] = maxRange; // No obstacle detected within range
            }
        }

        // Create LaserScan message
        LaserScanMsg scanMsg = new LaserScanMsg();
        scanMsg.header = new std_msgs.HeaderMsg(0, new builtin_interfaces.TimeMsg(), "lidar_link");
        scanMsg.angle_min = minAngle;
        scanMsg.angle_max = maxAngle;
        scanMsg.angle_increment = angleStep;
        scanMsg.time_increment = 0.0f; // For simulated data
        scanMsg.scan_time = 0.1f; // 10Hz
        scanMsg.range_min = 0.1f;
        scanMsg.range_max = maxRange;
        scanMsg.ranges = ranges;
        scanMsg.intensities = new float[numberOfRays]; // Initialize with zeros

        // Send the LIDAR message
        ros.Send(topicName, scanMsg);
    }
}
```

### Step 5: Create User Interface for Robot Control
Create a comprehensive UI system for robot teleoperation:

1. **Main Control Panel Script:**
```csharp
using UnityEngine;
using UnityEngine.UI;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

public class RobotControlPanel : MonoBehaviour
{
    ROSConnection ros;
    public Slider linearVelocitySlider;
    public Slider angularVelocitySlider;
    public Button moveButton;
    public Button stopButton;
    public Text statusText;
    public Text positionText;

    // Robot position tracking
    private Vector3 robotPosition = Vector3.zero;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        // Setup UI event listeners
        moveButton.onClick.AddListener(SendVelocityCommand);
        stopButton.onClick.AddListener(SendStopCommand);
        linearVelocitySlider.onValueChanged.AddListener(OnLinearVelocityChanged);
        angularVelocitySlider.onValueChanged.AddListener(OnAngularVelocityChanged);

        // Initialize UI
        UpdateUI();
    }

    void UpdateUI()
    {
        statusText.text = "Connected to ROS";
        positionText.text = $"Position: {robotPosition.x:F2}, {robotPosition.y:F2}, {robotPosition.z:F2}";
    }

    void SendVelocityCommand()
    {
        var twistMsg = new TwistMsg();
        twistMsg.linear = new Vector3Msg(linearVelocitySlider.value, 0, 0);
        twistMsg.angular = new Vector3Msg(0, 0, angularVelocitySlider.value);

        ros.Send("cmd_vel", twistMsg);
        statusText.text = "Command sent: Linear=" + linearVelocitySlider.value + ", Angular=" + angularVelocitySlider.value;
    }

    void SendStopCommand()
    {
        var twistMsg = new TwistMsg();
        twistMsg.linear = new Vector3Msg(0, 0, 0);
        twistMsg.angular = new Vector3Msg(0, 0, 0);

        ros.Send("cmd_vel", twistMsg);
        statusText.text = "Stop command sent";
    }

    void OnLinearVelocityChanged(float value)
    {
        // Update UI feedback
    }

    void OnAngularVelocityChanged(float value)
    {
        // Update UI feedback
    }
}
```

2. **Sensor Visualization Panel:**
```csharp
using UnityEngine;
using UnityEngine.UI;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class SensorVisualization : MonoBehaviour
{
    public RawImage cameraFeedImage;
    public Texture2D defaultTexture;
    public Image lidarVisualization;
    public Text sensorStatusText;

    void Start()
    {
        // Initialize camera feed image
        if (cameraFeedImage != null)
        {
            cameraFeedImage.texture = defaultTexture;
        }

        // Subscribe to camera topic
        ROSConnection.GetOrCreateInstance().Subscribe<ImageMsg>("camera/image_raw", UpdateCameraFeed);

        sensorStatusText.text = "Sensors initialized";
    }

    void UpdateCameraFeed(ImageMsg imageMsg)
    {
        if (cameraFeedImage != null)
        {
            // Create texture from image data
            Texture2D texture = new Texture2D((int)imageMsg.width, (int)imageMsg.height, TextureFormat.RGB24, false);

            // Note: This is a simplified approach; in practice, you'd need to properly decode the image data
            // based on the encoding format specified in the message
            if (imageMsg.encoding == "rgb8")
            {
                texture.LoadRawTextureData(imageMsg.data);
                texture.Apply();
                cameraFeedImage.texture = texture;
            }
        }
    }
}
```

### Step 6: Scene Setup and Testing
1. Create a new Unity scene
2. Add a robot model (or simple geometric shapes to represent the robot)
3. Attach the sensor scripts to appropriate GameObjects
4. Add the control panel UI to the scene
5. Configure the Canvas and UI elements
6. Test the connection with ROS 2

### Step 7: Testing with ROS 2
1. Start ROS 2 daemon:
```bash
source /opt/ros/humble/setup.bash
ros2 daemon start
```

2. Create a simple ROS 2 node to receive commands:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class RobotCommandReceiver(Node):
    def __init__(self):
        super().__init__('robot_command_receiver')
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.listener_callback,
            10)
        self.get_logger().info('Robot command receiver started')

    def listener_callback(self, msg):
        self.get_logger().info(f'Received cmd_vel: linear={msg.linear.x}, angular={msg.angular.z}')

def main(args=None):
    rclpy.init(args=args)
    node = RobotCommandReceiver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Assessment Questions
1. What are the advantages of using Unity for high-fidelity robotics simulation compared to Gazebo?
2. How does the Unity-ROS 2 bridge handle data transmission between the two systems?
3. What are the computational trade-offs when using photorealistic rendering in robotics simulation?
4. How would you implement collision detection between the Unity robot and real-world objects?

## What You Learned
In this lab, you learned how to set up Unity with ROS 2 integration, create photorealistic environments, implement advanced sensor simulation, and build user interfaces for robot teleoperation. You gained hands-on experience with Unity's capabilities for high-fidelity robotics simulation and learned how to integrate various sensor systems into your virtual environment.