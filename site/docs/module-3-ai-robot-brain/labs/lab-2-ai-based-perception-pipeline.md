---
title: "Lab 2: AI-Based Perception Pipeline"
sidebar_position: 2
---

# Lab 2: AI-Based Perception Pipeline

## Objective

In this lab, you will set up Isaac ROS perception nodes, implement object detection and semantic segmentation using GPU-accelerated AI models, process sensor data using GPU acceleration, and validate perception accuracy in simulation. This lab demonstrates how to build an end-to-end AI perception pipeline that leverages NVIDIA's hardware acceleration capabilities.

## Prerequisites

- Completed Lab 3.1: Isaac Sim Environment Setup
- NVIDIA GPU with CUDA support (RTX 3080 or higher recommended)
- Isaac ROS packages installed
- Basic knowledge of deep learning frameworks (PyTorch/TensorFlow)
- Understanding of computer vision concepts
- Docker and NVIDIA Container Toolkit

## Step-by-Step Instructions

### Step 1: Set Up Isaac ROS Perception Nodes

1. **Install Isaac ROS Perception Packages**
   ```bash
   sudo apt update
   sudo apt install ros-humble-isaac-ros-perception
   sudo apt install ros-humble-isaac-ros-visual-slam
   sudo apt install ros-humble-isaac-ros-bitmask-publisher
   ```

2. **Verify Installation**
   ```bash
   # Check available Isaac ROS packages
   ros2 pkg list | grep isaac_ros
   ```

3. **Launch Perception Stack**
   ```bash
   # Create a launch file for perception nodes
   mkdir -p ~/isaac_ws/src/perception_launch
   cd ~/isaac_ws/src/perception_launch
   ```

### Step 2: Configure Camera and Sensor Data Pipeline

1. **Set Up Camera Configuration**
   ```yaml
   # camera_config.yaml
   camera:
     ros__parameters:
       width: 640
       height: 480
       fps: 30
       fov: 1.047
       frame_id: "camera_link"
       rectified: true
   ```

2. **Create Camera Launch File**
   ```python
   # camera_launch.py
   from launch import LaunchDescription
   from launch_ros.actions import Node
   from ament_index_python.packages import get_package_share_directory
   import os

   def generate_launch_description():
       config = os.path.join(
           get_package_share_directory('perception_launch'),
           'config',
           'camera_config.yaml'
       )

       camera_node = Node(
           package='isaac_ros_stereo_image_proc',
           executable='disparity_node',
           name='disparity_node',
           parameters=[config]
       )

       return LaunchDescription([camera_node])
   ```

3. **Connect Camera to Isaac Sim**
   - Configure Isaac Sim to publish camera data to ROS topics
   - Verify camera data is being published at the expected rate
   - Test camera calibration and rectification

### Step 3: Implement Object Detection Pipeline

1. **Set Up TensorRT-based Object Detection**
   ```python
   # object_detection_pipeline.py
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import Image
   from vision_msgs.msg import Detection2DArray
   from cv_bridge import CvBridge
   import numpy as np
   import tensorrt as trt
   import pycuda.driver as cuda

   class ObjectDetectionNode(Node):
       def __init__(self):
           super().__init__('object_detection_node')
           self.bridge = CvBridge()

           # Subscribe to camera image
           self.image_sub = self.create_subscription(
               Image, '/camera/image_raw', self.image_callback, 10)

           # Publisher for detections
           self.detection_pub = self.create_publisher(
               Detection2DArray, '/detections', 10)

           # Initialize TensorRT engine
           self.initialize_tensorrt()

       def initialize_tensorrt(self):
           # Load TensorRT engine for object detection
           self.trt_logger = trt.Logger(trt.Logger.WARNING)
           with open("yolov8.engine", "rb") as f:
               self.engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(f.read())
           self.context = self.engine.create_execution_context()

       def image_callback(self, msg):
           cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

           # Run object detection
           detections = self.run_inference(cv_image)

           # Publish results
           self.publish_detections(detections)

       def run_inference(self, image):
           # Preprocess image
           input_tensor = self.preprocess(image)

           # Run inference on GPU
           output = self.inference(input_tensor)

           # Postprocess results
           detections = self.postprocess(output, image.shape)

           return detections

       def preprocess(self, image):
           # Resize and normalize image for model input
           resized = cv2.resize(image, (640, 640))
           normalized = resized.astype(np.float32) / 255.0
           return np.transpose(normalized, (2, 0, 1))

       def inference(self, input_tensor):
           # GPU inference implementation
           inputs, outputs, bindings, stream = self.allocate_buffers()
           np.copyto(inputs[0].host, input_tensor.ravel())

           # Run inference
           [output] = self.do_inference_v2(
               context=self.context,
               bindings=bindings,
               inputs=inputs,
               outputs=outputs,
               stream=stream
           )
           return output

       def postprocess(self, output, image_shape):
           # Convert raw output to detection format
           # Implementation depends on specific model output format
           pass

   def main(args=None):
       rclpy.init(args=args)
       node = ObjectDetectionNode()
       rclpy.spin(node)
       node.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

2. **Configure Object Detection Parameters**
   ```yaml
   # detection_config.yaml
   object_detection:
     ros__parameters:
       model_path: "/path/to/yolov8.engine"
       confidence_threshold: 0.5
       nms_threshold: 0.4
       input_width: 640
       input_height: 640
       max_batch_size: 1
   ```

### Step 4: Implement Semantic Segmentation

1. **Set Up Semantic Segmentation Node**
   ```python
   # semantic_segmentation_pipeline.py
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import Image
   from std_msgs.msg import Int32MultiArray
   from cv_bridge import CvBridge
   import numpy as np
   import tensorrt as trt

   class SemanticSegmentationNode(Node):
       def __init__(self):
           super().__init__('semantic_segmentation_node')
           self.bridge = CvBridge()

           # Subscribe to camera image
           self.image_sub = self.create_subscription(
               Image, '/camera/image_raw', self.image_callback, 10)

           # Publisher for segmentation mask
           self.mask_pub = self.create_publisher(
               Int32MultiArray, '/segmentation_mask', 10)

           # Initialize segmentation model
           self.initialize_segmentation_model()

       def initialize_segmentation_model(self):
           # Load TensorRT segmentation model
           self.trt_logger = trt.Logger(trt.Logger.WARNING)
           with open("deeplabv3.engine", "rb") as f:
               self.engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(f.read())
           self.context = self.engine.create_execution_context()

       def image_callback(self, msg):
           cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

           # Run semantic segmentation
           segmentation_mask = self.run_segmentation(cv_image)

           # Publish results
           self.publish_segmentation(segmentation_mask)

       def run_segmentation(self, image):
           # Preprocess image for segmentation
           input_tensor = self.preprocess_segmentation(image)

           # Run segmentation inference
           output = self.segmentation_inference(input_tensor)

           # Convert to class predictions
           mask = np.argmax(output, axis=0)
           return mask

   def main(args=None):
       rclpy.init(args=args)
       node = SemanticSegmentationNode()
       rclpy.spin(node)
       node.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

2. **Create Segmentation Launch Configuration**
   ```python
   # segmentation_launch.py
   from launch import LaunchDescription
   from launch_ros.actions import Node

   def generate_launch_description():
       segmentation_node = Node(
           package='your_package',
           executable='semantic_segmentation_pipeline',
           name='semantic_segmentation',
           parameters=[
               {'model_path': 'deeplabv3.engine'},
               {'input_width': 513},
               {'input_height': 513}
           ]
       )

       return LaunchDescription([segmentation_node])
   ```

### Step 5: Process Sensor Data Using GPU Acceleration

1. **Set Up GPU-Accelerated Point Cloud Processing**
   ```python
   # pointcloud_processing.py
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import PointCloud2, PointField
   from std_msgs.msg import Header
   import numpy as np
   import open3d as o3d
   from numba import cuda
   import cupy as cp

   class PointCloudProcessingNode(Node):
       def __init__(self):
           super().__init__('pointcloud_processing')

           # Subscribe to LIDAR point cloud
           self.pc_sub = self.create_subscription(
               PointCloud2, '/scan_cloud', self.pc_callback, 10)

           # Publisher for processed point cloud
           self.processed_pc_pub = self.create_publisher(
               PointCloud2, '/processed_scan_cloud', 10)

       @cuda.jit
       def remove_ground_kernel(self, points, ground_threshold, output):
           # CUDA kernel for ground plane removal
           idx = cuda.grid(1)
           if idx < points.shape[0]:
               if points[idx, 2] > ground_threshold:
                   output[idx] = 1
               else:
                   output[idx] = 0

       def pc_callback(self, msg):
           # Convert ROS PointCloud2 to numpy array
           points = self.pointcloud2_to_array(msg)

           # Process on GPU
           gpu_points = cp.asarray(points)
           ground_removed = self.remove_ground_gpu(gpu_points)

           # Convert back to ROS message
           processed_msg = self.array_to_pointcloud2(ground_removed, msg.header)
           self.processed_pc_pub.publish(processed_msg)

       def remove_ground_gpu(self, points):
           # Use GPU to remove ground plane
           ground_threshold = -0.1  # Adjust based on robot height
           mask = cp.zeros(points.shape[0], dtype=cp.int32)

           threads_per_block = 256
           blocks_per_grid = (points.shape[0] + threads_per_block - 1) // threads_per_block

           self.remove_ground_kernel[blocks_per_grid, threads_per_block](
               points, ground_threshold, mask)

           return points[mask == 1]

   def main(args=None):
       rclpy.init(args=args)
       node = PointCloudProcessingNode()
       rclpy.spin(node)
       node.destroy_node()
       rclpy.shutdown()

   if __name__ == '__main__':
       main()
   ```

2. **Implement Multi-Sensor Fusion**
   ```python
   # sensor_fusion.py
   import rclpy
   from rclpy.node import Node
   from sensor_msgs.msg import Image, PointCloud2
   from geometry_msgs.msg import PoseStamped
   import numpy as np
   from scipy.spatial.transform import Rotation as R

   class SensorFusionNode(Node):
       def __init__(self):
           super().__init__('sensor_fusion')

           # Subscribe to multiple sensors
           self.image_sub = self.create_subscription(
               Image, '/camera/image_raw', self.image_callback, 10)
           self.pc_sub = self.create_subscription(
               PointCloud2, '/scan_cloud', self.pc_callback, 10)
           self.imu_sub = self.create_subscription(
               Imu, '/imu/data', self.imu_callback, 10)

           # Publisher for fused data
           self.fused_data_pub = self.create_publisher(
               ObjectHypothesisArray, '/fused_objects', 10)

           # Initialize fusion parameters
           self.camera_intrinsics = np.array([
               [554.25, 0, 320],
               [0, 415.69, 240],
               [0, 0, 1]
           ])
           self.extrinsics = np.eye(4)  # Camera to LIDAR transform

       def fuse_camera_lidar(self, image_detections, pointcloud):
           # Project 3D points to 2D image space
           projected_points = self.project_points_to_image(pointcloud)

           # Associate detections with point cloud regions
           fused_objects = self.associate_detections(projected_points, image_detections)

           return fused_objects

       def project_points_to_image(self, pointcloud):
           # Transform 3D points to camera frame and project to 2D
           points_3d = self.pointcloud_to_array(pointcloud)
           points_4d = np.column_stack([points_3d, np.ones(len(points_3d))])

           # Apply extrinsic transform
           points_cam = (self.extrinsics @ points_4d.T).T[:, :3]

           # Project to image plane
           points_2d = (self.camera_intrinsics @ points_cam.T).T
           points_2d = points_2d[:, :2] / points_2d[:, 2:3]

           return points_2d
   ```

### Step 6: Validate Perception Accuracy

1. **Create Ground Truth Comparison Node**
   ```python
   # validation_node.py
   import rclpy
   from rclpy.node import Node
   from vision_msgs.msg import Detection2DArray
   from builtin_interfaces.msg import Time
   import numpy as np

   class PerceptionValidationNode(Node):
       def __init__(self):
           super().__init__('perception_validation')

           # Subscribe to detections and ground truth
           self.detection_sub = self.create_subscription(
               Detection2DArray, '/detections', self.detection_callback, 10)
           self.ground_truth_sub = self.create_subscription(
               Detection2DArray, '/ground_truth', self.ground_truth_callback, 10)

           # Publisher for validation metrics
           self.metrics_pub = self.create_publisher(
               String, '/perception_metrics', 10)

           self.detection_buffer = {}
           self.ground_truth_buffer = {}

       def detection_callback(self, msg):
           # Store detections with timestamp
           timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
           self.detection_buffer[timestamp] = msg

       def ground_truth_callback(self, msg):
           # Store ground truth with timestamp
           timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
           self.ground_truth_buffer[timestamp] = msg

       def calculate_metrics(self, detections, ground_truth):
           # Calculate precision, recall, mAP
           ious = self.calculate_ious(detections, ground_truth)

           # True positives, false positives, false negatives
           tp = np.sum(ious > 0.5)  # IoU threshold
           fp = len(detections) - tp
           fn = len(ground_truth) - tp

           precision = tp / (tp + fp) if (tp + fp) > 0 else 0
           recall = tp / (tp + fn) if (tp + fn) > 0 else 0
           f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

           return {
               'precision': precision,
               'recall': recall,
               'f1_score': f1_score,
               'mAP': self.calculate_map(detections, ground_truth, ious)
           }

       def calculate_ious(self, detections, ground_truth):
           # Calculate Intersection over Union for each detection-ground_truth pair
           ious = np.zeros((len(detections), len(ground_truth)))

           for i, det in enumerate(detections):
               for j, gt in enumerate(ground_truth):
                   iou = self.bbox_iou(det.bbox, gt.bbox)
                   ious[i, j] = iou

           return ious

       def bbox_iou(self, box1, box2):
           # Calculate IoU between two bounding boxes
           x1_inter = max(box1.center.x - box1.size_x/2, box2.center.x - box2.size_x/2)
           y1_inter = max(box1.center.y - box1.size_y/2, box2.center.y - box2.size_y/2)
           x2_inter = min(box1.center.x + box1.size_x/2, box2.center.x + box2.size_x/2)
           y2_inter = min(box1.center.y + box1.size_y/2, box2.center.y + box2.size_y/2)

           if x2_inter <= x1_inter or y2_inter <= y1_inter:
               return 0

           inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
           box1_area = box1.size_x * box1.size_y
           box2_area = box2.size_x * box2.size_y
           union_area = box1_area + box2_area - inter_area

           return inter_area / union_area if union_area > 0 else 0
   ```

2. **Run Validation Tests**
   ```bash
   # Launch validation test
   ros2 launch your_package validation_test.launch.py
   ```

## Expected Outcome

Upon completion of this lab, you should have:

- A complete AI-based perception pipeline with GPU acceleration
- Working object detection using TensorRT-optimized models
- Semantic segmentation capabilities with real-time performance
- Multi-sensor fusion combining camera and LIDAR data
- Validation system comparing perception results to ground truth
- Performance metrics showing the effectiveness of your perception system

The perception pipeline should be able to:
- Detect and classify objects in real-time at 30+ FPS
- Generate accurate semantic segmentation masks
- Fuse data from multiple sensors effectively
- Achieve high accuracy compared to ground truth data

## Troubleshooting

- **TensorRT engine not loading**: Verify model conversion and compatibility
- **GPU memory errors**: Reduce batch size or input resolution
- **Low inference speed**: Check CUDA installation and GPU utilization
- **Detection accuracy issues**: Verify model calibration and preprocessing steps

## Optional Extension Tasks

1. **Instance Segmentation**: Implement instance segmentation to distinguish between different objects of the same class.

2. **3D Object Detection**: Extend the pipeline to detect 3D objects from stereo or RGB-D data.

3. **Tracking Pipeline**: Add object tracking capabilities to maintain object identities across frames.

4. **Adversarial Testing**: Test the perception system with adversarial examples to evaluate robustness.

## Summary

This lab demonstrated the implementation of a comprehensive AI-based perception pipeline using Isaac ROS and GPU acceleration. You've learned to set up object detection, semantic segmentation, and multi-sensor fusion systems that can process sensor data in real-time. The validation framework provides metrics to evaluate perception accuracy, which is crucial for building reliable robotic systems.
