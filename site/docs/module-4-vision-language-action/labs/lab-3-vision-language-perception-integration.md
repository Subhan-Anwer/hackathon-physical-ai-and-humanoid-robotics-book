---
title: "Lab 3: Vision-Language Perception Integration"
sidebar_position: 3
---

# Lab 3: Vision-Language Perception Integration

## Overview
This lab focuses on implementing vision-language perception systems that combine computer vision with language understanding. You will combine computer vision with language understanding, implement object identification from natural language queries, integrate visual perception with LLM decision making, and test recognition accuracy and response time.

## Objectives
- Combine computer vision with language understanding
- Implement object identification from natural language queries
- Integrate visual perception with LLM decision making
- Test recognition accuracy and response time

## Prerequisites
- Completed Lab 1 and Lab 2
- Basic understanding of computer vision concepts
- Python programming experience with OpenCV and PyTorch
- Knowledge of transformer models and CLIP
- ROS 2 environment with vision processing capabilities

## Lab Setup

### 1. Install Required Dependencies
Install the necessary packages for vision-language integration:

```bash
pip install torch torchvision
pip install transformers
pip install openai
pip install opencv-python
pip install pillow
pip install clip @ git+https://github.com/openai/CLIP.git
pip install supervision
pip install ultralytics
```

### 2. Create Vision-Language Package
Create a new ROS 2 package for vision-language integration:

```bash
cd ~/voice_command_ws/src
ros2 pkg create --build-type ament_python vision_language_perception
cd vision_language_perception
```

### 3. Set Up Model Downloads
For this lab, we'll use pre-trained models. The CLIP model will be downloaded automatically when first used.

## Implementation Steps

### Step 1: Create the Vision-Language Perception Node
Create a ROS 2 node that integrates computer vision with language understanding:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import clip
from PIL import Image as PILImage
from typing import List, Dict, Tuple
import json

class VisionLanguagePerceptionNode(Node):
    def __init__(self):
        super().__init__('vision_language_perception_node')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Load CLIP model
        self.get_logger().info("Loading CLIP model...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.get_device())
        self.clip_model.eval()

        # Create subscribers and publishers
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )

        self.query_sub = self.create_subscription(
            String,
            'vision_language_query',
            self.query_callback,
            10
        )

        self.detection_pub = self.create_publisher(
            Detection2DArray,
            'vision_language_detections',
            10
        )

        self.result_pub = self.create_publisher(
            String,
            'vision_language_result',
            10
        )

        # Store the latest image for processing
        self.latest_image = None
        self.latest_query = None

        # Object detection model (using a simple approach for this example)
        self.object_detector = self._initialize_object_detector()

        self.get_logger().info("Vision-Language Perception Node initialized")

    def get_device(self):
        """Get the appropriate device (GPU if available)"""
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        return device

    def _initialize_object_detector(self):
        """Initialize object detection model (using YOLOv5 as example)"""
        # In a real implementation, you would load a pre-trained model
        # For this example, we'll create a mock detector
        class MockDetector:
            def detect(self, image):
                # This would be replaced with actual object detection
                # For now, we'll simulate detection of common objects
                height, width = image.shape[:2]
                detections = []

                # Simulate some common object detections
                detections.append({
                    'label': 'cup',
                    'confidence': 0.85,
                    'bbox': [width//4, height//4, width//3, height//3]  # [x, y, w, h]
                })
                detections.append({
                    'label': 'book',
                    'confidence': 0.78,
                    'bbox': [width//2, height//3, width//5, height//4]
                })
                detections.append({
                    'label': 'phone',
                    'confidence': 0.92,
                    'bbox': [width//3, height//2, width//6, height//6]
                })

                return detections

        return MockDetector()

    def image_callback(self, msg: Image):
        """Handle incoming camera images"""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cv_image
            self.get_logger().info("Received image for vision-language processing")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {str(e)}")

    def query_callback(self, msg: String):
        """Handle incoming vision-language queries"""
        query = msg.data
        self.latest_query = query
        self.get_logger().info(f"Received vision-language query: {query}")

        # Process the query if we have a recent image
        if self.latest_image is not None:
            self.process_vision_language_query(self.latest_image, query)

    def process_vision_language_query(self, image: np.ndarray, query: str):
        """Process a vision-language query using CLIP"""
        try:
            # Convert OpenCV image to PIL format for CLIP
            pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Prepare text for CLIP
            texts = [query, "background"]  # Include background as negative class
            text_tokens = clip.tokenize(texts).to(self.get_device())

            # Preprocess image for CLIP
            image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.get_device())

            # Get CLIP predictions
            with torch.no_grad():
                logits_per_image, logits_per_text = self.clip_model(image_input, text_tokens)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

            # Get the probability that the query matches the image content
            query_match_prob = probs[0]

            # Also run object detection to get bounding boxes
            detections = self.object_detector.detect(image)

            # Use CLIP to score each detected object against the query
            scored_detections = []
            for detection in detections:
                # Create a cropped image of the detected object
                x, y, w, h = detection['bbox']
                obj_image = image[y:y+h, x:x+w]

                # Convert to PIL and preprocess for CLIP
                obj_pil = PILImage.fromarray(cv2.cvtColor(obj_image, cv2.COLOR_BGR2RGB))
                obj_input = self.clip_preprocess(obj_pil).unsqueeze(0).to(self.get_device())

                # Get CLIP similarity for this specific object
                text_for_obj = [query, f"not {query}", "background"]
                text_tokens_obj = clip.tokenize(text_for_obj).to(self.get_device())

                with torch.no_grad():
                    logits_per_image_obj, _ = self.clip_model(obj_input, text_tokens_obj)
                    obj_probs = logits_per_image_obj.softmax(dim=-1).cpu().numpy()[0]

                # Use the probability of the query matching the object
                obj_score = obj_probs[0]

                scored_detection = {
                    'label': detection['label'],
                    'confidence': detection['confidence'],
                    'bbox': detection['bbox'],
                    'clip_score': float(obj_score),
                    'full_image_score': float(query_match_prob)
                }
                scored_detections.append(scored_detection)

            # Sort detections by CLIP score
            scored_detections.sort(key=lambda x: x['clip_score'], reverse=True)

            # Publish results
            self.publish_vision_language_results(scored_detections, query)

        except Exception as e:
            self.get_logger().error(f"Error processing vision-language query: {str(e)}")

    def publish_vision_language_results(self, detections: List[Dict], query: str):
        """Publish vision-language results"""
        # Create Detection2DArray message
        detection_array = Detection2DArray()
        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.header.frame_id = "camera_frame"

        for det in detections:
            detection_msg = Detection2D()
            detection_msg.header.stamp = detection_array.header.stamp
            detection_msg.header.frame_id = detection_array.header.frame_id

            # Set bounding box (convert from [x, y, w, h] to center + size)
            x, y, w, h = det['bbox']
            detection_msg.bbox.center.x = x + w / 2
            detection_msg.bbox.center.y = y + h / 2
            detection_msg.bbox.size_x = w
            detection_msg.bbox.size_y = h

            # Set results (object hypothesis with score)
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = det['label']
            hypothesis.score = det['clip_score']  # Use CLIP score as confidence

            detection_msg.results.append(hypothesis)
            detection_array.detections.append(detection_msg)

        self.detection_pub.publish(detection_array)

        # Also publish detailed results as JSON string
        result_msg = String()
        result_data = {
            'query': query,
            'detections': detections,
            'image_match_score': detections[0]['full_image_score'] if detections else 0.0
        }
        result_msg.data = json.dumps(result_data, indent=2)
        self.result_pub.publish(result_msg)

        # Log the results
        if detections:
            top_detection = detections[0]
            self.get_logger().info(
                f"Top match: {top_detection['label']} with CLIP score: {top_detection['clip_score']:.3f}"
            )
        else:
            self.get_logger().info("No objects detected matching the query")

def main(args=None):
    rclpy.init(args=args)
    node = VisionLanguagePerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Vision-Language Perception Node")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 2: Create the Object Identification Node
Create a node that specializes in identifying objects based on natural language descriptions:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import clip
from PIL import Image as PILImage
from typing import List, Dict, Tuple
import json
import re

class ObjectIdentificationNode(Node):
    def __init__(self):
        super().__init__('object_identification_node')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Load CLIP model
        self.get_logger().info("Loading CLIP model for object identification...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.get_device())
        self.clip_model.eval()

        # Create subscribers and publishers
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )

        self.identification_query_sub = self.create_subscription(
            String,
            'object_identification_query',
            self.identification_query_callback,
            10
        )

        self.identification_result_pub = self.create_publisher(
            String,
            'object_identification_result',
            10
        )

        self.detection_pub = self.create_publisher(
            Detection2DArray,
            'identified_objects',
            10
        )

        # Store the latest image
        self.latest_image = None
        self.latest_query = None

        # Object detection model
        self.object_detector = self._initialize_object_detector()

        self.get_logger().info("Object Identification Node initialized")

    def get_device(self):
        """Get the appropriate device (GPU if available)"""
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        return device

    def _initialize_object_detector(self):
        """Initialize object detection model"""
        # For this example, we'll use a mock detector
        # In a real implementation, you'd use YOLO, Detectron2, or similar
        class MockDetector:
            def detect(self, image):
                height, width = image.shape[:2]
                # Return mock detections: [label, confidence, [x, y, w, h]]
                return [
                    ['red cup', 0.85, [width//4, height//4, width//6, height//6]],
                    ['blue book', 0.78, [width//2, height//3, width//5, height//5]],
                    ['phone', 0.92, [width//3, height//2, width//8, width//8]],
                    ['pen', 0.65, [width//2, height//2, width//10, height//3]]
                ]

        return MockDetector()

    def image_callback(self, msg: Image):
        """Handle incoming camera images"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f"Error converting image: {str(e)}")

    def identification_query_callback(self, msg: String):
        """Handle object identification queries"""
        query = msg.data
        self.latest_query = query
        self.get_logger().info(f"Received identification query: {query}")

        if self.latest_image is not None:
            self.identify_objects_by_description(self.latest_image, query)

    def identify_objects_by_description(self, image: np.ndarray, description: str):
        """Identify objects based on a natural language description"""
        try:
            # First, detect all objects in the image
            all_detections = self.object_detector.detect(image)

            # Prepare CLIP text inputs based on the description
            # Include the description and some negative examples
            texts = [description, f"not {description}", "background", "unrelated object"]
            text_tokens = clip.tokenize(texts).to(self.get_device())

            # Preprocess the full image
            pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.get_device())

            # Get overall image match score
            with torch.no_grad():
                logits_per_image, _ = self.clip_model(image_input, text_tokens)
                overall_probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
                overall_match_score = overall_probs[0]

            # Now evaluate each detected object against the description
            matched_objects = []
            for label, conf, bbox in all_detections:
                # Extract the object region from the image
                x, y, w, h = bbox
                obj_region = image[y:y+h, x:x+w]

                # Convert to PIL and preprocess for CLIP
                obj_pil = PILImage.fromarray(cv2.cvtColor(obj_region, cv2.COLOR_BGR2RGB))
                obj_input = self.clip_preprocess(obj_pil).unsqueeze(0).to(self.get_device())

                # Get CLIP scores for this specific object
                with torch.no_grad():
                    logits_per_image_obj, _ = self.clip_model(obj_input, text_tokens)
                    obj_probs = logits_per_image_obj.softmax(dim=-1).cpu().numpy()[0]

                # Calculate the match score for this object
                obj_match_score = obj_probs[0]

                # Combine the detection confidence with CLIP score
                combined_score = (conf + obj_match_score) / 2.0

                matched_objects.append({
                    'label': label,
                    'bbox': bbox,
                    'detection_confidence': conf,
                    'clip_score': float(obj_match_score),
                    'combined_score': combined_score,
                    'description': description
                })

            # Sort by combined score
            matched_objects.sort(key=lambda x: x['combined_score'], reverse=True)

            # Filter results based on confidence threshold
            confident_matches = [obj for obj in matched_objects if obj['combined_score'] > 0.3]

            # Publish results
            self.publish_identification_results(confident_matches, description, overall_match_score)

        except Exception as e:
            self.get_logger().error(f"Error in object identification: {str(e)}")

    def publish_identification_results(self, matches: List[Dict], description: str, overall_score: float):
        """Publish object identification results"""
        # Create detailed result message
        result_msg = String()
        result_data = {
            'query': description,
            'overall_image_match_score': float(overall_score),
            'matched_objects': matches,
            'total_matches': len(matches)
        }
        result_msg.data = json.dumps(result_data, indent=2)
        self.identification_result_pub.publish(result_msg)

        # Also publish as detections for visualization
        detection_array = Detection2DArray()
        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.header.frame_id = "camera_frame"

        for match in matches:
            from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose

            detection_msg = Detection2D()
            detection_msg.header.stamp = detection_array.header.stamp
            detection_msg.header.frame_id = detection_array.header.frame_id

            # Set bounding box
            x, y, w, h = match['bbox']
            detection_msg.bbox.center.x = x + w / 2
            detection_msg.bbox.center.y = y + h / 2
            detection_msg.bbox.size_x = w
            detection_msg.bbox.size_y = h

            # Set result
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = match['label']
            hypothesis.score = match['combined_score']

            detection_msg.results.append(hypothesis)
            detection_array.detections.append(detection_msg)

        self.detection_pub.publish(detection_array)

        # Log results
        self.get_logger().info(f"Found {len(matches)} objects matching '{description}'")
        for i, match in enumerate(matches[:3]):  # Log top 3 matches
            self.get_logger().info(
                f"  {i+1}. {match['label']} - Score: {match['combined_score']:.3f}"
            )

def main(args=None):
    rclpy.init(args=args)
    node = ObjectIdentificationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Object Identification Node")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 3: Create the LLM Integration Node
Create a node that integrates visual perception with LLM decision making:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import clip
from PIL import Image as PILImage
from typing import List, Dict, Any
import json
import openai

class LLMVisionIntegrationNode(Node):
    def __init__(self):
        super().__init__('llm_vision_integration_node')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Load CLIP model
        self.get_logger().info("Loading CLIP model for LLM integration...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.get_device())
        self.clip_model.eval()

        # Create subscribers and publishers
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )

        self.vision_query_sub = self.create_subscription(
            String,
            'llm_vision_query',
            self.vision_query_callback,
            10
        )

        self.llm_decision_pub = self.create_publisher(
            String,
            'llm_decision',
            10
        )

        self.scene_description_pub = self.create_publisher(
            String,
            'scene_description',
            10
        )

        # Store latest data
        self.latest_image = None
        self.latest_query = None

        # Object detection model
        self.object_detector = self._initialize_object_detector()

        self.get_logger().info("LLM Vision Integration Node initialized")

    def get_device(self):
        """Get the appropriate device (GPU if available)"""
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        return device

    def _initialize_object_detector(self):
        """Initialize object detection model"""
        class MockDetector:
            def detect(self, image):
                height, width = image.shape[:2]
                return [
                    ['red cup', 0.85, [width//4, height//4, width//6, height//6]],
                    ['blue book', 0.78, [width//2, height//3, width//5, height//5]],
                    ['phone', 0.92, [width//3, height//2, width//8, width//8]]
                ]

        return MockDetector()

    def image_callback(self, msg: Image):
        """Handle incoming camera images"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f"Error converting image: {str(e)}")

    def vision_query_callback(self, msg: String):
        """Handle LLM vision queries"""
        query = msg.data
        self.latest_query = query
        self.get_logger().info(f"Received LLM vision query: {query}")

        if self.latest_image is not None:
            self.process_llm_vision_query(self.latest_image, query)

    def process_llm_vision_query(self, image: np.ndarray, query: str):
        """Process a query using both visual perception and LLM"""
        try:
            # First, analyze the scene visually
            scene_analysis = self.analyze_scene(image)

            # Create a prompt for the LLM that includes visual information
            llm_prompt = self.create_llm_prompt(query, scene_analysis)

            # Get LLM response
            llm_response = self.get_llm_response(llm_prompt)

            # Publish the LLM decision
            decision_msg = String()
            decision_msg.data = json.dumps({
                'query': query,
                'scene_analysis': scene_analysis,
                'llm_response': llm_response,
                'timestamp': self.get_clock().now().nanoseconds
            })
            self.llm_decision_pub.publish(decision_msg)

            # Publish scene description
            scene_desc_msg = String()
            scene_desc_msg.data = json.dumps(scene_analysis)
            self.scene_description_pub.publish(scene_desc_msg)

            self.get_logger().info(f"LLM decision published for query: {query}")

        except Exception as e:
            self.get_logger().error(f"Error in LLM vision processing: {str(e)}")

    def analyze_scene(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze the scene using visual perception"""
        # Detect objects
        detections = self.object_detector.detect(image)

        # Use CLIP to get more detailed scene understanding
        pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.get_device())

        # Define common scene categories for CLIP to classify
        scene_categories = [
            "indoor scene", "outdoor scene", "kitchen", "living room", "office", "bedroom",
            "cluttered", "organized", "bright", "dim", "empty", "occupied"
        ]

        text_tokens = clip.tokenize(scene_categories).to(self.get_device())

        with torch.no_grad():
            logits_per_image, _ = self.clip_model(image_input, text_tokens)
            scene_probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        # Get top scene categories
        top_scene_indices = np.argsort(scene_probs)[-3:][::-1]  # Top 3
        top_scenes = [(scene_categories[i], float(scene_probs[i])) for i in top_scene_indices]

        # Format detections
        formatted_detections = []
        for label, conf, bbox in detections:
            formatted_detections.append({
                'label': label,
                'confidence': float(conf),
                'bbox': [int(x) for x in bbox]  # Convert to int for JSON serialization
            })

        return {
            'objects': formatted_detections,
            'scene_categories': top_scenes,
            'image_shape': image.shape,
            'object_count': len(detections)
        }

    def create_llm_prompt(self, query: str, scene_analysis: Dict[str, Any]) -> str:
        """Create a prompt for the LLM with visual context"""
        prompt = f"""
        You are a robot assistant with visual perception capabilities. You can see the following scene:

        Scene Analysis:
        - Objects detected: {scene_analysis['objects']}
        - Scene categories: {[cat[0] for cat in scene_analysis['scene_categories']]}
        - Number of objects: {scene_analysis['object_count']}

        The user has asked: "{query}"

        Please provide a helpful response based on what you can see in the scene. If the query is about objects that are visible, provide specific information about their location and appearance. If the query cannot be answered with the current scene, explain why and suggest what might be needed.

        Response:
        """

        return prompt

    def get_llm_response(self, prompt: str) -> str:
        """Get response from LLM"""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful robot assistant with visual perception capabilities. Respond based on what you can see in the scene."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            self.get_logger().error(f"Error calling LLM: {str(e)}")
            return f"Error getting LLM response: {str(e)}"

def main(args=None):
    rclpy.init(args=args)
    node = LLMVisionIntegrationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down LLM Vision Integration Node")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 4: Create the Performance Monitor Node
Create a node that monitors and evaluates the performance of the vision-language system:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import time
from collections import deque
from typing import Dict, List
import json

class PerformanceMonitorNode(Node):
    def __init__(self):
        super().__init__('performance_monitor_node')

        # Create subscribers
        self.vision_result_sub = self.create_subscription(
            String,
            'vision_language_result',
            self.vision_result_callback,
            10
        )

        self.identification_result_sub = self.create_subscription(
            String,
            'object_identification_result',
            self.identification_result_callback,
            10
        )

        self.llm_decision_sub = self.create_subscription(
            String,
            'llm_decision',
            self.llm_decision_callback,
            10
        )

        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )

        # Performance tracking
        self.processing_times = deque(maxlen=100)  # Track last 100 processing times
        self.query_count = 0
        self.last_image_time = None
        self.last_query_time = None

        # Publishers for performance metrics
        self.performance_pub = self.create_publisher(
            String,
            'performance_metrics',
            10
        )

        # Timer for periodic performance reporting
        self.performance_timer = self.create_timer(5.0, self.report_performance)

        self.get_logger().info("Performance Monitor Node initialized")

    def image_callback(self, msg: Image):
        """Track image reception times"""
        current_time = time.time()
        if self.last_image_time:
            # Calculate frame rate
            frame_interval = current_time - self.last_image_time
            fps = 1.0 / frame_interval if frame_interval > 0 else 0
            self.get_logger().debug(f"Image FPS: {fps:.2f}")

        self.last_image_time = current_time

    def vision_result_callback(self, msg: String):
        """Track vision-language processing times"""
        try:
            result_data = json.loads(msg.data)
            if 'processing_time' in result_data:
                self.processing_times.append(result_data['processing_time'])
        except json.JSONDecodeError:
            pass

        self.query_count += 1
        self.last_query_time = time.time()

    def identification_result_callback(self, msg: String):
        """Track object identification processing"""
        self.query_count += 1

    def llm_decision_callback(self, msg: String):
        """Track LLM decision processing"""
        self.query_count += 1

    def report_performance(self):
        """Report performance metrics"""
        if not self.processing_times:
            avg_processing_time = 0.0
        else:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)

        metrics = {
            'average_processing_time': avg_processing_time,
            'processing_time_samples': len(self.processing_times),
            'query_count': self.query_count,
            'timestamp': time.time()
        }

        # Publish metrics
        metrics_msg = String()
        metrics_msg.data = json.dumps(metrics, indent=2)
        self.performance_pub.publish(metrics_msg)

        # Log metrics
        self.get_logger().info(
            f"Performance Metrics - Avg Processing Time: {avg_processing_time:.3f}s, "
            f"Samples: {len(self.processing_times)}, Queries: {self.query_count}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = PerformanceMonitorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Performance Monitor Node")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 5: Create Launch File
Create a launch file to start all vision-language perception nodes:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    return LaunchDescription([
        # Vision-Language Perception node
        Node(
            package='vision_language_perception',
            executable='vision_language_perception_node',
            name='vision_language_perception_node',
            output='screen'
        ),

        # Object Identification node
        Node(
            package='vision_language_perception',
            executable='object_identification_node',
            name='object_identification_node',
            output='screen'
        ),

        # LLM Integration node
        Node(
            package='vision_language_perception',
            executable='llm_vision_integration_node',
            name='llm_vision_integration_node',
            output='screen'
        ),

        # Performance Monitor node
        Node(
            package='vision_language_perception',
            executable='performance_monitor_node',
            name='performance_monitor_node',
            output='screen'
        )
    ])
```

## Testing and Evaluation

### 1. Basic Functionality Test
Test the vision-language perception system with simple queries:
- "Find the red cup"
- "Show me all the books"
- "What objects are on the table?"

### 2. Complex Query Test
Test with more complex natural language queries:
- "Find the largest object in the scene"
- "Which object is closest to the center of the image?"
- "Identify all electronic devices"

### 3. Performance Evaluation
Evaluate the system's performance metrics:
- Recognition accuracy for different object types
- Response time for various query complexities
- Memory usage during processing
- Frame rate with continuous processing

### 4. Accuracy Assessment
Test under different conditions:
- Varying lighting conditions
- Different object orientations
- Partial occlusions
- Multiple similar objects

## Optional Extensions

### 1. Real-time Processing Optimization
Implement optimizations for real-time performance:
- Model quantization
- Efficient preprocessing pipelines
- Multi-threading for parallel processing

### 2. Advanced Scene Understanding
Enhance scene understanding capabilities:
- 3D object localization
- Spatial relationship understanding
- Temporal consistency tracking

### 3. Multimodal Fusion
Add additional sensory modalities:
- Integration with depth sensors
- Audio-visual fusion
- Tactile feedback integration

## Assessment Questions
1. How does the CLIP-based vision-language model compare to traditional object detection methods for specific queries?
2. What are the main computational bottlenecks in vision-language perception systems?
3. How could you improve the system's ability to handle ambiguous or complex natural language queries?
4. What are the trade-offs between accuracy and speed in real-time vision-language systems?

## What You Learned
In this lab, you implemented a comprehensive vision-language perception system that combines computer vision with natural language understanding. You learned how to integrate CLIP models with ROS 2, identify objects based on natural language descriptions, incorporate LLM decision-making with visual perception, and evaluate system performance. You also explored the challenges and optimization strategies for real-time vision-language systems in robotics applications.