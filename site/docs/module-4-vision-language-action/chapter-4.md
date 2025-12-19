---
title: "Chapter 4: Vision-Language Integration for Robot Perception"
sidebar_position: 4
---

# Chapter 4: Vision-Language Integration for Robot Perception

## Learning Objectives

By the end of this chapter, you will be able to:
- Implement multimodal perception combining vision and language for robotic systems
- Design object recognition and identification systems using natural language descriptions
- Apply visual grounding and spatial reasoning techniques in robotics
- Integrate vision-language systems with Isaac ROS perception pipelines
- Build real-time perception-action loops for VLA systems

## Multimodal Perception Combining Vision and Language

### The Vision-Language Integration Paradigm

Vision-Language integration represents a fundamental advancement in robotic perception, enabling robots to understand their environment through both visual input and linguistic context. This multimodal approach allows robots to perform tasks that require both visual recognition and semantic understanding, such as identifying objects based on natural language descriptions or understanding spatial relationships expressed in human language.

The integration operates on multiple levels:

**Feature-Level Integration**: Combining visual and linguistic features at early processing stages to create joint representations.

**Decision-Level Integration**: Merging outputs from separate vision and language processing systems to make final decisions.

**Fusion-Level Integration**: Creating unified models that process both modalities simultaneously with cross-modal attention mechanisms.

### Cross-Modal Attention Mechanisms

Cross-modal attention allows vision and language systems to influence each other's processing, creating more robust and context-aware perception. The key mechanisms include:

**Visual-to-Language Attention**: Language understanding is guided by visual information, helping to resolve ambiguities in natural language.

**Language-to-Visual Attention**: Visual processing is guided by linguistic context, focusing attention on relevant parts of the visual scene.

**Bidirectional Attention**: Both modalities continuously influence each other throughout the processing pipeline.

### Vision-Language Model Architecture

```python
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

class VisionLanguageFusion(nn.Module):
    """Fusion model combining vision and language processing"""

    def __init__(self, vision_model_name='resnet50', language_model_name='bert-base-uncased'):
        super(VisionLanguageFusion, self).__init__()

        # Vision encoder (using ResNet for feature extraction)
        self.vision_encoder = models.resnet50(pretrained=True)
        # Remove the final classification layer
        self.vision_features_dim = self.vision_encoder.fc.in_features
        self.vision_encoder.fc = nn.Identity()

        # Language encoder (using BERT)
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        self.language_encoder = AutoModel.from_pretrained(language_model_name)

        # Vision-Language fusion layer
        self.fusion_layer = nn.Linear(self.vision_features_dim + self.language_encoder.config.hidden_size,
                                     self.vision_features_dim)

        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.vision_features_dim,
            num_heads=8,
            batch_first=True
        )

        # Output classifier for specific tasks
        self.classifier = nn.Linear(self.vision_features_dim, 1000)  # Adjust based on task

    def forward(self, images, text_descriptions):
        """
        Forward pass combining vision and language inputs
        images: batch of image tensors
        text_descriptions: list of text descriptions
        """
        # Process visual features
        vision_features = self.vision_encoder(images)  # [batch_size, vision_features_dim]

        # Process text features
        text_tokens = self.tokenizer(text_descriptions, return_tensors='pt', padding=True, truncation=True)
        text_outputs = self.language_encoder(**text_tokens)
        text_features = text_outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]

        # Cross-attention between vision and language
        vision_features_expanded = vision_features.unsqueeze(1)  # [batch_size, 1, vision_features_dim]
        text_features_expanded = text_features.unsqueeze(1)      # [batch_size, 1, hidden_size]

        # Apply cross-attention (vision attending to text)
        attended_vision, _ = self.cross_attention(
            vision_features_expanded, text_features_expanded, text_features_expanded
        )

        # Flatten the attended features
        attended_vision = attended_vision.squeeze(1)

        # Concatenate original vision features with attended features
        combined_features = torch.cat([vision_features, attended_vision], dim=1)

        # Apply fusion layer
        fused_features = self.fusion_layer(combined_features)

        # Apply classifier
        output = self.classifier(fused_features)

        return output

# Example usage
def example_vision_language_integration():
    model = VisionLanguageFusion()

    # Simulate batch of images (batch_size=2, channels=3, height=224, width=224)
    images = torch.randn(2, 3, 224, 224)

    # Simulate text descriptions
    text_descriptions = ["red cup on the table", "blue book near the lamp"]

    # Forward pass
    output = model(images, text_descriptions)
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    example_vision_language_integration()
```

## Object Recognition and Identification from Natural Language Descriptions

### Language-Guided Object Detection

Language-guided object detection enables robots to identify and locate objects based on natural language descriptions rather than pre-defined categories. This capability is crucial for VLA systems that need to understand and interact with objects mentioned in human commands.

The process involves:

1. **Text Parsing**: Understanding the object description and its attributes
2. **Visual Search**: Locating objects in the visual scene that match the description
3. **Matching**: Associating visual objects with the linguistic description
4. **Verification**: Confirming the match through additional processing

### Grounded Object Recognition System

```python
import cv2
import numpy as np
from typing import List, Dict, Tuple
import torch
from transformers import CLIPProcessor, CLIPModel

class GroundedObjectRecognizer:
    """Recognizes objects based on natural language descriptions"""

    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        # Initialize CLIP model for vision-language matching
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

        # Object detection model (using a simple approach for demonstration)
        self.object_detector = self._initialize_object_detector()

    def _initialize_object_detector(self):
        """Initialize object detection model"""
        # In practice, you'd use a model like YOLO, Detectron2, or similar
        # For this example, we'll simulate object detection
        class MockDetector:
            def detect(self, image):
                # Simulate object detection by returning bounding boxes and labels
                # In reality, this would use a real object detection model
                height, width = image.shape[:2]
                # Return mock detections: [x, y, w, h, confidence, class_name]
                return [
                    [width//4, height//4, width//4, height//4, 0.9, "cup"],
                    [width//2, height//3, width//5, height//5, 0.8, "book"],
                    [width//3, height//2, width//6, height//6, 0.7, "pen"]
                ]
        return MockDetector()

    def recognize_objects_by_description(self, image: np.ndarray, description: str) -> List[Dict]:
        """
        Recognize objects in image based on natural language description
        Returns list of matching objects with bounding boxes and confidence scores
        """
        # First, detect all objects in the image
        detected_objects = self.object_detector.detect(image)

        # Create candidate texts for each detected object
        candidate_texts = []
        for obj in detected_objects:
            x, y, w, h, conf, class_name = obj
            # Create descriptive text for each detected object
            candidate_texts.append(f"{class_name}")

        # Add the target description to the list of texts to compare against
        texts_to_compare = [description] + candidate_texts

        # Use CLIP to compare the target description with detected objects
        inputs = self.clip_processor(text=texts_to_compare, images=image, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self.clip_model(**inputs)

        # Get similarity scores between the image and each text
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        # Match the probabilities back to the detected objects
        results = []
        for i, obj in enumerate(detected_objects):
            x, y, w, h, conf, class_name = obj
            # Get the probability that this object matches the description
            # Index 0 is the target description, index i+1 is the i-th detected object
            similarity = probs[0] * probs[i+1]  # Combined similarity score

            results.append({
                'bbox': [int(x), int(y), int(x+w), int(y+h)],  # Convert to [x1, y1, x2, y2] format
                'class': class_name,
                'confidence': float(similarity),
                'description': description
            })

        # Sort by confidence score and return top matches
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results

    def identify_object_by_attributes(self, image: np.ndarray, color: str = None,
                                    shape: str = None, size: str = None) -> List[Dict]:
        """Identify objects based on specific visual attributes"""
        description_parts = []
        if color:
            description_parts.append(f"{color} object")
        if shape:
            description_parts.append(f"{shape} object")
        if size:
            description_parts.append(f"{size} object")

        if not description_parts:
            description_parts = ["object"]

        description = " ".join(description_parts)
        return self.recognize_objects_by_description(image, description)

# Example usage with ROS 2 integration
class VisionLanguagePerceptionNode:
    """ROS 2 node for vision-language perception"""

    def __init__(self):
        self.recognizer = GroundedObjectRecognizer()
        self.last_image = None
        self.last_description = None

    def process_image_with_description(self, image_data: np.ndarray, description: str) -> List[Dict]:
        """Process image with a natural language description to find matching objects"""
        try:
            matches = self.recognizer.recognize_objects_by_description(image_data, description)
            # Filter results based on confidence threshold
            confident_matches = [match for match in matches if match['confidence'] > 0.3]
            return confident_matches
        except Exception as e:
            print(f"Error in vision-language processing: {e}")
            return []

    def find_object_by_command(self, image: np.ndarray, command: str) -> List[Dict]:
        """Find objects based on a command that may contain object descriptions"""
        # Extract object descriptions from command
        # This is a simplified example - in practice, you'd use NLP techniques
        import re

        # Look for patterns like "red cup", "large box", etc.
        patterns = [
            r'\b(\w+)\s+(\w+)\b',  # color + object, e.g., "red cup"
            r'\b(\w+)\s+box\b',    # size + object, e.g., "large box"
            r'\b(\w+)\s+book\b',   # color + object, e.g., "blue book"
        ]

        found_descriptions = []
        for pattern in patterns:
            matches = re.findall(pattern, command.lower())
            for match in matches:
                found_descriptions.append(" ".join(match))

        if not found_descriptions:
            # If no specific object description found, look for general objects
            return self.process_image_with_description(image, "object")
        else:
            # Process with the first found description
            return self.process_image_with_description(image, found_descriptions[0])

# Example usage
def example_object_recognition():
    node = VisionLanguagePerceptionNode()

    # Simulate an image (in practice, this would come from a camera)
    simulated_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Test with a command that describes an object
    command = "Find the red cup on the table"
    results = node.find_object_by_command(simulated_image, command)

    print(f"Found {len(results)} matching objects:")
    for i, result in enumerate(results):
        print(f"  Object {i+1}: {result['class']} at {result['bbox']} with confidence {result['confidence']:.3f}")

if __name__ == "__main__":
    example_object_recognition()
```

## Visual Grounding and Spatial Reasoning

### Understanding Visual Grounding

Visual grounding is the process of connecting linguistic expressions to specific visual elements in an image or scene. For robots, this means understanding spatial relationships and object locations described in natural language commands.

Key components of visual grounding include:

**Spatial Reference Resolution**: Understanding phrases like "to the left of", "behind", "next to", etc.

**Object Localization**: Identifying the precise location of objects mentioned in commands.

**Scene Understanding**: Comprehending the overall spatial layout and relationships between objects.

### Spatial Reasoning Framework

```python
from typing import Dict, List, Tuple, Optional
import numpy as np

class SpatialReasoner:
    """Handles spatial reasoning and grounding for robotic perception"""

    def __init__(self):
        self.spatial_relations = {
            'left': lambda ref_pos, obj_pos: obj_pos[0] < ref_pos[0],
            'right': lambda ref_pos, obj_pos: obj_pos[0] > ref_pos[0],
            'above': lambda ref_pos, obj_pos: obj_pos[1] < ref_pos[1],  # y-axis inverted in image coordinates
            'below': lambda ref_pos, obj_pos: obj_pos[1] > ref_pos[1],
            'behind': lambda ref_pos, obj_pos: obj_pos[2] < ref_pos[2],  # assuming z-depth
            'in_front': lambda ref_pos, obj_pos: obj_pos[2] > ref_pos[2],
            'near': lambda ref_pos, obj_pos: np.linalg.norm(np.array(obj_pos[:2]) - np.array(ref_pos[:2])) < 50,  # pixels
            'far': lambda ref_pos, obj_pos: np.linalg.norm(np.array(obj_pos[:2]) - np.array(ref_pos[:2])) > 150,
        }

    def parse_spatial_command(self, command: str) -> Dict:
        """Parse spatial relationships from a command"""
        command_lower = command.lower()

        # Extract spatial relationships
        spatial_info = {
            'target_object': None,
            'reference_object': None,
            'spatial_relation': None,
            'spatial_distance': None
        }

        # Simple parsing - in practice, this would use more sophisticated NLP
        if 'left of' in command_lower:
            spatial_info['spatial_relation'] = 'left'
        elif 'right of' in command_lower:
            spatial_info['spatial_relation'] = 'right'
        elif 'above' in command_lower or 'on top of' in command_lower:
            spatial_info['spatial_relation'] = 'above'
        elif 'below' in command_lower or 'under' in command_lower:
            spatial_info['spatial_relation'] = 'below'
        elif 'behind' in command_lower:
            spatial_info['spatial_relation'] = 'behind'
        elif 'in front of' in command_lower:
            spatial_info['spatial_relation'] = 'in_front'
        elif 'near' in command_lower or 'next to' in command_lower:
            spatial_info['spatial_relation'] = 'near'
        elif 'far from' in command_lower:
            spatial_info['spatial_relation'] = 'far'

        # Extract object names (simplified)
        words = command.split()
        for i, word in enumerate(words):
            if word.lower() in ['the', 'a', 'an']:
                if i + 1 < len(words):
                    if not spatial_info['target_object']:
                        spatial_info['target_object'] = words[i + 1]
                    elif not spatial_info['reference_object']:
                        spatial_info['reference_object'] = words[i + 1]

        return spatial_info

    def resolve_spatial_query(self, objects: List[Dict], command: str) -> List[Dict]:
        """Resolve a spatial query against detected objects"""
        spatial_info = self.parse_spatial_command(command)

        if not spatial_info['spatial_relation'] or not spatial_info['reference_object']:
            # If no spatial relation, return objects matching target
            if spatial_info['target_object']:
                return [obj for obj in objects if spatial_info['target_object'] in obj.get('class', '').lower()]
            else:
                return objects

        # Find reference object
        reference_obj = None
        for obj in objects:
            if spatial_info['reference_object'] in obj.get('class', '').lower():
                reference_obj = obj
                break

        if not reference_obj:
            return []  # No reference object found

        # Get reference position
        ref_bbox = reference_obj['bbox']
        ref_center = [(ref_bbox[0] + ref_bbox[2]) // 2, (ref_bbox[1] + ref_bbox[3]) // 2]

        # Apply spatial relation to filter objects
        relation_func = self.spatial_relations.get(spatial_info['spatial_relation'])
        if not relation_func:
            return []

        # Calculate 3D position (simplified - using bbox center and area as depth proxy)
        ref_pos = [ref_center[0], ref_center[1], (ref_bbox[2] - ref_bbox[0]) * (ref_bbox[3] - ref_bbox[1])]

        matching_objects = []
        for obj in objects:
            if obj == reference_obj:  # Skip the reference object itself
                continue

            obj_bbox = obj['bbox']
            obj_center = [(obj_bbox[0] + obj_bbox[2]) // 2, (obj_bbox[1] + obj_bbox[3]) // 2]
            obj_size = (obj_bbox[2] - obj_bbox[0]) * (obj_bbox[3] - obj_bbox[1])
            obj_pos = [obj_center[0], obj_center[1], obj_size]

            if relation_func(ref_pos, obj_pos):
                # If target object specified, also match that
                if spatial_info['target_object']:
                    if spatial_info['target_object'] in obj.get('class', '').lower():
                        matching_objects.append(obj)
                else:
                    matching_objects.append(obj)

        return matching_objects

class VisualGroundingSystem:
    """Complete visual grounding system for robot perception"""

    def __init__(self):
        self.spatial_reasoner = SpatialReasoner()
        self.object_recognizer = GroundedObjectRecognizer()  # From previous section

    def ground_command(self, image: np.ndarray, command: str) -> Dict:
        """Ground a natural language command in the visual scene"""
        # First, detect objects in the image
        all_objects = self.object_recognizer.recognize_objects_by_description(image, "object")

        # Then, apply spatial reasoning to find relevant objects
        relevant_objects = self.spatial_reasoner.resolve_spatial_query(all_objects, command)

        # Return grounding result
        return {
            'command': command,
            'detected_objects': all_objects,
            'relevant_objects': relevant_objects,
            'spatial_info': self.spatial_reasoner.parse_spatial_command(command)
        }

# Example usage
def example_visual_grounding():
    grounding_system = VisualGroundingSystem()

    # Simulate an image
    simulated_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Test spatial commands
    commands = [
        "Find the cup to the left of the book",
        "Move to the object near the pen",
        "Identify what is behind the blue book"
    ]

    for command in commands:
        result = grounding_system.ground_command(simulated_image, command)
        print(f"\nCommand: {command}")
        print(f"Detected objects: {len(result['detected_objects'])}")
        print(f"Relevant objects: {len(result['relevant_objects'])}")
        print(f"Spatial info: {result['spatial_info']}")

if __name__ == "__main__":
    example_visual_grounding()
```

## Integration with Isaac ROS Perception Pipelines

### Isaac ROS Overview for VLA Systems

Isaac ROS provides a comprehensive set of perception packages optimized for robotics applications. For VLA systems, Isaac ROS offers specialized packages that can be integrated with vision-language models to create powerful perception capabilities.

Key Isaac ROS packages relevant to VLA:

- **Isaac ROS DNN**: Deep neural network inference for perception tasks
- **Isaac ROS Visual SLAM**: Visual simultaneous localization and mapping
- **Isaac ROS Manipulation**: Perception and planning for manipulation tasks
- **Isaac ROS Image Pipeline**: Optimized image processing pipelines

### ROS 2 Integration with Isaac ROS

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

class IsaacROSIntegrationNode(Node):
    """Integrates vision-language perception with Isaac ROS pipelines"""

    def __init__(self):
        super().__init__('isaac_ros_integration_node')

        # Create CV bridge for image conversion
        self.cv_bridge = CvBridge()

        # Create subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.command_sub = self.create_subscription(
            String,
            '/vl_command',
            self.command_callback,
            10
        )

        # Create publishers
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/vision_language_detections',
            10
        )

        self.result_pub = self.create_publisher(
            String,
            '/vl_result',
            10
        )

        # Initialize vision-language components
        self.vision_language_system = VisionLanguagePerceptionNode()
        self.visual_grounding_system = VisualGroundingSystem()

        # Store latest command
        self.latest_command = None

        self.get_logger().info("Isaac ROS Integration Node initialized")

    def image_callback(self, msg: Image):
        """Handle incoming camera images"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            if self.latest_command:
                # Process image with the latest command
                results = self.vision_language_system.find_object_by_command(
                    cv_image, self.latest_command
                )

                # Create detection message
                detection_msg = self.create_detection_message(results, msg.header)

                # Publish detections
                self.detection_pub.publish(detection_msg)

                # Also publish grounding result
                grounding_result = self.visual_grounding_system.ground_command(
                    cv_image, self.latest_command
                )

                result_msg = String()
                result_msg.data = str({
                    'command': self.latest_command,
                    'objects_found': len(results),
                    'relevant_objects': len(grounding_result['relevant_objects'])
                })
                self.result_pub.publish(result_msg)

                self.get_logger().info(f"Processed image with command: {self.latest_command}")
                self.get_logger().info(f"Found {len(results)} objects matching description")

        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

    def command_callback(self, msg: String):
        """Handle incoming vision-language commands"""
        self.latest_command = msg.data
        self.get_logger().info(f"Received command: {self.latest_command}")

    def create_detection_message(self, results: List[Dict], header) -> Detection2DArray:
        """Create Detection2DArray message from vision-language results"""
        detection_array = Detection2DArray()
        detection_array.header = header

        for result in results:
            detection = Detection2D()
            detection.header = header
            detection.results = []

            # Create object hypothesis
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = result['class']
            hypothesis.score = result['confidence']

            # Set bounding box (convert from [x1, y1, x2, y2] to center + size)
            bbox = result['bbox']
            detection.bbox.center.x = (bbox[0] + bbox[2]) / 2
            detection.bbox.center.y = (bbox[1] + bbox[3]) / 2
            detection.bbox.size_x = bbox[2] - bbox[0]
            detection.bbox.size_y = bbox[3] - bbox[1]

            detection.results.append(hypothesis)
            detection_array.detections.append(detection)

        return detection_array

def main(args=None):
    rclpy.init(args=args)
    node = IsaacROSIntegrationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Isaac ROS Integration Node")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Real-Time Perception-Action Loops

### Designing Perception-Action Integration

Real-time perception-action loops are critical for VLA systems that need to continuously process visual and linguistic input while executing actions. The loop must balance processing speed with accuracy to maintain responsive behavior.

Key components of the perception-action loop:

**Perception Module**: Processes visual and linguistic input to understand the current state
**Decision Module**: Determines appropriate actions based on perception and goals
**Action Module**: Executes selected actions and monitors their outcomes
**Feedback Module**: Updates the system based on action outcomes

### Real-Time VLA Loop Implementation

```python
import threading
import time
from queue import Queue, Empty
from typing import Dict, Any, Callable
import numpy as np

class RealTimeVLALoop:
    """Real-time Vision-Language-Action loop for robotic systems"""

    def __init__(self, perception_rate: float = 10.0,  # Hz
                 action_rate: float = 5.0,              # Hz
                 max_queue_size: int = 10):

        self.perception_rate = perception_rate
        self.action_rate = action_rate
        self.perception_period = 1.0 / perception_rate
        self.action_period = 1.0 / action_rate

        # Queues for inter-thread communication
        self.image_queue = Queue(maxsize=max_queue_size)
        self.command_queue = Queue(maxsize=max_queue_size)
        self.action_queue = Queue(maxsize=max_queue_size)

        # Components
        self.vision_language_system = VisionLanguagePerceptionNode()
        self.visual_grounding_system = VisualGroundingSystem()

        # State management
        self.current_state = {
            'objects': [],
            'robot_pose': None,
            'command': None,
            'action_queue': [],
            'execution_status': 'idle'
        }

        # Control flags
        self.running = False
        self.perception_thread = None
        self.action_thread = None

    def start(self):
        """Start the real-time VLA loop"""
        self.running = True

        # Start perception thread
        self.perception_thread = threading.Thread(target=self._perception_loop, daemon=True)
        self.perception_thread.start()

        # Start action thread
        self.action_thread = threading.Thread(target=self._action_loop, daemon=True)
        self.action_thread.start()

        self.get_logger().info("Real-time VLA loop started")

    def stop(self):
        """Stop the real-time VLA loop"""
        self.running = False
        if self.perception_thread:
            self.perception_thread.join(timeout=1.0)
        if self.action_thread:
            self.action_thread.join(timeout=1.0)
        self.get_logger().info("Real-time VLA loop stopped")

    def _perception_loop(self):
        """Main perception loop running at perception_rate"""
        last_time = time.time()

        while self.running:
            current_time = time.time()
            if current_time - last_time < self.perception_period:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                continue

            last_time = current_time

            try:
                # Get latest image (non-blocking)
                try:
                    image = self.image_queue.get_nowait()
                except Empty:
                    continue

                # Get latest command if available
                try:
                    command = self.command_queue.get_nowait()
                    self.current_state['command'] = command
                except Empty:
                    pass

                # Process perception
                if self.current_state['command']:
                    results = self.vision_language_system.find_object_by_command(
                        image, self.current_state['command']
                    )
                    self.current_state['objects'] = results

                    # Apply visual grounding
                    grounding_result = self.visual_grounding_system.ground_command(
                        image, self.current_state['command']
                    )
                    self.current_state['relevant_objects'] = grounding_result['relevant_objects']

                    # Generate actions based on perception
                    new_actions = self._generate_actions_from_perception()
                    self.current_state['action_queue'].extend(new_actions)

            except Exception as e:
                self.get_logger().error(f"Error in perception loop: {str(e)}")

    def _action_loop(self):
        """Main action execution loop running at action_rate"""
        last_time = time.time()

        while self.running:
            current_time = time.time()
            if current_time - last_time < self.action_period:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                continue

            last_time = current_time

            try:
                # Execute next action if available
                if self.current_state['action_queue']:
                    action = self.current_state['action_queue'].pop(0)
                    self._execute_action(action)

            except Exception as e:
                self.get_logger().error(f"Error in action loop: {str(e)}")

    def _generate_actions_from_perception(self) -> List[Dict]:
        """Generate actions based on current perception state"""
        actions = []

        if not self.current_state['command'] or not self.current_state['objects']:
            return actions

        command = self.current_state['command'].lower()

        # Generate navigation actions if objects are detected
        if 'go to' in command or 'move to' in command or 'navigate to' in command:
            for obj in self.current_state['objects'][:1]:  # Take first match
                actions.append({
                    'type': 'navigation',
                    'target': obj['bbox'],  # Use bounding box center as target
                    'description': f"Navigate to {obj['class']}"
                })

        # Generate manipulation actions if applicable
        elif 'pick up' in command or 'grasp' in command or 'take' in command:
            for obj in self.current_state['objects'][:1]:  # Take first match
                actions.append({
                    'type': 'manipulation',
                    'target': obj['bbox'],
                    'object': obj['class'],
                    'description': f"Grasp {obj['class']}"
                })

        # Generate observation actions for complex commands
        elif 'find' in command or 'locate' in command or 'look for' in command:
            for obj in self.current_state['objects']:
                actions.append({
                    'type': 'observation',
                    'target': obj['bbox'],
                    'object': obj['class'],
                    'description': f"Observe {obj['class']}"
                })

        return actions

    def _execute_action(self, action: Dict):
        """Execute a single action"""
        self.current_state['execution_status'] = f"executing_{action['type']}"

        try:
            if action['type'] == 'navigation':
                self._execute_navigation(action)
            elif action['type'] == 'manipulation':
                self._execute_manipulation(action)
            elif action['type'] == 'observation':
                self._execute_observation(action)
            else:
                self.get_logger().warn(f"Unknown action type: {action['type']}")

            self.get_logger().info(f"Completed action: {action['description']}")
            self.current_state['execution_status'] = 'idle'

        except Exception as e:
            self.get_logger().error(f"Error executing action {action['description']}: {str(e)}")
            self.current_state['execution_status'] = 'error'

    def _execute_navigation(self, action: Dict):
        """Execute navigation action"""
        # In a real system, this would interface with navigation stack
        bbox = action['target']
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        # Simulate navigation to object location
        self.get_logger().info(f"Navigating to object at ({center_x}, {center_y})")
        time.sleep(1.0)  # Simulate navigation time

    def _execute_manipulation(self, action: Dict):
        """Execute manipulation action"""
        # In a real system, this would interface with manipulation stack
        self.get_logger().info(f"Manipulating object: {action['object']}")
        time.sleep(1.5)  # Simulate manipulation time

    def _execute_observation(self, action: Dict):
        """Execute observation action"""
        # In a real system, this would interface with perception system
        self.get_logger().info(f"Observing object: {action['object']}")
        time.sleep(0.5)  # Simulate observation time

    def add_image(self, image: np.ndarray):
        """Add an image to the processing queue"""
        try:
            self.image_queue.put_nowait(image)
        except:
            # Queue is full, remove oldest and add new
            try:
                self.image_queue.get_nowait()
                self.image_queue.put_nowait(image)
            except:
                pass  # Queue might be empty, just continue

    def add_command(self, command: str):
        """Add a command to the processing queue"""
        try:
            self.command_queue.put_nowait(command)
        except:
            # Queue is full, remove oldest and add new
            try:
                self.command_queue.get_nowait()
                self.command_queue.put_nowait(command)
            except:
                pass

    def get_current_state(self) -> Dict:
        """Get the current state of the VLA system"""
        return self.current_state.copy()

    def get_logger(self):
        """Simple logger for the VLA loop"""
        class SimpleLogger:
            def info(self, msg):
                print(f"VLA-INFO: {msg}")
            def error(self, msg):
                print(f"VLA-ERROR: {msg}")
            def warn(self, msg):
                print(f"VLA-WARN: {msg}")
        return SimpleLogger()

# Example usage
def example_real_time_vla():
    vla_loop = RealTimeVLALoop(perception_rate=5.0, action_rate=2.0)  # 5Hz perception, 2Hz actions
    vla_loop.start()

    # Simulate adding images and commands
    for i in range(20):
        # Simulate camera images
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        vla_loop.add_image(image)

        # Add commands periodically
        if i % 10 == 0:
            vla_loop.add_command("Find the red cup in the room")
        elif i % 10 == 5:
            vla_loop.add_command("Go to the book on the table")

        time.sleep(0.1)  # Simulate real-time operation

    time.sleep(5)  # Let it run for a bit
    vla_loop.stop()

if __name__ == "__main__":
    example_real_time_vla()
```

## What You Learned

In this chapter, you've learned how to implement sophisticated vision-language integration for robotic perception systems. You now understand how to combine visual and linguistic inputs using cross-modal attention mechanisms, recognize objects based on natural language descriptions, apply visual grounding and spatial reasoning techniques, integrate with Isaac ROS perception pipelines, and build real-time perception-action loops for VLA systems. These capabilities enable robots to perceive and understand their environment in a more human-like way, bridging the gap between visual perception and linguistic understanding to create truly intelligent robotic systems.