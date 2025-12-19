---
title: "Lab 1: Voice Command Recognition System"
sidebar_position: 1
---

# Lab 1: Voice Command Recognition System

## Overview
This lab introduces the fundamental concepts of voice command recognition for robotics applications. You will implement a complete voice command recognition system using OpenAI Whisper, integrate it with ROS 2, and test its performance in various acoustic conditions.

## Objectives
- Set up OpenAI Whisper for real-time speech recognition
- Implement voice command parsing and classification
- Integrate voice commands with ROS 2 message passing
- Test and evaluate recognition accuracy in various conditions

## Prerequisites
- Basic understanding of ROS 2 concepts
- Python programming experience
- OpenAI API account (for Whisper API) or local Whisper model
- Microphone for audio input
- Basic knowledge of audio processing concepts

## Lab Setup

### 1. Install Required Dependencies
First, create a new ROS 2 workspace for the voice command project:

```bash
mkdir -p ~/voice_command_ws/src
cd ~/voice_command_ws
colcon build
source install/setup.bash
```

Install the required Python packages:

```bash
pip install openai
pip install whisper
pip install torch torchaudio
pip install pyaudio
pip install speechrecognition
pip install transformers
```

### 2. Set Up OpenAI API (if using Whisper API)
If you plan to use OpenAI's Whisper API:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Alternative: Install Local Whisper Model
For local processing without API dependency:

```bash
pip install git+https://github.com/openai/whisper.git
```

## Implementation Steps

### Step 1: Create the Audio Capture Node
Create a ROS 2 node that captures audio from the microphone and publishes it as messages:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import AudioData
import pyaudio
import numpy as np
import threading
import queue

class AudioCaptureNode(Node):
    def __init__(self):
        super().__init__('audio_capture_node')

        # Audio parameters
        self.rate = 16000  # Sample rate
        self.chunk = 1024  # Buffer size
        self.format = pyaudio.paInt16
        self.channels = 1

        # Create publisher for audio data
        self.audio_pub = self.create_publisher(AudioData, 'audio_input', 10)

        # Audio queue for processing
        self.audio_queue = queue.Queue()

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

        # Start audio capture thread
        self.capture_thread = threading.Thread(target=self.capture_audio, daemon=True)
        self.capture_thread.start()

        self.get_logger().info("Audio Capture Node initialized")

    def capture_audio(self):
        """Capture audio from microphone and publish to ROS topic"""
        stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        self.get_logger().info("Audio capture started")

        try:
            while rclpy.ok():
                # Read audio data
                data = stream.read(self.chunk, exception_on_overflow=False)

                # Create and publish AudioData message
                audio_msg = AudioData()
                audio_msg.data = data
                self.audio_pub.publish(audio_msg)

        except Exception as e:
            self.get_logger().error(f"Audio capture error: {str(e)}")
        finally:
            stream.stop_stream()
            stream.close()

def main(args=None):
    rclpy.init(args=args)
    node = AudioCaptureNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.audio.terminate()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Step 2: Create the Whisper Processing Node
Create a node that processes audio using Whisper and converts speech to text:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import AudioData
from std_msgs.msg import String
import whisper
import numpy as np
import threading
import queue
import io
from scipy.io import wavfile

class WhisperNode(Node):
    def __init__(self):
        super().__init__('whisper_node')

        # Load Whisper model (choose appropriate size for your hardware)
        self.get_logger().info("Loading Whisper model...")
        self.model = whisper.load_model("base")  # Options: tiny, base, small, medium, large

        # Create subscribers and publishers
        self.audio_sub = self.create_subscription(
            AudioData,
            'audio_input',
            self.audio_callback,
            10
        )

        self.transcript_pub = self.create_publisher(
            String,
            'transcript',
            10
        )

        self.status_pub = self.create_publisher(
            String,
            'whisper_status',
            10
        )

        # Audio processing queue
        self.audio_queue = queue.Queue()

        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.processing_thread.start()

        self.get_logger().info("Whisper Node initialized with model loaded")

    def audio_callback(self, msg):
        """Callback function to handle incoming audio data"""
        try:
            # Convert AudioData to numpy array
            audio_array = np.frombuffer(msg.data, dtype=np.int16)
            audio_array = audio_array.astype(np.float32) / 32768.0  # Normalize

            # Add to processing queue
            self.audio_queue.put(audio_array)
        except Exception as e:
            self.get_logger().error(f"Error processing audio callback: {str(e)}")

    def process_audio(self):
        """Process audio data from queue using Whisper"""
        accumulated_audio = np.array([])

        while rclpy.ok():
            try:
                # Get audio data from queue
                audio_chunk = self.audio_queue.get(timeout=1.0)

                # Accumulate audio for better recognition
                accumulated_audio = np.concatenate([accumulated_audio, audio_chunk])

                # Process accumulated audio when we have enough (0.5 seconds worth)
                if len(accumulated_audio) >= 8000:  # 0.5 seconds at 16kHz
                    # Transcribe the accumulated audio
                    result = self.model.transcribe(accumulated_audio)
                    transcription = result['text'].strip()

                    if transcription:  # Only publish non-empty transcriptions
                        self.get_logger().info(f"Transcribed: {transcription}")

                        # Publish transcription
                        transcript_msg = String()
                        transcript_msg.data = transcription
                        self.transcript_pub.publish(transcript_msg)

                        # Publish status
                        status_msg = String()
                        status_msg.data = f"Transcribed: {transcription}"
                        self.status_pub.publish(status_msg)

                    # Clear accumulated audio (keep some overlap for continuity)
                    accumulated_audio = accumulated_audio[-4000:]  # Keep last 0.25 seconds

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f"Error in audio processing: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = WhisperNode()

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

### Step 3: Create the Voice Command Parser Node
Create a node that parses transcribed text into structured commands:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Bool
import json
import re

class VoiceCommandParserNode(Node):
    def __init__(self):
        super().__init__('voice_command_parser_node')

        # Create subscribers and publishers
        self.transcript_sub = self.create_subscription(
            String,
            'transcript',
            self.transcript_callback,
            10
        )

        self.command_pub = self.create_publisher(
            String,
            'parsed_commands',
            10
        )

        self.validity_pub = self.create_publisher(
            Bool,
            'command_validity',
            10
        )

        # Define command patterns
        self.command_patterns = [
            {
                'pattern': r'go to (.+)',
                'intent': 'navigation',
                'action': 'navigate_to'
            },
            {
                'pattern': r'move to (.+)',
                'intent': 'navigation',
                'action': 'navigate_to'
            },
            {
                'pattern': r'navigate to (.+)',
                'intent': 'navigation',
                'action': 'navigate_to'
            },
            {
                'pattern': r'pick up (.+)',
                'intent': 'manipulation',
                'action': 'grasp_object'
            },
            {
                'pattern': r'grasp (.+)',
                'intent': 'manipulation',
                'action': 'grasp_object'
            },
            {
                'pattern': r'take (.+)',
                'intent': 'manipulation',
                'action': 'grasp_object'
            },
            {
                'pattern': r'find (.+)',
                'intent': 'perception',
                'action': 'detect_object'
            },
            {
                'pattern': r'locate (.+)',
                'intent': 'perception',
                'action': 'detect_object'
            },
            {
                'pattern': r'look for (.+)',
                'intent': 'perception',
                'action': 'detect_object'
            },
            {
                'pattern': r'stop',
                'intent': 'control',
                'action': 'stop_robot'
            },
            {
                'pattern': r'pause',
                'intent': 'control',
                'action': 'pause_robot'
            }
        ]

        self.get_logger().info("Voice Command Parser Node initialized")

    def transcript_callback(self, msg):
        """Parse incoming transcript into structured commands"""
        transcript = msg.data.lower().strip()

        if not transcript:
            return

        self.get_logger().info(f"Parsing transcript: {transcript}")

        # Try to match command patterns
        for pattern_info in self.command_patterns:
            match = re.search(pattern_info['pattern'], transcript)
            if match:
                # Extract the target (object, location, etc.)
                target = match.group(1).strip() if len(match.groups()) > 0 else ""

                # Create structured command
                command = {
                    'intent': pattern_info['intent'],
                    'action': pattern_info['action'],
                    'target': target,
                    'original_text': transcript,
                    'confidence': 0.9  # For now, assume high confidence
                }

                # Publish parsed command
                command_msg = String()
                command_msg.data = json.dumps(command)
                self.command_pub.publish(command_msg)

                # Publish validity
                validity_msg = Bool()
                validity_msg.data = True
                self.validity_pub.publish(validity_msg)

                self.get_logger().info(f"Parsed command: {command}")
                return

        # If no pattern matched, command is invalid
        self.get_logger().info(f"No command pattern matched: {transcript}")
        validity_msg = Bool()
        validity_msg.data = False
        self.validity_pub.publish(validity_msg)

def main(args=None):
    rclpy.init(args=args)
    node = VoiceCommandParserNode()

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

### Step 4: Create the Command Validation and Feedback Node
Create a node that validates commands and provides feedback:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from std_msgs.msg import Empty
import json
import time

class CommandValidationNode(Node):
    def __init__(self):
        super().__init__('command_validation_node')

        # Create subscribers and publishers
        self.command_sub = self.create_subscription(
            String,
            'parsed_commands',
            self.command_callback,
            10
        )

        self.validity_sub = self.create_subscription(
            Bool,
            'command_validity',
            self.validity_callback,
            10
        )

        self.feedback_pub = self.create_publisher(
            String,
            'command_feedback',
            10
        )

        self.ack_pub = self.create_publisher(
            Empty,
            'command_acknowledged',
            10
        )

        # Store recent commands for validation
        self.recent_commands = []
        self.last_validity = False

        self.get_logger().info("Command Validation Node initialized")

    def command_callback(self, msg):
        """Handle incoming parsed commands"""
        try:
            command_data = json.loads(msg.data)
            self.get_logger().info(f"Received command: {command_data}")

            # Validate command
            is_valid = self.validate_command(command_data)

            if is_valid:
                # Publish feedback
                feedback_msg = String()
                feedback_msg.data = f"Command acknowledged: {command_data['action']} {command_data.get('target', '')}"
                self.feedback_pub.publish(feedback_msg)

                # Publish acknowledgment
                ack_msg = Empty()
                self.ack_pub.publish(ack_msg)

                self.get_logger().info(f"Command validated and acknowledged: {command_data['action']}")
            else:
                feedback_msg = String()
                feedback_msg.data = f"Invalid command: {command_data}"
                self.feedback_pub.publish(feedback_msg)

        except json.JSONDecodeError:
            self.get_logger().error(f"Invalid JSON in command: {msg.data}")
        except Exception as e:
            self.get_logger().error(f"Error processing command: {str(e)}")

    def validity_callback(self, msg):
        """Update validity status"""
        self.last_validity = msg.data

    def validate_command(self, command_data):
        """Validate command based on various criteria"""
        # Check if action is recognized
        valid_actions = [
            'navigate_to', 'grasp_object', 'detect_object',
            'stop_robot', 'pause_robot'
        ]

        if command_data['action'] not in valid_actions:
            return False

        # Check if target is appropriate for the action
        if command_data['action'] in ['navigate_to', 'detect_object'] and not command_data.get('target'):
            return False

        # Check if command is not a duplicate of recent commands
        current_time = time.time()
        for recent_cmd in self.recent_commands:
            if (current_time - recent_cmd['timestamp'] < 2.0 and  # Within 2 seconds
                recent_cmd['command']['action'] == command_data['action'] and
                recent_cmd['command'].get('target') == command_data.get('target')):
                return False  # Duplicate command

        # Add to recent commands
        self.recent_commands.append({
            'command': command_data,
            'timestamp': current_time
        })

        # Clean up old commands (keep last 10 seconds)
        self.recent_commands = [
            cmd for cmd in self.recent_commands
            if current_time - cmd['timestamp'] < 10.0
        ]

        return True

def main(args=None):
    rclpy.init(args=args)
    node = CommandValidationNode()

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

### Step 5: Create Launch File
Create a launch file to start all nodes together:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    return LaunchDescription([
        # Audio capture node
        Node(
            package='voice_command_system',
            executable='audio_capture_node',
            name='audio_capture_node',
            output='screen'
        ),

        # Whisper processing node
        Node(
            package='voice_command_system',
            executable='whisper_node',
            name='whisper_node',
            output='screen'
        ),

        # Voice command parser node
        Node(
            package='voice_command_system',
            executable='voice_command_parser_node',
            name='voice_command_parser_node',
            output='screen'
        ),

        # Command validation node
        Node(
            package='voice_command_system',
            executable='command_validation_node',
            name='command_validation_node',
            output='screen'
        )
    ])
```

## Testing and Evaluation

### 1. Basic Functionality Test
Test the system with simple commands:
- "Go to the kitchen"
- "Find the red cup"
- "Stop"
- "Pick up the book"

### 2. Performance Evaluation
Test the system under different conditions:
- Quiet environment
- Noisy environment
- Different speaking volumes
- Various accents (if possible)

### 3. Accuracy Assessment
Record the following metrics:
- Recognition accuracy (correctly transcribed commands)
- Command parsing accuracy (correctly identified intents)
- Response time (from speech to command execution)

### 4. Troubleshooting Common Issues
- **No audio input**: Check microphone permissions and connections
- **Poor recognition**: Ensure proper microphone placement and audio quality
- **Command not recognized**: Verify command patterns in the parser node

## Assessment Questions
1. How does the Whisper model's performance vary with different model sizes (tiny, base, small)?
2. What are the main challenges in voice command recognition for robotics applications?
3. How could you improve the command parsing accuracy for ambiguous commands?

## What You Learned
In this lab, you implemented a complete voice command recognition system using OpenAI Whisper and integrated it with ROS 2. You learned how to capture audio, process speech to text, parse commands, and validate inputs. You also evaluated the system's performance under various conditions and identified potential improvements for real-world applications.