---
sidebar_position: 6
---

# Lab 1.2: Custom Message Types and Services

## Objective
Define custom message and service types, implement a client-server interaction, and test communication with rqt tools.

## Prerequisites
- Completion of Lab 1.1
- Understanding of basic ROS 2 concepts
- ROS 2 development tools installed

## Learning Outcomes
- Define custom message types using .msg files
- Create service definitions using .srv files
- Implement client-server communication
- Test communication using rqt tools

## Lab Steps

### Step 1: Define Custom Message Types
1. Create a .msg file for a custom message
2. Define fields with appropriate data types
3. Add the message to the package manifest
4. Build the package to generate message code

### Step 2: Define Custom Service Types
1. Create a .srv file for a custom service
2. Define request and response fields
3. Add the service to the package manifest
4. Build the package to generate service code

### Step 3: Implement Service Server
1. Create a service server node
2. Implement the service callback function
3. Process requests and return appropriate responses
4. Add error handling and validation

### Step 4: Implement Service Client
1. Create a service client node
2. Call the service with appropriate request data
3. Handle the response and any errors
4. Add appropriate logging

### Step 5: Test Communication
1. Use rqt tools to monitor service calls
2. Test various request scenarios
3. Validate response handling
4. Debug any communication issues

## Assessment
- Successfully define and use custom message types
- Create and implement a custom service
- Demonstrate client-server communication
- Test communication using rqt tools