"""Mock PyBullet environment for testing.

Provides a complete mock PyBullet physics simulation environment
for testing tactile processing and physics integration without
requiring actual PyBullet installation.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import Mock


class MockPyBullet:
    """Mock PyBullet physics simulation for testing."""
    
    def __init__(self):
        """Initialize mock PyBullet environment."""
        self.bodies = {}
        self.contacts = []
        self.gravity = [0, 0, -9.81]
        self.time_step = 1.0/240.0
        self.simulation_time = 0.0
        self.next_body_id = 1
        
        # Mock constants
        self.JOINT_REVOLUTE = 0
        self.JOINT_PRISMATIC = 1
        self.JOINT_FIXED = 4
        
        self.GUI = 1
        self.DIRECT = 2
        
        self.COV_ENABLE_GUI = 1
        self.COV_ENABLE_WIREFRAME = 3
        
    def connect(self, mode):
        """Mock connect to physics server."""
        return 0  # Physics client ID
    
    def disconnect(self):
        """Mock disconnect from physics server."""
        pass
    
    def setGravity(self, x, y, z):
        """Mock set gravity."""
        self.gravity = [x, y, z]
    
    def setTimeStep(self, time_step):
        """Mock set simulation time step."""
        self.time_step = time_step
    
    def loadURDF(self, urdf_path, basePosition=None, baseOrientation=None, **kwargs):
        """Mock URDF loading."""
        body_id = self.next_body_id
        self.next_body_id += 1
        
        self.bodies[body_id] = {
            'urdf_path': urdf_path,
            'position': basePosition or [0, 0, 0],
            'orientation': baseOrientation or [0, 0, 0, 1],
            'num_joints': np.random.randint(0, 10),
            'links': {}
        }
        
        return body_id
    
    def getNumJoints(self, body_id):
        """Mock get number of joints."""
        if body_id in self.bodies:
            return self.bodies[body_id]['num_joints']
        return 0
    
    def getJointInfo(self, body_id, joint_index):
        """Mock get joint information."""
        joint_info = [
            joint_index,  # joint index
            f"joint_{joint_index}",  # joint name
            self.JOINT_REVOLUTE,  # joint type
            -1,  # q index
            -1,  # u index
            0,   # flags
            0.1, # joint damping
            0.1, # joint friction
            -np.pi,  # joint lower limit
            np.pi,   # joint upper limit
            10.0,    # joint max force
            1.0,     # joint max velocity
            f"link_{joint_index}",  # link name
            [0, 0, 1],  # joint axis
            [0, 0, 0],  # parent frame position
            [0, 0, 0, 1],  # parent frame orientation
            joint_index - 1 if joint_index > 0 else -1  # parent index
        ]
        return joint_info
    
    def getBasePositionAndOrientation(self, body_id):
        """Mock get base position and orientation."""
        if body_id in self.bodies:
            body = self.bodies[body_id]
            return body['position'], body['orientation']
        return [0, 0, 0], [0, 0, 0, 1]
    
    def getLinkState(self, body_id, link_index):
        """Mock get link state."""
        # Returns: linkWorldPosition, linkWorldOrientation, localInertialFramePosition, 
        #         localInertialFrameOrientation, worldLinkFramePosition, worldLinkFrameOrientation
        position = [np.random.uniform(-1, 1) for _ in range(3)]
        orientation = [0, 0, 0, 1]
        
        return [
            position,  # link world position
            orientation,  # link world orientation
            position,  # local inertial frame position
            orientation,  # local inertial frame orientation
            position,  # world link frame position
            orientation  # world link frame orientation
        ]
    
    def getContactPoints(self, bodyA=None, bodyB=None, linkIndexA=None, linkIndexB=None):
        """Mock get contact points between bodies."""
        if not self.contacts:
            # Generate some default contact points if none exist
            self.contacts = self._generate_default_contacts(bodyA, bodyB)
        
        filtered_contacts = []
        for contact in self.contacts:
            # Filter based on parameters
            if bodyA is not None and contact[1] != bodyA and contact[2] != bodyA:
                continue
            if bodyB is not None and contact[1] != bodyB and contact[2] != bodyB:
                continue
            if linkIndexA is not None and contact[3] != linkIndexA:
                continue
            if linkIndexB is not None and contact[4] != linkIndexB:
                continue
            
            filtered_contacts.append(contact)
        
        return filtered_contacts
    
    def _generate_default_contacts(self, bodyA=None, bodyB=None):
        """Generate default contact points for testing."""
        contacts = []
        num_contacts = np.random.randint(1, 5)
        
        bodyA = bodyA or 1
        bodyB = bodyB or 2
        
        for i in range(num_contacts):
            # Contact point format:
            # [contactFlag, bodyUniqueIdA, bodyUniqueIdB, linkIndexA, linkIndexB,
            #  positionOnA, positionOnB, contactNormalOnB, contactDistance, normalForce,
            #  lateralFriction1, lateralFrictionDir1, lateralFriction2, lateralFrictionDir2]
            
            position = [np.random.uniform(-0.05, 0.05) for _ in range(3)]
            normal = [0, 0, 1]  # Up normal
            force = np.random.uniform(0.5, 5.0)
            
            contact = [
                0,  # contact flag
                bodyA,  # body A
                bodyB,  # body B
                i,  # link A
                0,  # link B
                position,  # position on A
                position,  # position on B
                normal,  # contact normal on B
                0.001,  # contact distance
                force,  # normal force
                0.0,  # lateral friction 1
                [1, 0, 0],  # lateral friction dir 1
                0.0,  # lateral friction 2
                [0, 1, 0]   # lateral friction dir 2
            ]
            contacts.append(contact)
        
        return contacts
    
    def setContactFilter(self, body_id, link_index, enable):
        """Mock set contact filter."""
        pass
    
    def addUserDebugLine(self, lineFromXYZ, lineToXYZ, lineColorRGB=None, lineWidth=1):
        """Mock add debug line."""
        return np.random.randint(1, 1000)  # Debug item ID
    
    def removeUserDebugItem(self, debug_item_id):
        """Mock remove debug item."""
        pass
    
    def stepSimulation(self):
        """Mock step physics simulation."""
        self.simulation_time += self.time_step
        
        # Update contact points with some randomness
        self._update_contacts()
    
    def _update_contacts(self):
        """Update contact points with random variations."""
        # Add some noise to existing contacts
        for contact in self.contacts:
            # Add noise to force
            contact[9] += np.random.normal(0, 0.1)
            contact[9] = max(0, contact[9])  # Keep force positive
            
            # Add small noise to position
            for i in range(3):
                contact[5][i] += np.random.normal(0, 0.001)
                contact[6][i] += np.random.normal(0, 0.001)
    
    def rayTest(self, rayFromPosition, rayToPosition):
        """Mock ray test."""
        # Return format: [[objectUniqueId, linkIndex, hit fraction, hit position, hit normal]]
        hit_fraction = np.random.uniform(0.1, 0.9)
        hit_position = [
            rayFromPosition[i] + hit_fraction * (rayToPosition[i] - rayFromPosition[i])
            for i in range(3)
        ]
        hit_normal = [0, 0, 1]
        
        return [[1, 0, hit_fraction, hit_position, hit_normal]]
    
    def configureDebugVisualizer(self, flag, enable):
        """Mock configure debug visualizer."""
        pass
    
    def resetDebugVisualizerCamera(self, distance, yaw, pitch, target):
        """Mock reset debug visualizer camera."""
        pass
    
    def setRealTimeSimulation(self, enable):
        """Mock set real-time simulation."""
        pass
    
    def getKeyboardEvents(self):
        """Mock get keyboard events."""
        return {}
    
    def getMouseEvents(self):
        """Mock get mouse events."""
        return []


class MockPyBulletEnvironment:
    """Complete mock PyBullet environment with robots and objects."""
    
    def __init__(self):
        """Initialize mock environment."""
        self.p = MockPyBullet()
        self.robot_id = None
        self.object_ids = []
        
    def setup_environment(self):
        """Setup complete mock environment."""
        # Connect to physics
        self.p.connect(self.p.DIRECT)
        
        # Set physics parameters
        self.p.setGravity(0, 0, -9.81)
        self.p.setTimeStep(1.0/240.0)
        
        # Load robot
        self.robot_id = self.p.loadURDF("robot.urdf", basePosition=[0, 0, 0])
        
        # Load objects
        self.object_ids.append(
            self.p.loadURDF("cube.urdf", basePosition=[0.3, 0, 0.5])
        )
        self.object_ids.append(
            self.p.loadURDF("sphere.urdf", basePosition=[-0.3, 0, 0.5])
        )
        
        return self.robot_id, self.object_ids
    
    def get_robot_contacts(self):
        """Get contacts between robot and objects."""
        all_contacts = []
        
        if self.robot_id is not None:
            for obj_id in self.object_ids:
                contacts = self.p.getContactPoints(bodyA=self.robot_id, bodyB=obj_id)
                all_contacts.extend(contacts)
        
        return all_contacts
    
    def simulate_grasp(self, target_object_id):
        """Simulate grasping interaction."""
        # Generate realistic contact points for grasping
        grasp_contacts = []
        
        # Finger tip contacts
        positions = [
            [0.02, 0, 0],      # Thumb
            [-0.01, 0.015, 0], # Index finger
            [-0.01, -0.015, 0] # Middle finger
        ]
        
        for i, pos in enumerate(positions):
            contact = [
                0,  # contact flag
                self.robot_id,  # robot body
                target_object_id,  # object body
                7 + i,  # finger link
                0,  # object link
                pos,  # position on robot
                pos,  # position on object
                [0, 0, 1],  # contact normal
                0.001,  # distance
                np.random.uniform(1.0, 3.0),  # force
                0.0,  # lateral friction 1
                [1, 0, 0],  # friction dir 1
                0.0,  # lateral friction 2
                [0, 1, 0]   # friction dir 2
            ]
            grasp_contacts.append(contact)
        
        self.p.contacts = grasp_contacts
        return grasp_contacts
    
    def simulate_touch(self, contact_force=1.0):
        """Simulate light touch interaction."""
        touch_contact = [
            0,  # contact flag
            self.robot_id,  # robot body
            self.object_ids[0],  # first object
            8,  # fingertip link
            0,  # object link
            [0, 0, 0],  # position on robot
            [0, 0, 0],  # position on object
            [0, 0, 1],  # contact normal
            0.001,  # distance
            contact_force,  # force
            0.0,  # lateral friction 1
            [1, 0, 0],  # friction dir 1
            0.0,  # lateral friction 2
            [0, 1, 0]   # friction dir 2
        ]
        
        self.p.contacts = [touch_contact]
        return [touch_contact]
    
    def step_simulation(self, steps=1):
        """Step the simulation forward."""
        for _ in range(steps):
            self.p.stepSimulation()
    
    def cleanup(self):
        """Clean up the mock environment."""
        self.p.disconnect()


def create_mock_pybullet():
    """Factory function to create mock PyBullet module."""
    mock_module = Mock()
    
    # Create mock PyBullet instance
    mock_pb = MockPyBullet()
    
    # Map all PyBullet functions
    mock_module.connect = mock_pb.connect
    mock_module.disconnect = mock_pb.disconnect
    mock_module.setGravity = mock_pb.setGravity
    mock_module.setTimeStep = mock_pb.setTimeStep
    mock_module.loadURDF = mock_pb.loadURDF
    mock_module.getNumJoints = mock_pb.getNumJoints
    mock_module.getJointInfo = mock_pb.getJointInfo
    mock_module.getBasePositionAndOrientation = mock_pb.getBasePositionAndOrientation
    mock_module.getLinkState = mock_pb.getLinkState
    mock_module.getContactPoints = mock_pb.getContactPoints
    mock_module.stepSimulation = mock_pb.stepSimulation
    mock_module.rayTest = mock_pb.rayTest
    mock_module.configureDebugVisualizer = mock_pb.configureDebugVisualizer
    mock_module.resetDebugVisualizerCamera = mock_pb.resetDebugVisualizerCamera
    mock_module.setRealTimeSimulation = mock_pb.setRealTimeSimulation
    mock_module.getKeyboardEvents = mock_pb.getKeyboardEvents
    mock_module.getMouseEvents = mock_pb.getMouseEvents
    
    # Constants
    mock_module.GUI = mock_pb.GUI
    mock_module.DIRECT = mock_pb.DIRECT
    mock_module.JOINT_REVOLUTE = mock_pb.JOINT_REVOLUTE
    mock_module.JOINT_PRISMATIC = mock_pb.JOINT_PRISMATIC
    mock_module.JOINT_FIXED = mock_pb.JOINT_FIXED
    mock_module.COV_ENABLE_GUI = mock_pb.COV_ENABLE_GUI
    mock_module.COV_ENABLE_WIREFRAME = mock_pb.COV_ENABLE_WIREFRAME
    
    return mock_module, mock_pb