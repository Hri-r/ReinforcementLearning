import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
from Box2D import b2World, b2PolygonShape, b2RevoluteJointDef


class LegEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.world = b2World(gravity=(0, -9.8), doSleep=True)
        self.timestep = 1.0 / 60.0
        self.max_steps = 5000
        self.step_count = 0

        # Action space: motor speeds for 4 joints
        self.action_space = spaces.Box(low=-5, high=5, shape=(4,), dtype=np.float32)

        # Observation: joint angles, speeds, body pos/vel
        high = np.array([np.inf] * 12, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Rendering
        self.patches = []
        self.fig = None
        self.ax = None
        if self.render_mode == "human":
            self._init_render()

        self._build_world()

    def _init_render(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-1, 7)
        self.ax.set_title("Legged Robot")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.grid(True)

        # Ground line
        self.ax.plot([-50, 50], [0, 0], 'k-', lw=3)

        for _ in range(6):  # body, link1–4, ground
            patch, = self.ax.plot([], [], lw=2)
            self.patches.append(patch)

    def _draw_body_patch(self, body, patch):
        for fixture in body.fixtures:
            shape = fixture.shape
            vertices = [body.transform * v for v in shape.vertices]
            xs, ys = zip(*vertices)
            xs += (xs[0],)
            ys += (ys[0],)
            patch.set_data(xs, ys)

    def _build_world(self):
        w = self.world
        self.ground = w.CreateStaticBody(
            position=(0, 0),
            shapes=b2PolygonShape(box=(50, 0))
        )

        body_length = 3.0
        L1, L2 = 2.0, 2.0

        # Central body
        self.body = w.CreateDynamicBody(position=(0, 4))
        self.body.CreatePolygonFixture(box=(body_length / 2, 0.4), density=1.0, friction=1.0)

        # Left leg
        link1_angle = 0
        link2_angle = 0
        self.link1 = w.CreateDynamicBody(position=(-body_length / 2 - L1 / 2, 4))
        self.link1.angle = link1_angle
        self.link1.CreatePolygonFixture(box=(L1 / 2, 0.1), density=1.0)

        self.link2 = w.CreateDynamicBody(position=(-body_length / 2 - L1 - L2 / 2, 4))
        self.link2.angle = link2_angle
        self.link2.CreatePolygonFixture(box=(L2 / 2, 0.1), density=1.0)

        # Right leg
        link3_angle = 0
        link4_angle = 0
        self.link3 = w.CreateDynamicBody(position=(body_length / 2 + L1 / 2, 4))
        self.link3.angle = link3_angle
        self.link3.CreatePolygonFixture(box=(L1 / 2, 0.1), density=1.0)

        self.link4 = w.CreateDynamicBody(position=(body_length / 2 + L1 + L2 / 2, 4))
        self.link4.angle = link4_angle
        self.link4.CreatePolygonFixture(box=(L2 / 2, 0.1), density=1.0)

        # Joints
        import math

        self.joint1 = w.CreateJoint(b2RevoluteJointDef(
            bodyA=self.body, bodyB=self.link1,
            localAnchorA=(-body_length / 2, 0),
            localAnchorB=(L1 / 2, 0),
            enableMotor=True, maxMotorTorque=100.0, motorSpeed=0.0,
            enableLimit=True, lowerAngle=-math.pi/3, upperAngle=math.pi/2
        ))
        self.joint2 = w.CreateJoint(b2RevoluteJointDef(
            bodyA=self.link1, bodyB=self.link2,
            localAnchorA=(-L1 / 2, 0),
            localAnchorB=(L2 / 2, 0),
            enableMotor=True, maxMotorTorque=100.0, motorSpeed=0.0,
            enableLimit=True, lowerAngle= 0, upperAngle=math.pi/2
        ))
        self.joint3 = w.CreateJoint(b2RevoluteJointDef(
            bodyA=self.body, bodyB=self.link3,
            localAnchorA=(body_length / 2, 0),
            localAnchorB=(-L1 / 2, 0),
            enableMotor=True, maxMotorTorque=100.0, motorSpeed=0.0,
            enableLimit=True, lowerAngle=-math.pi/2, upperAngle=math.pi/3
        ))
        self.joint4 = w.CreateJoint(b2RevoluteJointDef(
            bodyA=self.link3, bodyB=self.link4,
            localAnchorA=(L1 / 2, 0),
            localAnchorB=(-L2 / 2, 0),
            enableMotor=True, maxMotorTorque=100.0, motorSpeed=0.0,
            enableLimit=True, lowerAngle= -math.pi/2, upperAngle= 0
        ))

        self.joints = [self.joint1, self.joint2, self.joint3, self.joint4]

    def _get_obs(self):
        return np.array([
            self.joint1.angle,
            self.joint2.angle,
            self.joint3.angle,
            self.joint4.angle,
            self.joint1.speed,
            self.joint2.speed,
            self.joint3.speed,
            self.joint4.speed,
            self.body.position.x,
            self.body.position.y,
            self.body.linearVelocity.x,
            self.body.linearVelocity.y,
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        for body in self.world.bodies[:]:
            self.world.DestroyBody(body)

        self.step_count = 0
        self._build_world()

        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        self.step_count += 1

        print(action)

        for i, joint in enumerate(self.joints):
            joint.motorSpeed = float(action[i])

        # for i, joint in enumerate(self.joints):
        #     if i!=3:
        #         joint.motorSpeed = float(action[i])
        #     else:
        #         joint.motorSpeed = 1.0
        #         print(joint.angle)

        self.world.Step(self.timestep, 6, 2)
        obs = self._get_obs()

        reward = self.body.linearVelocity.x + 0.01
        terminated = self.body.position.y <= 0.5
        truncated = self.step_count >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != "human":
            return

        self._draw_body_patch(self.body, self.patches[0])
        self._draw_body_patch(self.link1, self.patches[1])
        self._draw_body_patch(self.link2, self.patches[2])
        self._draw_body_patch(self.link3, self.patches[3])
        self._draw_body_patch(self.link4, self.patches[4])
        self._draw_body_patch(self.ground, self.patches[5])
        plt.pause(0.001)

    def close(self):
        if self.render_mode == "human" and self.fig is not None:
            plt.close(self.fig)


# i = 0
# le = LegEnv(render_mode="human")
# while(i<1000000):
#     le.step([0,0,0,0])
#     i+=1