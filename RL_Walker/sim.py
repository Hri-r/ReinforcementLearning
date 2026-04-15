from Box2D import (
    b2World, b2PolygonShape, b2RevoluteJointDef, b2_staticBody
)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Create the Box2D world
world = b2World(gravity=(0, -9.8), doSleep=True)

# Lengths
L1, L2 = 2.0, 2.0
body_length = 3.0

patches = []


# Ground
ground = world.CreateStaticBody(
    position=(0, 0),
    shapes=b2PolygonShape(box=(50, 0))
)

# Body bar (initially horizontal)
body = world.CreateDynamicBody(position=(0, 4), angle=0.0)
body.CreatePolygonFixture(box=(body_length / 2, 0.4), density=1.0, friction=1.0, restitution=0.1)

# First leg link (base link)
link1 = world.CreateDynamicBody(position=(-body_length / 2, 4))
link1.CreatePolygonFixture(box=(L1 / 2, 0.1), density=1.0, friction=1.0, restitution=0.1)

# Second leg link
link2 = world.CreateDynamicBody(position=(-body_length / 2 - L1, 4))
link2.CreatePolygonFixture(box=(L2 / 2, 0.1), density=1.0, friction=1.0, restitution=0.1)

# Joint: body to link1 (hip)
joint1 = world.CreateJoint(b2RevoluteJointDef(
    bodyA=body,
    bodyB=link1,
    localAnchorA=(-body_length / 2, 0),
    localAnchorB=(L1 / 2, 0),
    enableMotor=True,
    maxMotorTorque=100.0,
    motorSpeed=0.0
))

# Joint: link1 to link2 (knee)
joint2 = world.CreateJoint(b2RevoluteJointDef(
    bodyA=link1,
    bodyB=link2,
    localAnchorA=(-L1 / 2, 0),
    localAnchorB=(L2 / 2, 0),
    enableMotor=True,
    maxMotorTorque=100.0,
    motorSpeed=0.0
))

# First leg link (base link)
link3 = world.CreateDynamicBody(position=(body_length / 2, 4))
link3.CreatePolygonFixture(box=(L1 / 2, 0.1), density=1.0, friction=1.0, restitution=0.1)

# Second leg link
link4 = world.CreateDynamicBody(position=(body_length / 2 + L1, 4))
link4.CreatePolygonFixture(box=(L2 / 2, 0.1), density=1.0, friction=1.0, restitution=0.1)

# Joint: body to link1 (hip)
joint3 = world.CreateJoint(b2RevoluteJointDef(
    bodyA=body,
    bodyB=link3,
    localAnchorA=(body_length / 2, 0),
    localAnchorB=(-L1 / 2, 0),
    enableMotor=True,
    maxMotorTorque=100.0,
    motorSpeed=0.0
))

# Joint: link1 to link2 (knee)
joint4 = world.CreateJoint(b2RevoluteJointDef(
    bodyA=link3,
    bodyB=link4,
    localAnchorA=(L1 / 2, 0),
    localAnchorB=(-L2 / 2, 0),
    enableMotor=True,
    maxMotorTorque=100.0,
    motorSpeed=0.0
))

# Matplotlib setup
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-15, 15)
ax.set_ylim(-1, 7)
line, = ax.plot([], [], 'o-', lw=4)
ground_line, = ax.plot([-50, 50], [0, 0], 'k-', lw=3)

for _ in range(6):  # body, link1, link2, ground
    patch, = ax.plot([], [], lw=2)
    patches.append(patch)

def draw_body_patch(body, patch):
    for fixture in body.fixtures:
        shape = fixture.shape
        vertices = [body.transform * v for v in shape.vertices]
        xs, ys = zip(*vertices)
        # Close the polygon
        xs += (xs[0],)
        ys += (ys[0],)
        patch.set_data(xs, ys)

def init():
    line.set_data([], [])
    return line, ground_line

def set_action(motor_speeds):
    joint1.motorSpeed = motor_speeds[0]
    joint2.motorSpeed = motor_speeds[1]
    joint3.motorSpeed = motor_speeds[2]
    joint4.motorSpeed = motor_speeds[3]

def dummy_policy(i):
    return [(-1)**np.random.randint(low = 1, high = 3)*np.random.random(), (-1)**np.random.randint(low = 1, high = 3)*np.random.random(), (-1)**np.random.randint(low = 1, high = 3)*np.random.random(), (-1)**np.random.randint(low = 1, high = 3)*np.random.random()]

def animate(i):

    action = dummy_policy(i)
    set_action(action)

    world.Step(1.0 / 60, 6, 2)


    draw_body_patch(body, patches[0])
    draw_body_patch(link1, patches[1])
    draw_body_patch(link2, patches[2])
    draw_body_patch(link3, patches[3])
    draw_body_patch(link4, patches[4])
    draw_body_patch(ground, patches[5])

    return patches

ani = animation.FuncAnimation(
    fig, animate, frames=6000, init_func=lambda: patches, blit=True, interval=1000 / 60
)

plt.title("SIM")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()
