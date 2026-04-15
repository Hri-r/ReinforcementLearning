import time

import mujoco
import mujoco.viewer

import numpy as np

m = mujoco.MjModel.from_xml_path('four_legged_bot.xml')
d = mujoco.MjData(m)
stepcount = 0

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 300:
    stepcount+=1
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    d.ctrl[0] = 1
    d.ctrl[1] = 0
    d.ctrl[2] = -1
    d.ctrl[3] = 0
    d.ctrl[4] = -1
    d.ctrl[5] = 0
    d.ctrl[6] = 1
    d.ctrl[7] = 0

    mujoco.mj_step(m, d)

    torso_pos = d.geom_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "torso_geom")]
    print(torso_pos)

    # torso_linvel = d.cvel[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "torso_geom")][:3]  # Linear velocity [vx, vy, vz]
    # if(stepcount%5==0):
    #   print(torso_linvel)

    joint_name_to_print = "front_left_knee_joint"

    # 2. Get the joint's unique ID
    joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name_to_print)

    # 3. Find the index (address) of the joint in the qpos array
    qpos_address = m.jnt_qposadr[joint_id]

    # 4. Access the angle value from d.qpos
    # The value is in radians.
    joint_angle_rad = d.qpos[qpos_address]

    # 5. Convert radians to degrees for easier interpretation
    joint_angle_deg = np.rad2deg(joint_angle_rad)

    # 6. Print the formatted result
    # print(f"Angle of '{joint_name_to_print}': {joint_angle_deg:.2f} degrees")
    
    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1
      viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
      

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)