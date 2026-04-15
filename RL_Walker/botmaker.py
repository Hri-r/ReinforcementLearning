from dm_control import mjcf
import re

# ------------------------------------------
# Create MJCF model
# ------------------------------------------
model = mjcf.RootElement()
model.model = 'four_legged_bot'

# Ground
model.worldbody.add('geom', name='ground', type='plane', size=[10, 10, 0.1], rgba=[0.2, 0.2, 0.2, 1])

# Torso
torso = model.worldbody.add('body', name='torso', pos=[0, 0, 0.5])
torso.add('geom', name='torso_geom', type='box', size=[0.3, 0.2, 0.1], rgba=[0.8, 0.6, 0.4, 1])
torso.add('joint', name='torso_free_joint', type='free', damping=1)

# Legs
leg_pos = {
    'front_left':  [ 0.2,  0.15, 0],
    'front_right': [ 0.2, -0.15, 0],
    'back_left':   [-0.2,  0.15, 0],
    'back_right':  [-0.2, -0.15, 0],
}

for name, pos in leg_pos.items():
    hip = torso.add('body', name=f'{name}_hip', pos=pos)
    hip.add('joint', name=f'{name}_hip_joint', type='hinge', axis=[0, 1, 0], range=[-45, 45], damping=1)
    hip.add('geom', name=f'{name}_hip_geom', type='capsule', fromto=[0, 0, 0, 0, 0, -0.2], size=[0.03], rgba=[0.8, 0.6, 0.4, 1])

    knee = hip.add('body', name=f'{name}_knee', pos=[0, 0, -0.2])
    knee.add('joint', name=f'{name}_knee_joint', type='hinge', axis=[0, 1, 0], range=[-90, 0], damping=1)
    knee.add('geom', name=f'{name}_knee_geom', type='capsule', fromto=[0, 0, 0, 0, 0, -0.2], size=[0.025], rgba=[0.8, 0.6, 0.4, 1])

# ------------------------------------------
# Generate XML and clean it up
# ------------------------------------------
xml = model.to_xml_string()

# Remove default block and class="/" attributes
xml = re.sub(r'<default>\s*<default class="/">.*?</default>\s*</default>', '', xml, flags=re.DOTALL)
xml = re.sub(r'\s*class="/"', '', xml)

# Save to file
with open("clean_four_legged_bot.xml", "w") as f:
    f.write(xml)

print("✅ Clean MJCF XML saved as clean_four_legged_bot.xml")
