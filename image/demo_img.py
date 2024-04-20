# Created by jing at 19.04.24
from PIL import Image, ImageDraw

# Image dimensions
width = 400
height = 200

# Create a blank RGB image
image = Image.new("RGB", (width, height), "white")

# Create a drawing object
draw = ImageDraw.Draw(image)

# Draw the ground
draw.rectangle([0, height - 10, width, height], fill="grey")

# Draw the stick figure agent
agent_head_radius = 10
agent_body_height = 50
agent_leg_length = 30
agent_arm_length = 30
agent_position = ((width - agent_head_radius) // 2, height - agent_body_height - agent_leg_length)
agent_head = (agent_position[0] + agent_head_radius, agent_position[1] - agent_head_radius)
draw.ellipse((agent_head[0] - agent_head_radius, agent_head[1] - agent_head_radius,
              agent_head[0] + agent_head_radius, agent_head[1] + agent_head_radius), fill="black")
draw.line((agent_position[0], agent_position[1] + agent_head_radius,
           agent_position[0], agent_position[1] + agent_head_radius + agent_body_height), fill="black")
draw.line((agent_position[0], agent_position[1] + agent_head_radius + agent_body_height,
           agent_position[0] - agent_arm_length, agent_position[1] + agent_head_radius + agent_body_height - agent_arm_length), fill="black")
draw.line((agent_position[0], agent_position[1] + agent_head_radius + agent_body_height,
           agent_position[0] + agent_arm_length, agent_position[1] + agent_head_radius + agent_body_height - agent_arm_length), fill="black")
draw.line((agent_position[0], agent_position[1] + agent_head_radius + agent_body_height,
           agent_position[0] - agent_leg_length, agent_position[1] + agent_head_radius + agent_body_height + agent_leg_length), fill="black")
draw.line((agent_position[0], agent_position[1] + agent_head_radius + agent_body_height,
           agent_position[0] + agent_leg_length, agent_position[1] + agent_head_radius + agent_body_height + agent_leg_length), fill="black")

# Draw the door
door_width = 80
door_height = 120
door_position = (width - 100, height - door_height)
draw.rectangle(door_position + (door_position[0] + door_width, height), fill="brown")

# Draw the key
key_width = 20
key_height = 10
key_position = (50, height - key_height)
draw.rectangle(key_position + (key_position[0] + key_width, height), fill="gold")

# Save the image
image.save("game_scene.png")

