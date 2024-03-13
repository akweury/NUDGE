import pygame
import random

# Initialize pygame
pygame.init()

# Set up the screen
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Avoidance Game")

# Define colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GRAY = (96, 96, 96)
# Define player properties
player_radius = 10
player_x = WIDTH // 2
player_y = HEIGHT // 2
player_speed = 5

# Define enemy properties
num_enemies = 10
enemy_radius = 5
enemy_speed = 3
enemies = []
enemy_directions = []
for _ in range(num_enemies):
    enemies.append([random.randint(0, WIDTH), random.randint(0, HEIGHT)])
    enemy_directions.append([random.choice([-1, 1]), random.choice([-1, 1])])

# Main game loop
running = True
clock = pygame.time.Clock()
while running:
    screen.fill(WHITE)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Move the player
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        player_x -= player_speed
    if keys[pygame.K_RIGHT]:
        player_x += player_speed
    if keys[pygame.K_UP]:
        player_y -= player_speed
    if keys[pygame.K_DOWN]:
        player_y += player_speed

    # Keep player within screen bounds
    player_x = max(player_radius, min(player_x, WIDTH - player_radius))
    player_y = max(player_radius, min(player_y, HEIGHT - player_radius))

    # Draw the player
    pygame.draw.circle(screen, RED, (player_x, player_y), player_radius)

    # Move enemies
    for i, enemy in enumerate(enemies):
        enemy[0] += enemy_speed * enemy_directions[i][0]
        enemy[1] += enemy_speed * enemy_directions[i][1]

        # Check for collision with screen borders
        if enemy[0] < enemy_radius or enemy[0] > WIDTH - enemy_radius:
            enemy_directions[i][0] *= -1
        if enemy[1] < enemy_radius or enemy[1] > HEIGHT - enemy_radius:
            enemy_directions[i][1] *= -1

        pygame.draw.circle(screen, GRAY, (enemy[0], enemy[1]), enemy_radius)

        # Check for collision with player
        distance = ((player_x - enemy[0]) ** 2 + (player_y - enemy[1]) ** 2) ** 0.5
        if distance < player_radius + enemy_radius:
            running = False

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)

# Quit pygame
pygame.quit()
