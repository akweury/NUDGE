import pygame
import random

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Define constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
MAZE_SIZE = 15
CELL_SIZE = SCREEN_WIDTH // MAZE_SIZE


class MazeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Maze Game")
        self.clock = pygame.time.Clock()
        self.maze = self.generate_maze(MAZE_SIZE)
        self.player_position = (MAZE_SIZE // 2, MAZE_SIZE // 2)
        self.coin_position = self.place_coin()

    def generate_maze(self, size):
        maze = [['#' for _ in range(size)] for _ in range(size)]
        for i in range(1, size - 1):
            for j in range(1, size - 1):
                maze[i][j] = ' ' if random.random() < 0.7 else '#'
        return maze

    def place_coin(self):
        empty_spaces = [(i, j) for i in range(1, MAZE_SIZE - 1) for j in range(1, MAZE_SIZE - 1) if
                        self.maze[i][j] == ' ']
        return random.choice(empty_spaces)

    def draw_maze(self):
        self.screen.fill(BLACK)
        for i in range(MAZE_SIZE):
            for j in range(MAZE_SIZE):
                if self.maze[i][j] == '#':
                    pygame.draw.rect(self.screen, WHITE, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(self.screen, GREEN, (
        self.player_position[1] * CELL_SIZE, self.player_position[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(self.screen, YELLOW,
                         (self.coin_position[1] * CELL_SIZE, self.coin_position[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.display.update()

    def move_player(self, direction):
        x, y = self.player_position
        if direction == 'up' and self.maze[x - 1][y] != '#':
            self.player_position = (x - 1, y)
        elif direction == 'down' and self.maze[x + 1][y] != '#':
            self.player_position = (x + 1, y)
        elif direction == 'left' and self.maze[x][y - 1] != '#':
            self.player_position = (x, y - 1)
        elif direction == 'right' and self.maze[x][y + 1] != '#':
            self.player_position = (x, y + 1)

    def play(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.move_player('up')
                    elif event.key == pygame.K_DOWN:
                        self.move_player('down')
                    elif event.key == pygame.K_LEFT:
                        self.move_player('left')
                    elif event.key == pygame.K_RIGHT:
                        self.move_player('right')

            self.draw_maze()
            if self.player_position == self.coin_position:
                print("Congratulations! You found the coin!")
                running = False

            self.clock.tick(30)

        pygame.quit()


# Play the game
game = MazeGame()
game.play()
