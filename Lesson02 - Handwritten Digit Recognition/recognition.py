# THIS GAME IS TAKEN FROM HARVARD CS50 INTRODUCTION TO AI

import numpy as np
import pygame
import sys
import tensorflow as tf

# == SET THE GAME UP ==
def load_model():
    """Load the trained MNIST model."""
    return tf.keras.models.load_model("model.h5")

def init_pygame():
    """Initialize Pygame and set up the screen."""
    pygame.init()
    size = width, height = 600, 400
    return pygame.display.set_mode(size), width, height


# Initialize Pygame FIRST before loading fonts
pygame.init()

# Define Fonts
OPEN_SANS = "assets/OpenSansRegular.ttf"
smallFont = pygame.font.Font(OPEN_SANS, 20)
largeFont = pygame.font.Font(OPEN_SANS, 40)
BLACK, WHITE = (0, 0, 0), (255, 255, 255)
ROWS, COLS, OFFSET, CELL_SIZE = 28, 28, 20, 10


def create_grid():
    """Initialize an empty 28x28 handwriting grid."""
    return [[0] * COLS for _ in range(ROWS)]

def draw_grid(screen, handwriting, mouse):
    """Draw the 28x28 pixel grid where the user can write."""
    for i in range(ROWS):
        for j in range(COLS):
            rect = pygame.Rect(OFFSET + j * CELL_SIZE, OFFSET + i * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            # If cell has been written on, darken it
            if handwriting[i][j]:
                shade = 255 - (handwriting[i][j] * 255)
                pygame.draw.rect(screen, (shade, shade, shade), rect)
            else:
                pygame.draw.rect(screen, WHITE, rect)

            pygame.draw.rect(screen, BLACK, rect, 1)

            # If user is writing, update grid
            if mouse and rect.collidepoint(mouse):
                handwriting[i][j] = 250 / 255
                if i + 1 < ROWS:
                    handwriting[i + 1][j] = 220 / 255
                if j + 1 < COLS:
                    handwriting[i][j + 1] = 220 / 255
                if i + 1 < ROWS and j + 1 < COLS:
                    handwriting[i + 1][j + 1] = 190 / 255

def create_button(screen, text, x, y):
    """Create a button and return its Rect object."""
    button = pygame.Rect(x, y, 100, 30)
    pygame.draw.rect(screen, WHITE, button)
    text_surface = smallFont.render(text, True, BLACK)
    text_rect = text_surface.get_rect(center=button.center)
    screen.blit(text_surface, text_rect)
    return button

def classify_handwriting(model, handwriting):
    """Use the trained model to classify the user's drawing."""
    return model.predict([np.array(handwriting).reshape(1, 28, 28, 1)]).argmax()

def display_classification(screen, classification, width):
    """Show the predicted digit on the right side of the screen."""
    if classification is not None:
        text_surface = largeFont.render(str(classification), True, WHITE)
        text_rect = text_surface.get_rect(center=(width - 150, 100))
        screen.blit(text_surface, text_rect)

def main():
    """Main game loop."""
    screen, width, height = init_pygame()
    model = load_model()
    handwriting = create_grid()
    classification = None

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        screen.fill(BLACK)
        mouse = pygame.mouse.get_pos() if pygame.mouse.get_pressed()[0] else None

        draw_grid(screen, handwriting, mouse)

        # Create buttons
        reset_button = create_button(screen, "Reset", 30, OFFSET + ROWS * CELL_SIZE + 30)
        classify_button = create_button(screen, "Classify", 150, OFFSET + ROWS * CELL_SIZE + 30)

        # Handle button clicks
        if mouse:
            if reset_button.collidepoint(mouse):
                handwriting = create_grid()
                classification = None
            elif classify_button.collidepoint(mouse):
                classification = classify_handwriting(model, handwriting)

        display_classification(screen, classification, width)
        pygame.display.flip()

if __name__ == "__main__":
    main()
