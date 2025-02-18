import pygame
import random


def draw_random_squares(num_squares=10, screen_size=(500, 500)):
    pygame.init()
    screen = pygame.display.set_mode(screen_size)
    screen.fill((255, 255, 255))  # Sfondo bianco

    squares = []  # Lista per memorizzare le posizioni dei quadrati

    for _ in range(num_squares):
        valid_position = False

        while not valid_position:
            size = 80  # Dimensione casuale
            x = random.randint(0, screen_size[0] - size)
            y = random.randint(0, screen_size[1] - size)

            # Controlla che il nuovo quadrato non si sovrapponga agli altri
            valid_position = all(
                not (x < sx + ssize and x + size > sx and y < sy + ssize and y + size > sy)
                for sx, sy, ssize in squares
            )

        squares.append((x, y, size))
        pygame.draw.rect(screen, (0, 0, 0), (x, y, size, size), 3)  # Bordi neri

    pygame.display.flip()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()


# Esegui la funzione per visualizzare i quadrati
draw_random_squares()
