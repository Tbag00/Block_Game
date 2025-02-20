import pygame
import numpy as np


def edit_matrix(matrix):
    pygame.init()

    CELL_SIZE = 50
    PADDING = 10
    FONT_SIZE = 30

    rows, cols = matrix.shape
    width = cols * CELL_SIZE + PADDING * 2
    height = rows * CELL_SIZE + PADDING * 2

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Modifica Matrice")
    font = pygame.font.Font(None, FONT_SIZE)

    running = True
    while running:
        screen.fill((255, 255, 255))

        for r in range(rows):
            for c in range(cols):
                rect = pygame.Rect(PADDING + c * CELL_SIZE, PADDING + r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, (0, 0, 0), rect, 2)
                text_surf = font.render(str(matrix[r, c]), True, (0, 0, 0))
                text_rect = text_surf.get_rect(center=rect.center)
                screen.blit(text_surf, text_rect)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    running = False
                elif event.key == pygame.K_r:
                    matrix = np.vstack((matrix, np.zeros((1, cols), dtype=int)))
                    rows += 1
                    height = rows * CELL_SIZE + PADDING * 2
                    screen = pygame.display.set_mode((width, height))
                elif event.key == pygame.K_c:
                    matrix = np.hstack((matrix, np.zeros((rows, 1), dtype=int)))
                    cols += 1
                    width = cols * CELL_SIZE + PADDING * 2
                    screen = pygame.display.set_mode((width, height))
                elif event.key == pygame.K_t:
                    if rows > 1:
                        matrix = matrix[:-1, :]
                        rows -= 1
                        height = rows * CELL_SIZE + PADDING * 2
                        screen = pygame.display.set_mode((width, height))
                elif event.key == pygame.K_v:
                    if cols > 1:
                        matrix = matrix[:, :-1]
                        cols -= 1
                        width = cols * CELL_SIZE + PADDING * 2
                        screen = pygame.display.set_mode((width, height))
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                col = (x - PADDING) // CELL_SIZE
                row = (y - PADDING) // CELL_SIZE
                if 0 <= row < rows and 0 <= col < cols:
                    matrix[row, col] = (matrix[row, col] + 1) % 7  # Cambia valore ciclicamente tra 0 e 6

    pygame.quit()
    return matrix


# Esempio di utilizzo
#if __name__ == "__main__":
# initial_matrix = np.zeros((3, 3), dtype=int)
# updated_matrix = edit_matrix(initial_matrix)
# print("Matrice aggiornata:")
# print(updated_matrix)
