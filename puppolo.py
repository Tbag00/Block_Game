import random
import sys
import numpy as np
import pygame

def assign_colors():
    return {
        0: (0, 0, 0),  # Nero per spazi vuoti
        1: (255, 0, 0),  # Rosso
        2: (255, 165, 0),  # Arancione
        3: (255, 255, 0),  # Giallo
        4: (0, 255, 0),  # Verde
        5: (0, 0, 255),  # Blu
        6: (128, 0, 128)  # Viola
    }


def draw_matrix(surface, matrix, color_map, start_x, start_y, block_size):
    font = pygame.font.Font(None, 36)
    size = matrix.shape[0]  # Matrice quadrata
    for i in range(size):
        for j in range(size):
            value = matrix[i, j]
            x = start_x + j * block_size
            y = start_y + i * block_size
            color = color_map.get(value, (255, 255, 255))  # Default bianco se valore non previsto
            pygame.draw.rect(surface, color, (x, y, block_size, block_size))
            pygame.draw.rect(surface, (255, 255, 255), (x, y, block_size, block_size), 2)
            if value != 0:
                text = font.render(str(value), True, (255, 255, 255))
                text_rect = text.get_rect(center=(x + block_size // 2, y + block_size // 2))
                surface.blit(text, text_rect)


def aggiorna_matrice(matrix, movimento):
    colonna_sorgente, colonna_destinazione = movimento

    # Trova il primo blocco non nullo nella colonna sorgente
    for i in range(matrix.shape[0]):
        if matrix[i, colonna_sorgente] != 0:
            valore_blocco = matrix[i, colonna_sorgente]
            matrix[i, colonna_sorgente] = 0  # Rimuovi il blocco dalla colonna sorgente
            break
    else:
        return matrix  # Se non ci sono blocchi nella colonna sorgente, non fare nulla

    # Trova la prima posizione vuota nella colonna di destinazione
    for j in range(matrix.shape[0] - 1, -1, -1):
        if matrix[j, colonna_destinazione] == 0:
            matrix[j, colonna_destinazione] = valore_blocco  # Inserisci il blocco nella colonna di destinazione
            break

    return matrix


def inverti_mossa(mossa):
    """Inverte la mossa scambiando la sorgente con la destinazione"""
    return (mossa[1], mossa[0])  # Scambia la colonna di partenza con quella di destinazione


def anima_matrice(matrix1, matrix2, movimenti):
    color_map = assign_colors()
    pygame.init()

    block_size = 50
    padding = 20
    size = matrix1.shape[0]  # Matrice quadrata

    width = 2 * size * block_size + 3 * padding
    height = size * block_size + 2 * padding

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Block World")

    clock = pygame.time.Clock()

    move_index = 0  # Tiene traccia della mossa corrente
    running = True

    while running:
        screen.fill((0, 0, 0))
        draw_matrix(screen, matrix1, color_map, padding, padding, block_size)
        draw_matrix(screen, matrix2, color_map, padding + size * block_size + padding, padding, block_size)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT and move_index < len(movimenti):
                    # Avanza alla prossima mossa
                    matrix1 = aggiorna_matrice(matrix1, movimenti[move_index])
                    move_index += 1
                elif event.key == pygame.K_LEFT and move_index > 0:
                    # Torna alla mossa precedente eseguendo la mossa inversa
                    move_index -= 1
                    matrix1 = aggiorna_matrice(matrix1, inverti_mossa(movimenti[move_index]))

        clock.tick(30)

    pygame.quit()
    sys.exit()


'''if __name__ == "__main__":
    size = 6  #np.random.randint(2, 7)
    matrix1 = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 6, 0],
               [2, 5, 0, 0, 3, 4]])
    matrix2 = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 6, 0, 0, 5],
               [3, 2, 4, 0, 0, 1]])

    # Esegui dei movimenti sulla matrice1
    movimenti = [(1, 2), (4, 2), (1, 5), (0, 1), (4, 0), (5, 4), (5, 3), (2, 5), (4, 5), (2, 5), (3, 2), (5, 4), (5, 4),
                 (5, 2), (4, 3), (4, 5), (3, 5)]  # Esempio di movimenti: (colonna_sorgente, colonna_destinazione)
    #movimenti = [(5, 2), (1, 5), (4, 2), (1, 5), (0, 1), (4, 0)]

    anima_matrice(matrix1, matrix2, movimenti)'''
