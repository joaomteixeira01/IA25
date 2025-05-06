# Ficheiro principal que executa o programa

from board import Board

def main():
    board = Board.parse_instance()
    board.print_instance()

    # Testar adjacent_regions
    print("\nRegiões adjacentes à região 3:")
    print(board.adjacent_regions(3))  # Espera-se que devolva uma lista com regiões vizinhas de 3

    # Testar adjacent_positions
    row, col = 2, 3  # exemplo: coordenadas linha 2, coluna 3
    print(f"\nPosições adjacentes à célula ({row}, {col}):")
    print(board.adjacent_positions(row, col))

    # Testar adjacent_values
    print(f"\nValores adjacentes à célula ({row}, {col}):")
    print(board.adjacent_values(row, col))

if __name__ == "__main__":
    main()
