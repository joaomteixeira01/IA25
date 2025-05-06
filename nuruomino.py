#-------- Grupo 34 --------#
# 97226 João Teixeira      #
# 110094 Francisco Fialho  #
#--------------------------#

from sys import stdin

class Board:
    """Representação interna de um tabuleiro do Puzzle Nuruomino"""

    def __init__(self, grid):
        self.grid = grid
        self.n = len(grid)

    def get_value(self, row, col):
        """Devolve o valor da célula na posição (row, col)"""
        return self.grid[row][col]

    def print_instance(self):
        """Imprime a grelha no formato esperado (com tabs)"""
        for row in self.grid:
            print('\t'.join(map(str, row)))
    
    def adjacent_regions(self, region: int) -> list:
        """Devolve uma lista das regiões que fazem fronteira com a região enviada no argumento"""
        adjacents = set()
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if self.grid[i][j] == str(region):  # célula da região procurada
                    for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # cima, baixo, esquerda, direita
                        ni, nj = i + di, j + dj
                        if 0 <= ni < len(self.grid) and 0 <= nj < len(self.grid[0]):
                            val = self.grid[ni][nj]
                            if val != str(region):
                                adjacents.add(int(val))  # converter para inteiro
        return list(adjacents)
    
    def adjacent_positions(self, row: int, col: int) -> list:
        """Devolve as posições adjacentes à região, em todas as direções, incluindo diagonais."""
        positions = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue  # ignora a própria célula
                ni, nj = row + di, col + dj
                if 0 <= ni < len(self.grid) and 0 <= nj < len(self.grid[0]):
                    positions.append((ni, nj))
        return positions
    
    def adjacent_values(self, row: int, col: int) -> list:
        """Devolve os valores das células adjacentes à região, em todas as direções, incluindo diagonais"""
        values = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue  # ignora a própria célula
                ni, nj = row + di, col + dj
                if 0 <= ni < len(self.grid) and 0 <= nj < len(self.grid[0]):
                    values.append(self.grid[ni][nj])
        return values

    @staticmethod
    def parse_instance():
        """Lê o input do stdin e cria uma instância de Board."""
        grid = []
        for line in stdin:
            if line.strip() == "":
                continue  # ignora linhas vazias
            grid.append(line.strip().split())  # divide por espaço ou tab
        return Board(grid)


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
