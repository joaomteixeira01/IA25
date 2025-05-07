#-------- Grupo 34 --------#
# 97226 João Teixeira      #
# 110094 Francisco Fialho  #
#--------------------------#

from sys import stdin
from search import*


class NuruominoState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = Nuroumino.state_id
        Nuroumino.state_id += 1

    def __lt__(self, other):
        """ Este método é utilizado em caso de empate na gestão da lista
        de abertos nas procuras informadas. """
        return self.id < other.id

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


class Nuruomino(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        #TODO
        pass 

    def actions(self, state: NuruominoState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        #TODO
        pass 

    def result(self, state: NuruominoState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""

        #TODO
        pass 
        

    def goal_test(self, state: NuruominoState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        #TODO
        pass 

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass


def main():
    board = Board.parse_instance()
    board.print_instance()

    # Testar adjacent_regions
    print("\nRegiões adjacentes à região 3:")
    print(board.adjacent_regions(3))  # Espera-se que devolva uma lista com regiões vizinhas de 3

    # Testar adjacent_positions
    row, col = 3, 3  # exemplo: coordenadas linha 2, coluna 3
    print(f"\nPosições adjacentes à célula ({row}, {col}):")
    print(board.adjacent_positions(row, col))

    # Testar adjacent_values
    print(f"\nValores adjacentes à célula ({row}, {col}):")
    print(board.adjacent_values(row, col))

if __name__ == "__main__":
    main()
