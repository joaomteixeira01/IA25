# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 34:
# 97226 João Teixeira 
# 110094 Francisco Fialho

from search import Node, Problem

# Tetraminos permitidos no puzzle (todas as rotações/reflexões)
TETRAMINOS = {
    'L': [
        [(0,0), (1,0), (2,0), (2,1)],
        [(0,0), (0,1), (0,2), (1,0)],
        [(0,0), (0,1), (1,1), (2,1)],
        [(0,2), (1,0), (1,1), (1,2)],
        [(0,0), (0,1), (1,0), (2,0)],
        [(0,0), (0,1), (0,2), (1,2)],
        [(0,1), (1,1), (2,0), (2,1)],
        [(0,0), (1,0), (1,1), (1,2)],
    ],
    'I': [
        [(0,0), (1,0), (2,0), (3,0)],
        [(0,0), (0,1), (0,2), (0,3)],
    ],
    'T': [
        [(0,0), (0,1), (0,2), (1,1)],
        [(0,1), (1,0), (1,1), (2,1)],
        [(1,0), (1,1), (1,2), (0,1)],
        [(0,0), (1,0), (1,1), (2,0)],
    ],
    'S': [
        [(0,1), (0,2), (1,0), (1,1)],
        [(0,0), (1,0), (1,1), (2,1)],
        [(1,1), (1,2), (2,0), (2,1)],
        [(0,1), (1,1), (1,2), (2,2)],
    ],
}


class NuruominoState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NuruominoState.state_id
        NuruominoState.state_id += 1

    def __lt__(self, other):
        """ Este método é utilizado em caso de empate na gestão da lista
        de abertos nas procuras informadas. """
        return self.id < other.id

class Board:
    """Representação interna de um tabuleiro do Puzzle Nuruomino."""

    def __init__(self, board):
        self.board = board
        self.n = len(board)

    def get_value(self, row, col):
        """Devolve o valor da célula na posição (row, col)"""
        return self.board[row][col]
    
    def print_instance(self):
        """Imprime a grelha no formato esperado (com tabs)"""
        for row in self.board:
            print('\t'.join(map(str, row)))

    def adjacent_regions(self, region:int) -> list:
        """Devolve uma lista das regiões que fazem fronteira com a região enviada no argumento."""
        adjacents = set()
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if self.board[i][j] == str(region):  # célula da região procurada
                    for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # cima, baixo, esquerda, direita
                        ni, nj = i + di, j + dj
                        if 0 <= ni < len(self.board) and 0 <= nj < len(self.board[0]):
                            val = self.board[ni][nj]
                            if val != str(region):
                                adjacents.add(int(val))  # converter para inteiro
        return list(adjacents)
    
    def adjacent_positions(self, row:int, col:int) -> list:
        """Devolve as posições adjacentes à região, em todas as direções, incluindo diagonais."""
        positions = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue  # ignora a própria célula
                ni, nj = row + di, col + dj
                if 0 <= ni < len(self.board) and 0 <= nj < len(self.board[0]):
                    positions.append((ni, nj))
        return positions

    def adjacent_values(self, row:int, col:int) -> list:
        """Devolve os valores das celulas adjacentes à região, em todas as direções, incluindo diagonais."""
        values = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue  # ignora a própria célula
                ni, nj = row + di, col + dj
                if 0 <= ni < len(self.board) and 0 <= nj < len(self.board[0]):
                    values.append(self.board[ni][nj])
        return values
    
    
    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 pipe.py < test-01.txt

            > from sys import stdin
            > line = stdin.readline().split()
        """
        from sys import stdin
        board = []
        for line in stdin:
            if line.strip() == "":
                continue  # ignora linhas vazias
            board.append(line.strip().split())  # divide por espaço ou tab
        return Board(board)   

    # TODO: outros metodos da classe Board
    def get_region_positions(self, region_id):
        """Devolve uma lista com as posições (i, j) da região dada."""
        positions = []
        for i in range(self.n):
            for j in range(self.n):
                if self.board[i][j] == str(region_id):
                    positions.append((i, j))
        return positions

    

class Nuruomino(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.board = board
        self.regions = self._extract_regions()
        initial_state = NuruominoState(board)
        super().__init__(initial_state)
    
    def _extract_regions(self):
        """Identifica os números das regiões únicas no tabuleiro."""
        unique_regions = set()
        for row in self.board.board:
            for cell in row:
                if cell.isdigit():
                    unique_regions.add(int(cell))
        return sorted(unique_regions)
    
    def is_valid_placement(self, region_id, shape):
        """Verifica se a forma pode ser colocada na região indicada."""
        region_cells = self.board.get_region_positions(region_id)

        for origin in region_cells:
            # Aplica o shape com esta célula como (0,0)
            shape_abs = [(origin[0] + dx, origin[1] + dy) for dx, dy in shape]

            # Verifica se todas as células estão dentro da região
            if all(pos in region_cells for pos in shape_abs):
                return True

        return False

    def actions(self, state: NuruominoState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        actions = []

        for region_id in self.regions:
            region_cells = self.board.get_region_positions(region_id)

            # Se a região já estiver preenchida com uma peça, ignora
            if not all(self.board.get_value(i, j).isdigit() for (i, j) in region_cells):
                continue

            for piece_letter, shapes in TETRAMINOS.items():
                for shape in shapes:
                    if self.is_valid_placement(region_id, shape):
                        actions.append((region_id, piece_letter, shape))

        return actions

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

if __name__ == "__main__":
    board = Board.parse_instance()
    
    print("Grelha lida do input:")
    board.print_instance()

    print("\nRegiões adjacentes à região 3:")
    print(board.adjacent_regions(3))

    row, col = 2, 3
    print(f"\nPosições adjacentes à célula ({row}, {col}):")
    print(board.adjacent_positions(row, col))

    print(f"\nValores adjacentes à célula ({row}, {col}):")
    print(board.adjacent_values(row, col))

    # Criar o problema e estado inicial
    problem = Nuruomino(board)
    initial_state = NuruominoState(board)

    print("\nAções válidas no estado inicial:")
    for action in problem.actions(initial_state):
        region_id, piece, shape = action
        print(f"Região {region_id} → peça {piece} com forma {shape}")