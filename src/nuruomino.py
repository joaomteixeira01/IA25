# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 34:
# 97226 João Teixeira 
# 110094 Francisco Fialho

from search import Node, Problem, astar_search, InstrumentedProblem

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
        [(0,0), (0,1), (1,1), (1,2)],
        [(0,1), (1,1), (1,0), (2,0)],
    ],
}

# Pontos críticos que formam cantos por peça e rotação
CORNER_OFFSETS = {
    'L': [
        [(1, 1)],  # [(0,0), (1,0), (2,0), (2,1)]
        [(1, 0)],  # [(0,0), (0,1), (0,2), (1,0)]
        [(1, 0)],  # [(0,0), (0,1), (1,1), (2,1)]
        [(0, 0)],  # [(0,2), (1,0), (1,1), (1,2)]
        [(1, 1)],  # [(0,0), (0,1), (1,0), (2,0)]
        [(1, 2)],  # [(0,0), (0,1), (0,2), (1,2)]
        [(2, 0)],  # [(0,1), (1,1), (2,0), (2,1)]
        [(1, 0)],  # [(0,0), (1,0), (1,1), (1,2)]
    ],
    'T': [
        [(1, 0), (1, 2)],  # [(0,0), (0,1), (0,2), (1,1)]
        [(0, 0), (2, 0)],  # [(0,1), (1,0), (1,1), (2,1)]
        [(0, 0), (0, 2)],  # [(1,0), (1,1), (1,2), (0,1)]
        [(0, 1), (2, 1)],  # [(0,0), (1,0), (1,1), (2,0)]
    ],
    'S': [
        [(0, 0), (1, 2)],  # [(0,1), (0,2), (1,0), (1,1)]
        [(0, 1), (2, 0)],  # [(0,0), (1,0), (1,1), (2,1)]
        [(1, 0), (0, 2)],  # [(0,0), (0,1), (1,1), (1,2)]
        [(0, 0), (2, 1)],  # [(0,1), (1,1), (1,0), (2,0)]
    ]
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
    
    def is_valid_placement_on_board(self, board: Board, region_id, shape):
        """Verifica se a forma pode ser colocada na região indicada, num tabuleiro arbitrário."""
        region_cells = board.get_region_positions(region_id)

        for origin in region_cells:
            shape_abs = [(origin[0] + dx, origin[1] + dy) for dx, dy in shape]
            if all(pos in region_cells and board.get_value(pos[0], pos[1]) != 'X' for pos in shape_abs):
                return True
        return False


    def actions(self, state: NuruominoState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        actions = []
        board = state.board

        # Priorizar regiões com exatamente 4 células
        regions_sorted = sorted(self.regions, key=lambda r: len(board.get_region_positions(r)) != 4)

        for region_id in regions_sorted:
            region_cells = board.get_region_positions(region_id)

            # Se a região já estiver preenchida com uma peça, ignora
            if not all(board.get_value(i, j).isdigit() for (i, j) in region_cells):
                continue

            for piece_letter, shapes in TETRAMINOS.items():
                for shape in shapes:
                    if self.is_valid_placement_on_board(board, region_id, shape):
                        actions.append((region_id, piece_letter, shape, shapes.index(shape)))

        return actions


    def result(self, state: NuruominoState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""

        import copy
        region_id, piece_letter, shape, index = action

        new_board_data = copy.deepcopy(state.board.board)
        region_cells = state.board.get_region_positions(region_id)

        for origin in region_cells:
            shape_abs = [(origin[0] + dx, origin[1] + dy) for dx, dy in shape]

            if all(pos in region_cells for pos in shape_abs):
                # Verificar se alguma célula da peça iria cobrir um 'X'
                if any(new_board_data[i][j] == 'X' for (i, j) in shape_abs):
                    continue  # Ignora esta tentativa

                # Aplicar a peça
                for i, j in shape_abs:
                    new_board_data[i][j] = piece_letter

                # Marcar os cantos, se houver
                if piece_letter in CORNER_OFFSETS:
                    corner_offsets = CORNER_OFFSETS[piece_letter][index]
                    for corner_offset in corner_offsets:
                        ci, cj = origin[0] + corner_offset[0], origin[1] + corner_offset[1]
                        if 0 <= ci < len(new_board_data) and 0 <= cj < len(new_board_data[0]):
                            if new_board_data[ci][cj].isdigit():
                                new_board_data[ci][cj] = 'X'

                # Criar novo tabuleiro e novo estado
                new_board = Board(new_board_data)
                return NuruominoState(new_board)

        raise ValueError(f"Ação inválida: peça {piece_letter} não encaixa na região {region_id}")

    # Game rules 
    def _all_filled_with_pieces(self, board):
        for row in board.board:
            for cell in row:
                if cell.isdigit():
                    return False
        return True

    def _has_no_2x2_blocks(self, board):
        for i in range(board.n - 1):
            for j in range(board.n - 1):
                block = {
                    board.get_value(i, j),
                    board.get_value(i+1, j),
                    board.get_value(i, j+1),
                    board.get_value(i+1, j+1)
                }
                if len(block) == 1 and block.pop() in "LITS":
                    return False
        return True

    def _is_connected(self, board):
        from collections import deque

        visited = set()
        queue = deque()

        for i in range(board.n):
            for j in range(board.n):
                if board.get_value(i, j) in "LITS":
                    queue.append((i, j))
                    visited.add((i, j))
                    break
            if queue:
                break

        while queue:
            i, j = queue.popleft()
            for ni, nj in board.adjacent_positions(i, j):
                if board.get_value(ni, nj) in "LITS" and (ni, nj) not in visited:
                    visited.add((ni, nj))
                    queue.append((ni, nj))

        total_filled = sum(
            1 for i in range(board.n)
            for j in range(board.n)
            if board.get_value(i, j) in "LITS"
        )

        return len(visited) == total_filled

    def _no_same_piece_adjacent(self, board):
        for i in range(board.n):
            for j in range(board.n):
                current = board.get_value(i, j)
                if current not in "LITS":
                    continue
                for (ni, nj) in [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]:
                    if 0 <= ni < board.n and 0 <= nj < board.n:
                        neighbor = board.get_value(ni, nj)
                        if neighbor == current:
                            # Verifica se pertencem a regiões diferentes
                            if board.board[i][j] != board.board[ni][nj]:
                                return False
        return True

        
    def goal_test(self, state: NuruominoState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
        board = state.board
        return (self._all_filled_with_pieces(board) and 
                self._has_no_2x2_blocks(board) and 
                self._is_connected(board) and 
                self._no_same_piece_adjacent(board))

    '''def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        board = node.state.board
        score = 0

        # Penalizar regiões cujo tamanho não é múltiplo de 4
        for region_id in self.regions:
            region_cells = board.get_region_positions(region_id)
            # Região ainda não preenchida
            if any(board.get_value(i, j).isdigit() for (i, j) in region_cells):
                if len(region_cells) % 4 != 0:
                    score += 50  # Penalização alta: impossível de resolver

        # Penalizar 'X' expostos (cantos perigosos)
        for i in range(board.n):
            for j in range(board.n):
                if board.get_value(i, j) == 'X':
                    # Verifica se ainda tem vizinhos preenchíveis
                    for (ni, nj) in board.adjacent_positions(i, j):
                        if board.get_value(ni, nj).isdigit():
                            score += 10  # Penalização por X vulnerável
                            break

        # Penalizar buracos: células livres com poucos vizinhos livres
        for i in range(board.n):
            for j in range(board.n):
                if board.get_value(i, j).isdigit():
                    free_adj = sum(
                        1 for (ni, nj) in board.adjacent_positions(i, j)
                        if board.get_value(ni, nj).isdigit()
                    )
                    score += (4 - free_adj) ** 2  # Penalização quadrática

        return score'''
    
    # def h(self, node: Node):
    #     board = node.state.board
    #     score = 0

    #     # 1. Penaliza regiões não preenchidas (prioridade máxima)
    #     unfinished_regions = sum(
    #         1 for region_id in self.regions
    #         if any(board.get_value(i, j).isdigit() for (i, j) in board.get_region_positions(region_id))
    #     )
    #     score += unfinished_regions * 20  # Peso alto

    #     # 2. Penaliza desconexão (prioridade alta)
    #     if not self._is_connected(board):
    #         score += 50

    #     # 3. Penaliza blocos 2x2 (prioridade alta)
    #     if not self._has_no_2x2_blocks(board):
    #         score += 40

    #     # 4. Penaliza peças iguais adjacentes (prioridade média)
    #     if not self._no_same_piece_adjacent(board):
    #         score += 30

    #     # 5. Penaliza cantos 'X' com vizinhos não preenchidos (prioridade baixa)
    #     for i in range(board.n):
    #         for j in range(board.n):
    #             if board.get_value(i, j) == 'X':
    #                 adj_digits = sum(
    #                     1 for (ni, nj) in board.adjacent_positions(i, j)
    #                     if board.get_value(ni, nj).isdigit()
    #                 )
    #                 if adj_digits > 0:
    #                     score += 5 * adj_digits

    #     return score


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

    problem = Nuruomino(board)
    initial_state = NuruominoState(board)

    print("\nAções válidas no estado inicial:")
    for action in problem.actions(initial_state):
        region_id, piece, shape, index = action
        print(f"Região {region_id} → peça {piece} com forma {shape}")

    print("\nTestar result(): aplicar primeira ação possível")
    actions = problem.actions(initial_state)
    if actions:
        action = actions[0]
        print("Ação aplicada:", action)
        new_state = problem.result(initial_state, action)
        print("Novo estado do tabuleiro:")
        new_state.board.print_instance()
    else:
        print("Nenhuma ação disponível.")
    
    # instrumented = InstrumentedProblem(problem)
    # goal_node = astar_search(instrumented)
    # print(f"Nós gerados: {instrumented.states}")
    # print(f"Nós expandidos: {instrumented.succs}")
    # if goal_node:
    #     print("\nSolução encontrada:")
    #     goal_node.state.board.print_instance()
    # else:
    #     print("Nenhuma solução encontrada.")