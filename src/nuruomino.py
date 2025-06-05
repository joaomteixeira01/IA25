# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 34:
# 97226 João Teixeira 
# 110094 Francisco Fialho

from search import Node, Problem, astar_search, InstrumentedProblem, breadth_first_graph_search, depth_first_graph_search, depth_first_tree_search
import copy
import time
import tracemalloc

TETRAMINOS = {
    'L': [
        [(0, 0), (-1, 0), (-1, 1), (-1, 2)],
        [(0, 0), (0, -1), (1, -1), (2, -1)],
        [(0, 0), (0, -1), (-1, -1), (-2, -1)],
        [(0, 0), (-1, 0), (-1, -1), (-1, -2)],
        [(0, 0), (0, 1), (-1, 1), (-2, 1)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (1, -1), (1, -2)],
    ],
    'I': [
        [(0, 0), (1, 0), (2, 0), (3, 0)],
        [(0, 0), (0, 1), (0, 2), (0, 3)],
    ],
    'T': [
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(0, 0), (1, -1), (1, 0), (2, 0)],
        [(0, 0), (1, 0), (1, 1), (1, -1)],
        [(0, 0), (1, 0), (1, 1), (2, 0)],
    ],
    'S': [
        [(0, 0), (0, -1), (-1, -1), (-1, -2)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(0, 0), (0, 1), (-1, 1), (-1, 2)],
        [(0, 0), (1, -1), (1, 0), (2, -1)],
    ]
}
  
CORNER_OFFSETS = {
    'L': [
        [(0,1)],    
        [(1,0)],
        [(-1,0)],
        [(0,-1)],
        [(-1,0)],
        [(1,0)],
        [(0,1)],
        [(0,-1)],
    ],
    'I': [
        [],                # I vertical: [(0,0), (1,0), (2,0), (3,0)]
        [],                # I horizontal: [(0,0), (0,1), (0,2), (0,3)]
    ],
    'T': [
        [(1,0), (1,2)],    
        [(0,-1), (2,-1)],
        [(0,-1), (0,1)],
        [(0,1), (2,1)],

    ],
    'S': [
        [(-2,0), (1,0)],     
        [(0,1), (2,0)],
        [(0,2), (-1,0)],
        [(0,-1), (2,0)]
    ],
}


LAST_STATES = []


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
    
    def __eq__(self, other):
        if not isinstance(other, NuruominoState):
            return False
        return all(self.board.get_value(i, j) == other.board.get_value(i, j)
                for i in range(self.board.n) for j in range(self.board.n))

    def __hash__(self):
        return hash(tuple(tuple(row) for row in self.board.board))

class Board:
    """Representação interna de um tabuleiro do Puzzle Nuruomino."""

    def __init__(self, board):
        self.board = board
        self.n = len(board)
        self.regiao_original = copy.deepcopy(board)  # guarda a grelha original para referência

    def get_value(self, row, col):
        """Devolve o valor da célula na posição (row, col)"""
        return self.board[row][col]
    
    def print_instance(self):
        """Imprime a grelha no formato esperado (com tabs), sem \n final"""
        output = '\n'.join('\t'.join(map(str, row)) for row in self.board)
        print(output, end='')

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
                            if val != str(region) and val.isdigit():  # só adiciona se for número
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
        positions = []
        for i in range(self.n):
            for j in range(self.n):
                if self.regiao_original[i][j] == str(region_id):
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


    # def actions2(self, state: NuruominoState):
    #     """Retorna uma lista de ações que podem ser executadas a
    #     partir do estado passado como argumento."""
    #     actions = []
    #     board = state.board

    #     regions_sorted = sorted(self.regions, key=lambda r: len(board.get_region_positions(r)) != 4)

    #     for region_id in regions_sorted:
    #         region_cells = board.get_region_positions(region_id)

    #         # Check if region already has some LITS letters placed
    #         has_letters = any(
    #             board.get_value(i, j) in "LITS"
    #             for (i, j) in region_cells
    #         )

    #         # If region already has some letters but not exactly 4, skip it
    #         # This prevents adding multiple tetrominoes to a single region
    #         piece_positions = [
    #             (i, j) for (i, j) in region_cells 
    #             if board.get_value(i, j) in "LITS"
    #         ]
            
    #         # Skip if region already has some letters but not a complete tetromino
    #         if has_letters and len(piece_positions) != 4:
    #             continue

    #         for piece_letter, shapes in TETRAMINOS.items():
    #             for index, shape in enumerate(shapes):
    #                 for origin in region_cells:
    #                     shape_abs = [(origin[0] + dx, origin[1] + dy) for dx, dy in shape]

    #                     # DEBUG: Mostra cada tentativa de encaixe
    #                     print(f"Tentando regiao {region_id} com peça {piece_letter} (forma {index}) na origem {origin}: {shape_abs}")

    #                     # Verifica se a peça encaixa na própria região e em células ainda disponíveis
    #                     if all(
    #                         pos in region_cells and
    #                         board.get_value(pos[0], pos[1]) not in "LITSX"  # pode ser número ou '?'
    #                         for pos in shape_abs
    #                     ):
    #                         '''if self._would_create_2x2_block(shape_abs, board):
    #                             continue'''
    #                         if self._would_touch_equal_piece(shape_abs, piece_letter, board):
    #                             continue
    #                         actions.append((region_id, piece_letter, shape, index, shape_abs))
    #                     # else:
    #                     #     # Para debug
    #                     #     falhas = []
    #                     #     for pos in shape_abs:
    #                     #         if pos not in region_cells:
    #                     #             falhas.append(f"Posicao {pos} nao esta na regiao {region_id}")
    #                     #         elif board.get_value(pos[0], pos[1]) in "LITSX":
    #                     #             falhas.append(f"Posicao {pos} ja tem valor {board.get_value(pos[0], pos[1])}")
                            
    #                     #     if falhas:
    #                     #         print(f"Rejeicao: {', '.join(falhas)}")

    #     return actions
    
    def actions(self, state: NuruominoState):
        board = state.board
        best_region = None
        best_actions = None
        min_actions = float('inf')

        for region_id in self.regions:
            region_cells = board.get_region_positions(region_id)
            piece_positions = [(i, j) for (i, j) in region_cells if board.get_value(i, j) in "LITS"]
            if len(piece_positions) == 4 or (piece_positions and len(piece_positions) != 4):
                continue

            actions_for_region = []

            for piece_letter, shapes in TETRAMINOS.items():
                for index, shape in enumerate(shapes):
                    for origin in region_cells:
                        shape_abs = [(origin[0] + dx, origin[1] + dy) for dx, dy in shape]
                        if all(
                            pos in region_cells and
                            board.get_value(pos[0], pos[1]) not in "LITSX"
                            for pos in shape_abs
                        ):
                            if not self._would_touch_equal_piece(shape_abs, piece_letter, board) and not self._would_create_2x2_block(shape_abs, board):
                                # --- Forward checking só para regiões vizinhas ---
                                next_state = self.result(state, (region_id, piece_letter, shape, index, shape_abs))
                                blocked = False
                                for neighbor_region in board.adjacent_regions(region_id):
                                    neighbor_cells = board.get_region_positions(neighbor_region)
                                    neighbor_piece_positions = [(i, j) for (i, j) in neighbor_cells if next_state.board.get_value(i, j) in "LITS"]
                                    if len(neighbor_piece_positions) == 4 or (neighbor_piece_positions and len(neighbor_piece_positions) != 4):
                                        continue
                                    found = False
                                    for n_piece_letter, n_shapes in TETRAMINOS.items():
                                        for n_index, n_shape in enumerate(n_shapes):
                                            for n_origin in neighbor_cells:
                                                n_shape_abs = [(n_origin[0] + dx, n_origin[1] + dy) for dx, dy in n_shape]
                                                if all(
                                                    pos in neighbor_cells and
                                                    next_state.board.get_value(pos[0], pos[1]) not in "LITSX"
                                                    for pos in n_shape_abs
                                                ):
                                                    if not self._would_touch_equal_piece(n_shape_abs, n_piece_letter, next_state.board) and not self._would_create_2x2_block(n_shape_abs, next_state.board):
                                                        found = True
                                                        break
                                            if found:
                                                break
                                        if found:
                                            break
                                    if not found:
                                        blocked = True
                                        break
                                if not blocked:
                                    actions_for_region.append((region_id, piece_letter, shape, index, shape_abs))
            if 0 < len(actions_for_region) <= 2:
                #print(f"[DEBUG] Região {region_id} tem {len(actions_for_region)} ações possíveis (restrita, devolve já)")
                return actions_for_region
            if 0 < len(actions_for_region) < min_actions:
                min_actions = len(actions_for_region)
                best_region = region_id
                best_actions = actions_for_region

        if best_actions:
            #print(f"[DEBUG] Focando na região {best_region} com {min_actions} ações possíveis")
            return best_actions
        return []


    # def actions(self, state: NuruominoState):
    #     board = state.board
    #     best_region = None
    #     best_actions = []
    #     min_actions = float('inf')

    #     for region_id in self.regions:
    #         region_cells = board.get_region_positions(region_id)
    #         letras = [board.get_value(i, j) for (i, j) in region_cells if board.get_value(i, j) in "LITS"]

    #         if len(letras) == 4 or (letras and len(letras) != 4):
    #             continue

    #         actions_for_region = []

    #         # Só gerar ações a partir de células "ativas" (com adjacente preenchido)
    #         candidate_origins = [cell for cell in region_cells if any(
    #             board.get_value(ni, nj) in "LITS" for (ni, nj) in board.adjacent_positions(cell[0], cell[1])
    #         )]
    #         if not candidate_origins:
    #             candidate_origins = region_cells  # fallback

    #         for piece_letter, shapes in TETRAMINOS.items():
    #             for index, shape in enumerate(shapes):
    #                 for origin in candidate_origins:
    #                     shape_abs = [(origin[0] + dx, origin[1] + dy) for dx, dy in shape]
    #                     if not all(pos in region_cells and board.get_value(pos[0], pos[1]) not in "LITSX" for pos in shape_abs):
    #                         continue
    #                     if self._would_touch_equal_piece(shape_abs, piece_letter, board):
    #                         continue
    #                     if self._would_create_2x2_block(shape_abs, board):
    #                         continue

    #                     # --- Forward checking só para regiões vizinhas ---
    #                     next_state = self.result(state, (region_id, piece_letter, shape, index, shape_abs))
    #                     blocked = False
    #                     for neighbor_region in board.adjacent_regions(region_id):
    #                         neighbor_cells = board.get_region_positions(neighbor_region)
    #                         neighbor_piece_positions = [(i, j) for (i, j) in neighbor_cells if next_state.board.get_value(i, j) in "LITS"]
    #                         if len(neighbor_piece_positions) == 4 or (neighbor_piece_positions and len(neighbor_piece_positions) != 4):
    #                             continue
    #                         found = False
    #                         for n_piece_letter, n_shapes in TETRAMINOS.items():
    #                             for n_index, n_shape in enumerate(n_shapes):
    #                                 for n_origin in neighbor_cells:
    #                                     n_shape_abs = [(n_origin[0] + dx, n_origin[1] + dy) for dx, dy in n_shape]
    #                                     if all(
    #                                         pos in neighbor_cells and
    #                                         next_state.board.get_value(pos[0], pos[1]) not in "LITSX"
    #                                         for pos in n_shape_abs
    #                                     ):
    #                                         if not self._would_touch_equal_piece(n_shape_abs, n_piece_letter, next_state.board) and not self._would_create_2x2_block(n_shape_abs, next_state.board):
    #                                             found = True
    #                                             break
    #                                 if found:
    #                                     break
    #                             if found:
    #                                 break
    #                         if not found:
    #                             blocked = True
    #                             break
    #                     if not blocked:
    #                         actions_for_region.append((region_id, piece_letter, shape, index, shape_abs))

    #         if 0 < len(actions_for_region) <= 2:
    #             return actions_for_region

    #         if 0 < len(actions_for_region) < min_actions:
    #             min_actions = len(actions_for_region)
    #             best_region = region_id
    #             best_actions = actions_for_region

    #     return best_actions


    def result(self, state: NuruominoState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        region_id, piece_letter, shape, index, shape_abs = action

        # Copia só as linhas afetadas
        old_board = state.board.board
        new_board_data = [row[:] for row in old_board]  # shallow copy das linhas

        # Aplica a peça diretamente nas posições definidas
        for i, j in shape_abs:
            new_board_data[i][j] = piece_letter

        if piece_letter in CORNER_OFFSETS:
            corner_offsets = CORNER_OFFSETS[piece_letter][index]
            origin_i = shape_abs[0][0] - shape[0][0]
            origin_j = shape_abs[0][1] - shape[0][1]
            for offset_i, offset_j in corner_offsets:
                ci = origin_i + offset_i
                cj = origin_j + offset_j
                if 0 <= ci < len(new_board_data) and 0 <= cj < len(new_board_data[0]):
                    if new_board_data[ci][cj].isdigit():
                        new_board_data[ci][cj] = 'X'

        new_board = Board(new_board_data)
        new_board.regiao_original = state.board.regiao_original
        LAST_STATES.append(new_board)
        if len(LAST_STATES) > 10:
            LAST_STATES.pop(0)
        return NuruominoState(new_board)

    # Game rules 
    def _all_regions_have_one_piece(self, board):
        """Verifica se cada região tem exatamente uma peça tetraminó (4 células com letra), ignorando 'X'."""
        from collections import deque
    
        def orthogonal_neighbors(i, j):
            return [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
    
        for region_id in self.regions:
            region_cells = board.get_region_positions(region_id)
    
            # Filtrar apenas as posições com letras LITS
            piece_positions = [
                (i, j)
                for (i, j) in region_cells
                if board.get_value(i, j) in "LITS"
            ]
    
            if len(piece_positions) == 0:
                return False  # região sem peça
    
            if len(piece_positions) != 4:
                return False
    
            # Verificar se estão conectadas entre si ortogonalmente
            visited = set()
            queue = deque([piece_positions[0]])
            visited.add(piece_positions[0])
    
            while queue:
                i, j = queue.popleft()
                for ni, nj in orthogonal_neighbors(i, j):
                    if (ni, nj) in piece_positions and (ni, nj) not in visited:
                        visited.add((ni, nj))
                        queue.append((ni, nj))
    
            if len(visited) != 4:
                return False
    
        return True

    def _has_no_2x2_blocks(self, board):
        '''
            Verifica se existem blocos 2x2 formados por celulas com peças, ou seja, com letras pertencentes a LITS
        '''
        for i in range(board.n - 1):
            for j in range(board.n - 1):
                values = [
                    board.get_value(i, j),
                    board.get_value(i+1, j),
                    board.get_value(i, j+1),
                    board.get_value(i+1, j+1)
                ]
                if all(v in "LITS" for v in values):
                    #print(f"[DEBUG] Bloco 2x2 inválido encontrado em ({i},{j}) com letras: {values}")
                    return False
        return True


    def _is_connected(self, board, debug=False):
        '''
            Verifica se as peças estao conectadas ortagonalmente, isto é, tem de formar uma "ilha"
        '''
        from collections import deque

        def orthogonal_neighbors(i, j):
            return [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]

        visited = set()
        queue = deque()
        piece_connections = set()

        # Encontrar a primeira célula preenchida com peça
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
            current_piece = board.get_value(i, j)

            for ni, nj in orthogonal_neighbors(i, j):
                if 0 <= ni < board.n and 0 <= nj < board.n:
                    neighbor_val = board.get_value(ni, nj)

                    if neighbor_val in "LITS":
                        # Adiciona a ligação entre peças diferentes
                        if neighbor_val != current_piece:
                            connection = tuple(sorted([current_piece, neighbor_val]))
                            piece_connections.add(connection)

                        if (ni, nj) not in visited:
                            visited.add((ni, nj))
                            queue.append((ni, nj))

        total_filled = sum(
            1 for i in range(board.n)
            for j in range(board.n)
            if board.get_value(i, j) in "LITS"
        )

        # if debug:
        #     print(f"\n[DEBUG] Peças conectadas ortogonalmente:")
        #     for a, b in sorted(piece_connections):
        #         print(f"  -> {a} conectado a {b}")
        #     print(f"Total de peças conectadas: {len(visited)} / {total_filled}")

        return len(visited) == total_filled
        


    def _no_same_piece_adjacent(self, board):
        '''
            Verifica se nao existem pelas iguais conectadas
        '''
        for i in range(board.n):
            for j in range(board.n):
                current = board.get_value(i, j)
                if current not in "LITS":
                    continue
                for (ni, nj) in [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]:
                    if 0 <= ni < board.n and 0 <= nj < board.n:
                        neighbor = board.get_value(ni, nj)
                        if neighbor == current:
                            # Só é problema se estiverem em regiões diferentes
                            reg1 = board.regiao_original[i][j]
                            reg2 = board.regiao_original[ni][nj]
                            if reg1 != reg2:
                                return False
        return True
    
    def _would_create_2x2_block(self, positions, board):
        for (i, j) in positions:
            for di in [0, -1]:
                for dj in [0, -1]:
                    square = [
                        (i + di, j + dj),
                        (i + di + 1, j + dj),
                        (i + di, j + dj + 1),
                        (i + di + 1, j + dj + 1)
                    ]
                    if all(
                        0 <= x < board.n and 0 <= y < board.n and
                        ((x, y) in positions or board.get_value(x, y) in "LITS")
                        for (x, y) in square
                    ):
                        return True  # Um bloco 2x2 seria formado
        return False  # Nenhum bloco 2x2 seria formado

    def _would_touch_equal_piece(self, positions, piece_letter, board: Board):
        for (i, j) in positions:
            for (ni, nj) in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
                if 0 <= ni < board.n and 0 <= nj < board.n:
                    if (ni, nj) not in positions:
                        val = board.get_value(ni, nj)
                        if val == piece_letter:
                            # Só bloqueia se estiverem em regiões diferentes
                            reg1 = board.regiao_original[i][j]
                            reg2 = board.regiao_original[ni][nj]
                            if reg1 != reg2:
                                return True
        return False
        
    def goal_test(self, state: NuruominoState):
        board = state.board
        #print("\n[DEBUG] Estado atual do tabuleiro:")
        #board.print_instance()
        
        all_pieces = self._all_regions_have_one_piece(board)
        no_2x2 = self._has_no_2x2_blocks(board) 
        connected = self._is_connected(board, debug=True)  # Ative o debug
        no_adj = self._no_same_piece_adjacent(board)
        
        # print(f"\nGoal Test Results:")
        # print(f"- Todas regiões têm peça: {all_pieces}")
        # print(f"- Sem blocos 2x2: {no_2x2}")
        # print(f"- Peças conectadas: {connected}")
        # print(f"- Sem peças iguais adjacentes: {no_adj}")
        
        return all([all_pieces, no_2x2, connected, no_adj])

    '''def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        board = node.state.board
        score = 0
        return score'''

    def h(self, node: Node):
        board = node.state.board
        h_n = 0
        

        # Conta o número de regiões ainda não preenchidas
        incomplete_regions = 0
        for region_id in self.regions:
            region_cells = board.get_region_positions(region_id)
            if any(board.get_value(i, j).isdigit() for (i, j) in region_cells):
                incomplete_regions += 1

        h_n += incomplete_regions * 10  # encoraja a resolver regiões cedo

        # Penaliza células vazias com menos vizinhos livres (pontos de bloqueio)
        for i in range(board.n):
            for j in range(board.n):
                if board.get_value(i, j).isdigit():
                    free_adj = sum(
                        1 for (ni, nj) in board.adjacent_positions(i, j)
                        if board.get_value(ni, nj).isdigit()
                    )
                    if free_adj <= 1:
                        h_n += 10  # beco sem saída

        # Verifica só se o estado está quase completo
        filled = sum(
            1 for i in range(board.n) for j in range(board.n)
            if board.get_value(i, j) in "LITS"
        )
        total = board.n * board.n
        ratio = filled / total

        if ratio >= 0.75:
            # Corta se houver blocos 2x2
            if not self._has_no_2x2_blocks(board):
                return float('inf')

            # Corta se peças iguais estiverem ortogonalmente adjacentes
            if not self._no_same_piece_adjacent(board):
                return float('inf')

            # Corta se peças não estiverem ligadas (nurilkabe inválido)
            if not self._is_connected(board):
                return float('inf')

        return h_n
    

# def preenche_regioes_de_4_celulas(board: Board, problem: Nuruomino) -> Board:
#     new_board = copy.deepcopy(board)

#     for region_id in problem.regions:
#         region_cells = new_board.get_region_positions(region_id)
#         if len(region_cells) == 4 and all(new_board.get_value(i, j).isdigit() or new_board.get_value(i, j) == '?' for (i, j) in region_cells):
#             temp_state = NuruominoState(new_board)
#             actions = problem.actions(temp_state)
            
#             # Tentar todas as ações para esta região antes de passar para a próxima
#             for action in actions:
#                 if action[0] == region_id:
#                     print(f"Tentando ação na região {region_id}: {action}")
#                     temp_state_result = problem.result(temp_state, action)
#                     # Verificar se a ação preencheu a região corretamente
#                     if all(temp_state_result.board.get_value(i, j) != '?' for (i, j) in region_cells):
#                         new_board = temp_state_result.board
#                         print(f"Região {region_id} preenchida com sucesso!")
#                         break  # Sai do loop assim que uma ação válida for encontrada
#             else:
#                 print(f"Nenhuma ação válida encontrada para a região {region_id}")

#     return new_board


def marcar_celulas_comuns(board: Board, problem: Nuruomino):
    """Marca no tabuleiro as células que são comuns a todas as ações possíveis por região."""
    state = NuruominoState(board)
    acoes = problem.actions(state)
    count = 0

    from collections import defaultdict
    regioes_shapes = defaultdict(list)

    # Agrupar as formas por região
    for (reg_id, piece_letter, shape, index, shape_abs) in acoes:
        regioes_shapes[reg_id].append(set(shape_abs))

    for reg_id, lista_de_posicoes in regioes_shapes.items():
        if not lista_de_posicoes:
            continue
        
        # Interseção de todas as posições possíveis
        comuns = set.intersection(*lista_de_posicoes)

        # Marcar as posições comuns no board (ex: com '?')
        for (i, j) in comuns:
            val = board.get_value(i, j)
            if val.isdigit():  # só marcamos se estiver ainda por preencher
                board.board[i][j] = '?'
    marcar_cantos_comuns_invalidos(board)



def marcar_cantos_comuns_invalidos(board: Board):
    """Marca com 'X' todas as células que, se preenchidas, formariam blocos 2x2 com 'letras' ou '?'."""
    for i in range(board.n - 1):
        for j in range(board.n - 1):
            square = [
                (i, j), (i+1, j), (i, j+1), (i+1, j+1)
            ]

            letras_ou_certas = []
            digitos = []

            for (x, y) in square:
                val = board.get_value(x, y)
                if val in "LITS?":
                    letras_ou_certas.append((x, y))
                elif val.isdigit():
                    digitos.append((x, y))

            if len(letras_ou_certas) == 3 and len(digitos) == 1:
                (dx, dy) = digitos[0]
                board.board[dx][dy] = 'X'

def limpar_celulas_interrogacao(board: Board):
    for i in range(board.n):
        for j in range(board.n):
            if board.get_value(i, j) == '?':
                board.board[i][j] = board.regiao_original[i][j]

def limpar_X(board: Board):
    for i in range(board.n):
        for j in range(board.n):
            if board.get_value(i, j) == 'X':
                board.board[i][j] = board.regiao_original[i][j]


if __name__ == "__main__":
    board = Board.parse_instance()

    # print("Grelha lida do input:")
    # board.print_instance()

    tracemalloc.start()
    tic = time.perf_counter()

    # Guardar um copia original
    original_board_copy = copy.deepcopy(board.board)
    board.regiao_original = original_board_copy

    problem = Nuruomino(board)
    #state = NuruominoState(board)

    # print("\nNúmero de células por região:")
    # for region_id in problem.regions:
    #     positions = board.get_region_positions(region_id)
    #     print(f" - Região {region_id}: {len(positions)} células")

    # Aplicamos peças nas regiões com 4 células
    # board = preenche_regioes_de_4_celulas(board, problem)

    #Atribuir novamente a cópia original ao novo board criado
    # board.regiao_original = original_board_copy

    # print("\nTabuleiro após preencher todas as regiões de 4 células:")
    # board.print_instance()

    # print("\nAções válidas no estado atual:")
    # state_temp = NuruominoState(board)
    # for action in problem.actions(state_temp):
    #     region_id, piece, shape, index, shape_abs = action
    #     print(f"Região {region_id} -> peça {piece} com forma {shape} na posição {shape_abs}")

    # # Marcar células deduzidas por interseção de possibilidades
    # print("\nMarcar células comuns nas regiões restantes:")
    # marcar_celulas_comuns(board, problem)
    # board.print_instance()

    # # We use the ? to help us filter the actions, after that help they´re no longer needed
    # # therefore we turn them back to numbers to get the new list of actions
    # print("\nTabuleiro apos limpar celulas '?':")
    # limpar_celulas_interrogacao(board)
    # board.print_instance()

    # Criamos o novo estado inicial com as peças já aplicadas
    state = NuruominoState(board)
    problem = Nuruomino(board)  # novo problem com board atualizado
    problem.initial = state

    # print("\nAções válidas no estado atual:")
    # for action in problem.actions(state):
    #     region_id, piece, shape, index, shape_abs = action
    #     print(f"Região {region_id} -> peça {piece} com forma {shape} na posição {shape_abs}")


    # print("\nTabuleiro apos limpar celulas 'X':")
    # limpar_X(board)
    # board.print_instance()

    instrumented = InstrumentedProblem(problem)
    # goal_node = astar_search(instrumented)
    # print("A testar com *A...")

    # goal_node = breadth_first_graph_search(instrumented)
    # print("A testar com BFS...")

    goal_node = depth_first_graph_search(instrumented)
    # print("A testar com DFS...")
    
    toc = time.perf_counter()
    print(f"  Programa executado em {toc - tic:0.4f} segundos")
    print(f"  Memória usada: {tracemalloc.get_traced_memory()[1] // 1024} kB")
    print(f"  Nós gerados: {instrumented.states}")
    print(f"  Nós expandidos: {instrumented.succs}")
    if goal_node:
        #print("\nSolução encontrada:")
        limpar_X(goal_node.state.board)
        goal_node.state.board.print_instance()
    # else:
    #     print("Nenhuma solução encontrada.")


    '''print("\nÚltimos 10 estados gerados:")
    for idx, b in enumerate(LAST_STATES):
        print(f"\nEstado {idx+1}:")
        b.print_instance()'''

    '''def debug_goal_test_details(problem):
        print("\n[DEBUG] Diagnóstico do `goal_test` nos últimos 10 estados:")

        for idx, board in enumerate(LAST_STATES):
            print(f"\nEstado {idx + 1}:")
            board.print_instance()
            state = NuruominoState(board)

            all_pieces = problem._all_regions_have_one_piece(board)
            no_2x2 = problem._has_no_2x2_blocks(board)
            connected = problem._is_connected(board)
            no_adj = problem._no_same_piece_adjacent(board)

            if all([all_pieces, no_2x2, connected, no_adj]):
                print("  goal_test: TRUE — Estado objetivo atingido.")
            else:
                print("  goal_test: FALSE — Falhas detetadas:")

                # 1. Regiões incompletas
                if not all_pieces:
                    print("    - Regiões com número incorreto de letras:")
                    for region_id in problem.regions:
                        region_cells = board.get_region_positions(region_id)
                        letras = [board.get_value(i, j) for i, j in region_cells if board.get_value(i, j) in "LITS"]
                        if len(letras) != 4:
                            print(f"      Região {region_id}: {len(letras)} letras (esperado: 4)")

                # 2. Blocos 2x2 inválidos
                if not no_2x2:
                    print("    - Blocos 2x2 com letras:")
                    for i in range(board.n - 1):
                        for j in range(board.n - 1):
                            v1 = board.get_value(i, j)
                            v2 = board.get_value(i + 1, j)
                            v3 = board.get_value(i, j + 1)
                            v4 = board.get_value(i + 1, j + 1)
                            if all(v in "LITS" for v in [v1, v2, v3, v4]):
                                print(f"      → Bloco 2x2 com letras em ({i},{j}) [{v1}, {v2}, {v3}, {v4}]")

                # 3. Conectividade
                if not connected:
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
                        for ni, nj in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                            if 0 <= ni < board.n and 0 <= nj < board.n:
                                if board.get_value(ni, nj) in "LITS" and (ni, nj) not in visited:
                                    visited.add((ni, nj))
                                    queue.append((ni, nj))

                    total_letras = [
                        (i, j) for i in range(board.n)
                        for j in range(board.n)
                        if board.get_value(i, j) in "LITS"
                    ]
                    print(f"    - Peças conectadas: {len(visited)} / {len(total_letras)}")
                    print("      Células desconectadas:")
                    for cell in total_letras:
                        if cell not in visited:
                            print(f"        {cell} = {board.get_value(cell[0], cell[1])}")

                # 4. Peças iguais adjacentes
                if not no_adj:
                    print("    - Peças iguais ortogonalmente adjacentes:")
                    for i in range(board.n):
                        for j in range(board.n):
                            val = board.get_value(i, j)
                            if val not in "LITS":
                                continue
                            for ni, nj in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                                if 0 <= ni < board.n and 0 <= nj < board.n:
                                    if board.get_value(ni, nj) == val:
                                        print(f"      → ({i},{j}) e ({ni},{nj}) = '{val}' (inválido)")

    
    debug_goal_test_details(problem)'''
