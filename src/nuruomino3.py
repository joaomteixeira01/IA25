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

# Pre-compute orthogonal directions for efficiency
ORTHOGONAL_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

LAST_STATES = []


class NuruominoState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NuruominoState.state_id
        NuruominoState.state_id += 1
        self.unfinished_regions = self._get_unfinished_regions()
        # Cache for hash computation
        self._hash = None

    def __lt__(self, other):
        """ Este método é utilizado em caso de empate na gestão da lista
        de abertos nas procuras informadas. """
        return self.id < other.id
    
    def __eq__(self, other):
        if not isinstance(other, NuruominoState):
            return False
        # Quick check with hash first
        if hash(self) != hash(other):
            return False
        return all(self.board.get_value(i, j) == other.board.get_value(i, j)
                for i in range(self.board.n) for j in range(self.board.n))

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(tuple(tuple(row) for row in self.board.board))
        return self._hash
    
    def _get_unfinished_regions(self):
        """Devolve uma lista de regiões que ainda não têm todas as peças preenchidas."""
        unfinished = []
        for region_id in self.board.regions:
            # Use cached positions
            positions = self.board.get_region_positions(region_id)
            
            # Verificar se a região não tem nenhuma letra LITS
            has_piece = any(self.board.get_value(i, j) in "LITS" for (i, j) in positions)
            
            if not has_piece:
                unfinished.append(region_id)
                
        return unfinished

class Board:
    """Representação interna de um tabuleiro do Puzzle Nuruomino."""

    def __init__(self, board, preserve_original=True):
        self.board = board
        self.n = len(board)
        self._region_positions_cache = {}  # Cache para posições de regiões
        self._adjacent_regions_cache = {}  # Cache para regiões adjacentes
        
        # Só faz cópia se necessário
        if preserve_original:
            self.regiao_original = [row[:] for row in board]  # shallow copy das linhas
        else:
            self.regiao_original = board
        
        # Extract regions with optimization
        self.regions = self._extract_regions()  
        
        # Pre-compute region positions for all regions
        for region_id in self.regions:
            self.get_region_positions(region_id)
        
        #lista de ações por região
        self.region_actions_list = {}
        for region_id in self.regions:
            self.region_actions_list[region_id] = self.region_actions(region_id)
        
        # Ordenar regiões por número de ações (menos ações primeiro)
        region_action_counts = [(region_id, len(actions)) 
                               for region_id, actions in self.region_actions_list.items()]
        region_action_counts.sort(key=lambda x: x[1])
        self.regions = [region_id for region_id, _ in region_action_counts]
        

    def _extract_regions(self):
        """Identifica os números das regiões únicas no tabuleiro."""
        unique_regions = set()
        for row in self.board:
            for cell in row:
                if cell.isdigit():
                    unique_regions.add(int(cell))
        return sorted(unique_regions)


    def get_value(self, row, col):
        """Devolve o valor da célula na posição (row, col)"""
        return self.board[row][col]
    

    def print_instance(self):
        """Imprime a grelha no formato esperado (com tabs), sem \n final"""
        output = '\n'.join('\t'.join(map(str, row)) for row in self.board)
        print(output, end='')


    def adjacent_regions(self, region:int) -> list:
        """Devolve uma lista das regiões que fazem fronteira com a região enviada no argumento."""
        if region not in self._adjacent_regions_cache:
            adjacents = set()
            region_positions = self.get_region_positions(region)
            
            for i, j in region_positions:
                for di, dj in ORTHOGONAL_DIRS:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.n and 0 <= nj < self.n:
                        val = self.board[ni][nj]
                        if val != str(region) and val.isdigit():
                            adjacents.add(int(val))
            
            self._adjacent_regions_cache[region] = list(adjacents)
        return self._adjacent_regions_cache[region]
    
    def adjacent_positions(self, row:int, col:int) -> list:
        """Devolve as posições adjacentes à região, em todas as direções, incluindo diagonais."""
        positions = []
        for di, dj in ALL_DIRS:
            ni, nj = row + di, col + dj
            if 0 <= ni < self.n and 0 <= nj < self.n:
                positions.append((ni, nj))
        return positions

    def adjacent_values(self, row:int, col:int) -> list:
        """Devolve os valores das celulas adjacentes à região, em todas as direções, incluindo diagonais."""
        values = []
        for di, dj in ALL_DIRS:
            ni, nj = row + di, col + dj
            if 0 <= ni < self.n and 0 <= nj < self.n:
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
        if region_id not in self._region_positions_cache:
            positions = []
            region_str = str(region_id)
            for i in range(self.n):
                for j in range(self.n):
                    if self.regiao_original[i][j] == region_str:
                        positions.append((i, j))
            self._region_positions_cache[region_id] = positions
        return self._region_positions_cache[region_id]
    
    
    def region_actions(self, region_id):
        """Calcula todas as ações possíveis para uma região (versão do region_actions original)"""
        actions = []
        piece_letters = list(TETRAMINOS.keys())
        region_positions = self.get_region_positions(region_id)
        region_positions_set = set(region_positions)  # Convert to set for O(1) lookup
        
        for piece_letter in piece_letters:
            for index, shape in enumerate(TETRAMINOS[piece_letter]):
                for anchor_pos in region_positions:
                    anchor_i, anchor_j = anchor_pos
                    
                    shape_abs = []
                    valid_placement = True
                    
                    for rel_i, rel_j in shape:
                        abs_i = anchor_i + rel_i
                        abs_j = anchor_j + rel_j
                        
                        if 0 <= abs_i < self.n and 0 <= abs_j < self.n:
                            shape_abs.append((abs_i, abs_j))
                        else:
                            valid_placement = False
                            break
                    
                    if valid_placement and all(pos in region_positions_set for pos in shape_abs):
                        actions.append((region_id, piece_letter, shape, index, shape_abs))
        
        return actions
    

class Nuruomino(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.board = board
        self.regions = board.regions 
        initial_state = NuruominoState(board)
        super().__init__(initial_state)
    
    def actions(self, state: NuruominoState):  
        """Devolve uma lista de ações possíveis para o estado atual."""
        #se não houver regiões por preencher, devolve lista vazia
        if not state.unfinished_regions:
            return []
        
        # Escolher região com menos ações possíveis (MRV - Most Restricting Variable)
        best_region = min(state.unfinished_regions, key=lambda r: len(self.board.region_actions_list[r]))
        all_actions = self.board.region_actions_list[best_region]
        valid_actions = []
        
        for action in all_actions:
            if self.is_valid_action(action, state):
                #se for a ultima região por preencher, adiciona a ação
                if len(state.unfinished_regions) == 1:
                    new_state = self.result(state, action)
                    if not self.goal_test(new_state):
                        continue  # não adiciona ações que não levam a um estado objetivo
                else:
                    #forward checking: verifica se as regiioes adjacentes à região atual ainda têm ações válidas
                    new_state = self.result(state, action)
                    if not self._has_future_solutions(new_state, action[0]):
                        continue
                valid_actions.append(action)

        return valid_actions            
    

    def _has_future_solutions(self, state, filled_region):
        """Verifica rapidamente se regiões adjacentes ainda têm soluções viáveis"""
        adjacent_regions = state.board.adjacent_regions(filled_region)
        
        for adj_region in adjacent_regions:
            if adj_region in state.unfinished_regions:
                # Verificação rápida: pelo menos uma ação deve ser válida
                actions = self.board.region_actions_list[adj_region]
                if not any(self.is_valid_action(action, state) for action in actions):  
                    return False
        return True


    def result(self, state: NuruominoState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        region_id, piece_letter, shape, index, shape_abs = action

        # Optimized board copying - only copy what's needed
        old_board = state.board.board
        new_board_data = [row[:] for row in old_board]  # shallow copy das linhas

        # Aplica a peça diretamente nas posições definidas
        for i, j in shape_abs:
            new_board_data[i][j] = piece_letter

        # Shallow copy optimization
        new_board = copy.copy(state.board)
        new_board.board = new_board_data
        new_board.regiao_original = state.board.regiao_original  # Reutiliza a referência

        # Optimized action list copying
        new_board.region_actions_list = {
            reg_id: actions[:] for reg_id, actions in state.board.region_actions_list.items()
        }

        # Remove a região preenchida da lista
        if region_id in new_board.region_actions_list:
            del new_board.region_actions_list[region_id]

        # Atualiza as ações das regiões adjacentes removendo as inválidas
        self._update_adjacent_actions(new_board, region_id, piece_letter, shape_abs)
        
        # Reordena as regiões por número de ações (menos ações primeiro)
        remaining_regions = [r for r in new_board.regions if r in new_board.region_actions_list]
        remaining_regions.sort(key=lambda r: len(new_board.region_actions_list[r]))
        new_board.regions = remaining_regions
        
        # Optimized state tracking
        if len(LAST_STATES) >= 10:
            LAST_STATES.pop(0)
        LAST_STATES.append(new_board)

        new_state = NuruominoState(new_board)
        return new_state



    def _update_adjacent_actions(self, board, filled_region, piece_letter, placed_positions):
        adjacent_regions = board.adjacent_regions(filled_region)

        # Conjunto de posições afetadas (para ver sobreposição com ações existentes)
        placed_set = set(placed_positions)

        for adj_region in adjacent_regions:
            if adj_region not in board.region_actions_list:
                continue

            old_actions = board.region_actions_list[adj_region]
            new_actions = []

            for action in old_actions:
                _, act_piece, _, _, act_positions = action

                # Se esta ação toca em alguma célula da peça colocada:
                if any(pos in placed_set for pos in act_positions):
                    # Verifica se ainda é válida
                    if self._is_action_still_valid(action, piece_letter, placed_positions, board):
                        new_actions.append(action)
                    # Senão, ignora (ação inválida)
                else:
                    # Ação não foi afetada, mantém
                    new_actions.append(action)

            board.region_actions_list[adj_region] = new_actions
    

    def _is_action_still_valid(self, action, new_piece_letter, new_positions, board):
        """Verifica se uma ação ainda é válida após colocar uma nova peça."""
        region_id, piece_letter, shape, index, shape_abs = action
        
        # Reutiliza seus métodos existentes com pequenas adaptações:
        
        # 1. Verifica blocos 2x2 (pode usar seu método existente)
        if self._would_create_2x2_block(shape_abs, board):
            return False
        
        # 2. Verifica adjacência de peças iguais (pode usar seu método existente)
        if self._would_touch_equal_piece(shape_abs, piece_letter, board):
            return False
        
        # 3. NOVA VERIFICAÇÃO: interação específica entre a nova peça e a ação
        if piece_letter == new_piece_letter:
            # Verifica se as duas peças (mesma letra) tocariam em regiões diferentes
            for (i1, j1) in shape_abs:
                for (i2, j2) in new_positions:
                    if abs(i1 - i2) + abs(j1 - j2) == 1:  # adjacentes ortogonalmente
                        reg1 = board.regiao_original[i1][j1]
                        reg2 = board.regiao_original[i2][j2]
                        if reg1 != reg2:  # regiões diferentes
                            return False
        
        return True


    

    def is_valid_action(self, action, state: NuruominoState):
        """Verifica se a ação é válida no estado atual."""
        region_id, piece_letter, shape, index, shape_abs = action
        board = state.board
    
        # Verificação mais rápida primeiro
        if self._would_touch_equal_piece(shape_abs, piece_letter, board):
            return False
            
        # Verificação mais pesada por último
        if self._would_create_2x2_block(shape_abs, board):
            return False
    
        return True  # A ação é válida
    

    def _would_create_2x2_block(self, positions, board):
        """Optimized 2x2 block detection"""
        positions_set = set(positions)
        
        for (i, j) in positions:
            # Check only squares where this position could be part of
            for di in [0, -1]:
                for dj in [0, -1]:
                    square = [
                        (i + di, j + dj),
                        (i + di + 1, j + dj),
                        (i + di, j + dj + 1),
                        (i + di + 1, j + dj + 1)
                    ]
                    
                    # Check bounds and conditions in one pass
                    if all(
                        0 <= x < board.n and 0 <= y < board.n and
                        ((x, y) in positions_set or board.get_value(x, y) in "LITS")
                        for (x, y) in square
                    ):
                        return True
        return False


    def _would_touch_equal_piece(self, positions, piece_letter, board: Board):
        """Optimized equal piece adjacency check"""
        positions_set = set(positions)
        
        for (i, j) in positions:
            for di, dj in ORTHOGONAL_DIRS:
                ni, nj = i + di, j + dj
                if (0 <= ni < board.n and 0 <= nj < board.n and 
                    (ni, nj) not in positions_set):
                    val = board.get_value(ni, nj)
                    if val == piece_letter:
                        # Só bloqueia se estiverem em regiões diferentes
                        reg1 = board.regiao_original[i][j]
                        reg2 = board.regiao_original[ni][nj]
                        if reg1 != reg2:
                            return True
        return False

   # Game rules 
    def _all_regions_have_one_piece(self, board, state):
        """Verifica se cada região tem exatamente uma peça tetraminó (4 células com letra)."""
        # Se ainda há regiões por preencher, automaticamente não todas têm peças
        if state.unfinished_regions:
            return False
        return True


    def _has_no_2x2_blocks(self, board):
        """Optimized 2x2 block detection for final state"""
        for i in range(board.n - 1):
            for j in range(board.n - 1):
                if all(board.get_value(i+x, j+y) in "LITS" for x in [0,1] for y in [0,1]):
                    return False
        return True


    def _is_connected(self, board, debug=False):
        """Verifica se as peças estao conectadas ortagonalmente, isto é, tem de formar uma "ilha" """
        from collections import deque

        visited = set()
        queue = deque()
        piece_connections = set()

        # Encontrar a primeira célula preenchida com peça
        start_found = False
        for i in range(board.n):
            if start_found:
                break
            for j in range(board.n):
                if board.get_value(i, j) in "LITS":
                    queue.append((i, j))
                    visited.add((i, j))
                    start_found = True
                    break

        while queue:
            i, j = queue.popleft()
            current_piece = board.get_value(i, j)

            for di, dj in ORTHOGONAL_DIRS:
                ni, nj = i + di, j + dj
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

        # Count total filled cells
        total_filled = sum(
            1 for i in range(board.n)
            for j in range(board.n)
            if board.get_value(i, j) in "LITS"
        )

        if debug:
            print(f"\n[DEBUG] Peças conectadas ortogonalmente:")
            for a, b in sorted(piece_connections):
                print(f"  -> {a} conectado a {b}")
            print(f"Total de peças conectadas: {len(visited)} / {total_filled}")

        return len(visited) == total_filled
        


    def _no_same_piece_adjacent(self, board):
        """Verifica se nao existem pelas iguais conectadas"""
        for i in range(board.n):
            for j in range(board.n):
                current = board.get_value(i, j)
                if current not in "LITS":
                    continue
                for di, dj in ORTHOGONAL_DIRS:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < board.n and 0 <= nj < board.n:
                        neighbor = board.get_value(ni, nj)
                        if neighbor == current:
                            # Só é problema se estiverem em regiões diferentes
                            reg1 = board.regiao_original[i][j]
                            reg2 = board.regiao_original[ni][nj]
                            if reg1 != reg2:
                                return False
        return True
    
        
    def goal_test(self, state: NuruominoState):
        board = state.board
        print("\n[DEBUG] Estado atual do tabuleiro:")
        board.print_instance()

        if state.unfinished_regions:  # Se ainda há regiões por preencher
            return False
        
        if not self._all_regions_have_one_piece(state.board, state):
            print(f"- Todas regiões têm peça: {False}")
            return False
        if not self._no_same_piece_adjacent(state.board):
            print(f"- Peças iguais adjacentes: {False}")
            return False 
        if not self._has_no_2x2_blocks(state.board):
            print(f"- Sem blocos 2x2: {False}")
            return False
        if not self._is_connected(state.board, debug=True):
            print(f"- Peças conectadas: {False}")
            return False
        return True


def marcar_celulas_comuns(board: Board, problem: Nuruomino):
    """Marca no tabuleiro as células que são comuns a todas as ações possíveis por região."""
    state = NuruominoState(board)
    acoes = problem.actions(state)

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


def main():
    board = Board.parse_instance()

    print("Grelha lida do input:")
    board.print_instance()

    tracemalloc.start()
    tic = time.perf_counter()

    problem = Nuruomino(board)

    print("\nNúmero de células por região:")
    for region_id in problem.regions:
        positions = board.get_region_positions(region_id)
        print(f" - Região {region_id}: {len(positions)} células")

    # Criamos o novo estado inicial
    state = NuruominoState(board)
    problem = Nuruomino(board)  # novo problem com board atualizado
    problem.initial = state

    print("\nAções válidas no estado atual:")
    for action in problem.actions(state):
        region_id, piece, shape, index, shape_abs = action
        print(f"Região {region_id} -> peça {piece} com forma {shape} na posição {shape_abs}")

    
    # print("\nTabuleiro apos limpar celulas 'X':")
    # limpar_X(board)
    # board.print_instance()

    instrumented = InstrumentedProblem(problem)
    # goal_node = astar_search(instrumented)
    # print("A testar com *A...")

    # goal_node = breadth_first_graph_search(instrumented)
    # print("A testar com BFS...")

    goal_node = depth_first_graph_search(instrumented)
    print("A testar com DFS...")
    
    toc = time.perf_counter()
    print(f"  Programa executado em {toc - tic:0.4f} segundos")
    print(f"  Memória usada: {tracemalloc.get_traced_memory()[1] // 1024} kB")
    print(f"  Nós gerados: {instrumented.states}")
    print(f"  Nós expandidos: {instrumented.succs}")
    if goal_node:
        print("\nSolução encontrada:")
        limpar_X(goal_node.state.board)  # Limpa os 'X' antes de imprimir
        goal_node.state.board.print_instance()
    else:
        print("Nenhuma solução encontrada.")

if __name__ == "__main__":
    import cProfile
    import pstats
    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr)
    stats.sort_stats("cumulative").print_stats(30)