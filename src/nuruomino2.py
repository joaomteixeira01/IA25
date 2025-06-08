# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 34:
# 97226 João Teixeira 
# 110094 Francisco Fialho

from search import Node, Problem, InstrumentedProblem, depth_first_graph_search
import copy
from sys import stdin

# Pré-computa as direções ortogonais (cima, baixo, esquerda, direita)
ORTHOGONAL_DIRS = [(-1,0),(1,0),(0,-1),(0,1)]

# Define as formas das peças como matrizes booleanas
SHAPES = {
    'L1':  [[True, False],
            [True, False],
            [True, True]],
    'L2': [[True, True],
           [True, False],
           [True, False]],
    'L3': [[False, True],
           [False, True],
           [True, True]],
    'L4':[[True, True],
          [False, True],
          [False, True]],
    'L5':[[True, False, False],
          [True, True, True]],
    'L6':[[True, True, True],
          [True, False, False]],
    'L7':[[False, False, True],
          [True, True, True]],
    'L8':[[True, True, True],
          [False, False, True]],
    'I1':  [[True],
            [True],
            [True],
            [True]],
    'I2': [[True, True, True, True]],
    'T1': [[False, True, False],
           [True, True, True]],
    'T2': [[True, True, True],
           [False, True, False]],
    'T3': [[True, False],
           [True, True],
           [True, False]],
    'T4': [[False, True],
           [True, True],
           [False, True]],
    'S1':  [[False, True, True],
            [True, True, False]],
    'S2': [[True, True, False],
           [False, True, True]],
    'S3': [[False, True],
           [True, True],
           [True, False]],
    'S4': [[True, False],
           [True, True],
           [False, True]],
}

class Action:
    """Representa uma ação: colocar uma peça específica numa posição específica."""
    def __init__(self, shape_name, board_position, region_id):
        self.shape_name = shape_name          # Nome da forma da peça
        self.board_position = board_position  # Posição no tabuleiro (linha, coluna)
        self.region_id = region_id           # ID da região onde colocar
        self.n_adjacent = 0                  # Número de regiões adjacentes afetadas

    def __repr__(self):
        return f"Action(shape={self.shape_name}, pos={self.board_position}, region={self.region_id})"

class NuruominoState:
    """Estado do jogo Nuruomino."""
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NuruominoState.state_id
        NuruominoState.state_id += 1

    def __lt__(self, other):
        """Método usado para desempate na gestão da lista de abertos nas procuras informadas."""
        return self.id < other.id

class Board:
    """Representa o tabuleiro do jogo Nuruomino."""
    
    def __init__(self, board_matrix):
        """Inicializa o tabuleiro a partir de uma matriz."""
        self.board = board_matrix                    # Matriz do tabuleiro
        self.rows = len(board_matrix)               # Número de linhas
        self.cols = len(board_matrix[0]) if self.rows > 0 else 0  # Número de colunas
        self.regions = {}                           # Dicionário de regiões
        self.finished_regions = set()               # Regiões já preenchidas
        self.invalid = False                        # Flag de estado inválido
        self.regiao_original = [row[:] for row in board_matrix]  # Cópia do tabuleiro original
        self._shape_cache = {}  #Cache para posições absolutas de formas

        # Identifica e agrupa células por região
        for i in range(self.rows):
            for j in range(self.cols):
                region_id = board_matrix[i][j]
                if region_id not in self.regions:
                    self.regions[region_id] = {
                        'cells': [(i, j)],      # Células da região
                        'adjacents': set(),     # Regiões adjacentes
                        'domain': [],           # Ações possíveis
                        'action': None          # Ação escolhida      
                    }
                else:
                    self.regions[region_id]['cells'].append((i, j))

        # Calcula ações possíveis e adjacências para cada região
        for rid in self.regions:
            self.calculate_possible_actions(rid)
            self.find_adjacent_regions(rid)


    @staticmethod
    def parse_instance():
        """Lê uma instância do problema a partir do stdin."""
        from sys import stdin
        lines = [line.strip() for line in stdin if line.strip()]
        board_matrix = [line.split() for line in lines]
        return Board(board_matrix)


    def duplicate_board(self):
        """Cria uma cópia otimizada do tabuleiro."""
        new_board = Board.__new__(Board)
        
        # Copia a matriz do tabuleiro 
        new_board.board = [row[:] for row in self.board]
        new_board.rows = self.rows
        new_board.cols = self.cols
        new_board.invalid = self.invalid
        new_board.regiao_original = self.regiao_original  
        new_board._shape_cache = {}  # Cache para posições absolutas de formas
        
        # Copia as regiões 
        new_board.regions = {}
        for region_id, info in self.regions.items():
            new_board.regions[region_id] = {
                'cells': info['cells'],  
                'adjacents': info['adjacents'].copy(),  
                'domain': info['domain'],  
                'action': info['action']  
            }
        
        new_board.finished_regions = self.finished_regions.copy()
        return new_board
    

    def find_adjacent_regions(self, id_regiao):
        """Encontra todas as regiões adjacentes a uma região específica."""
        region = self.regions[id_regiao]
        for (linha, coluna) in region['cells']:
            # Verifica as 4 direções ortogonais
            for (dl, dc) in ORTHOGONAL_DIRS:
                ni, nj = linha + dl, coluna + dc
                # Se está dentro dos limites e é uma região diferente
                if 0 <= ni < self.rows and 0 <= nj < self.cols:
                    adj_id = self.board[ni][nj]
                    if adj_id != id_regiao:
                        region['adjacents'].add(adj_id)


    def piece_adjacents(self, shape_name, position, region_id, original):
        """Determina que regiões ficam adjacentes a uma peça colocada numa posição."""
        shape = SHAPES[shape_name]
        cells = self._get_absolute_shape_cells(shape, position)
        
        connected_shapes = set()  # Regiões com peças já colocadas
        connected_regions = set() # Todas as regiões adjacentes

        # Para cada célula da peça
        for row, col in cells:
            # Verifica os 4 vizinhos ortogonais
            for dr, dc in ORTHOGONAL_DIRS:
                nr = row + dr
                nc = col + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    neighbor_val = self.board[nr][nc]      # Valor atual no tabuleiro
                    neighbor_region = original[nr][nc]     # Região original
                    if neighbor_region == region_id:      # Ignora a própria região     
                        continue
                    # Se é uma peça diferente já colocada
                    if neighbor_val.isalpha() and neighbor_val != shape_name[0]:
                        connected_shapes.add(neighbor_region)
                    connected_regions.add(neighbor_region)

        return connected_shapes, connected_regions
    

    def _get_absolute_shape_cells(self, shape, top_left_pos):
        """Converte posições relativas da forma em posições absolutas no tabuleiro."""
        # Usa cache para evitar recálculos repetidos
        cache_key = (id(shape), top_left_pos)
        if cache_key in self._shape_cache:
            return self._shape_cache[cache_key]
        
        row0, col0 = top_left_pos
        absolute_cells = []
        for i, row in enumerate(shape):
            for j, val in enumerate(row):
                if val:  
                    absolute_cells.append((row0 + i, col0 + j))
        
        self._shape_cache[cache_key] = absolute_cells
        return absolute_cells


    def calculate_possible_actions(self, region_id):
        """Calcula todas as ações possíveis para uma região."""
        region = self.regions[region_id]
        # Tenta todas as formas
        for shape_name in SHAPES:
            # Em todas as células da região
            for cell in region['cells']:
                self._try_place_shape(shape_name, cell, region_id)


    def _try_place_shape(self, shape_name, cell, region_id):
        """Tenta colocar uma forma específica usando uma célula como âncora."""
        shape = SHAPES[shape_name]
        anchor = self._find_anchor(shape)  
        if anchor is None:
            return
        
        ai, aj = anchor
        base_i, base_j = cell
        # Calcula posição superior-esquerda da forma
        top_i = base_i - ai
        top_j = base_j - aj
        
        # Verifica se cabe no tabuleiro e na região
        if not self._shape_within_bounds(shape, top_i, top_j):
            return
        if not self._shape_fits_region(shape, top_i, top_j, region_id):
            return
        
        # Cria ação e calcula influência
        action = Action(shape_name, (top_i, top_j), region_id)
        _, connected = self.piece_adjacents(shape_name, (top_i, top_j), region_id, self.regiao_original)
        action.influence_count = len(connected)
        self.regions[region_id]['domain'].append(action)


    def _find_anchor(self, shape):
        """Encontra a primeira célula preenchida de uma forma (âncora)."""
        for i, row in enumerate(shape):
            for j, val in enumerate(row):
                if val:
                    return (i, j)
        return None


    def _shape_within_bounds(self, shape, row, col):
        """Verifica se uma forma cabe dentro dos limites do tabuleiro."""
        return 0 <= row and 0 <= col and row + len(shape) <= self.rows and col + len(shape[0]) <= self.cols


    def _shape_fits_region(self, shape, top_i, top_j, region_id):
        """Verifica se uma forma cabe inteiramente dentro de uma região."""
        for i, row in enumerate(shape):
            for j, filled in enumerate(row):
                if filled:
                    if self.board[top_i + i][top_j + j] != region_id:
                        return False
        return True


    def validate_placement_rules(self, shape_name, position):
        """Valida se colocar uma peça numa posição respeita as regras do jogo."""
        if self._touches_identical_piece(shape_name, position):
            return False  
        if self.detect_create_2x2_block(shape_name, position):
            return False  
        return True


    def _touches_identical_piece(self, shape_name, position):
        """Verifica se uma peça tocaria numa peça idêntica."""
        shape = SHAPES[shape_name]
        base_row, base_col = position

        for i, row in enumerate(shape):
            for j, filled in enumerate(row):
                if not filled:
                    continue
                cell_row = base_row + i
                cell_col = base_col + j
                # Verifica vizinhos ortogonais
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    neighbor_row = cell_row + dr
                    neighbor_col = cell_col + dc
                    if 0 <= neighbor_row < self.rows and 0 <= neighbor_col < self.cols:
                        if self.board[neighbor_row][neighbor_col] == shape_name[0]:
                            return True
        return False


    def detect_create_2x2_block(self, shape_name, position):
        """Deteta se colocar uma peça criaria um bloco 2x2 preenchido."""
        shape = SHAPES[shape_name]
        top_row, left_col = position
        # Calcula área onde procurar blocos 2x2
        r_start, r_end, c_start, c_end = self._get_block_area_bounds(top_row, left_col, shape)

        # Verifica cada posição possível de um bloco 2x2
        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                if r + 1 < self.rows and c + 1 < self.cols:
                    if self._is_full_filled_square(r, c, shape_name, position, shape):
                        return True
        return False


    def _get_block_area_bounds(self, row, col, shape):
        """Calcula os limites da área onde procurar blocos 2x2."""
        height = len(shape)
        width = len(shape[0])
        # Área de busca: uma célula antes da peça até uma célula depois
        r_start = max(0, row - 1)
        r_end = min(self.rows - 1, row + height)
        c_start = max(0, col - 1)
        c_end = min(self.cols - 1, col + width)
        return r_start, r_end, c_start, c_end


    def _is_full_filled_square(self, r, c, shape_name, position, shape):
        """Verifica se um quadrado 2x2 específico ficaria totalmente preenchido."""
        top_row, left_col = position
        # Verifica as 4 células do quadrado 2x2
        for dr in (0, 1):
            for dc in (0, 1):
                i = r + dr
                j = c + dc
                cell_val = self.board[i][j]
                # Se é uma célula vazia (região)
                if cell_val.isdigit():
                    # Verifica se seria preenchida pela nova peça
                    if top_row <= i < top_row + len(shape) and left_col <= j < left_col + len(shape[0]):
                        si = i - top_row
                        sj = j - left_col
                        if not shape[si][sj]:  
                            return False
                    else:  
                        return False
        return True


    def remove_connection(self, region_a, region_b):
        """Remove a conexão entre duas regiões."""
        self.regions[region_a]["adjacents"].discard(region_b)
        if region_b in self.regions:
            adjacent_set = self.regions[region_b]["adjacents"]
            if region_a in adjacent_set:
                adjacent_set.remove(region_a)


    def restrict_action_domain(self, region_id):
        """Restringe o domínio de ações de uma região baseado nas regras."""
        region_info = self.regions[region_id]
        updated_domain = []
        # Filtra apenas ações que respeitam as regras
        for candidate_action in region_info["domain"]:
            if self.validate_placement_rules(candidate_action.shape_name, candidate_action.board_position):
                updated_domain.append(candidate_action)
        # Se não há ações válidas e região não tem peça, estado é inválido
        if region_info["action"] is None and not updated_domain:
            self.invalid = True
        region_info["domain"] = updated_domain


    def apply_restrictions(self, action: Action, initial_board):
        """Aplica restrições após colocar uma peça."""
        region_id = action.region_id
        self.regions[region_id]['action'] = action

        # Determina que regiões ficam conectadas
        shapes_touching, regions_touching = self.piece_adjacents(
            action.shape_name, action.board_position, region_id, initial_board
        )

        adjacent_ids = list(self.regions[region_id]['adjacents'])
        # Determina que conexões devem ser cortadas
        to_disconnect = self._determine_disconnections(adjacent_ids, shapes_touching, regions_touching)

        # Restringe domínios das regiões que continuam conectadas
        for neighbor_id in adjacent_ids:
            if neighbor_id not in to_disconnect and neighbor_id not in self.finished_regions:
                self.restrict_action_domain(neighbor_id)

        # Remove conexões cortadas
        self._remove_disconnected_neighbors(region_id, to_disconnect)

        # Verifica se o grafo continua conectado
        if not self.verify_graph_connectivity():
            self.invalid = True


    def _determine_disconnections(self, neighbors, connected_shapes, connected_regions):
        """Determina que regiões devem ser desconectadas."""
        cut_list = []
        for nid in neighbors:
            region_data = self.regions[nid]
            # Se região não tem peça e não está adjacente à nova peça
            if region_data['action'] is None and nid not in connected_regions:
                cut_list.append(nid)
            # Se região tem peça mas não toca na nova peça
            elif region_data['action'] is not None and nid not in connected_shapes:
                cut_list.append(nid)
        return cut_list


    def _remove_disconnected_neighbors(self, region_id, cut_list):
        """Remove conexões com regiões desconectadas."""
        for nid in cut_list:
            self.remove_connection(region_id, nid)


    def verify_graph_connectivity(self):
        """Verifica se o grafo de regiões continua conectado."""
        region_keys = list(self.regions.keys())
        if len(region_keys) <= 1:
            return True
    
        # Usa DFS iterativo em vez de recursivo para evitar stack overflow
        visited = set()
        stack = [region_keys[0]]
        
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            
            # Adiciona vizinhos não visitados à pilha
            for neighbor in self.regions[current]['adjacents']:
                if neighbor not in visited:
                    stack.append(neighbor)
    
        # Todas as regiões devem ser alcançáveis
        return len(visited) == len(region_keys)


    def get_value(self, row, col):
        """Obtém o valor numa posição do tabuleiro."""
        return self.board[row][col]


    def print_instance(self):
        """Imprime o tabuleiro formatado."""
        output = '\n'.join('\t'.join(map(str,row)) for row in self.board)
        print(output, end='')


class Nuruomino(Problem):
    """Classe principal do problema Nuruomino."""
    
    def __init__(self, board: Board):
        self.initial = NuruominoState(board)


    def get_sorted_actions_from_best_region(self, state: NuruominoState):
        """Obtém ações ordenadas da melhor região (mais restringida primeiro)."""
        board = state.board
        best_region = None
        min_domain_size = float('inf')
    
        # Encontra região com menor domínio (mais restringida primeiro)
        for region_id, region in board.regions.items():
            if region_id in board.finished_regions:
                continue
    
            domain_size = len(region['domain'])
            if domain_size == 0:
                continue
                
            if domain_size < min_domain_size:
                min_domain_size = domain_size
                best_region = region
                
            # Se o domínio tem tamanho 1, usar imediatamente (movimento forçado)
            if domain_size == 1:
                break
    
        if best_region is None:
            return []
    
        # Ordena por menor influência (menos restritivo primeiro)
        return sorted(best_region['domain'], key=lambda a: a.n_adjacent)


    def actions(self, state: NuruominoState):
        """Retorna ações possíveis num estado."""
        board = state.board
        if board.invalid:
            return []  
        return self.get_sorted_actions_from_best_region(state)


    def result(self, state: NuruominoState, action: Action):
        """Aplica uma ação e retorna o novo estado."""
        original_board = state.board
        new_board = original_board.duplicate_board()
        shape = SHAPES[action.shape_name]
        anchor_row, anchor_col = action.board_position

        # Coloca a peça no tabuleiro
        for dx, row in enumerate(shape):
            for dy, filled in enumerate(row):
                if filled:
                    new_board.board[anchor_row + dx][anchor_col + dy] = action.shape_name[0]

        # Aplica restrições e marca região como terminada
        new_board.apply_restrictions(action, self.initial.board.board)
        new_board.finished_regions.add(action.region_id)

        return NuruominoState(new_board)


    def goal_test(self, state: NuruominoState):
        """Testa se um estado é objetivo (todas as regiões preenchidas e válidas)."""
        return (len(state.board.finished_regions) == len(state.board.regions) and not state.board.invalid)
    

def main():
    """Função principal."""
    board = Board.parse_instance()
    
    # import tracemalloc
    # import time
    # # # Inicia medição de tempo e memória
    # tracemalloc.start()
    # tic = time.perf_counter()

    state = NuruominoState(board)
    problem = Nuruomino(board)
    problem.initial = state

    # Resolve o problema usando procura em profundidade
    instrumented = InstrumentedProblem(problem)
    goal_node = depth_first_graph_search(instrumented)

    # # Calcula tempo e memória usados
    # toc = time.perf_counter()
    # print(f"  Programa executado em {toc - tic:0.4f} segundos")
    # print(f"  Memória usada: {tracemalloc.get_traced_memory()[1] // 1024} kB")
    
    # Imprime resultado
    if goal_node:
        goal_node.state.board.print_instance()
    else:
        print("Nenhuma solução encontrada.")

if __name__ == "__main__":
    main()
