# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 25:
# 109981 Afonso Marques
# 109623 Tiago Lobo

from search import Problem, Node
from sys import stdin
import numpy as np # type: ignore
from numpy.lib.stride_tricks import sliding_window_view # type: ignore
import search as sh # type: ignore

SHAPES = {    
    'L': np.array([
        [True, False],
        [True, False],
        [True, True]
    ]),
    'LI': np.array([
        [True, True],
        [True, False],
        [True, False]
    ]),
    'LS': np.array([
        [False, True],
        [False, True],
        [True, True]
    ]),
    'LIS': np.array([
        [True, True],
        [False, True],
        [False, True]
    ]),
    'LLU': np.array([
        [True, False, False],
        [True, True, True]
    ]),
    'LLD': np.array([
        [True, True, True],
        [True, False, False]
    ]),
    'LRU': np.array([
        [False, False, True],
        [True, True, True]
    ]),
    'LRD': np.array([
        [True, True, True],
        [False, False, True]
    ]),
    'I': np.array([
        [True],
        [True],
        [True],
        [True]
    ]),
    'IH': np.array([
        [True, True, True, True],
    ]),
    'TU': np.array([
        [False, True, False],
        [True, True, True]
    ]),
    'TD': np.array([
        [True, True, True],
        [False, True, False]
    ]),
    'TR': np.array([
        [True, False],
        [True, True],
        [True, False]
    ]),
    'TL': np.array([
        [False, True],
        [True, True],
        [False, True]
    ]),
    'S': np.array([
        [False, True, True],
        [True, True, False]
    ]),
    'SI': np.array([
        [True, True, False],
        [False, True, True]
    ]),
    'SR': np.array([
        [False, True],
        [True, True],
        [True, False]
    ]),
    'SL': np.array([
        [True, False],
        [True, True],
        [False, True]
    ]),
}

class Action:
    def __init__(self, shape_name, board_position, region_id):
        self.shape_name = shape_name
        self.board_position = board_position
        self.region_id = region_id
        self.influence_count = 0

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

    def __init__(self, board: np.ndarray, regions: dict, assigned = []):
        """Construtor da classe Board.

        :param board: Matriz numpy que representa o tabuleiro.
        :param rows: Número de linhas do tabuleiro.
        :param cols: Número de colunas do tabuleiro.
        :param regions: Dicionário com as regiões identificadas no tabuleiro.
        """
        self.board = board
        self.rows, self.cols = board.shape
        self.regions = regions
        self.assigned = assigned
        self.invalid = False
        
    def clone(self):
        """Cria uma cópia do tabuleiro."""
        new_board = np.copy(self.board)
        new_regions = {}
        for region_id, data in self.regions.items():
            new_regions[region_id] = {
                'cells': data['cells'].copy(),
                'adjacents': data['adjacents'].copy(),
                'domain': data['domain'].copy(),
                'action': data['action']
            }
        new_assigned = self.assigned.copy()
        return Board(new_board, new_regions, new_assigned)

    @staticmethod
    def parse_instance():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 pipe.py < test-01.txt

            > from sys import stdin
            > line = stdin.readline().split()
        """
        board = np.array([line.split() for line in stdin.readlines()])
        rows, cols = board.shape
        regions = {}
        for row in range(rows):
            for col in range(cols):
                id = board[row][col]
                if id not in regions:
                    regions[id] = {'cells': [(row, col)], 'adjacents': set(), 'domain': [], 'action': None}
                else:
                    regions[id]['cells'].append((row, col))
        board_instance = Board(board, regions)
        for region_id, data in board_instance.regions.items():
            board_instance.calculate_domain(region_id)
            board_instance.calculate_adjacencies(region_id)
        return board_instance
    
    def adjacent_regions(self, region:int) -> list:
        """Devolve uma lista das regiões que fazem fronteira com a região enviada no argumento."""
        #TODO
        pass
    
    def adjacent_positions(self, row:int, col:int) -> list:
        """Devolve as posições adjacentes à região, em todas as direções, incluindo diagonais."""
        #TODO
        pass

    def adjacent_values(self, row:int, col:int) -> list:
        """Devolve os valores das celulas adjacentes à região, em todas as direções, incluindo diagonais."""
        #TODO
        pass

    def shape_surrowndings(self, shape_name:str, pos:tuple, region_id, initial):

        row, col = pos
        surrownding_shapes = set()  # ids das regiões com peças adjacentes
        surrownding_regions = set()  # ids das regiões adjacentes
        directions = [(1, 0), (-1, 0), (0, -1), (0, 1)]
        shape = SHAPES[shape_name]
        shape_rows, shape_cols = shape.shape

        for i in range(shape_rows):
            for j in range(shape_cols):
                if shape[i][j]:
                    board_row = row + i
                    board_col = col + j
                    
                    for dr, dc in directions:
                        d_row = board_row + dr
                        d_col = board_col + dc
                        if (0 <= d_row < self.rows and 0 <= d_col < self.cols):
                            adj_value = self.board[d_row][d_col]
                            adj_region_id = initial[d_row][d_col]
                            
                            # Ignorar células da própria região
                            if adj_region_id == region_id:
                                continue
                            
                            # Se é uma letra (peça colocada), adicionar às shapes conectadas
                            if adj_value.isalpha() and adj_value != shape_name[0]:
                                surrownding_shapes.add(adj_region_id)
                            
                            # Adicionar todas as regiões adjacentes
                            surrownding_regions.add(adj_region_id)

        return surrownding_shapes, surrownding_regions
    
    def check_cell_adjacency(self, shape_name:str, pos:tuple) -> list:
        """Verifica se há uma shape igual a shape, adjacente à posição dada."""
        row, col = pos
        directions = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
        for d_row, d_col in directions:
            if (0 <= d_row < self.rows and 0 <= d_col < self.cols):
                adj_shape = self.board[d_row][d_col]
                if shape_name[0] == adj_shape: return True
        return False
    
    def check_for_squares(self, shape_name: str, pos: tuple):
        """Versão otimizada que verifica apenas a área afetada pela peça"""
        shape = SHAPES[shape_name]
        shape_rows, shape_cols = shape.shape
        row, col = pos
        
        min_row = max(0, row - 1)
        max_row = min(self.rows - 1, row + shape_rows)
        min_col = max(0, col - 1)
        max_col = min(self.cols - 1, col + shape_cols)
        
        for r in range(min_row, max_row):
            for c in range(min_col, max_col):
                if r + 1 < self.rows and c + 1 < self.cols:
                    square_filled = True
                    for dr in range(2):
                        for dc in range(2):
                            cell_r, cell_c = r + dr, c + dc
                            cell_val = self.board[cell_r, cell_c]
                            
                            # Verificar se a célula estará preenchida
                            if cell_val.isdigit():
                                # Verificar se a nova peça cobre esta célula
                                if (row <= cell_r < row + shape_rows and 
                                    col <= cell_c < col + shape_cols):
                                    shape_r, shape_c = cell_r - row, cell_c - col
                                    if not shape[shape_r, shape_c]:
                                        square_filled = False
                                        break
                                else:
                                    square_filled = False
                                    break
                        if not square_filled:
                            break
                    
                    if square_filled:
                        return True
        
        return False
    
    def check_restrictions(self, shape_name, pos, squares:bool):
        #if squares: return not self.check_for_squares(shape_name, pos)
        shape = SHAPES[shape_name]
        for r in range(shape.shape[0]):
            for c in range(shape.shape[1]):
                if not shape[r, c]: continue
                has_adjacent = self.check_cell_adjacency(shape_name, (r+pos[0], c+pos[1]))
                if has_adjacent:
                    return False
        return not self.check_for_squares(shape_name, pos)
    
    def adjacency_graph_connectivity(self):
        nodes = set(r for r in self.regions.keys())

        region = next(iter(nodes))
        queue = [region]
        visited = {region}

        while queue:
            current = queue.pop(0)
            for adjacent in self.regions[current]['adjacents']:
                if adjacent in nodes and adjacent not in visited:
                    visited.add(adjacent)
                    queue.append(adjacent)
                    
        if len(visited) < len(nodes):
            return False
        
        return True

    def _get_window(self, shape, pos):
        # Encontra a janela do tabuleiro que corresponde à forma (shape) na posição (pos).
        shape_indices = np.argwhere(shape)
        anchor_c, anchor_r = shape_indices[0]
        pos_c = pos[0] - anchor_c
        pos_r = pos[1] - anchor_r
        shape_rows, shape_cols = shape.shape

        if (pos_c < 0 or pos_r < 0 or shape_rows > (self.rows - pos_c) or shape_cols > (self.cols - pos_r)):
            return None
        window = self.board[pos_c:pos_c+shape_rows, pos_r:pos_r+shape_cols]
        return window, (pos_c, pos_r)

    def calculate_domain(self, region_id):
        region_data = self.regions[region_id]
        for cell in region_data['cells']:
            for shape_name, raw_shape in SHAPES.items():
                shape = raw_shape
                matching_info = self._get_window(shape, cell)
                if matching_info is None:
                    continue
                window, placement = matching_info
                if np.all(shape <= (window == region_id)):
                    action = Action(shape_name, placement, region_id)
                    _, connected = self.shape_surrowndings(action.shape_name, action.board_position, region_id, self.board)
                    action.influence_count = len(connected)
                    region_data['domain'].append(action)

    def calculate_adjacencies(self, region_id):
        region_data = self.regions[region_id]
        for cell in region_data['cells']:
            row, col = cell
            directions = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
            for d_row, d_col in directions:
                if 0 <= d_row < self.rows and 0 <= d_col < self.cols:
                    adjacent_id = self.board[d_row][d_col]
                    if adjacent_id != region_id:
                        region_data['adjacents'].add(adjacent_id)

    def filter_domain(self, region_id, squares=False):
        region_data = self.regions[region_id]
        to_remove = []
        for action in region_data['domain']:
            if not self.check_restrictions(action.shape_name, action.board_position, squares):
                to_remove.append(action)
        if region_data['action'] is None and len(region_data['domain']) == 0:
            self.invalid = True
        for action in to_remove:
            region_data['domain'].remove(action)

    def cut_connections(self, region_id, adjacent_id):
        self.regions[region_id]['adjacents'].discard(adjacent_id)
        self.regions[adjacent_id]['adjacents'].discard(region_id)

    def propagate(self, action: Action, initial_board):
        region_id = action.region_id
        region_data = self.regions[region_id]
        region_data['action'] = action
        connected_shapes, connected_regions = self.shape_surrowndings(action.shape_name, action.board_position, region_id, initial_board)
        connections_to_cut = []
        
        for adjacent_id in list(region_data['adjacents']):
            adjacent_data = self.regions[adjacent_id]
            if adjacent_data['action'] is None:
                if adjacent_id not in connected_regions:
                    connections_to_cut.append(adjacent_id)
                    continue
            else:
                if adjacent_id not in connected_shapes:
                    connections_to_cut.append(adjacent_id)
                    continue

            if adjacent_id not in self.assigned:
                self.filter_domain(adjacent_id)

        for c in connections_to_cut:
            self.cut_connections(region_id, c)
        if not self.adjacency_graph_connectivity():
            self.invalid = True

class Nuruomino(Problem):
    contador = 0
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        self.initial = NuruominoState(board)

    def select_next_region(self, state: NuruominoState):
        regions = [data for id, data in state.board.regions.items() if id not in state.board.assigned]
        if not regions:
            return None
        def region_score(region_data):
            domain_size = len(region_data['domain'])
            empty_adjacents = sum(1 for ad_id in region_data['adjacents'] if ad_id not in state.board.assigned)
            return (domain_size, -empty_adjacents)

        best_region = min(regions, key=region_score)
        return best_region
    
    def sort_actions(self, actions):
        def action_score(action: Action):
            return (action.influence_count)
        return sorted(actions, key=action_score)

    def actions(self, state: NuruominoState):
        """Retorna uma lista de ações que podem ser executadas a partir do estado passado como argumento."""
        if state.board.invalid: 
            return []
        best_region = self.select_next_region(state)
        if best_region is None:
            return []
        actions = self.sort_actions(best_region['domain'])
        return actions

    def result(self, state: NuruominoState, action: Action):
        """Retorna o estado resultante de executar a 'action' sobre 'state' passado como argumento.
        A ação a executar deve ser uma das presentes na lista obtida pela execução de self.actions(state)."""
        new_board = state.board.clone()
        shape_name = action.shape_name
        pos_c, pos_r = action.board_position
        region_id = action.region_id
        shape = SHAPES[shape_name]
        shape_rows, shape_cols = shape.shape
        
        # Colocar peça no tabuleiro
        for i in range(shape_rows):
            for j in range(shape_cols):
                if shape[i, j]:
                    new_board.board[pos_c + i, pos_r + j] = shape_name[0]
        
        # Propagar restrições
        new_board.propagate(action, self.initial.board.board)
        new_board.assigned.append(region_id)
        
        new_state = NuruominoState(new_board)
        return new_state
        
    def goal_test(self, state: NuruominoState): 
        """Retorna True se e só se o estado passado como argumento é um estado objetivo.
        Deve verificar se todas as posições do tabuleiro estão preenchidas de acordo com as regras do problema."""
        if len(state.board.assigned) == len(state.board.regions.keys()):
            Nuruomino.contador += 1
        return len(state.board.assigned) == len(state.board.regions.keys()) and not state.board.invalid

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        state = node.state
        board = state.board
        
        unfilled_regions = len(board.regions) - len(board.assigned)
        remaining_pieces = sum(len(data['domain']) for id, data in board.regions.items() if id not in board.assigned)

        if state.board.invalid:
            return float('inf')

        if unfilled_regions > 2:
            unfilled_regions += 100
        
        return unfilled_regions + remaining_pieces

if __name__ == "__main__":
    board = Board.parse_instance()
    problem = Nuruomino(board)
    result = sh.depth_first_tree_search(problem)
    #result = sh.astar_search(problem, h=problem.h, display=False)
    import time
    import tracemalloc
    tracemalloc.start()
    tic = time.perf_counter()
    toc = time.perf_counter()
    print(f"  Programa executado em {toc - tic:0.4f} segundos")
    print(f"  Memória usada: {tracemalloc.get_traced_memory()[1] // 1024} kB")

    if not result:
        print("No solution found.")
    for row in result.state.board.board:
        print('\t'.join(row))
