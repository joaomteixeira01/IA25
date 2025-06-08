# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 34:
# 97226 João Teixeira 
# 110094 Francisco Fialho


from search import Node, Problem, InstrumentedProblem, depth_first_graph_search
import copy
from sys import stdin
import time
import tracemalloc

# Pre-compute orthogonal directions
ORTHOGONAL_DIRS = [(-1,0),(1,0),(0,-1),(0,1)]

# Define shapes as boolean matrices (list-of-lists of booleans)
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
    def __init__(self, shape_name, board_position, region_id):
        self.shape_name = shape_name
        self.board_position = board_position
        self.region_id = region_id
        self.n_adjacent = 0

    def __repr__(self):
        return f"Action(shape={self.shape_name}, pos={self.board_position}, region={self.region_id})"

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
    def __init__(self, board_matrix):
        self.board = board_matrix
        self.rows = len(board_matrix)
        self.cols = len(board_matrix[0]) if self.rows > 0 else 0
        self.regions = {}
        self.finished_regions = set()
        self.invalid = False
        self.regiao_original = [row[:] for row in board_matrix]

        for i in range(self.rows):
            for j in range(self.cols):
                region_id = board_matrix[i][j]
                if region_id not in self.regions:
                    self.regions[region_id] = {
                        'cells': [(i, j)],
                        'adjacents': set(),
                        'domain': [],
                        'action': None
                    }
                else:
                    self.regions[region_id]['cells'].append((i, j))

        for rid in self.regions:
            self.calculate_possible_actions(rid)
            self.find_adjacent_regions(rid)


    @staticmethod
    def parse_instance():
        from sys import stdin
        lines = [line.strip() for line in stdin if line.strip()]
        board_matrix = [line.split() for line in lines]
        return Board(board_matrix)

    def duplicate_board(self):
        # Create new board instance without calling __init__
        new_board = Board.__new__(Board)

        # Shallow copy the board matrix (each row list is copied)
        new_board.board = [row[:] for row in self.board]

        # Copy primitive attributes
        new_board.rows = self.rows
        new_board.cols = self.cols
        new_board.invalid = self.invalid

        # Reference the immutable original (never modified after creation)
        new_board.regiao_original = self.regiao_original

        # Optimized region copying: reuse immutable cells, copy sets and lists
        new_board.regions = {}
        for region_id, info in self.regions.items():
            new_board.regions[region_id] = {
                'cells': info['cells'],                # immutable tuples, safe to share
                'adjacents': set(info['adjacents']),  # copy set
                'domain': info['domain'][:],          # copy domain list
                'action': info['action']              # None or Action (immutable)
            }

        # Copy finished regions
        new_board.finished_regions = set(self.finished_regions)

        return new_board


    def find_adjacent_regions(self, id_regiao):
        region = self.regions[id_regiao]
        for (linha, coluna) in region['cells']:
            for (dl, dc) in ORTHOGONAL_DIRS:
                ni, nj = linha + dl, coluna + dc
                if 0 <= ni < self.rows and 0 <= nj < self.cols:
                    adj_id = self.board[ni][nj]
                    if adj_id != id_regiao:
                        region['adjacents'].add(adj_id)

    
    def piece_adjacents(self, shape_name, position, region_id, original):
        shape = SHAPES[shape_name]
        cells = self._get_absolute_shape_cells(shape, position)

        connected_shapes = set()
        connected_regions = set()

        # For each cell of the placed shape, check orthogonal neighbors
        for (row, col) in cells:
            for (dr, dc) in ORTHOGONAL_DIRS:
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    neighbor_val = self.board[nr][nc]
                    neighbor_region = original[nr][nc]
                    # Skip same region
                    if neighbor_region == region_id:
                        continue
                    # If neighbor cell is a letter and not the same shape letter, record shape adjacency
                    if neighbor_val.isalpha() and neighbor_val != shape_name[0]:
                        connected_shapes.add(neighbor_region)
                    # Always record region adjacency
                    connected_regions.add(neighbor_region)

        return connected_shapes, connected_regions

    def _get_absolute_shape_cells(self, shape, top_left_pos):
        row0, col0 = top_left_pos
        absolute_cells = []
        for i, row in enumerate(shape):
            for j, val in enumerate(row):
                if val:
                    absolute_cells.append((row0 + i, col0 + j))
        return absolute_cells


    def calculate_possible_actions(self, region_id):
        region = self.regions[region_id]
        for shape_name in SHAPES:
            for cell in region['cells']:
                self._try_place_shape(shape_name, cell, region_id)


    def _try_place_shape(self, shape_name, cell, region_id):
        shape = SHAPES[shape_name]
        anchor = self._find_anchor(shape)
        if anchor is None:
            return
        ai, aj = anchor
        base_i, base_j = cell
        top_i = base_i - ai
        top_j = base_j - aj
        if not self._shape_within_bounds(shape, top_i, top_j):
            return
        if not self._shape_fits_region(shape, top_i, top_j, region_id):
            return
        action = Action(shape_name, (top_i, top_j), region_id)
        _, connected = self.piece_adjacents(shape_name, (top_i, top_j), region_id, self.regiao_original)
        action.influence_count = len(connected)
        self.regions[region_id]['domain'].append(action)
        if not self.validate_placement_rules(shape_name, (top_i, top_j)):
            return  # Don’t add invalid actions to domain


    def _find_anchor(self, shape):
        for i, row in enumerate(shape):
            for j, val in enumerate(row):
                if val:
                    return (i, j)
        return None


    def _shape_within_bounds(self, shape, row, col):
        return 0 <= row and 0 <= col and row + len(shape) <= self.rows and col + len(shape[0]) <= self.cols

    def _shape_fits_region(self, shape, top_i, top_j, region_id):
        for i, row in enumerate(shape):
            for j, filled in enumerate(row):
                if filled and self.board[top_i + i][top_j + j] != region_id:
                    return False
        return True

    def validate_placement_rules(self, shape_name, position):
        # Check no identical shapes touch
        if self._touches_identical_piece(shape_name, position):
            return False
        # Check no 2x2 block is formed
        if self.detect_create_2x2_block(shape_name, position):
            return False
        return True

    def _touches_identical_piece(self, shape_name, position):
        shape = SHAPES[shape_name]
        base_row, base_col = position

        for i, row in enumerate(shape):
            for j, filled in enumerate(row):
                if not filled:
                    continue
                cell_row = base_row + i
                cell_col = base_col + j
                # Check four orthogonal neighbors for same letter
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cell_row + dr, cell_col + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        if self.board[nr][nc] == shape_name[0]:
                            return True
        return False

    def detect_create_2x2_block(self, shape_name, position):
        # New shape cells (letter placed) may form a new 2x2 block
        shape = SHAPES[shape_name]
        base_row, base_col = position
        # For each cell of the shape, check 2x2 including that cell
        for i, row in enumerate(shape):
            for j, filled in enumerate(row):
                if not filled:
                    continue
                r, c = base_row + i, base_col + j
                # Check the four possible 2x2 blocks that include (r,c)
                for dr in (0, -1):
                    for dc in (0, -1):
                        top_r, top_c = r + dr, c + dc
                        # Ensure within grid for a 2x2 block
                        if (0 <= top_r < self.rows - 1) and (0 <= top_c < self.cols - 1):
                            # Check all four cells of the block
                            if (self.board[top_r][top_c].isalpha() and
                                self.board[top_r+1][top_c].isalpha() and
                                self.board[top_r][top_c+1].isalpha() and
                                self.board[top_r+1][top_c+1].isalpha()):
                                return True
        return False

    def remove_connection(self, region_a, region_b):
        self.regions[region_a]["adjacents"].discard(region_b)
        if region_b in self.regions:
            self.regions[region_b]["adjacents"].discard(region_a)

    def restrict_action_domain(self, region_id):
        region_info = self.regions[region_id]
        # Filter domain with list comprehension (faster than manual loop):contentReference[oaicite:7]{index=7}
        updated_domain = [
            action for action in region_info["domain"]
            if self.validate_placement_rules(action.shape_name, action.board_position)
        ]
        region_info["domain"] = updated_domain
        # If no action is set and domain is empty, mark invalid
        if region_info["action"] is None and not updated_domain:
            self.invalid = True

    def apply_restrictions(self, action: Action, initial_board):
        region_id = action.region_id
        self.regions[region_id]['action'] = action

        shapes_touching, regions_touching = self.piece_adjacents(
            action.shape_name, action.board_position, region_id, initial_board
        )

        adjacent_ids = list(self.regions[region_id]['adjacents'])
        to_disconnect = self._determine_disconnections(adjacent_ids, shapes_touching, regions_touching)

        # Restrict neighbors' domains
        for neighbor_id in adjacent_ids:
            if neighbor_id not in to_disconnect and neighbor_id not in self.finished_regions:
                self.restrict_action_domain(neighbor_id)
                if self.invalid:
                    break  # early exit if already invalid

        # Remove disconnected neighbors
        if to_disconnect:
            self._remove_disconnected_neighbors(region_id, to_disconnect)
            # Check connectivity only if something changed
            if not self.verify_graph_connectivity():
                self.invalid = True

    def _determine_disconnections(self, neighbors, connected_shapes, connected_regions):
        cut_list = []
        for nid in neighbors:
            region_data = self.regions[nid]
            if region_data['action'] is None and nid not in connected_regions:
                cut_list.append(nid)
            elif region_data['action'] is not None and nid not in connected_shapes:
                cut_list.append(nid)
        return cut_list

    def _remove_disconnected_neighbors(self, region_id, cut_list):
        for nid in cut_list:
            self.remove_connection(region_id, nid)

    def verify_graph_connectivity(self):
        region_keys = set(self.regions.keys())
        if not region_keys:
            return True
        visited = set()
        stack = [next(iter(region_keys))]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.regions[current]['adjacents']:
                if neighbor in region_keys and neighbor not in visited:
                    stack.append(neighbor)
        return len(visited) == len(region_keys)

    def get_value(self, row, col):
        return self.board[row][col]

    def print_instance(self):
        print("\n".join("\t".join(map(str, row)) for row in self.board), end='')

class Nuruomino(Problem):
    def __init__(self, board: Board):
        self.initial = NuruominoState(board)

    def get_sorted_actions_from_best_region(self, state: NuruominoState):
        board = state.board
        best_region = None
        best_score = None
        for region_id, region in board.regions.items():
            if region_id in board.finished_regions:
                continue
            domain_size = len(region['domain'])
            unfilled_neighbors = sum(
                1 for adj_id in region['adjacents'] if adj_id not in board.finished_regions
            )
            score = (domain_size, -unfilled_neighbors)
            if best_region is None or score < best_score:
                best_region = region
                best_score = score
        if best_region is None:
            return []
        return sorted(best_region['domain'], key=lambda action: action.influence_count)

    def actions(self, state: NuruominoState):
        if state.board.invalid:
            return []
        return self.get_sorted_actions_from_best_region(state)

    def result(self, state: NuruominoState, action: Action):
        original_board = state.board
        new_board = original_board.duplicate_board()
        shape = SHAPES[action.shape_name]
        r0, c0 = action.board_position
        for dx, row in enumerate(shape):
            for dy, filled in enumerate(row):
                if filled:
                    new_board.board[r0 + dx][c0 + dy] = action.shape_name[0]
        new_board.apply_restrictions(action, self.initial.board.board)
        new_board.finished_regions.add(action.region_id)
        return NuruominoState(new_board)

    def goal_test(self, state: NuruominoState):
        return (len(state.board.finished_regions) == len(state.board.regions)
                and not state.board.invalid)


def main():
    board = Board.parse_instance()
    tracemalloc.start()
    tic = time.perf_counter()
    state = NuruominoState(board)
    problem = Nuruomino(board)
    problem.initial = state
    instrumented = InstrumentedProblem(problem)
    goal_node = depth_first_graph_search(instrumented)
    toc = time.perf_counter()
    print(f"  Programa executado em {toc - tic:0.4f} segundos")
    print(f"  Memória usada: {tracemalloc.get_traced_memory()[1] // 1024} kB")
    if goal_node:
        goal_node.state.board.print_instance()
    else:
        print("Nenhuma solução encontrada.")

if __name__ == "__main__":
    import cProfile, pstats
    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr)
    stats.sort_stats("cumulative").print_stats(30)