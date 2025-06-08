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
ORTHOGONAL_DIRS = [(-1,0),(1,0),(0,-1),(0,1)]
ALL_DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


LAST_STATES = []



class NuruominoState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NuruominoState.state_id
        NuruominoState.state_id += 1
        self.unfinished_regions = self._get_unfinished_regions()
        self._hash = None

    def __lt__(self, other):
        return self.id < other.id
    
    def __eq__(self, other):
        if not isinstance(other, NuruominoState):
            return False
        if hash(self) != hash(other):
            return False
        # Optimized: direct list access instead of get_value calls
        n = self.board.n
        for i in range(n):
            # using direct indexing avoids function-call overhead
            row_self = self.board.board[i]
            row_other = other.board.board[i]
            for j in range(n):
                if row_self[j] != row_other[j]:
                    return False
        return True
    
    def __hash__(self):
        if self._hash is None:
            # Hash the board's tuple-of-tuples
            self._hash = hash(tuple(tuple(row) for row in self.board.board))
        return self._hash

    def _get_unfinished_regions(self):
        unfinished = []
        for region_id in self.board.regions:
            positions = self.board.get_region_positions(region_id)
            # Check if region has any letter LITS
            has_piece = any(self.board.board[i][j] in "LITS" for (i,j) in positions)
            if not has_piece:
                unfinished.append(region_id)
        return unfinished

class Board:
    def __init__(self, board, preserve_original=True):
        self.board = board
        self.n = len(board)
        self._region_positions_cache = {}
        self._adjacent_regions_cache = {}
        self.regiao_original = [row[:] for row in board] if preserve_original else board
        self.regions = self._extract_regions()
        # Precompute region positions and actions
        for rid in self.regions:
            self.get_region_positions(rid)
        self.region_actions_list = {rid: self.region_actions(rid) for rid in self.regions}
        # Sort regions by fewest actions (MRV heuristic)
        counts = [(rid, len(actions)) for rid, actions in self.region_actions_list.items()]
        counts.sort(key=lambda x: x[1])
        self.regions = [rid for (rid,_) in counts]

    def _extract_regions(self):
        unique_regions = set()
        for row in self.board:
            for cell in row:
                if cell.isdigit():
                    unique_regions.add(int(cell))
        return sorted(unique_regions)

    def get_value(self, row, col):
        return self.board[row][col]

    def print_instance(self):
        output = '\n'.join('\t'.join(map(str,row)) for row in self.board)
        print(output, end='')

    def get_region_positions(self, region_id):
        if region_id not in self._region_positions_cache:
            positions = []
            region_str = str(region_id)
            for i in range(self.n):
                for j in range(self.n):
                    if self.regiao_original[i][j] == region_str:
                        positions.append((i,j))
            self._region_positions_cache[region_id] = positions
        return self._region_positions_cache[region_id]

    def adjacent_regions(self, region:int) -> list:
        if region not in self._adjacent_regions_cache:
            adjacents = set()
            positions = self.get_region_positions(region)
            for (i,j) in positions:
                for (di,dj) in ORTHOGONAL_DIRS:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < self.n and 0 <= nj < self.n:
                        val = self.board[ni][nj]
                        if val != str(region) and val.isdigit():
                            adjacents.add(int(val))
            self._adjacent_regions_cache[region] = list(adjacents)
        return self._adjacent_regions_cache[region]

    def adjacent_positions(self, row:int, col:int) -> list:
        positions = []
        for (di,dj) in ALL_DIRS:
            ni, nj = row+di, col+dj
            if 0 <= ni < self.n and 0 <= nj < self.n:
                positions.append((ni,nj))
        return positions

    def adjacent_values(self, row:int, col:int) -> list:
        values = []
        for (di,dj) in ALL_DIRS:
            ni, nj = row+di, col+dj
            if 0 <= ni < self.n and 0 <= nj < self.n:
                values.append(self.board[ni][nj])
        return values

    @staticmethod
    def parse_instance():
        # (Reading from stdin omitted for brevity)
        from sys import stdin
        board = []
        for line in stdin:
            if line.strip()=="":
                continue
            board.append(line.strip().split())
        return Board(board)

    def region_actions(self, region_id):
        actions = []
        letters = list(TETRAMINOS.keys())
        positions = self.get_region_positions(region_id)
        pos_set = set(positions)  # O(1) membership tests:contentReference[oaicite:6]{index=6}
        for letter in letters:
            for index, shape in enumerate(TETRAMINOS[letter]):
                for (ai,aj) in positions:
                    # Compute absolute positions for this shape
                    shape_abs = []
                    valid = True
                    for (ri,rj) in shape:
                        x, y = ai+ri, aj+rj
                        if 0 <= x < self.n and 0 <= y < self.n:
                            shape_abs.append((x,y))
                        else:
                            valid = False
                            break
                    if valid and all(pos in pos_set for pos in shape_abs):
                        actions.append((region_id, letter, shape, index, shape_abs))
        return actions

class Nuruomino(Problem):
    def __init__(self, board: Board):
        self.board = board
        self.regions = board.regions
        initial_state = NuruominoState(board)
        super().__init__(initial_state)

    def actions(self, state: NuruominoState):
        if not state.unfinished_regions:
            return []
        # Most-restricted region first
        best = min(state.unfinished_regions, key=lambda r: len(self.board.region_actions_list[r]))
        candidates = self.board.region_actions_list[best]
        valid_actions = []
        for action in candidates:
            if self.is_valid_action(action, state):
                if len(state.unfinished_regions) == 1:
                    # Last region: check goal (pruned)
                    new_state = self.result(state, action)
                    if not self.goal_test(new_state):
                        continue
                else:
                    new_state = self.result(state, action)
                    if not self._has_future_solutions(new_state, best):
                        continue
                valid_actions.append(action)
        return valid_actions

    def _has_future_solutions(self, state, filled_region):
        adj_regions = state.board.adjacent_regions(filled_region)
        for adj in adj_regions:
            if adj in state.unfinished_regions:
                # Use updated actions list from state.board (faster pruning)
                actions = state.board.region_actions_list[adj]  # **Optimized**: use updated list
                if not any(self.is_valid_action(act, state) for act in actions):
                    return False
        return True

    def result(self, state: NuruominoState, action):
        region_id, piece_letter, shape, index, shape_abs = action
        old_board = state.board.board
        # Copy board rows
        new_board_data = [row[:] for row in old_board]
        # Place the piece
        for (i,j) in shape_abs:
            new_board_data[i][j] = piece_letter
        # Shallow-copy Board and lists of actions
        new_board = copy.copy(state.board)
        new_board.board = new_board_data
        new_board.regiao_original = state.board.regiao_original
        # Copy region_actions_list (each list shallow-copied)
        new_board.region_actions_list = {rid: acts[:] 
                                          for rid, acts in state.board.region_actions_list.items()}
        # Remove filled region
        if region_id in new_board.region_actions_list:
            del new_board.region_actions_list[region_id]
        # Update adjacent regions' actions
        self._update_adjacent_actions(new_board, region_id, piece_letter, shape_abs)
        # Re-sort remaining regions by new action count
        remaining = [r for r in new_board.regions if r in new_board.region_actions_list]
        remaining.sort(key=lambda r: len(new_board.region_actions_list[r]))
        new_board.regions = remaining
        # Track states (unchanged)
        if len(LAST_STATES) >= 10:
            LAST_STATES.pop(0)
        LAST_STATES.append(new_board)
        return NuruominoState(new_board)


    def _update_adjacent_actions(self, board, filled_region, piece_letter, placed_positions):
        adj_regions = board.adjacent_regions(filled_region)
        placed_set = set(placed_positions)
        for adj in adj_regions:
            if adj not in board.region_actions_list:
                continue
            old_actions = board.region_actions_list[adj]
            new_actions = []
            for action in old_actions:
                (_, act_piece, _, _, act_pos) = action
                if any(pos in placed_set for pos in act_pos):
                    # If overlapping with the new piece, re-check validity
                    if self._is_action_still_valid(action, piece_letter, placed_positions, board):
                        new_actions.append(action)
                    # else drop it
                else:
                    new_actions.append(action)
            board.region_actions_list[adj] = new_actions


    def _is_action_still_valid(self, action, new_piece_letter, new_positions, board):
        (region_id, piece_letter, shape, index, shape_abs) = action
        # Reuse existing checks
        if self._would_create_2x2_block(shape_abs, board): return False
        if self._would_touch_equal_piece(shape_abs, piece_letter, board): return False
        # New check: two same-letter pieces touching from diff regions
        if piece_letter == new_piece_letter:
            for (i1,j1) in shape_abs:
                for (i2,j2) in new_positions:
                    if abs(i1-i2)+abs(j1-j2) == 1:
                        if board.regiao_original[i1][j1] != board.regiao_original[i2][j2]:
                            return False
        return True

    def is_valid_action(self, action, state: NuruominoState):
        (region_id, piece_letter, shape, index, shape_abs) = action
        board = state.board
        if self._would_touch_equal_piece(shape_abs, piece_letter, board): 
            return False
        if self._would_create_2x2_block(shape_abs, board): 
            return False
        return True

    def _would_create_2x2_block(self, positions, board):
        # Check for any 2x2 block fully filled after placing this piece
        n = board.n
        grid = board.board  # direct reference
        pos_set = set(positions)
        for (i,j) in positions:
            for di in (0, -1):
                for dj in (0, -1):
                    # Coordinates of 2x2 square
                    x0, y0 = i+di, j+dj
                    if not (0 <= x0 < n-1 and 0 <= y0 < n-1):
                        continue
                    # Check 4 cells
                    if ((x0,y0) in pos_set or grid[x0][y0] in "LITS") and \
                       ((x0+1,y0) in pos_set or grid[x0+1][y0] in "LITS") and \
                       ((x0,y0+1) in pos_set or grid[x0][y0+1] in "LITS") and \
                       ((x0+1,y0+1) in pos_set or grid[x0+1][y0+1] in "LITS"):
                        return True
        return False

    def _would_touch_equal_piece(self, positions, piece_letter, board):
        # Check orthogonal neighbors for same letter in a different region
        pos_set = set(positions)
        grid = board.board
        for (i,j) in positions:
            for (di,dj) in ORTHOGONAL_DIRS:
                ni, nj = i+di, j+dj
                if 0 <= ni < board.n and 0 <= nj < board.n and (ni,nj) not in pos_set:
                    if grid[ni][nj] == piece_letter:
                        # Only invalid if different original region
                        if board.regiao_original[i][j] != board.regiao_original[ni][nj]:
                            return True
        return False

    def _has_no_2x2_blocks(self, board):
        # Final check for any 2x2 block of letters
        grid = board.board
        n = board.n
        for i in range(n-1):
            for j in range(n-1):
                if grid[i][j] in "LITS" and grid[i+1][j] in "LITS" \
                   and grid[i][j+1] in "LITS" and grid[i+1][j+1] in "LITS":
                    return False
        return True

    def _is_connected(self, board, debug=False):
        # BFS to check all placed pieces form one connected component
        from collections import deque
        grid = board.board
        n = board.n
        visited = set()
        queue = deque()
        # Find first filled cell
        found = False
        for i in range(n):
            if found: break
            for j in range(n):
                if grid[i][j] in "LITS":
                    queue.append((i,j))
                    visited.add((i,j))
                    found = True
                    break
        # Traverse neighbors
        while queue:
            i,j = queue.popleft()
            current = grid[i][j]
            for (di,dj) in ORTHOGONAL_DIRS:
                ni, nj = i+di, j+dj
                if 0 <= ni < n and 0 <= nj < n:
                    val = grid[ni][nj]
                    if val in "LITS" and (ni,nj) not in visited:
                        visited.add((ni,nj))
                        queue.append((ni,nj))
        # Count all filled cells
        total = 0
        for i in range(n):
            for j in range(n):
                if grid[i][j] in "LITS":
                    total += 1
        return len(visited) == total

    def _no_same_piece_adjacent(self, board):
        grid = board.board
        n = board.n
        for i in range(n):
            for j in range(n):
                curr = grid[i][j]
                if curr not in "LITS": 
                    continue
                for (di,dj) in ORTHOGONAL_DIRS:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < n and 0 <= nj < n:
                        if grid[ni][nj] == curr:
                            # invalid if original region differs
                            if board.regiao_original[i][j] != board.regiao_original[ni][nj]:
                                return False
        return True

    def goal_test(self, state: NuruominoState):
        if state.unfinished_regions or not self._is_connected(state.board):
            return False
        return True


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
    # import cProfile
    # import pstats
    # with cProfile.Profile() as pr:
    #     main()
    # stats = pstats.Stats(pr)
    # stats.sort_stats("cumulative").print_stats(30)
    board = Board.parse_instance()

    tracemalloc.start()
    tic = time.perf_counter()

    problem = Nuruomino(board)
    # Criamos o novo estado inicial
    state = NuruominoState(board)
    problem.initial = state


    instrumented = InstrumentedProblem(problem)
    goal_node = depth_first_graph_search(instrumented)

    toc = time.perf_counter()
    print(f"  Programa executado em {toc - tic:0.4f} segundos")
    print(f"  Memória usada: {tracemalloc.get_traced_memory()[1] // 1024} kB")
    print(f"  Nós gerados: {instrumented.states}")
    print(f"  Nós expandidos: {instrumented.succs}")

    if goal_node:
        limpar_X(goal_node.state.board)  # Limpa os 'X' antes de imprimir
        goal_node.state.board.print_instance()
    # else:
    #     print("Nenhuma solução encontrada.")