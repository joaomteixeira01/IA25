# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 34:
# 97226 João Teixeira 
# 110094 Francisco Fialho

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
        """ Método utilizado para resolver empates na lista de estados abertos durante procuras informadas. """
        return self.id < other.id

class Board:
    """Classe que representa internamente um tabuleiro do Puzzle Nuruomino."""

    def __init__(self, board: np.ndarray, regions: dict, assigned = []):
        """Inicialização da classe Board.

        :param board: Array numpy representando o tabuleiro.
        :param rows: Linhas do tabuleiro.
        :param cols: Colunas do tabuleiro.
        :param regions: Dicionário contendo as regiões do tabuleiro.
        """
        self.board = board
        self.rows, self.cols = board.shape
        self.regions = regions
        self.assigned = assigned
        self.invalid = False

    @staticmethod
    def parse_instance():
        """Interpreta o input do standard input (stdin) passado como argumento
        e devolve uma instância da classe Board.

        Exemplo de uso:
            $ python3 pipe.py < test-01.txt

            > from sys import stdin
            > line = stdin.readline().split()
        """
        matriz_tabuleiro = np.array([line.split() for line in stdin.readlines()])
        num_linhas, num_colunas = matriz_tabuleiro.shape
        mapa_regioes = {}
        for linha in range(num_linhas):
            for coluna in range(num_colunas):
                identificador = matriz_tabuleiro[linha][coluna]
                if identificador not in mapa_regioes:
                    mapa_regioes[identificador] = {'cells': [(linha, coluna)], 'adjacents': set(), 'domain': [], 'action': None}
                else:
                    mapa_regioes[identificador]['cells'].append((linha, coluna))
        instancia_tabuleiro = Board(matriz_tabuleiro, mapa_regioes)
        for id_regiao, informacao in instancia_tabuleiro.regions.items():
            instancia_tabuleiro.compute_possible_actions(id_regiao)
            instancia_tabuleiro.find_neighboring_regions(id_regiao)
        return instancia_tabuleiro
        
    def duplicate_board(self):
        """Gera uma cópia do tabuleiro atual."""
        novo_tabuleiro = np.copy(self.board)
        novas_regioes = {}
        for id_regiao, informacao in self.regions.items():
            novas_regioes[id_regiao] = {
                'cells': informacao['cells'].copy(),
                'adjacents': informacao['adjacents'].copy(),
                'domain': informacao['domain'].copy(),
                'action': informacao['action']
            }
        nova_lista_atribuida = self.assigned.copy()
        return Board(novo_tabuleiro, novas_regioes, nova_lista_atribuida)
    
    def get_neighboring_regions(self, regiao:int) -> list:
        """Retorna uma lista das regiões fronteiriças com a região especificada."""
        #TODO
        pass
    
    def get_adjacent_coordinates(self, linha:int, coluna:int) -> list:
        """Retorna as coordenadas adjacentes à posição, incluindo diagonais."""
        #TODO
        pass

    def get_neighbor_values(self, linha:int, coluna:int) -> list:
        """Retorna os valores das células vizinhas à posição, incluindo diagonais."""
        #TODO
        pass

    def analyze_piece_environment(self, nome_forma:str, posicao:tuple, id_regiao, original):

        linha, coluna = posicao
        formas_adjacentes = set()  # ids das regiões com peças vizinhas
        regioes_vizinhas = set()  # ids das regiões adjacentes
        direcoes = [(1, 0), (-1, 0), (0, -1), (0, 1)]
        forma = SHAPES[nome_forma]
        linhas_forma, colunas_forma = forma.shape

        for i in range(linhas_forma):
            for j in range(colunas_forma):
                if forma[i][j]:
                    linha_tabuleiro = linha + i
                    coluna_tabuleiro = coluna + j
                    
                    for dl, dc in direcoes:
                        linha_destino = linha_tabuleiro + dl
                        coluna_destino = coluna_tabuleiro + dc
                        if (0 <= linha_destino < self.rows and 0 <= coluna_destino < self.cols):
                            valor_adjacente = self.board[linha_destino][coluna_destino]
                            id_regiao_adjacente = original[linha_destino][coluna_destino]
                            
                            # Ignorar células da mesma região
                            if id_regiao_adjacente == id_regiao:
                                continue
                            
                            # Se é uma letra (peça posicionada), adicionar às shapes conectadas
                            if valor_adjacente.isalpha() and valor_adjacente != nome_forma[0]:
                                formas_adjacentes.add(id_regiao_adjacente)
                            
                            # Adicionar todas as regiões adjacentes
                            regioes_vizinhas.add(id_regiao_adjacente)

        return formas_adjacentes, regioes_vizinhas
    
    def verify_adjacent_similarity(self, nome_forma:str, posicao:tuple) -> list:
        """Verifica se existe uma forma igual adjacente à posição especificada."""
        linha, coluna = posicao
        direcoes = [(linha-1, coluna), (linha+1, coluna), (linha, coluna-1), (linha, coluna+1)]
        for linha_destino, coluna_destino in direcoes:
            if (0 <= linha_destino < self.rows and 0 <= coluna_destino < self.cols):
                forma_adjacente = self.board[linha_destino][coluna_destino]
                if nome_forma[0] == forma_adjacente: return True
        return False
    
    def detect_square_formation(self, nome_forma: str, posicao: tuple):
        """Versão melhorada que verifica apenas a área impactada pela peça"""
        forma = SHAPES[nome_forma]
        linhas_forma, colunas_forma = forma.shape
        linha, coluna = posicao
        
        linha_min = max(0, linha - 1)
        linha_max = min(self.rows - 1, linha + linhas_forma)
        coluna_min = max(0, coluna - 1)
        coluna_max = min(self.cols - 1, coluna + colunas_forma)
        
        for r in range(linha_min, linha_max):
            for c in range(coluna_min, coluna_max):
                if r + 1 < self.rows and c + 1 < self.cols:
                    quadrado_preenchido = True
                    for dr in range(2):
                        for dc in range(2):
                            linha_celula, coluna_celula = r + dr, c + dc
                            valor_celula = self.board[linha_celula, coluna_celula]
                            
                            # Verificar se a célula será preenchida
                            if valor_celula.isdigit():
                                # Verificar se a nova peça cobre esta célula
                                if (linha <= linha_celula < linha + linhas_forma and 
                                    coluna <= coluna_celula < coluna + colunas_forma):
                                    linha_forma, coluna_forma = linha_celula - linha, coluna_celula - coluna
                                    if not forma[linha_forma, coluna_forma]:
                                        quadrado_preenchido = False
                                        break
                                else:
                                    quadrado_preenchido = False
                                    break
                        if not quadrado_preenchido:
                            break
                    
                    if quadrado_preenchido:
                        return True
        
        return False
    
    def validate_placement_rules(self, nome_forma, posicao, quadrados:bool):
        #if quadrados: return not self.detect_square_formation(nome_forma, posicao)
        forma = SHAPES[nome_forma]
        for r in range(forma.shape[0]):
            for c in range(forma.shape[1]):
                if not forma[r, c]: continue
                tem_adjacente = self.verify_adjacent_similarity(nome_forma, (r+posicao[0], c+posicao[1]))
                if tem_adjacente:
                    return False
        return not self.detect_square_formation(nome_forma, posicao)
    
    def verify_graph_connectivity(self):
        nos = set(r for r in self.regions.keys())

        regiao = next(iter(nos))
        fila = [regiao]
        visitados = {regiao}

        while fila:
            atual = fila.pop(0)
            for adjacente in self.regions[atual]['adjacents']:
                if adjacente in nos and adjacente not in visitados:
                    visitados.add(adjacente)
                    fila.append(adjacente)
                    
        if len(visitados) < len(nos):
            return False
        
        return True

    def extract_shape_window(self, forma, posicao):
        # Localiza a janela do tabuleiro correspondente à forma na posição especificada.
        indices_forma = np.argwhere(forma)
        ancora_c, ancora_r = indices_forma[0]
        posicao_c = posicao[0] - ancora_c
        posicao_r = posicao[1] - ancora_r
        linhas_forma, colunas_forma = forma.shape

        if (posicao_c < 0 or posicao_r < 0 or linhas_forma > (self.rows - posicao_c) or colunas_forma > (self.cols - posicao_r)):
            return None
        janela = self.board[posicao_c:posicao_c+linhas_forma, posicao_r:posicao_r+colunas_forma]
        return janela, (posicao_c, posicao_r)

    def find_neighboring_regions(self, id_regiao):
        dados_regiao = self.regions[id_regiao]
        for celula in dados_regiao['cells']:
            linha, coluna = celula
            direcoes = [(linha-1, coluna), (linha+1, coluna), (linha, coluna-1), (linha, coluna+1)]
            for linha_destino, coluna_destino in direcoes:
                if 0 <= linha_destino < self.rows and 0 <= coluna_destino < self.cols:
                    id_adjacente = self.board[linha_destino][coluna_destino]
                    if id_adjacente != id_regiao:
                        dados_regiao['adjacents'].add(id_adjacente)

    def compute_possible_actions(self, id_regiao):
        dados_regiao = self.regions[id_regiao]
        for celula in dados_regiao['cells']:
            for nome_forma, forma_bruta in SHAPES.items():
                forma = forma_bruta
                info_correspondencia = self.extract_shape_window(forma, celula)
                if info_correspondencia is None:
                    continue
                janela, posicionamento = info_correspondencia
                if np.all(forma <= (janela == id_regiao)):
                    acao = Action(nome_forma, posicionamento, id_regiao)
                    _, conectados = self.analyze_piece_environment(acao.shape_name, acao.board_position, id_regiao, self.board)
                    acao.influence_count = len(conectados)
                    dados_regiao['domain'].append(acao)

    def remove_connection(self, id_regiao, id_adjacente):
        self.regions[id_regiao]['adjacents'].discard(id_adjacente)
        self.regions[id_adjacente]['adjacents'].discard(id_regiao)

    def restrict_action_domain(self, id_regiao, quadrados=False):
        dados_regiao = self.regions[id_regiao]
        para_remover = []
        for acao in dados_regiao['domain']:
            if not self.validate_placement_rules(acao.shape_name, acao.board_position, quadrados):
                para_remover.append(acao)
        if dados_regiao['action'] is None and len(dados_regiao['domain']) == 0:
            self.invalid = True
        for acao in para_remover:
            dados_regiao['domain'].remove(acao)

    def apply_constraints(self, acao: Action, tabuleiro_inicial):
        id_regiao = acao.region_id
        dados_regiao = self.regions[id_regiao]
        dados_regiao['action'] = acao
        formas_conectadas, regioes_conectadas = self.analyze_piece_environment(acao.shape_name, acao.board_position, id_regiao, tabuleiro_inicial)
        conexoes_para_cortar = []
        
        for id_adjacente in list(dados_regiao['adjacents']):
            dados_adjacente = self.regions[id_adjacente]
            if dados_adjacente['action'] is None:
                if id_adjacente not in regioes_conectadas:
                    conexoes_para_cortar.append(id_adjacente)
                    continue
            else:
                if id_adjacente not in formas_conectadas:
                    conexoes_para_cortar.append(id_adjacente)
                    continue

            if id_adjacente not in self.assigned:
                self.restrict_action_domain(id_adjacente)

        for c in conexoes_para_cortar:
            self.remove_connection(id_regiao, c)
        if not self.verify_graph_connectivity():
            self.invalid = True

class Nuruomino(Problem):
    contador = 0
    def __init__(self, board: Board):
        """O construtor define o estado inicial."""
        self.initial = NuruominoState(board)

    def choose_optimal_region(self, estado: NuruominoState):
        regioes = [dados for id, dados in estado.board.regions.items() if id not in estado.board.assigned]
        if not regioes:
            return None
        def pontuacao_regiao(dados_regiao):
            tamanho_dominio = len(dados_regiao['domain'])
            adjacentes_vazios = sum(1 for id_adj in dados_regiao['adjacents'] if id_adj not in estado.board.assigned)
            return (tamanho_dominio, -adjacentes_vazios)

        melhor_regiao = min(regioes, key=pontuacao_regiao)
        return melhor_regiao
    
    def order_actions_by_priority(self, acoes):
        def pontuacao_acao(acao: Action):
            return (acao.influence_count)
        return sorted(acoes, key=pontuacao_acao)

    def actions(self, state: NuruominoState):
        """Devolve uma lista de ações executáveis a partir do estado fornecido."""
        if state.board.invalid: 
            return []
        melhor_regiao = self.choose_optimal_region(state)
        if melhor_regiao is None:
            return []
        acoes = self.order_actions_by_priority(melhor_regiao['domain'])
        return acoes

    def result(self, state: NuruominoState, action: Action):
        """Devolve o estado resultante da execução da 'action' sobre o 'state' fornecido.
        A ação deve ser uma das presentes na lista obtida por self.actions(state)."""
        novo_tabuleiro = state.board.duplicate_board()
        nome_forma = action.shape_name
        posicao_c, posicao_r = action.board_position
        id_regiao = action.region_id
        forma = SHAPES[nome_forma]
        linhas_forma, colunas_forma = forma.shape
        
        # Posicionar peça no tabuleiro
        for i in range(linhas_forma):
            for j in range(colunas_forma):
                if forma[i, j]:
                    novo_tabuleiro.board[posicao_c + i, posicao_r + j] = nome_forma[0]
        
        # Aplicar restrições
        novo_tabuleiro.apply_constraints(action, self.initial.board.board)
        novo_tabuleiro.assigned.append(id_regiao)
        
        novo_estado = NuruominoState(novo_tabuleiro)
        return novo_estado
        
    def goal_test(self, state: NuruominoState): 
        """Retorna True se o estado fornecido é um estado objetivo.
        Verifica se todas as posições do tabuleiro estão preenchidas seguindo as regras."""
        if len(state.board.assigned) == len(state.board.regions.keys()):
            Nuruomino.contador += 1
        return len(state.board.assigned) == len(state.board.regions.keys()) and not state.board.invalid

    def h(self, node: Node):
        """Função heurística para a procura A*."""
        estado = node.state
        tabuleiro = estado.board
        
        regioes_nao_preenchidas = len(tabuleiro.regions) - len(tabuleiro.assigned)
        pecas_restantes = sum(len(dados['domain']) for id, dados in tabuleiro.regions.items() if id not in tabuleiro.assigned)

        if estado.board.invalid:
            return float('inf')

        if regioes_nao_preenchidas > 2:
            regioes_nao_preenchidas += 100
        
        return regioes_nao_preenchidas + pecas_restantes

if __name__ == "__main__":
    import time
    import tracemalloc
    board = Board.parse_instance()
    # tracemalloc.start()
    # tic = time.perf_counter()
    problem = Nuruomino(board)
    result = sh.depth_first_tree_search(problem)
    # #result = sh.astar_search(problem, h=problem.h, display=False)
    # toc = time.perf_counter()
    # print(f"  Programa executado em {toc - tic:0.4f} segundos")
    # print(f"  Memória usada: {tracemalloc.get_traced_memory()[1] // 1024} kB")

    if not result:
        print("No solution found.", end="")
    else:
        for i, row in enumerate(result.state.board.board):
            if i == len(result.state.board.board) - 1:
                print('\t'.join(row), end="")
            else:
                print('\t'.join(row))