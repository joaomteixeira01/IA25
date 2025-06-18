# Nuruomino AI Solver – Projeto IA 2024/2025

Este repositório contém a implementação da resolução automática do puzzle **Nuruomino** desenvolvida no âmbito da unidade curricular de **Inteligência Artificial (IA)**, ano letivo 2024/2025.

**Nota final do projeto: 16 valores**

## Grupo 34
- João Teixeira – 97226  
- Francisco Fialho – 110094

## Objetivo
O objetivo do projeto é resolver automaticamente puzzles do tipo **Nuruomino**, onde tetraminós devem ser colocados em regiões de um tabuleiro, respeitando regras específicas de posicionamento:

- Cada região deve conter exatamente uma peça tetraminó;
- Não podem existir blocos 2x2 preenchidos;
- Peças idênticas não se podem tocar ortogonalmente;
- O grafo das regiões deve manter-se conexo após cada jogada.

A resolução é feita através de **procura em profundidade** com técnicas de **forward checking** e **restrição de domínios**.

---

**Comando para correr o projeto**
  python3 src/nuruomino.py < public-tests/test04.txt (Para correr o teste publico 4 neste caso)
  ***NOTA: necessario ter o python 3.7+***

## 📁 Estrutura do Projeto

```bash
.
├── public-tests/           # Casos de teste fornecidos (.txt e .out)
│   ├── imagens/            # Visualizações dos testes
│   ├── testXX.txt          # Ficheiros de input
│   └── testXX.out          # Ficheiros de output esperado
├── src/
│   ├── nuruomino.py        # Ficheiro principal com a lógica do puzzle
│   ├── search.py           # Implementação de algoritmos de procura
│   └── utils.py            # Funções auxiliares 
├── tests/                  # Testes adicionais 
├── .gitignore
└── IA_24_25__Projeto_Nuruomino_2_Maio.pdf  # Enunciado do projeto
