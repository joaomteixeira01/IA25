# Nuruomino - AI Project 2024/2025

This repository contains the implementation of an automatic solver for the **Nuruomino** puzzle, developed as part of the **Artificial Intelligence** course (2024/2025 academic year).

**Final project grade: 16 out of 20**

## Group 34
- João Teixeira – 97226  
- Francisco Fialho – 110094

## Objective
The goal of this project is to automatically solve **Nuruomino** puzzles, where tetrominoes must be placed in distinct regions of a board while respecting specific placement rules:

- Each region must contain exactly one tetromino piece;
- No filled 2x2 square blocks are allowed;
- Identical pieces cannot touch orthogonally;
- The region graph must remain connected after each move.

The solution is achieved using **depth-first search** combined with **forward checking** and **domain restriction** techniques.

---

**Command to run the project**  
```bash
python3 src/nuruomino.py < public-tests/test04.txt  # Example: runs public test 4
```
***NOTE: Requires Python 3.7+***

---
**Project structure**
```bash
.
├── public-tests/           # Provided test cases (.txt input and .out expected output)
│   ├── imagens/            # Visual representations of test cases
│   ├── testXX.txt          # Input files
│   └── testXX.out          # Expected output files
├── src/
│   ├── nuruomino.py        # Main file containing puzzle-solving logic
│   ├── search.py           # Implementation of search algorithms
│   └── utils.py            # Auxiliary functions 
├── tests/                  # Additional test cases 
├── .gitignore
└── IA_24_25__Projeto_Nuruomino_2_Maio.pdf  # Original project description (in Portuguese)
