import copy
import math
import random
from math import log, sqrt, inf
from random import randrange
import numpy as np
from rich.table import Table
from rich.progress import track
from rich.console import Console
from rich.progress import Progress

import classes.logic as logic

# When implementing a new strategy add it to the `str2strat`
# dictionary at the end of the file

class PlayerStrat:
    def __init__(self, _board_state, player):
        self.root_state = _board_state
        self.player = player

    def start(self):
        """
        This function select a tile from the board.

        @returns    (x, y) A tuple of integer corresponding to a valid
                    and free tile on the board.
        """
        raise NotImplementedError

class Node(object):
    """
    This class implements the main object that you will manipulate : nodes.
    Nodes include the state of the game (i.e. the 2D board), children (i.e. other children nodes), a list of
    untried moves, etc...
    """
    def __init__(self, board, move=(None, None),
                 wins=0, visits=0, children=None):
        # Save the #wins:#visited ratio
        self.state = board
        self.move = move
        self.wins = wins
        self.visits = visits
        self.children = children or []
        self.parent = None
        self.untried_moves = logic.get_possible_moves(board)

    def add_child(self, child):
        child.parent = self
        self.children.append(child)


class Random(PlayerStrat):
    def start(self):
        available_tiles = []
        board_size = len(self.root_state)  # Obtenir la taille du tableau

        for x in range(board_size):
            for y in range(board_size):
                if self.root_state[x, y] == 0:  # Vérifier si la tuile est vide (représentée par 0)
                    available_tiles.append((x, y))

        if available_tiles:
            return random.choice(available_tiles)
        else:
            return None



class MiniMax(PlayerStrat):
    def __init__(self, _board_state, player, max_depth=5):
        super().__init__(_board_state, player)
        self.max_depth = max_depth

    def start(self):
        best_move, best_value = self.minmax(self.root_state, self.max_depth, -float('inf'), float('inf'), True)
        print(f"Je suis le Joueur {self.player}, mon placement {best_move}")
        return best_move

    def minmax(self, board, depth, alpha, beta, max_player):
        if depth == 0 or logic.is_game_over(self.player, board) or logic.is_game_over(3 - self.player, board):
            evaluation = self.evaluate(board)
            return None, evaluation if evaluation is not None else (-float('inf') if max_player else float('inf'))

        if max_player:
            best_value = -math.inf
            best_move = None
            for move in logic.get_possible_moves(board):
                new_board = np.copy(board)
                new_board[move[0], move[1]] = self.player
                _, value = self.minmax(new_board, depth - 1, alpha, beta, False)
                new_board[move[0], move[1]] = 0 
                if value > best_value:
                    best_value = value
                    best_move = move
            return best_move, best_value  

        else:
            best_value = math.inf
            best_move = None
            for move in logic.get_possible_moves(board):
                new_board = np.copy(board)
                new_board[move[0], move[1]] = 3 - self.player
                _, value = self.minmax(new_board, depth - 1, alpha, beta, True)
                new_board[move[0], move[1]] = 0  
                if value < best_value:
                    best_value = value
                    best_move = move
            return best_move, best_value

    def evaluate(self, board):
        if logic.is_game_over(self.player, board):
            return 1
        elif logic.is_game_over(3 - self.player, board):
            return -1
        else:
            player_pieces = np.count_nonzero(board == self.player)
            opponent_pieces = np.count_nonzero(board == 3 - self.player)
            return player_pieces - opponent_pieces
        
class MiniMaxAlphaBeta(PlayerStrat):
    def __init__(self, _board_state, player, max_depth=100):
        super().__init__(_board_state, player)
        self.max_depth = max_depth

    def start(self):
        best_move, best_value = self.alphabeta(self.root_state, self.max_depth, -float('inf'), float('inf'), True)
        print(f"Je suis le Joueur {self.player}, mon placement {best_move}")
        return best_move

    def alphabeta(self, board, depth, alpha, beta, max_player):
        if depth == 0 or logic.is_game_over(self.player, board) or logic.is_game_over(3 - self.player, board):
            return None, self.evaluate(board) 
        # if self.evaluate(board) is not None else (-float('inf') if max_player else float('inf'))

        if max_player:
            best_value = -math.inf
            best_move = None
            for move in logic.get_possible_moves(board):
                new_board = np.copy(board)
                new_board[move[0], move[1]] = self.player
                _, value = self.alphabeta(new_board, depth - 1, alpha, beta, False)
                new_board[move[0], move[1]] = 0 
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return best_move, best_value
        else:
            best_value = math.inf
            best_move = None
            for move in logic.get_possible_moves(board):
                new_board = np.copy(board)
                new_board[move[0], move[1]] = 3 - self.player
                _, value = self.alphabeta(new_board, depth - 1, alpha, beta, True)
                new_board[move[0], move[1]] = 0  
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return best_move, best_value

    def evaluate(self, board):
            if logic.is_game_over(self.player, board):
                return 1
            elif logic.is_game_over(3 - self.player, board):
                return -1
    def count_pieces(self, board):
        return sum(1 for x in range(len(board)) for y in range(len(board[x])) if board[x][y] == self.player)
    
        

class MyStrategyPlayer(PlayerStrat):
    def __init__(self, _board_state, player):
        super().__init__(_board_state, player)
        self._board_state = _board_state  # Initialisation de self._board_state

    def start(self):
        # Votre implémentation spécifique pour cette stratégie
        # Ici, nous supposons une implémentation basique qui choisit la première tuile libre du plateau
        for x in range(self._board_state):
            for y in range(self._board_state):
                if self.board.is_empty(x, y):
                    return x, y

str2strat: dict[str, PlayerStrat] = {
        "human": None,
        "random": Random,
        "minimax": MiniMax,
        "my_new_ai_strat":MyStrategyPlayer,
        "minimax_ab": MiniMaxAlphaBeta,
}