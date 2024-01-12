import os
import pickle
import argparse
import logging
from rich import print
from rich.logging import RichHandler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from classes.strategy import MiniMax, Random,McPlayer,MiniMaxAlphaBeta
import pygame

# Hide Pygame welcome message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame
import pandas as pd

from classes.logic import player2str
from classes.game import Game

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()]
)


class Tournament:
    def __init__(self, args: list):
        """
        Initialises a tournament with:
           * the size of the board,
           * the players strategies, eg., ("human", "random"),
           * the game counter,
           * the number of games to play.
        """
        self.args = args
        (self.BOARD_SIZE, self.STRAT, self.GAME_COUNT,
         self.N_GAMES, self.USE_UI) = args

        if self.USE_UI:
            pygame.init()
            pygame.display.set_caption("Polyline")

    def single_game(self, black_starts: bool=True) -> int:
        """
        Runs a single game between two opponents.

        @return   The number of the winner, either 1 or 2, for black
                  and white respectively.
        """

        game = Game(board_size=self.BOARD_SIZE,
                    black_starts=black_starts, 
                    strat=self.STRAT,
                    use_ui=self.USE_UI)
        game.print_game_info(
            [self.BOARD_SIZE, self.STRAT, self.GAME_COUNT]
        )
        while game.winner is None:
            game.play()

        print(f"{player2str[game.winner]} player ({self.STRAT[game.winner-1]}) wins!")

        return game.winner

    def championship(self):
        """
        Runs a number of games between the same two opponents.
        """
        tournament_results = pd.DataFrame()  

        scores = [0, 0]

        for _ in range(self.N_GAMES):
            self.GAME_COUNT = _

            # First half of the tournament started by one player.
            # Remaining half started by other player (see "no pie
            #  rule")
            winner = self.single_game(
                black_starts=self.GAME_COUNT < self.N_GAMES / 2
            )
            scores[winner-1] += 1

        log = logging.getLogger("rich")

        # TODO Design your own evaluation measure!
        # https://pyformat.info/
        log.info("Design your own evaluation measure!")
        # print(scores)

         # Calcul des pourcentages de victoire
        total_games = sum(scores)
        win_percentages = [score / total_games for score in scores]

       
        parser = argparse.ArgumentParser()
        parser.add_argument("--player", help="Name of the first player")
        parser.add_argument("--other", help="Name of the second player")
        parser.add_argument("--size", type=int, help="Size of the board")
        args = parser.parse_args()

                # Use the command line arguments in your data dictionary
        names = [args.player, args.other]
        size = self.BOARD_SIZE 
        data = {
            'Joueur1': [names[0]],
            'Joueur2': [names[1]],
            'size': [size],
            '%win_Joueur1': [win_percentages[0]],
            '%win_Joueur2': [win_percentages[1]]
        }
        tournament_results = pd.concat([tournament_results, pd.DataFrame(data)])

        tournament_results = tournament_results[["Joueur1", "Joueur2", "%win_Joueur1", "%win_Joueur2", "size"]]
        tournament_results.to_csv('tournament_results.csv', sep=',', index=False, mode='a', header=False)
        # Read the CSV file into a DataFrame
        df = pd.read_csv("tournament_results.csv")

        print(df)
       
    
        matrix = df.pivot_table(index="Joueur1", columns="Joueur2", values=["%win_Joueur1", "%win_Joueur2"])

        fig, ax = plt.subplots()
        sns.heatmap(matrix, annot=True, fmt=".2f", ax=ax, cmap='coolwarm')

        ax.set_title('Pourcentage de victoires par joueur')
        ax.set_xlabel('Joueur2')
        ax.set_ylabel('Joueur1')

        plt.show()
