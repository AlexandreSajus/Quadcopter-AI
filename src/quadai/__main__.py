"""
2D Quadcopter AI by Alexandre Sajus

More information at:
https://github.com/AlexandreSajus/2D-Quadcopter-AI

This is the main game where you can compete with AI agents
Collect as many balloons within the time limit
"""

import sys
import time
import quadai
from quadai.balloon import balloon
from quadai.snowglobe import snowglobe


def main(game: str = "balloon") -> None:
    """
    Runs the selected game.

    Args:
        game (str): The game to run (balloon, snowglobe)
    """
    if game == "balloon":
        return balloon()
    elif game == "snowglobe":
        return snowglobe()
    else:
        print(f"Unknown tracking library: {game} (expected: balloon or snowglobe)")


if __name__ == "__main__":
    print("")
    print("---------------------------------------------------")
    print("")
    print("2D Quadcopter AI by Alexandre Sajus")
    print("More information at: https://github.com/AlexandreSajus/Quadcopter-AI")
    print("")
    print("Please select a game:")
    print(
        "1. Balloon: Control a drone with the arrow keys to collect as many balloons as possible"
    )
    print("2. Snowglobe: Control a drone with the mouse to move snow particles around")
    print("")
    print("Enter 1 or 2 to select a game:")
    game = input()
    print("")
    print("Launching game... (Don't forget to click on the game window to activate it)")
    time.sleep(3)
    if game == "1":
        chosen_game = "balloon"
    elif game == "2":
        chosen_game = "snowglobe"
    else:
        print("Invalid game choice")
        sys.exit(1)

    while True:
        ended = main(chosen_game)
        print("")
        print("Game Over")
        print("")
        print("Enter 1 or 2 to select a game:")
        game = input()
        if game == "1":
            chosen_game = "balloon"
        elif game == "2":
            chosen_game = "snowglobe"
        else:
            print("Invalid game choice")
            sys.exit(1)
