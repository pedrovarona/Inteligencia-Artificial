# This code is derived from the AIMA Python repository
# Original source: https://github.com/aimacode/aima-python/
# Licensed under MIT License
# Copyright (c) 2016 aima-python contributors

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional


class GameState(ABC):
    """Abstract base class defining the interface for game states."""

    @abstractmethod
    def get_legal_moves(self) -> list[Any]:
        """
        Return a list of all legal moves from the current state.

        :return: List of all legal moves available from this state
        :rtype: list[Any]
        """
        pass

    @abstractmethod
    def make_move(self, move: Any) -> "GameState":
        """
        Apply a move to the current state and return the resulting state.

        :param move: The move to apply
        :type move: Any
        :return: The new game state after applying the move
        :rtype: GameState
        :raises ValueError: If the move is not legal
        """
        pass

    @abstractmethod
    def is_over(self) -> bool:
        """
        Return True if the game is finished, False otherwise.

        :return: True if the game has ended, False if still in progress
        :rtype: bool
        """
        pass

    @abstractmethod
    def get_current_player(self) -> Any:
        """
        Return the identifier of the player whose turn it is.

        :return: The current player's identifier
        :rtype: Any
        """
        pass

    @abstractmethod
    def get_winner(self) -> Optional[Any]:
        """
        Return the winner of the game if it's over, None otherwise.

        :return: The player identifier of the winner, draw, or None if game is not over
        :rtype: Optional[Any]
        """
        pass

    @abstractmethod
    def copy(self) -> "GameState":
        """
        Return a deep copy of the game state.

        :return: A deep copy of this game state
        :rtype: GameState
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a string representation of the game state.

        :return: String representation of the current game state
        :rtype: str
        """
        pass


class GameProblem(ABC):
    """
    A game-problem is similar to a problem, but it has a terminal test instead of
    a goal test, and a utility for each terminal state. To create a game,
    subclass this class and implement `utility`. You will also need to set
    the `.initial` attribute to the initial state; this can be done in the
    constructor.
    """

    def actions(self, state: GameState) -> list[Any]:
        """
        Return a collection of the allowable moves from this state.

        :param state: The current game state
        :type state: GameState
        :return: List of legal moves available from the given state
        :rtype: list[Any]
        """
        return state.get_legal_moves()

    def result(self, state: GameState, move: Any) -> GameState:
        """
        Return the state that results from making a move from a state.

        :param state: The current game state
        :type state: GameState
        :param move: The move to apply
        :type move: Any
        :return: The new game state after applying the move
        :rtype: GameState
        """
        return state.copy().make_move(move)

    def is_terminal(self, state: GameState) -> bool:
        """
        Return True if this is a final state for the game.

        :param state: The game state to check
        :type state: GameState
        :return: True if the game has ended, False otherwise
        :rtype: bool
        """
        return state.is_over()

    @abstractmethod
    def utility(self, state: GameState, player: Any) -> float:
        """
        Return the value of this final state to player.

        :param state: The terminal game state
        :type state: GameState
        :param player: The player for whom to calculate utility
        :type player: Any
        :return: The utility value for the specified player
        :rtype: float
        """
        pass


def play_game(
    game: GameProblem,
    strategies: dict[Any, Callable[[GameProblem, GameState], Any]],
    verbose: bool = False,
) -> GameState:
    """
    Play a turn-taking game using the provided strategies for each player.

    :param game: The game problem instance containing initial state and rules
    :type game: GameProblem
    :param strategies: Dictionary mapping player IDs to strategy functions.
                      Each strategy function takes a GameProblem and GameState
                      as arguments and returns a move
    :type strategies: Dict[Any, Callable[[GameProblem, GameState], Any]]
    :param verbose: If True, print moves and game states during play
    :type verbose: bool
    :return: The final terminal state of the completed game
    :rtype: GameState

    .. note::
       - The game continues until a terminal state is reached
       - Strategy functions are called with signature `function(game, state)`
       - If verbose is True, each move and resulting state will be printed
    """
    state = game.initial
    while not game.is_terminal(state):
        player = state.get_current_player()
        move = strategies[player](game, state)
        state = game.result(state, move)
        if verbose:
            print("Player", player, "move:", move)
            print(state)
    return state
