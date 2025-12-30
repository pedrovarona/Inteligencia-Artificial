import random
from typing import Any, NamedTuple

from game_problems import GameProblem, GameState


def random_moves(game: "GameProblem", state: Any) -> Any:
    """
    A player that chooses a legal move at random.

    This strategy function randomly selects one of the available legal moves
    from the current game state. It provides no strategic consideration and
    is primarily useful for testing, baseline comparisons, or educational purposes.

    :param game: The game problem instance containing rules and available actions
    :type game: GameProblem
    :param state: The current game state to choose a move from
    :type state: Any
    :return: A randomly selected legal move
    :rtype: Any
    """
    return random.choice(game.actions(state))


class AdversarialSearchResult(NamedTuple):
    """
    Result of adversarial search (minimax or alpha-beta) containing utility value and best move.

    :param value: The utility value of the position for the searching player
    :type value: float
    :param move: The best move to make, or None if terminal state
    :type move: Any | None
    """

    value: float
    move: Any | None


def minimax_search(game: "GameProblem", state: "GameState") -> Any:
    """
    Search game tree to determine best move using minimax algorithm.

    The minimax algorithm searches the complete game tree to find the optimal
    move by assuming both players play perfectly. It alternates between
    maximizing and minimizing players at each level of the tree.

    :param game: The game problem instance containing rules and utility function
    :type game: GameProblem
    :param state: The current game state to analyze
    :type state: GameState
    :return: The best move for the current player
    :rtype: Any
    """
    player = state.get_current_player()

    def max_value(state: "GameState") -> AdversarialSearchResult:
        """
        Calculate the maximum value achievable from this state.

        :param state: The game state to evaluate
        :type state: GameState
        :return: The maximum value and corresponding best move
        :rtype: AdversarialSearchResult
        """
        if game.is_terminal(state):
            return AdversarialSearchResult(game.utility(state, player), None)

        best_value = float("-inf")
        best_move = None

        for action in game.actions(state):
            result = min_value(game.result(state, action))
            if result.value > best_value:
                best_value = result.value
                best_move = action

        return AdversarialSearchResult(best_value, best_move)

    def min_value(state: "GameState") -> AdversarialSearchResult:
        """
        Calculate the minimum value achievable from this state.

        :param state: The game state to evaluate
        :type state: GameState
        :return: The minimum value and corresponding best move
        :rtype: AdversarialSearchResult
        """
        if game.is_terminal(state):
            return AdversarialSearchResult(game.utility(state, player), None)

        best_value = float("inf")
        best_move = None

        for action in game.actions(state):
            result = max_value(game.result(state, action))
            if result.value < best_value:
                best_value = result.value
                best_move = action

        return AdversarialSearchResult(best_value, best_move)

    search_result = max_value(state)
    return search_result.move


def alphabeta_search(game: "GameProblem", state: "GameState") -> Any:
    """
    Search game tree to determine best move using alpha-beta pruning.

    Alpha-beta pruning is an optimization of the minimax algorithm that
    eliminates branches that cannot possibly influence the final decision.
    It maintains two values: alpha (best value for maximizing player) and
    beta (best value for minimizing player) to prune unnecessary branches.

    :param game: The game problem instance containing rules and utility function
    :type game: GameProblem
    :param state: The current game state to analyze
    :type state: GameState
    :return: The best move for the current player
    :rtype: Any

    .. seealso::
       Based on Figure 5.7 from "Artificial Intelligence: A Modern Approach"
    """
    player = state.get_current_player()

    def max_value(
        state: "GameState", alpha: float, beta: float
    ) -> AdversarialSearchResult:
        """
        Calculate the maximum value achievable from this state with alpha-beta pruning.

        :param state: The game state to evaluate
        :type state: GameState
        :param alpha: The best value the maximizing player can guarantee so far
        :type alpha: float
        :param beta: The best value the minimizing player can guarantee so far
        :type beta: float
        :return: The maximum value and corresponding best move
        :rtype: AdversarialSearchResult
        """
        if game.is_terminal(state):
            return AdversarialSearchResult(game.utility(state, player), None)

        best_value = float("-inf")
        best_move = None

        for action in game.actions(state):
            result = min_value(game.result(state, action), alpha, beta)
            if result.value > best_value:
                best_value = result.value
                best_move = action
                alpha = max(alpha, best_value)

            if alpha >= beta:
                return AdversarialSearchResult(best_value, best_move)

        return AdversarialSearchResult(best_value, best_move)

    def min_value(
        state: "GameState", alpha: float, beta: float
    ) -> AdversarialSearchResult:
        """
        Calculate the minimum value achievable from this state with alpha-beta pruning.

        :param state: The game state to evaluate
        :type state: GameState
        :param alpha: The best value the maximizing player can guarantee so far
        :type alpha: float
        :param beta: The best value the minimizing player can guarantee so far
        :type beta: float
        :return: The minimum value and corresponding best move
        :rtype: AdversarialSearchResult
        """
        if game.is_terminal(state):
            return AdversarialSearchResult(game.utility(state, player), None)

        best_value = float("inf")
        best_move = None

        for action in game.actions(state):
            result = max_value(game.result(state, action), alpha, beta)
            if result.value < best_value:
                best_value = result.value
                best_move = action
                beta = min(beta, best_value)

            if alpha >= beta:
                return AdversarialSearchResult(best_value, best_move)

        return AdversarialSearchResult(best_value, best_move)

    search_result = max_value(state, float("-inf"), float("inf"))
    return search_result.move
