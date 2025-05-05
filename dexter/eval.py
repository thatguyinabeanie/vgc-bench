import argparse
import asyncio

import numpy as np
import torch
from poke_env import cross_evaluate
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client import AccountConfiguration, ServerConfiguration
from scipy.optimize import minimize
from src.agent import Agent
from src.llm import LLMPlayer
from src.teams import RandomTeamBuilder
from src.utils import battle_format
from stable_baselines3 import PPO


def eval(teams: list[int], port: int):
    players = []
    for cls_ in [RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer]:
        player = cls_(
            server_configuration=ServerConfiguration(
                f"ws://localhost:{port}/showdown/websocket",
                "https://play.pokemonshowdown.com/action.php?",
            ),
            battle_format=battle_format,
            log_level=25,
            max_concurrent_battles=10,
            accept_open_team_sheet=True,
            open_timeout=None,
            team=RandomTeamBuilder(teams, battle_format),
        )
        players += [player]
    llm_player = LLMPlayer(
        device="cuda:0",
        server_configuration=ServerConfiguration(
            f"ws://localhost:{port}/showdown/websocket",
            "https://play.pokemonshowdown.com/action.php?",
        ),
        battle_format=battle_format,
        log_level=25,
        accept_open_team_sheet=True,
        open_timeout=None,
        team=RandomTeamBuilder(teams, battle_format),
    )
    players += [llm_player]
    agent_files = {
        "sp": f"results/saves-sp/{','.join([str(t) for t in teams])}-teams/5013504",
        "fp": f"results/saves-fp/{','.join([str(t) for t in teams])}-teams/5013504",
        "do": f"../UT-masters-thesis3/results/saves-do/{','.join([str(t) for t in teams])}-teams/5013504",
        "bc": f"results/saves-bc/29",
        "bc-sp": f"../UT-masters-thesis2/results/saves-bc-sp/{','.join([str(t) for t in teams])}-teams/5013504",
        "bc-fp": f"../UT-masters-thesis3/results/saves-bc-fp/{','.join([str(t) for t in teams])}-teams/5013504",
        "bc-do": f"../UT-masters-thesis2/results/saves-bc-do/{','.join([str(t) for t in teams])}-teams/5013504",
    }
    for name, f in agent_files.items():
        agent = Agent(
            num_frames=1,
            device=torch.device("cuda:0"),
            account_configuration=AccountConfiguration(name, None),
            server_configuration=ServerConfiguration(
                f"ws://localhost:{port}/showdown/websocket",
                "https://play.pokemonshowdown.com/action.php?",
            ),
            battle_format=battle_format,
            log_level=25,
            max_concurrent_battles=10,
            accept_open_team_sheet=True,
            open_timeout=None,
            team=RandomTeamBuilder(teams, battle_format),
        )
        agent.set_policy(PPO.load(f).policy)
        players += [agent]
    results = asyncio.run(cross_evaluate(players, 100))
    payoff_matrix = [
        [r if r is not None else 0 for r in result.values()] for result in results.values()
    ]
    elos = elo_from_payoff(payoff_matrix)
    print(payoff_matrix)
    print(elos)


def elo_from_payoff(A, K=400, eps=1e-9):
    """
    Fit Elo ratings to a payoff matrix of win-fractions.

    Parameters
    ----------
    A : (n, n) array_like
        Payoff matrix.  For i != j,
          - if 0 <= A[i,j] <= 1 : treated as fraction of wins by i over j,
          - if A[i,j] > 1       : treated as integer win-counts (will be normalized).
        Diagonal entries are ignored.
    K : float
        Elo scaling factor (default 400).
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    r : ndarray, shape (n,)
        Fitted Elo ratings (mean-centered).
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]

    # If counts (>1), normalize each pair to a fraction
    for i in range(n):
        for j in range(i + 1, n):
            total = A[i, j] + A[j, i]
            if total > 0 and (A[i, j] > 1 or A[j, i] > 1):
                A[i, j] /= total
                A[j, i] /= total

    def expected_matrix(r):
        diff = (r[:, None] - r[None, :]) / K
        return 1.0 / (1.0 + 10.0 ** (-diff))

    def neg_log_likelihood(r):
        E = expected_matrix(r)
        mask = ~np.eye(n, dtype=bool)
        ll = np.sum(A[mask] * np.log(E[mask] + eps) + (1 - A[mask]) * np.log(1 - E[mask] + eps))
        return -ll

    # Optimize starting from zero
    res = minimize(neg_log_likelihood, np.zeros(n), method="BFGS")
    r = res.x
    # Center ratings so mean = 0
    return r - np.mean(r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Pokémon AI model")
    parser.add_argument("--teams", nargs="+", type=int, help="Indices of teams to train with")
    parser.add_argument("--num_teams", type=int, help="Number of teams to train with")
    parser.add_argument("--port", type=int, default=8000, help="Port to run showdown server on")
    args = parser.parse_args()
    assert (args.teams is None) != (
        args.num_teams is None
    ), "Only pass one of --teams and --num_teams in"
    teams = args.teams if args.teams is not None else list(range(args.num_teams))
    eval(teams, args.port)
