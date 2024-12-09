import asyncio
import json
import pickle
from imitation.data.types import Trajectory
from src.reader import LogReader


def process_logs(log_jsons: dict[str, tuple[str, str]]) -> list[Trajectory]:
    trajs = []
    total = 0
    for i, (tag, (_, log)) in enumerate(log_jsons.items()):
        print(f"progress: {round(100 * i / len(log_jsons), ndigits=2)}%", end="\r")
        player1 = LogReader(battle_format="gen9vgc2024regh", log_level=51, accept_open_team_sheet=True)
        player2 = LogReader(battle_format="gen9vgc2024regh", log_level=51, accept_open_team_sheet=True)
        result1 = asyncio.run(player1.follow_log(tag, log, "p1"))
        result2 = asyncio.run(player2.follow_log(tag, log, "p2"))
        if result1 is not None:
            states1, actions1 = result1
            total += len(states1)
            trajs += [Trajectory(obs=states1, acts=actions1, infos=None, terminal=True)]
        if result2 is not None:
            states2, actions2 = result2
            total += len(states2)
            trajs += [Trajectory(obs=states2, acts=actions2, infos=None, terminal=True)]
    print(f"prepared {len(trajs)} trajectories with {total} total state-action pairs")
    return trajs


if __name__ == "__main__":
    with open("data/logs.json", "r") as f:
        logs = json.load(f)
    trajs = process_logs(logs)
    with open("data/trajs.pkl", "wb") as f:
        pickle.dump(trajs, f)
