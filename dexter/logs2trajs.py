import asyncio
import json
import pickle

import numpy as np
import numpy.typing as npt
from imitation.data.types import Trajectory
from poke_env import to_id_str
from poke_env.environment import AbstractBattle, DoubleBattle
from poke_env.player import BattleOrder, DoubleBattleOrder, DoublesEnv, Player
from poke_env.ps_client import AccountConfiguration
from scrape_logs import battle_formats
from src.agent import Agent
from src.utils import doubles_chunk_obs_len, frame_stack, num_frames


class LogReader(Player):
    states: list[npt.NDArray[np.float32]]
    actions: list[npt.NDArray[np.int64]]
    msg: str | None
    teampreview_draft: list[str]

    def __init__(self, *args, **kwargs):
        super().__init__(start_listening=False, *args, **kwargs)
        self.states = []
        self.actions = []
        self.next_msg = None
        self.teampreview_draft = []

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        assert self.next_msg is not None
        assert isinstance(battle, DoubleBattle)
        order1 = self.get_order(battle, self.next_msg, False)
        order2 = self.get_order(battle, self.next_msg, True)
        order = DoubleBattleOrder(order1, order2)
        state = Agent.embed_battle(battle, self.teampreview_draft)
        action = DoublesEnv.order_to_action(order, battle, fake=True)
        if action[0] != 0 and action[1] != 0:
            self.states += [state]
            self.actions += [action]
        return order

    async def _handle_battle_request(
        self, battle: AbstractBattle, from_teampreview_request: bool = False
    ):
        pass

    @staticmethod
    def get_order(battle: DoubleBattle, msg: str, is_right: bool) -> BattleOrder | None:
        pos = "b" if is_right else "a"
        move_lines = [
            l
            for l in msg.split("\n")
            if f"|move|{battle.player_role}{pos}: " in l and "[from]" not in l
        ]
        if move_lines:
            active = battle.active_pokemon[int(is_right)]
            assert active is not None, battle.active_pokemon
            [_, _, identifier, move, target_identifier, *_] = move_lines[0].split("|")
            if to_id_str(move) == "struggle":
                move = list(active.moves.values())[0]
            else:
                move = (
                    active.moves[to_id_str(move)]
                    if to_id_str(move) in active.moves
                    else active.moves["metronome"]
                )
            target_lines = [l for l in msg.split("\n") if f"|switch|{target_identifier}" in l]
            target_details = target_lines[0].split("|")[3] if target_lines else ""
            target = (
                battle.get_pokemon(target_identifier, details=target_details)
                if ": " in target_identifier
                else None
            )
            did_tera = f"|-terastallize|{identifier}|" in msg
            return BattleOrder(
                move, terastallize=did_tera, move_target=battle.to_showdown_target(move, target)
            )
        elif f"|switch|{battle.player_role}{pos}: " in msg:
            start = msg.index(f"|switch|{battle.player_role}{pos}: ")
            end = msg.index("\n", start) if "\n" in msg[start:] else len(msg)
            [_, _, identifier, details, *_] = msg[start:end].split("|")
            mon = battle.get_pokemon(identifier, details=details, request=battle.last_request)
            return BattleOrder(mon)
        else:
            return None

    def teampreview(self, battle: AbstractBattle) -> str:
        assert self.next_msg is not None
        assert isinstance(battle, DoubleBattle)
        id1 = self.get_teampreview_order(battle, self.next_msg, False)
        id2 = self.get_teampreview_order(battle, self.next_msg, True)
        all_choices = [str(c) for c in range(1, 7)]
        all_choices.remove(str(id1))
        all_choices.remove(str(id2))
        order_str = f"/team {id1}{id2}{all_choices[0]}{all_choices[1]}"
        order1 = BattleOrder(list(battle.team.values())[id1 - 1])
        order2 = BattleOrder(list(battle.team.values())[id2 - 1])
        order = DoubleBattleOrder(order1, order2)
        state = Agent.embed_battle(battle, self.teampreview_draft)
        assert state.shape == (num_frames, 12, doubles_chunk_obs_len) if frame_stack else (12, doubles_chunk_obs_len)
        action = DoublesEnv.order_to_action(order, battle, fake=True)
        self.states += [state]
        self.actions += [action]
        self.teampreview_draft = [
            p.name for i, p in enumerate(battle.team.values()) if i + 1 in [id1, id2]
        ]
        return order_str

    @staticmethod
    def get_teampreview_order(battle: AbstractBattle, msg: str, is_right: bool) -> int:
        pos = "b" if is_right else "a"
        start = msg.index(f"|switch|{battle.player_role}{pos}: ")
        end = msg.index("\n", start)
        [_, _, identifier, details, *_] = msg[start:end].split("|")
        mon = battle.get_pokemon(identifier, details=details, request=battle.last_request)
        index = list(battle.team.values()).index(mon)
        return index + 1

    async def follow_log(
        self, tag: str, log: str
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]] | None:
        self.states = []
        self.actions = []
        self.teampreview_draft = []
        tag = f"battle-{tag}"
        messages = [f">{tag}\n" + m for m in log.split("\n|\n")]
        if "|win|" not in messages[1]:
            for i in range(len(messages) - 1):
                split_messages = [m.split("|") for m in messages[i].split("\n")]
                self.next_msg = messages[i + 1]
                if i == 0:
                    battle = await self._create_battle(f">{tag}".split("-"))
                    battle.logger = None
                    battle._teampreview = True
                    await self._handle_battle_message(split_messages)
                    self.teampreview(battle)
                else:
                    battle = self.battles[tag]
                    battle._teampreview = False
                    await self._handle_battle_message(split_messages)
                    if "|switch|" in messages[i + 1] or "|move|" in messages[i + 1]:
                        self.choose_move(battle)
            split_messages = [m.split("|") for m in messages[-1].split("\n")]
            await self._handle_battle_message(split_messages)
            last_state = Agent.embed_battle(self.battles[tag], self.teampreview_draft)
            self.states += [last_state]
            return np.stack(self.states, axis=0), np.stack(self.actions, axis=0)


def process_logs(
    log_jsons: dict[str, tuple[str, str]], battle_format: str, strict: bool = False
) -> list[Trajectory]:
    trajs = []
    total = 0
    num_errors = 0
    for i, (tag, (_, log)) in enumerate(log_jsons.items()):
        print(f"progress: {i}/{len(log_jsons)}", end="\r")
        try:
            start_index = log.find("|win|") + 5
            username = log[start_index : log.find("\n", start_index)]
            player = LogReader(
                account_configuration=AccountConfiguration(username, None),
                battle_format=battle_format,
                log_level=51,
                accept_open_team_sheet=True,
            )
            result = asyncio.run(player.follow_log(tag, log))
            if result is not None:
                states, actions = result
                total += len(states)
                trajs += [Trajectory(obs=states, acts=actions, infos=None, terminal=True)]
        except KeyboardInterrupt:
            raise
        except SystemExit:
            raise
        except Exception as e:
            if strict:
                raise e
            else:
                num_errors += 1
    print(
        f"prepared {len(trajs)} trajectories "
        f"with {total} total state-action pairs "
        f"(and {num_errors} games thrown away)"
    )
    return trajs


if __name__ == "__main__":
    trajs = []
    for f in battle_formats:
        with open(f"data/logs-{f}.json", "r") as file:
            logs = json.load(file)
        trajs += process_logs(logs, f, strict=False)
    with open("data/trajs.pkl", "wb") as f:
        pickle.dump(trajs, f)
