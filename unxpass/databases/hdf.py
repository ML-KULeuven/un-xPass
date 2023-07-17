"""HDF store interface."""
import json
import os
from typing import Literal, Optional, Tuple, Union

import pandas as pd
from socceraction.spadl.config import field_length, field_width

from unxpass.config import logger as log

from .base import (
    TABLE_ACTIONS,
    TABLE_COMPETITIONS,
    TABLE_GAMES,
    TABLE_PLAYER_GAMES,
    TABLE_PLAYERS,
    TABLE_TEAMS,
    Database,
)


class HDFDatabase(Database):
    """Wrapper for a HDF database holding the raw data.

    Parameters
    ----------
    db_path : path-like object
        The path to the database file to be opened. Pass ":memory:" to open
        a connection to a database that is in RAM instead of on disk.
    mode : {'r', 'w', 'a'}, default 'r'
        The mode to open the database with.

    Attributes
    ----------
    conn : sqlite3.Connection
        The connection to the database.
    cursor : sqlite3.Cursor
        The cursor for the connection.
    """

    def __init__(
        self, db_path: Union[Literal[":memory:"], os.PathLike[str]] = ":memory:", mode: str = "r"
    ):
        super().__init__(mode)
        if db_path == ":memory:":
            self.store = pd.HDFStore(
                "sample.h5", mode=mode, driver="H5FD_CORE", driver_core_backing_store=0
            )
        else:
            self.store = pd.HDFStore(db_path, mode)

    def close(self) -> None:
        self.store.close()

    def _import_competitions(self, competitions: pd.DataFrame) -> None:
        competitions = competitions[TABLE_COMPETITIONS].copy()
        try:
            competitions = pd.concat([self.store["competitions"], competitions])
        except KeyError:
            pass
        competitions.drop_duplicates(
            subset=["competition_id", "season_id"], keep="last", inplace=True
        )
        self.store.put("competitions", competitions, format="table", data_columns=True)
        log.debug("Imported %d competitions", len(competitions))

    def _import_games(self, games: pd.DataFrame) -> None:
        games = games[TABLE_GAMES].copy()
        try:
            games = pd.concat([self.store["games"], games])
        except KeyError:
            pass
        games.drop_duplicates(subset=["game_id"], keep="last", inplace=True)
        self.store.put("games", games, format="table", data_columns=True)
        log.debug("Imported %d games", len(games))

    def _import_teams(self, teams: pd.DataFrame) -> None:
        teams = teams[TABLE_TEAMS].copy()
        try:
            teams = pd.concat([self.store["teams"], teams])
        except KeyError:
            pass
        teams.drop_duplicates(subset=["team_id"], keep="last", inplace=True)
        self.store.put("teams", teams, format="table", data_columns=True)
        log.debug("Imported %d teams", len(teams))

    def _import_players(self, players: pd.DataFrame) -> None:
        playerids = players[TABLE_PLAYERS].copy()
        try:
            playerids = pd.concat([self.store["players"], playerids])
        except KeyError:
            pass
        playerids.drop_duplicates(subset=["player_id"], keep="last", inplace=True)
        self.store.put("players", playerids, format="table", data_columns=True)
        log.debug("Imported %d players", len(players))

        player_games = players[TABLE_PLAYER_GAMES].copy()
        try:
            player_games = pd.concat([self.store["player_games"], player_games])
        except KeyError:
            pass
        player_games.drop_duplicates(subset=["player_id", "game_id"], keep="last", inplace=True)
        self.store.put("player_games", player_games, format="table", data_columns=True)

    def _import_actions(self, actions: pd.DataFrame) -> None:
        actions = actions[TABLE_ACTIONS].copy()
        actions["extra"] = actions["extra"].apply(json.dumps).astype("str")
        actions["visible_area_360"] = actions["visible_area_360"].apply(json.dumps).astype("str")
        actions["freeze_frame_360"] = actions["freeze_frame_360"].apply(json.dumps).astype("str")
        self.store.put(
            f"actions/game_{actions.game_id.iloc[0]}",
            actions,
            format="table",
            data_columns=True,
        )
        log.debug("Imported %d actions", len(actions))

    def games(
        self, competition_id: Optional[int] = None, season_id: Optional[int] = None
    ) -> pd.DataFrame:
        games = self.store["games"]
        if competition_id is not None:
            games = games[games["competition_id"] == competition_id]
        if season_id is not None:
            games = games[games["season_id"] == season_id]
        return games.set_index("game_id")

    def actions(self, game_id: int) -> pd.DataFrame:
        try:
            df_actions = self.store[f"actions/game_{game_id}"].set_index(["game_id", "action_id"])
        except KeyError:
            raise IndexError(f"No game found with ID={game_id}")
        # Conversion to python data structures is slow
        df_actions["extra"] = df_actions["extra"].apply(json.loads)
        df_actions["visible_area_360"] = df_actions["visible_area_360"].apply(json.loads)
        df_actions["freeze_frame_360"] = df_actions["freeze_frame_360"].apply(json.loads)
        return df_actions

    def freeze_frame(self, game_id: int, action_id: int, ltr: bool = False) -> pd.DataFrame:
        try:
            team, freeze_frame = self.store.select(
                f"actions/game_{game_id}",
                columns=["team_id", "freeze_frame_360"],
                where=[f"action_id == {action_id}"],
            ).values[0]
        except KeyError:
            raise IndexError(f"No action found with ID={action_id} in game with ID={game_id}")
        freeze_frame = json.loads(freeze_frame)
        if freeze_frame is None or len(freeze_frame) == 0:
            return pd.DataFrame(columns=["teammate", "actor", "keeper", "x", "y"])
        freezedf = pd.DataFrame(freeze_frame).fillna(
            {"teammate": False, "actor": False, "keeper": False}
        )
        if ltr:
            home_team_id, _ = self.get_home_away_team_id(game_id)
            if home_team_id != team:
                freezedf["x"] = field_length - freezedf["x"].values
                freezedf["y"] = field_width - freezedf["y"].values
        return freezedf

    def get_home_away_team_id(self, game_id: int) -> Tuple[int, int]:
        try:
            home_team_id, away_team_id = self.store.select(
                "games", columns=["home_team_id", "away_team_id"], where=[f"game_id == {game_id}"]
            ).values[0]
            return home_team_id, away_team_id
        except IndexError:
            raise IndexError(f"No game found with ID={game_id}")
