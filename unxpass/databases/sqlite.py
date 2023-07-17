"""SQLite database interface."""
import json
import os
import sqlite3
from typing import Literal, Optional, Tuple, Union

import pandas as pd
from socceraction.spadl.config import field_length, field_width

from .base import (
    TABLE_ACTIONS,
    TABLE_COMPETITIONS,
    TABLE_GAMES,
    TABLE_PLAYER_GAMES,
    TABLE_PLAYERS,
    TABLE_TEAMS,
    Database,
)


class SQLiteDatabase(Database):
    """Wrapper for a SQLite database holding the raw data.

    Parameters
    ----------
    db_path : path-like object, optional
        The path to the database file to be opened. Pass ":memory:" to open
        a connection to a database that is in RAM instead of on disk.

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
        self.conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.cursor = self.conn.cursor()
        self.create_schema()

    def create_schema(self) -> None:
        """Create the database schema."""
        sql = """
        CREATE TABLE IF NOT EXISTS competitions (
            competition_id INTEGER,
            competition_name TEXT,
            season_id INTEGER,
            season_name TEXT,
            country_name TEXT,
            competition_gender TEXT,
            PRIMARY KEY (competition_id, season_id)
        );
        CREATE TABLE IF NOT EXISTS games (
            game_id INTEGER PRIMARY KEY,
            season_id INTEGER,
            competition_id INTEGER,
            game_day INTEGER,
            game_date DATETIME,
            home_team_id INTEGER,
            away_team_id INTEGER,
            competition_stage TEXT,
            home_score INTEGER,
            away_score INTEGER,
            venue TEXT,
            referee TEXT,
            FOREIGN KEY(competition_id) REFERENCES competitions(competition_id),
            FOREIGN KEY(season_id) REFERENCES competitions(season_id)
        );
        CREATE TABLE IF NOT EXISTS teams (
            team_id INTEGER PRIMARY KEY,
            team_name TEXT
        );
        CREATE TABLE IF NOT EXISTS players (
            player_id INTEGER PRIMARY KEY,
            player_name TEXT,
            nickname TEXT
        );
        CREATE TABLE IF NOT EXISTS player_games (
            player_id INTEGER,
            game_id INTEGER,
            team_id INTEGER,
            is_starter BOOLEAN,
            minutes_played INTEGER,
            starting_position_id INTEGER,
            starting_position_name TEXT,
            jersey_number INTEGER,
            PRIMARY KEY(player_id, game_id),
            FOREIGN KEY(player_id) REFERENCES players(player_id),
            FOREIGN KEY(game_id) REFERENCES games(game_id),
            FOREIGN KEY(team_id) REFERENCES teams(team_id)
        );
        CREATE TABLE IF NOT EXISTS actions (
            game_id INTEGER,
            original_event_id TEXT,
            action_id INTEGER,
            period_id INTEGER,
            time_seconds INTEGER,
            team_id INTEGER,
            player_id INTEGER,
            start_x REAL,
            start_y REAL,
            end_x REAL,
            end_y REAL,
            bodypart_id INTEGER,
            type_id INTEGER,
            result_id INTEGER,
            possession_team_id INTEGER,
            play_pattern_name TEXT,
            under_pressure BOOLEAN,
            extra TEXT,
            visible_area_360 TEXT,
            in_visible_area_360 BOOLEAN,
            freeze_frame_360 TEXT,
            PRIMARY KEY (game_id, action_id),
            FOREIGN KEY(player_id) REFERENCES players(player_id),
            FOREIGN KEY(game_id) REFERENCES games(game_id),
            FOREIGN KEY(team_id) REFERENCES teams(team_id)
            FOREIGN KEY(possession_team_id) REFERENCES teams(team_id)
        );
        """
        self.cursor.executescript(sql)
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.cursor.close()
        self.conn.close()

    def _import_competitions(self, competitions: pd.DataFrame) -> None:
        self.cursor.executemany(
            "REPLACE INTO competitions VALUES(?,?,?,?,?,?);",
            competitions[TABLE_COMPETITIONS].itertuples(index=False),
        )
        self.conn.commit()

    def _import_games(self, games: pd.DataFrame) -> None:
        self.cursor.executemany(
            "REPLACE INTO games VALUES(?,?,?,?,?,?,?,?,?,?,?,?);",
            games[TABLE_GAMES].astype({"game_date": str}).itertuples(index=False),
        )
        self.conn.commit()

    def _import_teams(self, teams: pd.DataFrame) -> None:
        self.cursor.executemany(
            "REPLACE INTO teams VALUES(?,?);",
            teams[TABLE_TEAMS].itertuples(index=False),
        )
        self.conn.commit()

    def _import_players(self, players: pd.DataFrame) -> None:
        self.cursor.executemany(
            "REPLACE INTO players VALUES(?,?,?);",
            players[TABLE_PLAYERS].drop_duplicates(subset="player_id").itertuples(index=False),
        )
        self.cursor.executemany(
            "REPLACE INTO player_games VALUES(?,?,?,?,?,?,?,?);",
            players[TABLE_PLAYER_GAMES].itertuples(index=False),
        )
        self.conn.commit()

    def _import_actions(self, actions: pd.DataFrame) -> None:
        actions["extra"] = actions["extra"].apply(json.dumps).astype("str")
        actions["visible_area_360"] = actions["visible_area_360"].apply(json.dumps).astype("str")
        actions["freeze_frame_360"] = actions["freeze_frame_360"].apply(json.dumps).astype("str")
        self.cursor.executemany(
            "REPLACE INTO actions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);",
            actions[TABLE_ACTIONS].itertuples(index=False),
        )
        self.conn.commit()

    def games(
        self, competition_id: Optional[int] = None, season_id: Optional[int] = None
    ) -> pd.DataFrame:
        query = "SELECT * FROM games"
        filters = []
        if competition_id is not None:
            filters.append(f"competition_id = {competition_id}")
        if season_id is not None:
            filters.append(f"season_id = {season_id}")
        if len(filters):
            query += " WHERE " + " AND ".join(filters)
        return pd.read_sql_query(query, self.conn).set_index("game_id")

    def actions(self, game_id: int) -> pd.DataFrame:
        query = f"SELECT * FROM actions WHERE game_id = {game_id}"
        df_actions = pd.read_sql_query(query, self.conn).set_index(["game_id", "action_id"])
        if df_actions.empty:
            raise IndexError(f"No game found with ID={game_id}")
        df_actions["extra"] = df_actions["extra"].apply(json.loads)
        df_actions["visible_area_360"] = df_actions["visible_area_360"].apply(json.loads)
        df_actions["freeze_frame_360"] = df_actions["freeze_frame_360"].apply(json.loads)
        return df_actions

    def freeze_frame(self, game_id: int, action_id: int, ltr: bool = False) -> pd.DataFrame:
        query = f"SELECT team_id, freeze_frame_360 FROM actions WHERE game_id = {game_id} AND action_id = {action_id}"
        self.cursor.execute(query)
        res = self.cursor.fetchone()
        if res:
            freeze_frame = json.loads(res[1])
            if freeze_frame is None or len(freeze_frame) == 0:
                return pd.DataFrame(columns=["teammate", "actor", "keeper", "x", "y"])
            freezedf = pd.DataFrame(freeze_frame).fillna(
                {"teammate": False, "actor": False, "keeper": False}
            )
            if ltr:
                home_team_id, _ = self.get_home_away_team_id(game_id)
                if home_team_id != res[0]:
                    freezedf["x"] = field_length - freezedf["x"].values
                    freezedf["y"] = field_width - freezedf["y"].values
            return freezedf
        raise IndexError(f"No action found with ID={action_id} in game with ID={game_id}")

    def get_home_away_team_id(self, game_id: int) -> Tuple[int, int]:
        query = f"""
            SELECT home_team_id, away_team_id
            FROM games
            WHERE game_id = {game_id}
        """
        try:
            home_team_id, away_team_id = pd.read_sql_query(query, self.conn).loc[0]
            return home_team_id, away_team_id
        except KeyError:
            raise IndexError(f"No game found with ID={game_id}")
