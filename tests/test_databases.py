"""Tests for the unxpass.databases module."""
from typing import Generator

import pandas as pd
import pytest
from pytest import fixture

from unxpass.databases import Database, HDFDatabase, SQLiteDatabase
from unxpass.databases.base import _sb_freeze_frame_to_spadl, _sb_visible_area_to_spadl


def test_create_sqlite_dataset() -> None:
    """It should add the requested data to the database."""
    with SQLiteDatabase(":memory:") as db:
        db.import_data(getter="remote", competition_id=55, season_id=43, game_id=3795107)
        df_competitions = pd.read_sql_query("SELECT * FROM competitions", db.conn)
        assert len(df_competitions) == 1
        df_games = pd.read_sql_query("SELECT * FROM competitions", db.conn)
        assert len(df_games) == 1
        df_teams = pd.read_sql_query("SELECT * FROM teams", db.conn)
        assert len(df_teams) == 2
        df_players = pd.read_sql_query("SELECT * FROM players", db.conn)
        assert len(df_players) == 30
        df_player_games = pd.read_sql_query("SELECT * FROM player_games", db.conn)
        assert len(df_player_games) == 30
        df_actions = pd.read_sql_query("SELECT * FROM actions", db.conn)
        assert len(df_actions) > 0


def test_create_hdf_dataset(tmp_path) -> None:
    """It should add the requested data to the database."""
    db_path = tmp_path / "db.h5"
    with HDFDatabase(db_path, mode="w") as db:
        db.import_data(getter="remote", competition_id=55, season_id=43, game_id=3795107)
        df_competitions = db.store["competitions"]
        assert len(df_competitions) == 1
        df_games = db.store["games"]
        assert len(df_games) == 1
        df_teams = db.store["teams"]
        assert len(df_teams) == 2
        df_players = db.store["players"]
        assert len(df_players) == 30
        df_player_games = db.store["player_games"]
        assert len(df_player_games) == 30
        df_actions = db.store["actions/game_3795107"]
        assert len(df_actions) > 0


@fixture(scope="session")
def sqlitedb() -> Generator[SQLiteDatabase, None, None]:
    """Create a SQLite dataset with BEL v ITA at EURO2020."""
    with SQLiteDatabase(":memory:", mode="w") as db:
        db.import_data(getter="remote", competition_id=55, season_id=43, game_id=3795107)
        yield db


@fixture(scope="session")
def hdfdb() -> Generator[HDFDatabase, None, None]:
    """Create a HDF dataset with BEL v ITA at EURO2020."""
    with HDFDatabase(":memory:", mode="w") as db:
        db.import_data(getter="remote", competition_id=55, season_id=43, game_id=3795107)
        yield db


database_interfaces = [pytest.lazy_fixture("sqlitedb"), pytest.lazy_fixture("hdfdb")]


@pytest.mark.parametrize("db", database_interfaces)
def test_create_dataset_duplicates(db: Database) -> None:
    """The database should not contain duplicate data."""
    db.import_data(getter="remote", competition_id=55, season_id=43, game_id=3795107)
    db.import_data(getter="remote", competition_id=55, season_id=43, game_id=3795107)
    df_games = db.games()
    assert len(df_games) == 1


@pytest.mark.parametrize("db", database_interfaces)
def test_games(db: Database) -> None:
    """It should return the games for a given competition and season."""
    df_games = db.games()
    assert len(df_games) == 1
    df_games = db.games(competition_id=0)
    assert len(df_games) == 0
    df_games = db.games(season_id=0)
    assert len(df_games) == 0


@pytest.mark.parametrize("db", database_interfaces)
def test_actions(db: Database) -> None:
    """It should return the actions for a given game."""
    df_actions = db.actions(game_id=3795107)
    assert len(df_actions) > 0
    with pytest.raises(IndexError, match="No game found with ID=0"):
        db.actions(game_id=0)


@pytest.mark.parametrize("db", database_interfaces)
def test_freeze_frame(db: Database) -> None:
    """It should return the freeze frame for a given action."""
    df_frame = db.freeze_frame(game_id=3795107, action_id=0)
    assert len(df_frame) > 0
    with pytest.raises(IndexError, match="No action found with ID=0 in game with ID=0"):
        db.freeze_frame(game_id=0, action_id=0)


@pytest.mark.parametrize("db", database_interfaces)
def test_get_home_away_team_id(db: Database) -> None:
    """It should return the ID of the home and away team in a given game."""
    home_team_id, away_team_id = db.get_home_away_team_id(game_id=3795107)
    assert home_team_id == 782
    assert away_team_id == 914
    with pytest.raises(IndexError, match="No game found with ID=0"):
        db.get_home_away_team_id(game_id=0)


def test_convert_visible_area_to_spadl() -> None:
    """It should convert the visible area to SPADL coordinates."""
    visible_area = [
        109.0,
        0.0,
        74.4100848692153,
        75.9521430531334,
        45.789439204546,
        76.7273428944324,
        10.4477314711676,
        0.0,
        109.0,
        0.0,
    ]
    converted_visible_area = _sb_visible_area_to_spadl(visible_area)
    assert converted_visible_area is not None
    assert len(converted_visible_area) == len(visible_area) / 2
    assert len(converted_visible_area[0]) == 2
    assert converted_visible_area[0] == (95.375, 68)


def test_convert_freeze_frame_to_spadl() -> None:
    """It should convert the freeze frame coordinates to SPADL coordinates."""
    freeze_frame = [{"teammate": True, "actor": False, "keeper": False, "location": [120.0, 80.0]}]
    converted_freeze_frame = _sb_freeze_frame_to_spadl(freeze_frame)
    assert converted_freeze_frame is not None
    assert converted_freeze_frame == [
        {"teammate": True, "actor": False, "keeper": False, "x": 105.0, "y": 0.0}
    ]
    inverted_converted_freeze_frame = _sb_freeze_frame_to_spadl(freeze_frame, invert=True)
    assert inverted_converted_freeze_frame is not None
    assert inverted_converted_freeze_frame == [
        {"teammate": True, "actor": False, "keeper": False, "x": 0.0, "y": 68.0}
    ]
