"""Configuration for pytest."""
from typing import Generator

from pytest import fixture

from unxpass.databases import SQLiteDatabase


@fixture(scope="session")
def db() -> Generator[SQLiteDatabase, None, None]:
    """Create a dataset with BEL v ITA at EURO2020."""
    with SQLiteDatabase(":memory:") as db:
        db.import_data(getter="remote", competition_id=55, season_id=43, game_id=3795107)
        yield db
