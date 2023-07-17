"""Utility functions for data loading and preprocessing operations."""
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.path as mpl_path
import numpy as np
import numpy.typing as npt
import pandas as pd
import socceraction.spadl as spadl
from rich.progress import track
from socceraction.data.statsbomb import StatsBombLoader
from socceraction.spadl.config import field_length, field_width

from unxpass.config import logger


TABLE_COMPETITIONS = [
    "competition_id",
    "competition_name",
    "season_id",
    "season_name",
    "country_name",
    "competition_gender",
]
TABLE_GAMES = [
    "game_id",
    "season_id",
    "competition_id",
    "game_day",
    "game_date",
    "home_team_id",
    "away_team_id",
    "competition_stage",
    "home_score",
    "away_score",
    "venue",
    "referee",
]
TABLE_TEAMS = [
    "team_id",
    "team_name",
]
TABLE_PLAYERS = [
    "player_id",
    "player_name",
    "nickname",
]
TABLE_PLAYER_GAMES = [
    "player_id",
    "game_id",
    "team_id",
    "is_starter",
    "minutes_played",
    "starting_position_id",
    "starting_position_name",
    "jersey_number",
]
TABLE_ACTIONS = [
    "game_id",
    "original_event_id",
    "action_id",
    "period_id",
    "time_seconds",
    "team_id",
    "player_id",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "bodypart_id",
    "type_id",
    "result_id",
    "possession_team_id",
    "play_pattern_name",
    "under_pressure",
    "extra",
    "visible_area_360",
    "in_visible_area_360",
    "freeze_frame_360",
]


def _sb_events_to_spadl(events: pd.DataFrame, home_team_id: int) -> pd.DataFrame:
    # TODO: should we use atomic spadl here instead?
    # convert StatsBomb events to SPADL actions
    actions = spadl.statsbomb.convert_to_actions(events, home_team_id)
    # extend the actions with StatsBomb special data features
    actions = pd.merge(
        actions,
        events[
            [
                "event_id",
                "possession_team_id",
                "play_pattern_name",
                "under_pressure",
                "extra",
                "visible_area_360",
                "freeze_frame_360",
            ]
        ].rename(columns={"event_id": "original_event_id"}),
        on="original_event_id",
        how="left",
    )
    actions["under_pressure"] = actions["under_pressure"].fillna(False)
    # convert coordinates in freeze frames to SPADL coordinates
    actions["visible_area_360"] = actions["visible_area_360"].apply(_sb_visible_area_to_spadl)
    actions["in_visible_area_360"] = actions.apply(
        lambda x: (
            _inside_polygon(x.start_x, x.start_y, x.visible_area_360)
            and _inside_polygon(x.end_x, x.end_y, x.visible_area_360)
        ),
        axis=1,
    )
    actions["freeze_frame_360"] = actions.assign(away_idx=(actions.team_id != home_team_id)).apply(
        lambda x: _sb_freeze_frame_to_spadl(x.freeze_frame_360, x.away_idx), axis=1
    )
    return actions


def _sb_coordinates_to_spadl(
    sb_x: npt.NDArray[np.float64], sb_y: npt.NDArray[np.float64]
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # TODO: not sure if SB also uses cell-based coordinates here
    spadl_x = (sb_x / 120) * field_length
    spadl_y = field_width - (sb_y / 80) * field_width
    return spadl_x, spadl_y


def _sb_visible_area_to_spadl(
    visible_area: Optional[List[float]],
) -> Optional[List[Tuple[float, float]]]:
    if not isinstance(visible_area, list):
        return None
    np_visible_area = np.array(visible_area)
    visible_area_x, visible_area_y = _sb_coordinates_to_spadl(
        np_visible_area[::2], np_visible_area[1::2]
    )
    return list(zip(visible_area_x, visible_area_y))


def _sb_freeze_frame_to_spadl(
    freeze_frame: Optional[List[Dict[str, Any]]],
    invert: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(freeze_frame, list):
        return None
    freezedf = pd.DataFrame(
        {
            "teammate": e["teammate"],
            "actor": e["actor"],
            "keeper": e["keeper"],
            "x": e["location"][0],
            "y": e["location"][1],
        }
        for e in freeze_frame
    )
    freezedf["x"], freezedf["y"] = _sb_coordinates_to_spadl(freezedf["x"], freezedf["y"])
    if invert:
        freezedf["x"] = field_length - freezedf["x"].values
        freezedf["y"] = field_width - freezedf["y"].values
    return list(freezedf.to_dict("records"))


def _inside_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
    try:
        path = mpl_path.Path(polygon)
    except ValueError:
        return False
    inside = path.contains_points([(x, y)])
    return inside[0]


class Database(ABC):
    """Wrapper for a database holding the raw data.

    Parameters
    ----------
    mode : {'r', 'w', 'a'}, default 'r'
        The mode to open the database with.
    """

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def __enter__(self):  # type: ignore
        """Make a database connection and return it."""
        return self

    def __exit__(  # type: ignore
        self,
        exc_type,
        exc_val,
        exc_tb,
    ):
        """Close the database connection."""
        self.close()

    def close(self) -> None:
        """Close the database connection."""

    def import_data(
        self,
        getter: str = "local",
        root: Optional[os.PathLike[str]] = None,
        creds: Optional[Dict[str, str]] = None,
        competition_id: Optional[int] = None,
        season_id: Optional[int] = None,
        game_id: Optional[int] = None,
    ) -> None:
        """Import data from API or local folder."""
        if getter == "remote":
            # Disable private API warnings
            import warnings

            from statsbombpy.api_client import NoAuthWarning

            warnings.filterwarnings(action="ignore", category=NoAuthWarning, module="statsbombpy")

        # Set up the StatsBombLoader
        data_loader = StatsBombLoader(getter=getter, root=root, creds=creds)

        # Retrieve all available competitions
        competitions = data_loader.competitions()
        if competition_id is not None:
            competitions = competitions[competitions.competition_id == competition_id]
        if season_id is not None:
            competitions = competitions[competitions.season_id == season_id]

        # Store competitions
        self._import_competitions(competitions)

        # Retrieve games from all selected competitions
        games = pd.concat(
            [
                data_loader.games(row.competition_id, row.season_id)
                for row in competitions.itertuples()
            ]
        )
        if game_id is not None:
            games = games[games.game_id == game_id]
        if games.empty:
            logger.exception("No games found with given criteria.")
            return

        # Load and convert match data
        games_verbose = track(list(games.itertuples()), description="Loading game data...")
        teams, players = [], []
        for game in games_verbose:
            try:
                teams.append(data_loader.teams(game.game_id))
                players.append(data_loader.players(game.game_id))
                events = data_loader.events(game.game_id, load_360=True)
                # Store actions
                actions = _sb_events_to_spadl(events, game.home_team_id)
                self._import_actions(actions)
                self._import_games(games[games.game_id == game.game_id])
            except FileNotFoundError:
                logger.exception("Error adding game %s.", game.game_id)

        # Store teams
        self._import_teams(pd.concat(teams).drop_duplicates(subset="team_id"))

        # Store players
        self._import_players(pd.concat(players))

    @abstractmethod
    def _import_competitions(self, competitions: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def _import_games(self, games: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def _import_teams(self, teams: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def _import_players(self, players: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def _import_actions(self, actions: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def games(
        self, competition_id: Optional[int] = None, season_id: Optional[int] = None
    ) -> pd.DataFrame:
        """Get a DataFrame of games.

        Parameters
        ----------
        competition_id : int, optional
            The ID of the competition.
        season_id : int, optional
            The ID of the season.

        Returns
        -------
        pandas.DataFrame
            A DataFrame of games.
        """

    @abstractmethod
    def actions(self, game_id: int) -> pd.DataFrame:
        """Get a DataFrame of actions.

        Parameters
        ----------
        game_id : int
            The ID of the game.

        Returns
        -------
        pandas.DataFrame
            A DataFrame of actions.
        """

    @abstractmethod
    def freeze_frame(self, game_id: int, action_id: int, ltr: bool = False) -> pd.DataFrame:
        """Return the 360 freeze frame for a given action.

        Parameters
        ----------
        game_id : int
            The ID of a game.
        action_id : int
            The ID of an action.
        ltr : bool, default: False
            Whether to return the freeze frame for the team executing the action
            in left-to-right direction.

        Returns
        -------
        pd.DataFrame
            The 360 freeze frame.

        Raises
        ------
        IndexError
            If no (game, action) pair exists with the provided IDs.
        """

    @abstractmethod
    def get_home_away_team_id(self, game_id: int) -> Tuple[int, int]:
        """Return the id of the home and away team in a given game.

        Parameters
        ----------
        game_id : int
            The ID of a game.

        Returns
        -------
        (int, int)
            The ID of the home and away team.

        Raises
        ------
        IndexError
            If no game exists with the provided ID.
        """
