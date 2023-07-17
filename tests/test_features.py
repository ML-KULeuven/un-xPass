"""Tests for the unxpass.features module."""
import numpy as np
import pandas as pd
import pytest
from socceraction.spadl.utils import add_names
from socceraction.vaep.features import gamestates as to_gamestates

import unxpass.features as fs
from unxpass.utils import play_left_to_right


@pytest.fixture
def gamestates(db):
    """Create the SPADL game state representation for BEL v ITA at EURO2020."""
    game_id = 3795107
    nb_prev_actions = 3
    # retrieve actions from database
    actions = add_names(db.actions(game_id))
    # convert actions to gamestates
    home_team_id, _ = db.get_home_away_team_id(game_id)
    gamestates = play_left_to_right(to_gamestates(actions, nb_prev_actions), home_team_id)
    return gamestates


def test_get_features(db):
    """It should return the features for a given game."""
    df_features = fs.get_features(db, game_id=3795107)
    assert len(df_features) > 0


def test_required_fields():
    """Each transformer should have the fields it requires as an attribute."""
    assert fs.startlocation.required_fields == ["start_x", "start_y"]
    assert fs.relative_startlocation.required_fields == ["start_x", "start_y"]
    assert fs.intended_endlocation.required_fields == ["end_x", "end_y"]


def test_angle():
    """It should return the angle between the start and end location."""
    # forward pass
    df = pd.DataFrame([{"start_x": 40, "start_y": 68 / 2, "end_x": 60, "end_y": 68 / 2}])
    assert fs.angle(df).loc[0, "angle_a0"] == 0
    # backward pass
    df = pd.DataFrame([{"start_x": 60, "start_y": 68 / 2, "end_x": 40, "end_y": 68 / 2}])
    assert fs.angle(df).loc[0, "angle_a0"] == np.pi
    # pass to left side
    df = pd.DataFrame([{"start_x": 40, "start_y": 68 / 2, "end_x": 60, "end_y": 68}])
    assert fs.angle(df).loc[0, "angle_a0"] > 0
    # pass to right side
    df = pd.DataFrame([{"start_x": 40, "start_y": 68 / 2, "end_x": 60, "end_y": 0}])
    assert fs.angle(df).loc[0, "angle_a0"] < 0


def test_under_pressure(gamestates):
    """It should return the under pressure flag."""
    df = fs.under_pressure(gamestates)
    assert "under_pressure_a0" in df.columns
    assert len(df.columns) == len(gamestates)
    assert not df.loc[(3795107, 0), "under_pressure_a0"]
    assert df.loc[(3795107, 5), "under_pressure_a0"]


def test_ball_height(gamestates):
    """It should return whether a pass is a ground, low or high pass."""
    df = fs.ball_height(gamestates)
    assert "ball_height_a0" in df.columns
    assert len(df.columns) == len(gamestates)
    assert df["ball_height_a0"].isin(["ground", "low", "high", None]).all()
    assert df.loc[(3795107, 0), "ball_height_a0"] == "ground"


def test_player_possession_time(gamestates):
    """It should return the time a player has possession of the ball."""
    df = fs.player_possession_time(gamestates)
    assert "player_possession_time_a0" in df.columns
    assert len(df.columns) == len(gamestates)
    assert df.loc[(3795107, 0), "player_possession_time_a0"] == 0
    assert df.loc[(3795107, 1), "player_possession_time_a0"] == 0
    assert df.loc[(3795107, 2), "player_possession_time_a0"] == 1


def test_packing_rate(gamestates):
    """It should return the packing rate of the pass."""
    df = fs.packing_rate(gamestates)
    assert "packing_rate_a0" in df.columns
    assert len(df.columns) == len(gamestates)
    assert df.loc[(3795107, 0), "packing_rate_a0"] == 0
    assert df.loc[(3795107, 1), "packing_rate_a0"] is None
    assert df.loc[(3795107, 10), "packing_rate_a0"] == 2


def test_defenders_in_radius(gamestates):
    """It should return the number of defenders in a <x> meter radius."""
    df = fs.defenders_in_5m_radius(gamestates)
    assert "nb_defenders_start_5m_a0" in df.columns
    assert "nb_defenders_end_5m_a0" in df.columns
    assert len(df.columns) == 2 * len(gamestates)
    assert df.loc[(3795107, 15), "nb_defenders_start_5m_a0"] == 2
    assert df.loc[(3795107, 15), "nb_defenders_end_5m_a0"] == 1


def test_freeze_frame_360(gamestates):
    """It should return the raw freeze frame in LTR playing direction."""
    freeze_frames = fs.freeze_frame_360(gamestates)
    assert "freeze_frame_360_a0" in freeze_frames.columns
    assert len(freeze_frames.columns) == len(gamestates)
    # In LTR, the keeper of the attacking team should have low x coordinates.
    att_keeper_coo = []
    for (game_id, action_id), ff in freeze_frames.iterrows():
        if ff["freeze_frame_360_a0"] is not None:
            df = pd.DataFrame.from_records(ff["freeze_frame_360_a0"])
            df[["game_id", "action_id"]] = game_id, action_id
            att_keeper_coo.append(df.loc[df.teammate & df.keeper])
    att_keeper_coo = pd.concat(att_keeper_coo)
    assert sum(att_keeper_coo["x"] > (105 - 105 / 3)) <= 8  # The data has some bugs ¯\_(ツ)_/¯
    assert att_keeper_coo["x"].mean() < (105 / 3)


def test_intended_rename():
    """It should rename the wrapped function."""
    assert fs.intended(fs.endlocation).__name__ == "intended_endlocation"


def test_intended_end_location(gamestates):
    """It should modify the end location of failed passes."""
    df_actual = fs.endlocation(gamestates)
    df_intended = fs.intended(fs.endlocation)(gamestates)
    diff = pd.concat([df_actual, df_intended]).drop_duplicates(keep=False).index
    assert len(diff) > 0
    assert (gamestates[0].loc[diff, "result_name"] != "success").all()
    # plot the action with the most likely receiver
    PLOT = False
    if PLOT:
        from matplotlib import pyplot as plt

        from unxpass.visualization import plot_action

        sample = (3795107, 1974)
        action = gamestates[0].loc[sample]
        receiver = df_intended.loc[sample]
        _, ax = plt.subplots(1)
        plot_action(action, ax=ax)
        ax.plot(
            action["end_x"],
            action["end_y"],
            marker="o",
            markersize=20,
            markeredgecolor="green",
            markerfacecolor=(1, 1, 0, 0.5),
        )
        ax.plot(
            receiver["end_x_a0"],
            receiver["end_y_a0"],
            marker="o",
            markersize=20,
            markeredgecolor="red",
            markerfacecolor=(1, 1, 0, 0.5),
        )
        plt.show()


def test_nb_opp_in_path(gamestates):
    """It should return the number of opponents in the path of the pass."""
    df = fs.nb_opp_in_path(gamestates)
    assert "nb_opp_in_path_a0" in df.columns
    assert len(df.columns) == len(gamestates)
    assert df.loc[(3795107, 0), "nb_opp_in_path_a0"] == 0
    assert df.loc[(3795107, 36), "nb_opp_in_path_a0"] == 1


def test_dist_defender(gamestates):
    """It should return the distance to the closest defender."""
    df = fs.dist_defender(gamestates)
    assert "dist_defender_start_a0" in df.columns
    assert "dist_defender_end_a0" in df.columns
    assert "dist_defender_action_a0" in df.columns
    assert len(df.columns) == 3 * len(gamestates)
