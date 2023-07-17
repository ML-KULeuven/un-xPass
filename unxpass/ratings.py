from typing import Callable

import numpy as np
import pandas as pd

from unxpass.components import pass_selection, pass_success, pass_value
from unxpass.datasets import PassesDataset


def typical_pass(pass_selection_surface):
    """Get typical pass"""
    # get cell with max value
    y_t, x_t = np.unravel_index(pass_selection_surface.argmax(), pass_selection_surface.shape)
    # map cell index to pitch coordinates
    y_dim, x_dim = pass_selection_surface.shape
    y_t = y_t / y_dim * 68 + 68 / y_dim / 2
    x_t = x_t / x_dim * 105 + 105 / x_dim / 2
    return x_t, y_t


def _get_cell_indexes(x, y, x_bins=104, y_bins=68):
    x_bin = np.clip(x / 105 * x_bins, 0, x_bins - 1).astype(np.uint8)
    y_bin = np.clip(y / 68 * y_bins, 0, y_bins - 1).astype(np.uint8)
    return x_bin, y_bin


class CreativeDecisionRating:
    def __init__(
        self,
        pass_selection_component: pass_selection.SoccerMapComponent,
        pass_success_component: pass_success.XGBoostComponent,
        pass_value_component: pass_value.VaepModel,
    ):
        self.pass_value_component = pass_value_component
        self.pass_selection_component = pass_selection_component
        self.pass_success_component = pass_success_component

    def rate(self, db, dataset: Callable):
        # get the actual start and end location of each pass
        data = dataset(
            xfns={
                "startlocation": ["start_x_a0", "start_y_a0"],
                "endlocation": ["end_x_a0", "end_y_a0"],
            },
            yfns=["success"],
        )
        df_ratings = pd.concat([data.features, data.labels], axis=1).rename(
            columns={
                "start_x_a0": "start_x",
                "start_y_a0": "start_y",
                "end_x_a0": "true_end_x",
                "end_y_a0": "true_end_y",
            }
        )

        # get pass selection probabilities
        pass_selection_surfaces = self.pass_selection_component.predict_surface(dataset)

        # get the typical pass
        for game_id in pass_selection_surfaces:
            for action_id in pass_selection_surfaces[game_id]:
                surface = pass_selection_surfaces[game_id][action_id]
                df_ratings.loc[
                    (game_id, action_id), ["typical_end_x", "typical_end_y"]
                ] = typical_pass(surface)

        # get pass success probabilities
        data_pass_success = self.pass_success_component.initialize_dataset(dataset)
        feat_true_pass_succes = data_pass_success
        feat_typical_pass_succes = data_pass_success.apply_overrides(
            db,
            df_ratings[["typical_end_x", "typical_end_y"]].rename(
                columns={"typical_end_x": "end_x", "typical_end_y": "end_y"}
            ),
        )
        df_ratings["true_p_success"] = self.pass_success_component.predict(feat_true_pass_succes)
        df_ratings["typical_p_success"] = self.pass_success_component.predict(
            feat_typical_pass_succes
        )

        # get pass value
        data_pass_value = self.pass_value_component.offensive_model.initialize_dataset(dataset)
        feat_true_pass_value_succes = data_pass_value.apply_overrides(
            db,
            df_ratings[[]].assign(result_id=1, result_name="success"),
        )
        df_ratings["true_value_success"] = self.pass_value_component.predict(
            feat_true_pass_value_succes
        )
        feat_typical_pass_value_succes = data_pass_value.apply_overrides(
            db,
            df_ratings[["typical_end_x", "typical_end_y"]]
            .rename(columns={"typical_end_x": "end_x", "typical_end_y": "end_y"})
            .assign(result_id=1, result_name="success"),
        )
        df_ratings["typical_value_success"] = self.pass_value_component.predict(
            feat_typical_pass_value_succes
        )
        feat_true_pass_value_fail = data_pass_value.apply_overrides(
            db,
            df_ratings[[]].assign(result_id=0, result_name="fail"),
        )
        df_ratings["true_value_fail"] = self.pass_value_component.predict(
            feat_true_pass_value_fail
        )
        feat_typical_pass_value_fail = data_pass_value.apply_overrides(
            db,
            df_ratings[["typical_end_x", "typical_end_y"]]
            .rename(columns={"typical_end_x": "end_x", "typical_end_y": "end_y"})
            .assign(result_id=0, result_name="fail"),
        )
        df_ratings["typical_value_fail"] = self.pass_value_component.predict(
            feat_typical_pass_value_fail
        )

        df_ratings["CDR"] = (
            df_ratings["true_p_success"] * df_ratings["true_value_success"]
            + (1 - df_ratings["true_p_success"]) * df_ratings["true_value_fail"]
            - df_ratings["typical_p_success"] * df_ratings["typical_value_success"]
            + (1 - df_ratings["typical_p_success"]) * df_ratings["typical_value_fail"]
        )

        return df_ratings
