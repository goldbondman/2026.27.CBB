"""Frozen R1/R2 date-batched recursive offense/defense state model."""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class RecursiveOD:
    base_k: float = 0.12
    maturity_amplitude: float = 0.20
    maturity_horizon: int = 10
    multiplier_min: float = 0.5
    multiplier_max: float = 1.5
    use_r2: bool = True
    offense: dict[str, float] = field(default_factory=dict)
    defense: dict[str, float] = field(default_factory=dict)
    games: dict[str, int] = field(default_factory=dict)

    def initialize_season(
        self,
        previous: dict[str, tuple[float, float]],
        returning_minutes: dict[str, float] | None = None,
        transfer_experience_z: dict[str, float] | None = None,
        transition_scale: float = 1.20,
        returning_coefficient: float = 0.30,
        returning_center: float = 0.375,
        transfer_coefficient: float = 0.06,
    ) -> None:
        """Apply the frozen R1 net transition modifiers to prior O/D states.

        The modifier scales the prior-state pair symmetrically; callers must supply
        independently reconstructed, prior-season-only roster inputs.
        """
        returning_minutes = returning_minutes or {}
        transfer_experience_z = transfer_experience_z or {}
        self.offense.clear()
        self.defense.clear()
        self.games.clear()
        for team in sorted(previous):
            modifier = transition_scale
            modifier += returning_coefficient * (
                returning_minutes.get(team, returning_center) - returning_center
            )
            modifier += transfer_coefficient * transfer_experience_z.get(team, 0.0)
            modifier = min(max(modifier, 0.55), 1.45)
            offense, defense = previous[team]
            self.offense[team] = offense * modifier
            self.defense[team] = defense * modifier

    def predict(self, team: str, opponent: str) -> float:
        return 100.0 + self.offense.get(team, 0.0) - self.defense.get(opponent, 0.0)

    def _k(self, team: str, possessions: float) -> float:
        if not self.use_r2:
            return self.base_k
        maturity = min(self.games.get(team, 0) / self.maturity_horizon, 1.0)
        multiplier = (possessions / 70.0) * (1 + self.maturity_amplitude * (1 - 2 * maturity))
        multiplier = min(max(multiplier, self.multiplier_min), self.multiplier_max)
        return self.base_k * multiplier

    def run(self, rows: pd.DataFrame) -> pd.DataFrame:
        required = {"date", "game_id", "team_id", "opponent_id", "ortg", "possessions"}
        if required - set(rows):
            raise ValueError(f"missing model fields: {sorted(required - set(rows))}")
        ordered = rows.sort_values(["date", "game_id", "team_id"], kind="mergesort").copy()
        predictions: list[dict] = []
        for date, batch in ordered.groupby("date", sort=True):
            updates = []
            for row in batch.itertuples(index=False):
                prediction = self.predict(str(row.team_id), str(row.opponent_id))
                predictions.append(
                    {
                        "date": date,
                        "game_id": row.game_id,
                        "team_id": row.team_id,
                        "prediction": prediction,
                        "actual": row.ortg,
                        "prior_games": self.games.get(str(row.team_id), 0),
                    }
                )
                updates.append(
                    (
                        str(row.team_id),
                        str(row.opponent_id),
                        float(row.ortg) - prediction,
                        self._k(str(row.team_id), float(row.possessions)),
                    )
                )
            # Crucial: prediction of the entire date precedes every state update.
            # Frozen R1/R2 semantics apply the FULL k*residual to O_A and D_B.
            for team, opponent, residual, k in updates:
                self.offense[team] = self.offense.get(team, 0.0) + k * residual
                self.defense[opponent] = self.defense.get(opponent, 0.0) - k * residual
                self.games[team] = self.games.get(team, 0) + 1
        return pd.DataFrame(predictions)
