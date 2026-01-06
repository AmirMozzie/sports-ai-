"""
player_impact.py
----------------
Phase 3: Player Impact Estimation

Responsibilities:
- Quantify individual player impact on match outcome
- Combine talent score, visual score, and context
"""

class PlayerImpactEstimator:
    """
    Computes Player Impact Score (PIS) using an explainable formula.
    """

    def __init__(self,
                 talent_weight: float = 0.45,
                 visual_weight: float = 0.35,
                 context_weight: float = 0.20):

        self.talent_weight = talent_weight
        self.visual_weight = visual_weight
        self.context_weight = context_weight

    def compute_impact(self,
                       talent_score: float,
                       visual_score: float,
                       context_factor: float) -> dict:
        """
        Compute Player Impact Score.

        Parameters
        ----------
        talent_score : float
        visual_score : float
        context_factor : float
            Match context (0–1), e.g. opponent difficulty

        Returns
        -------
        dict
        """

        impact_score = (
            self.talent_weight * talent_score +
            self.visual_weight * visual_score +
            self.context_weight * context_factor
        )

        return {
            "player_impact_score": round(impact_score, 3),
            "components": {
                "talent_contribution": round(
                    self.talent_weight * talent_score, 3
                ),
                "visual_contribution": round(
                    self.visual_weight * visual_score, 3
                ),
                "context_contribution": round(
                    self.context_weight * context_factor, 3
                )
            }
        }
