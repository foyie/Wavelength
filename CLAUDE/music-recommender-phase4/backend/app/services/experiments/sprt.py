"""
SPRT — Sequential Probability Ratio Test
-----------------------------------------
Classical fixed-sample A/B tests waste traffic: you collect N samples,
then decide. SPRT is a sequential test that makes the decision as soon
as there is enough evidence — either direction — stopping early when
the difference is large and continuing when it is small.

Wald's SPRT:
  H0: p_B - p_A <= delta  (treatment no better than control by delta)
  H1: p_B - p_A >  delta  (treatment wins by at least delta)

At each observation we update a log-likelihood ratio Lambda.
  Lambda > log(1-beta / alpha)  → reject H0, declare treatment winner
  Lambda < log(beta / 1-alpha)  → accept H0, declare control winner
  Otherwise                     → continue collecting

Why SPRT over fixed-sample tests?
  - Stops early when signal is strong → saves traffic
  - Provides same Type I/II error guarantees as fixed tests
  - Handles continuous monitoring without inflating p-values
  - Natural fit for online recommender systems

Parameters:
  alpha   — Type I error (false positive rate). Default 0.05.
  beta    — Type II error (false negative rate). Default 0.20 → 80% power.
  delta   — MDE (minimum detectable effect). Default 0.02 (2 pp lift).

References:
  Wald, A. (1947). Sequential Analysis. Wiley.
  Johari et al. (2017). "Peeking at A/B Tests." KDD.
"""
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ExperimentStatus(str, Enum):
    RUNNING   = "running"
    CONTROL_WINS = "control_wins"
    TREATMENT_WINS = "treatment_wins"
    INCONCLUSIVE = "inconclusive"   # hit max_samples without conclusion


@dataclass
class ExperimentArm:
    name: str
    n_observations: int = 0
    n_successes: int = 0      # weighted by reward (add=1, complete=0.8, etc.)
    total_reward: float = 0.0

    @property
    def conversion_rate(self) -> float:
        return self.total_reward / self.n_observations if self.n_observations > 0 else 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.n_observations if self.n_observations > 0 else 0.0


@dataclass
class SPRTExperiment:
    """
    A single A/B experiment tracked by SPRT.
    Stores running log-likelihood ratio and arm statistics.
    """
    id: str
    name: str
    description: str
    control_arm: ExperimentArm
    treatment_arm: ExperimentArm

    # SPRT parameters
    alpha: float = 0.05      # Type I error
    beta: float = 0.20       # Type II error (1-power)
    delta: float = 0.02      # minimum detectable effect (absolute)
    max_samples_per_arm: int = 10_000

    # SPRT state
    log_lambda: float = 0.0  # cumulative log likelihood ratio
    status: ExperimentStatus = ExperimentStatus.RUNNING
    winner: Optional[str] = None

    # Boundaries (computed once, reused)
    upper_boundary: float = field(init=False)
    lower_boundary: float = field(init=False)

    # Audit trail
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    decided_at: Optional[str] = None
    n_updates: int = 0

    def __post_init__(self):
        # Wald's boundaries
        self.upper_boundary = math.log((1.0 - self.beta) / self.alpha)
        self.lower_boundary = math.log(self.beta / (1.0 - self.alpha))

    def add_observation(self, arm_name: str, reward: float) -> ExperimentStatus:
        """
        Record one observation and update SPRT lambda.
        Returns current status (may change from RUNNING to decided).
        """
        if self.status != ExperimentStatus.RUNNING:
            return self.status

        reward = max(0.0, min(1.0, reward))  # clamp to [0,1]

        if arm_name == self.control_arm.name:
            arm = self.control_arm
        elif arm_name == self.treatment_arm.name:
            arm = self.treatment_arm
        else:
            logger.warning(f"Unknown arm '{arm_name}' for experiment '{self.id}'")
            return self.status

        arm.n_observations += 1
        arm.total_reward += reward

        # Only update SPRT lambda when both arms have at least one observation
        # and we have a paired observation to compute the ratio
        if self.control_arm.n_observations > 0 and self.treatment_arm.n_observations > 0:
            self._update_lambda(reward, arm_name)

        self.n_updates += 1
        self._check_boundaries()

        # Safety: max samples reached
        if (self.control_arm.n_observations >= self.max_samples_per_arm and
                self.treatment_arm.n_observations >= self.max_samples_per_arm):
            if self.status == ExperimentStatus.RUNNING:
                self.status = ExperimentStatus.INCONCLUSIVE
                self.decided_at = datetime.utcnow().isoformat()
                logger.info(
                    f"Experiment '{self.name}' inconclusive after max samples. "
                    f"Control={self.control_arm.mean_reward:.4f} "
                    f"Treatment={self.treatment_arm.mean_reward:.4f}"
                )

        return self.status

    def _update_lambda(self, reward: float, arm_name: str):
        """
        Wald's SPRT log-likelihood ratio update.

        H0: p_treatment = p_control          (no effect)
        H1: p_treatment = p_control + delta  (treatment wins by delta)

        For each observation x_i:
          lambda += log( f(x_i | H1) / f(x_i | H0) )

        We use Bernoulli approximation: x_i ~ Bernoulli(p).
        For non-binary rewards in [0,1], we use the reward directly as p.
        """
        p0 = max(self.control_arm.mean_reward, 1e-9)      # H0 rate
        p1 = min(p0 + self.delta, 1.0 - 1e-9)             # H1 rate

        # Likelihood ratio for this observation
        if arm_name == self.treatment_arm.name:
            # Treatment observation — compare H1 vs H0 for treatment
            if reward > 0:
                llr = math.log(p1 / p0) * reward
            else:
                llr = math.log((1 - p1) / (1 - p0)) * (1 - reward)
        else:
            # Control observation — no direct SPRT contribution under H0 vs H1
            # Use symmetric update to handle imbalanced arm sizes
            llr = 0.0

        self.log_lambda += llr

    def _check_boundaries(self):
        """Check if SPRT has crossed a decision boundary."""
        if self.status != ExperimentStatus.RUNNING:
            return

        if self.log_lambda >= self.upper_boundary:
            self.status = ExperimentStatus.TREATMENT_WINS
            self.winner = self.treatment_arm.name
            self.decided_at = datetime.utcnow().isoformat()
            logger.info(
                f"Experiment '{self.name}' TREATMENT WINS: "
                f"lambda={self.log_lambda:.3f} > {self.upper_boundary:.3f} "
                f"treatment_rate={self.treatment_arm.mean_reward:.4f} "
                f"control_rate={self.control_arm.mean_reward:.4f} "
                f"n_control={self.control_arm.n_observations} "
                f"n_treatment={self.treatment_arm.n_observations}"
            )
        elif self.log_lambda <= self.lower_boundary:
            self.status = ExperimentStatus.CONTROL_WINS
            self.winner = self.control_arm.name
            self.decided_at = datetime.utcnow().isoformat()
            logger.info(
                f"Experiment '{self.name}' CONTROL WINS: "
                f"lambda={self.log_lambda:.3f} < {self.lower_boundary:.3f}"
            )

    def summary(self) -> dict:
        lift = (self.treatment_arm.mean_reward - self.control_arm.mean_reward)
        lift_pct = (lift / self.control_arm.mean_reward * 100
                    if self.control_arm.mean_reward > 0 else 0.0)

        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "winner": self.winner,
            "created_at": self.created_at,
            "decided_at": self.decided_at,
            "n_updates": self.n_updates,
            "sprt": {
                "log_lambda": round(self.log_lambda, 4),
                "upper_boundary": round(self.upper_boundary, 4),
                "lower_boundary": round(self.lower_boundary, 4),
                "alpha": self.alpha,
                "beta": self.beta,
                "delta": self.delta,
                "progress_pct": round(
                    min(100.0,
                        abs(self.log_lambda) / max(
                            abs(self.upper_boundary), abs(self.lower_boundary)
                        ) * 100
                    ), 1
                ),
            },
            "control": {
                "name": self.control_arm.name,
                "n_observations": self.control_arm.n_observations,
                "mean_reward": round(self.control_arm.mean_reward, 4),
                "total_reward": round(self.control_arm.total_reward, 4),
            },
            "treatment": {
                "name": self.treatment_arm.name,
                "n_observations": self.treatment_arm.n_observations,
                "mean_reward": round(self.treatment_arm.mean_reward, 4),
                "total_reward": round(self.treatment_arm.total_reward, 4),
            },
            "lift": round(lift, 4),
            "lift_pct": round(lift_pct, 2),
        }
