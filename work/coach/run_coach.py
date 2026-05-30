#!/usr/bin/env python3
"""
UNIQA Conversion Coach — Journey Simulation & LLM-Powered Intervention
======================================================================

Simulates a persona walking through the 15-step UNIQA health insurance
online calculator funnel. Detects hesitation signals, decides whether
and how to intervene, and generates a coaching message via a local LLM
(Qwen2.5-7B-Instruct running via transformers on the Leonardo GPU cluster).

Architecture (3 layers + LLM):
  1. Journey State Machine — 15 funnel steps as a dataclass/enum
  2. Detection Layer — computes hesitation score from behavioral signals
  3. Decision Layer — rule-based intervention type selection
  4. LLM Call — generates the coaching message text (not the decision)

The LLM only writes the message — it never makes decisions about whether
or how to intervene. That logic belongs to the Detection and Decision layers.

Usage:
    python run_coach.py --test                  # Quick LLM smoke test
    python run_coach.py --persona franz         # Full simulation, Franz
    python run_coach.py --persona judith        # Full simulation, Judith
    python run_coach.py --persona peter         # Full simulation, Peter
    python run_coach.py --persona franz --seed 42  # Reproducible run

References:
    - tracks/insurance-uniqa/personas.json
    - tracks/insurance-uniqa/uniqa-funnel-doc_en.md
    - tracks/insurance-uniqa/Track_AI_Guided_Conversion_Flow_EN.md
    - tracks/insurance-uniqa/personas_comparison_matrix.md
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path resolution — works from any working directory
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TRACKS_DIR = REPO_ROOT / "tracks" / "insurance-uniqa"
PERSONAS_PATH = TRACKS_DIR / "personas.json"
RESULTS_DIR = SCRIPT_DIR / "results"

# Model path: env var with fallback to Leonardo scratch default
DEFAULT_MODEL_PATH = (
    "/leonardo_scratch/large/usertrain/a08trc28/models/Qwen--Qwen2.5-7B-Instruct"
)

# Drop-off rates from UNIQA funnel analysis (Dec 2025 – Feb 2026)
# These are CONDITIONAL on reaching that step, not absolute.
DROPOFF_INITIAL_PRICE = 0.66  # Step 4  — tariff selection, first price display
DROPOFF_ADDON_COVERAGE = 0.24  # Step 5  — add-on coverage (hospital path only)
DROPOFF_FINAL_PRICE = 0.78  # Step 7  — final price after health questions

# Conversion baseline: ~5.6% (1,000 starters → ~56 completions)
BASELINE_CONVERSION = 0.056

# Hesitation threshold — scores above this trigger an intervention
HESITATION_THRESHOLD = 0.65

# LLM generation parameters
MAX_NEW_TOKENS = 200
LLM_TEMPERATURE = 0.7
TOP_P = 0.9

# Persona → segment key mapping for CLI
PERSONA_SEGMENT_MAP = {
    "judith": "segment_1",
    "franz": "segment_2",
    "peter": "segment_3",
}


# ===================================================================
# 1. FUNNEL STEP DEFINITION (Journey State Machine)
# ===================================================================


class FunnelStep(Enum):
    """
    The 15 steps of the UNIQA health insurance online calculator.

    Each step carries:
      - num:        ordinal position in the funnel
      - key:        machine-readable identifier
      - label:      human-readable description
      - base_risk:  inherent hesitation risk for this step (0.0–1.0)
      - dwell_min:  minimum expected dwell time (seconds)
      - dwell_max:  maximum expected dwell time (seconds)

    Steps 5 and 8–11 are the hospital/adviser path and are OUT OF SCOPE
    for coaching. The in-scope path (private doctor + myself + Start/Optimal)
    is: 1 → 2 → 3 → 4 → 6 → 7 → 12 → 13 → 14 → 15.
    """

    # Phase: Inputs
    COVERAGE_TYPE = (
        1, "coverage_type", "Where do you want coverage?",
        0.10, 5, 10,
    )
    INSURED_PERSON = (
        2, "insured_person", "Who should be insured?",
        0.10, 3, 7,
    )
    PERSONAL_DATA = (
        3, "personal_data", "Personal data for premium estimate",
        0.30, 15, 35,
    )
    # Phase: Product — first price display, 66 % real drop-off
    TARIFF_SELECTION = (
        4, "tariff_selection", "Tariff selection – first price display",
        0.70, 25, 65,
    )
    # Phase: Product — hospital path add-ons (OUT OF SCOPE)
    ADDON_COVERAGE = (
        5, "addon_coverage", "Add-on coverage selection (hospital path only)",
        0.00, 10, 25,
    )
    # Phase: Inputs — health assessment
    HEALTH_QUESTIONS = (
        6, "health_questions", "Health questions",
        0.35, 30, 90,
    )
    # Phase: Recommendation — final price, 78 % real drop-off
    FINAL_PRICE = (
        7, "final_price", "Final price after health assessment",
        0.80, 20, 55,
    )
    # Phase: Advisor path (OUT OF SCOPE)
    ADVISOR_LOCATION = (
        8, "advisor_location", "Consultation location preference",
        0.00, 8, 18,
    )
    ADVISOR_CUSTOMER_STATUS = (
        9, "advisor_customer_status", "Customer status",
        0.00, 5, 12,
    )
    ADVISOR_PROVINCE_SERVICE = (
        10, "advisor_province_service", "Province & service selection",
        0.00, 6, 15,
    )
    ADVISOR_DATE_BOOKING = (
        11, "advisor_date_booking", "Appointment date selection",
        0.00, 8, 20,
    )
    # Phase: Closing — online purchase (IN SCOPE)
    PERSONAL_DETAILS = (
        12, "personal_details", "Personal details (name, address, contact)",
        0.15, 20, 45,
    )
    CONTRACT_TERMS = (
        13, "contract_terms", "Contract terms & insurance start date",
        0.20, 12, 30,
    )
    PAYMENT_DETAILS = (
        14, "payment_details", "Payment details",
        0.20, 15, 35,
    )
    CONFIRMATION = (
        15, "confirmation", "Purchase confirmation",
        0.10, 8, 20,
    )

    def __init__(
        self, num: int, key: str, label: str,
        base_risk: float, dwell_min: int, dwell_max: int,
    ):
        self.num = num
        self.key = key
        self.label = label
        self.base_risk = base_risk
        self.dwell_min = dwell_min
        self.dwell_max = dwell_max

    @property
    def is_price_step(self) -> bool:
        """Steps where a concrete price is displayed."""
        return self in (FunnelStep.TARIFF_SELECTION, FunnelStep.FINAL_PRICE)

    @property
    def is_in_scope(self) -> bool:
        """Whether this step belongs to the coaching scope."""
        return self not in {
            FunnelStep.ADDON_COVERAGE,
            FunnelStep.ADVISOR_LOCATION,
            FunnelStep.ADVISOR_CUSTOMER_STATUS,
            FunnelStep.ADVISOR_PROVINCE_SERVICE,
            FunnelStep.ADVISOR_DATE_BOOKING,
        }

    @property
    def is_closing_step(self) -> bool:
        """Steps in the Closing phase (12–15)."""
        return self.num >= 12

    @property
    def is_early_step(self) -> bool:
        """Steps before the first price display."""
        return self.num <= 3

    @classmethod
    def by_key(cls, key: str) -> Optional["FunnelStep"]:
        """Look up a step by its string key."""
        for step in cls:
            if step.key == key:
                return step
        return None


# The in-scope path: steps the persona actually walks through
# for the private-doctor + myself + Start/Optimal journey.
IN_SCOPE_PATH: List[FunnelStep] = [
    FunnelStep.COVERAGE_TYPE,
    FunnelStep.INSURED_PERSON,
    FunnelStep.PERSONAL_DATA,
    FunnelStep.TARIFF_SELECTION,
    FunnelStep.HEALTH_QUESTIONS,
    FunnelStep.FINAL_PRICE,
    FunnelStep.PERSONAL_DETAILS,
    FunnelStep.CONTRACT_TERMS,
    FunnelStep.PAYMENT_DETAILS,
    FunnelStep.CONFIRMATION,
]

# Tariff data matching the real UNIQA calculator (May 2026)
TARIFFS = {
    "Start": {
        "annual_max_eur": 1400,
        "monthly_premium_eur": 38.74,
        "daily_eur": 1.27,
        "online_purchasable": True,
    },
    "Optimal": {
        "annual_max_eur": 2800,
        "monthly_premium_eur": 68.14,
        "daily_eur": 2.24,
        "online_purchasable": True,
    },
    "Opt. Plus": {
        "annual_max_eur": 4200,
        "monthly_premium_eur": 96.66,
        "daily_eur": 3.18,
        "online_purchasable": False,
    },
    "Premium": {
        "annual_max_eur": 8400,
        "monthly_premium_eur": 140.16,
        "daily_eur": 4.61,
        "online_purchasable": False,
    },
}


# ===================================================================
# 2. INTERVENTION TYPES (Decision Layer output)
# ===================================================================


class InterventionType(Enum):
    """
    What kind of coaching message to send. Determined entirely by
    rule-based logic in the Decision Layer — the LLM never chooses this.
    """
    PRICE_CONCERN = "price_concern"
    COVERAGE_CONFUSION = "coverage_confusion"
    GENERAL_HESITATION = "general_hesitation"


# Guidance snippets used in LLM prompt construction per intervention type
INTERVENTION_GUIDANCE: Dict[InterventionType, str] = {
    InterventionType.PRICE_CONCERN: (
        "The customer is concerned about the price. Address the cost directly — "
        "reframe it psychologically (daily cost), mention the value they get, "
        "and reassure them that they are getting a fair deal. "
        "If appropriate, mention the cheaper Start tariff as an alternative."
    ),
    InterventionType.COVERAGE_CONFUSION: (
        "The customer seems confused about what is covered. Clarify what the "
        "selected tariff includes in simple, concrete terms. "
        "Focus on the most relevant benefits for their profile "
        "(e.g., medical services, medications, therapeutic treatments, aids). "
        "Avoid insurance jargon."
    ),
    InterventionType.GENERAL_HESITATION: (
        "The customer is hesitating. Provide gentle reassurance and social proof. "
        "Remind them they can complete the purchase online right now, "
        "and that many customers in their situation found the right coverage here. "
        "Keep the tone warm and supportive, never pushy."
    ),
}


# ===================================================================
# 3. PERSONA PROFILE
# ===================================================================


@dataclass
class PersonaProfile:
    """
    Extracted persona traits that drive simulation behavior.

    All values are derived from UNIQA's segmentation research (n=4,004,
    Oct–Nov 2025) and persona sheets (May 2026), as recorded in
    personas.json and personas_comparison_matrix.md.
    """

    name: str
    segment: str
    segment_label: str
    age: int
    location: str
    occupation: str
    income_eur: int
    typical_quote: str

    # Behavioral parameters (0.0–1.0)
    price_sensitivity: float
    back_nav_probability: float
    overwhelm_tendency: float
    coaching_aversion: float

    # Which step this persona is most likely to drop off at
    primary_drop_off_step: str

    # Whether this persona actually wants human contact
    prefers_advisor: bool

    # Key concerns — used in LLM prompt construction
    key_concerns: List[str]

    @classmethod
    def from_persona_data(cls, persona_key: str, data: dict) -> "PersonaProfile":
        """
        Build a PersonaProfile from the personas.json segment data.

        The behavioral parameters are derived from the quantitative segment
        data and the qualitative persona archetypes. They are NOT one-to-one
        mappings — they are reasonable proxies tuned to produce realistic
        journey simulations.
        """
        seg = data["personas"][persona_key]
        arch = seg["persona_archetype"]
        income = seg["demographics"]["income_monthly_eur"]

        if persona_key == "segment_1":
            # Judith Berger — Rising Hybrids
            # High income, researches independently, wants advisor trust
            # Drops at initial price display; 19 % online purchase share
            return cls(
                name=arch["name"],
                segment=persona_key,
                segment_label=seg["name_short"],
                age=arch["age"],
                location=arch["location"],
                occupation=arch["occupation"],
                income_eur=income,
                typical_quote=arch["typical_quote"],
                price_sensitivity=0.55,
                back_nav_probability=0.18,
                overwhelm_tendency=0.25,
                coaching_aversion=0.15,
                primary_drop_off_step="tariff_selection",
                prefers_advisor=True,
                key_concerns=[
                    "Researches independently online but wants advisor trust for final decisions",
                    "Price-performance balance is critical (87 % top decision driver)",
                    "Values tailored products and individualization — dislikes one-size-fits-all",
                    "Drops off at initial price display — price shock or "
                    "'better tariffs require advisor' frustration",
                    "81 % of her segment completes purchase in person — "
                    "online is for research, not necessarily for buying",
                ],
            )

        elif persona_key == "segment_2":
            # Franz Huber — Online Affine
            # Digital-first, price-sensitive, dislikes advisor handoff
            # Drops at final price; 69 % online purchase share
            return cls(
                name=arch["name"],
                segment=persona_key,
                segment_label=seg["name_short"],
                age=arch["age"],
                location=arch["location"],
                occupation=arch["occupation"],
                income_eur=income,
                typical_quote=arch["typical_quote"],
                price_sensitivity=0.80,
                back_nav_probability=0.10,
                overwhelm_tendency=0.12,
                coaching_aversion=0.30,
                primary_drop_off_step="final_price",
                prefers_advisor=False,
                key_concerns=[
                    "Wants to complete everything online — fast, simple, transparent",
                    "Highly price-sensitive, compares offers across multiple sites (82 %)",
                    "Dislikes advisor handoff — 47 % of this segment has no advisor at all",
                    "Drops off at final price when it exceeds the initial estimate",
                    "Online purchase option is a primary purchase criterion (36 % vs 11 % for S1)",
                ],
            )

        else:
            # Peter Wagner — Service Affine
            # Low engagement, easily overwhelmed, needs human reassurance
            # Drops very early; 34 % online purchase share but prefers service
            return cls(
                name=arch["name"],
                segment=persona_key,
                segment_label=seg["name_short"],
                age=arch["age"],
                location=arch["location"],
                occupation=arch["occupation"],
                income_eur=income,
                typical_quote=arch["typical_quote"],
                price_sensitivity=0.45,
                back_nav_probability=0.30,
                overwhelm_tendency=0.70,
                coaching_aversion=0.05,
                primary_drop_off_step="personal_data",
                prefers_advisor=True,
                key_concerns=[
                    "Does not want to deal with insurance complexity — "
                    "'just tell me what I need'",
                    "Gets overwhelmed by too many decisions and form fields",
                    "Prefers customer service contact over self-service at every journey step",
                    "Drops off very early — before even reaching tariff selection",
                    "43 % hospitalization rate in last 3 years — genuinely needs coverage "
                    "but may not complete online",
                    "Conversion for this segment = qualified service contact, "
                    "not necessarily online purchase",
                ],
            )


# ===================================================================
# 4. JOURNEY RECORDS
# ===================================================================


@dataclass
class StepRecord:
    """What happened at a single funnel step."""

    step: FunnelStep
    dwell_time: float
    hesitation_score: float
    back_navigated: bool
    dropped: bool
    intervention_triggered: bool
    intervention_type: Optional[InterventionType] = None
    coach_message: Optional[str] = None
    selected_option: Optional[str] = None


@dataclass
class JourneyResult:
    """Complete outcome of a simulated journey."""

    persona_name: str
    persona_segment: str
    steps: List[StepRecord]
    total_time: float
    outcome: str  # converted | dropped | intervened+converted | intervened+dropped | out_of_scope
    interventions: List[Tuple[str, InterventionType, Optional[str]]]
    llm_prompt: Optional[str] = None
    llm_response: Optional[str] = None
    seed: int = 0


# ===================================================================
# 5. DETECTION LAYER
# ===================================================================


class DetectionLayer:
    """
    Computes a hesitation score (0.0–1.0) from behavioral signals.

    The score is a weighted combination of three components:

        hesitation = W_DWELL * dwell_score
                   + W_BACK_NAV * back_nav_score
                   + W_STEP_RISK * step_risk_score

    Where:
      - dwell_score:    how much actual dwell exceeds expected threshold
      - back_nav_score: binary — did the persona navigate backwards?
      - step_risk_score: inherent risk of the step, amplified by persona traits

    This is entirely rule-based. The LLM is never involved in computing
    hesitation or deciding whether to intervene.
    """

    W_DWELL = 0.40
    W_BACK_NAV = 0.30
    W_STEP_RISK = 0.30

    @staticmethod
    def compute_hesitation(
        step: FunnelStep,
        dwell_time: float,
        back_navigated: bool,
        persona: PersonaProfile,
    ) -> float:
        """
        Compute hesitation score for the current step.

        Parameters
        ----------
        step : FunnelStep
            The current funnel step.
        dwell_time : float
            Time the persona spent on this step (seconds).
        back_navigated : bool
            Whether the persona navigated backwards from this step.
        persona : PersonaProfile
            The persona whose behavior is being simulated.

        Returns
        -------
        float
            Hesitation score between 0.0 and 1.0.
        """
        # --- Dwell-time component ---
        # Use the midpoint of expected min/max as the "normal" threshold.
        threshold = (step.dwell_min + step.dwell_max) / 2.0
        # Ratio of actual to threshold, capped at 1.0
        dwell_ratio = min(dwell_time / max(threshold, 0.5), 1.5)
        # Sub-linear ramp — moderate overshoots don't spike the score,
        # but extreme dwell times push it high.
        if dwell_ratio <= 1.0:
            dwell_score = dwell_ratio * 0.5  # below threshold: mild contribution
        else:
            dwell_score = 0.5 + (dwell_ratio - 1.0) * 1.0  # above threshold: steeper
        dwell_score = min(dwell_score, 1.0)

        # --- Back-navigation component ---
        back_nav_score = 1.0 if back_navigated else 0.0

        # --- Step-risk component ---
        step_risk = step.base_risk

        # Price-sensitive personas perceive price steps as riskier
        if step.is_price_step:
            step_risk *= 1.0 + persona.price_sensitivity * 0.4

        # Overwhelmed personas perceive early form steps as riskier
        if step.is_early_step:
            step_risk *= 1.0 + persona.overwhelm_tendency * 0.5

        # Clamp
        step_risk = min(step_risk, 1.0)

        # --- Weighted sum ---
        score = (
            DetectionLayer.W_DWELL * dwell_score
            + DetectionLayer.W_BACK_NAV * back_nav_score
            + DetectionLayer.W_STEP_RISK * step_risk
        )

        return round(min(max(score, 0.0), 1.0), 3)


# ===================================================================
# 6. DECISION LAYER
# ===================================================================


class DecisionLayer:
    """
    Rule-based decision engine: given a step and hesitation score,
    determine WHAT kind of coaching message to send.

    The decision is based on:
      - Which funnel step triggered the intervention
      - The persona's overwhelm tendency and preferences
      - The step type (price step, data-entry step, closing step)

    The LLM is NEVER consulted for this decision — it only generates the
    final message text after the decision has been made.
    """

    @staticmethod
    def decide(
        step: FunnelStep,
        hesitation_score: float,
        persona: PersonaProfile,
    ) -> Optional[InterventionType]:
        """
        Decide whether and how to intervene.

        Returns None if hesitation is below threshold (no intervention needed).
        """
        if hesitation_score < HESITATION_THRESHOLD:
            return None

        # --- Price steps (4: tariff selection, 7: final price) ---
        if step == FunnelStep.TARIFF_SELECTION:
            # At tariff selection, price is the primary concern.
            # Overwhelmed personas additionally struggle with coverage understanding.
            if persona.overwhelm_tendency > 0.4:
                return InterventionType.COVERAGE_CONFUSION
            return InterventionType.PRICE_CONCERN

        if step == FunnelStep.FINAL_PRICE:
            # At final price, always a price concern — especially when
            # it differs from the provisional estimate.
            return InterventionType.PRICE_CONCERN

        # --- Data-entry steps with trust barriers ---
        if step in (FunnelStep.HEALTH_QUESTIONS, FunnelStep.PERSONAL_DATA):
            if persona.overwhelm_tendency > 0.5:
                return InterventionType.GENERAL_HESITATION
            return InterventionType.COVERAGE_CONFUSION

        # --- Early branching steps ---
        if step in (FunnelStep.COVERAGE_TYPE, FunnelStep.INSURED_PERSON):
            return InterventionType.COVERAGE_CONFUSION

        # --- Closing steps — push over the finish line ---
        if step.is_closing_step:
            return InterventionType.GENERAL_HESITATION

        # Fallback
        return InterventionType.GENERAL_HESITATION


# ===================================================================
# 7. JOURNEY SIMULATOR
# ===================================================================


class JourneySimulator:
    """
    Simulates a persona walking through the UNIQA health insurance funnel.

    At each step the simulator:
      1. Generates a randomized dwell time (persona-weighted)
      2. Optionally simulates back-navigation
      3. Computes a hesitation score via the Detection layer
      4. Decides whether to drop off
      5. If hesitation > threshold, triggers the Decision layer
    """

    def __init__(self, persona: PersonaProfile, seed: int = None):
        self.persona = persona
        self.rng = random.Random(seed if seed is not None else hash(persona.name) % (2 ** 31))
        self.detector = DetectionLayer()
        self.decider = DecisionLayer()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dwell_time(self, step: FunnelStep) -> float:
        """Generate a randomized dwell time for this step, modified by persona traits."""
        base = self.rng.uniform(step.dwell_min, step.dwell_max)

        if step.is_price_step:
            # Price-sensitive personas stare longer at price tables
            factor = 1.0 + self.persona.price_sensitivity * self.rng.uniform(0.7, 1.6)
            base *= factor
        elif step.is_early_step:
            # Overwhelmed personas take longer on early forms
            factor = 1.0 + self.persona.overwhelm_tendency * self.rng.uniform(0.4, 2.2)
            base *= factor
        elif step.is_closing_step and not self.persona.prefers_advisor:
            # Digital-native personas breeze through closing
            base *= self.rng.uniform(0.55, 0.90)

        return round(max(base, 0.5), 1)

    def _should_back_navigate(self, step: FunnelStep) -> bool:
        """Determine whether the persona navigates backwards from this step."""
        prob = self.persona.back_nav_probability

        if step.is_price_step:
            prob += self.persona.price_sensitivity * 0.22
        if step.is_early_step:
            prob += self.persona.overwhelm_tendency * 0.28
        if step == FunnelStep.FINAL_PRICE:
            # Gap between provisional and final price triggers re-checking
            prob += 0.14

        return self.rng.random() < min(prob, 0.62)

    def _should_drop(self, step: FunnelStep, hesitation: float) -> bool:
        """
        Determine whether the persona drops off at this step.

        Drop probability is a function of:
          - The step's base risk (scaled down — simulation is not 1:1 with reality)
          - Whether this is the persona's primary drop-off step
          - Persona-specific modifiers (price sensitivity, overwhelm)
          - Whether an intervention was triggered (reduces drop-off chance)
        """
        base_prob = step.base_risk * 0.45

        # Boost at the persona's known weak point
        if step.key == self.persona.primary_drop_off_step:
            base_prob += 0.22

        if step.is_price_step:
            base_prob += self.persona.price_sensitivity * 0.20
        if step.is_early_step:
            base_prob += self.persona.overwhelm_tendency * 0.22

        # Coach intervention reduces the chance of dropping
        if hesitation >= HESITATION_THRESHOLD:
            base_prob *= 0.50

        return self.rng.random() < min(base_prob, 0.88)

    def _pick_tariff(self) -> str:
        """Simulate which tariff the persona selects at step 4."""
        r = self.rng.random()

        if self.persona.prefers_advisor and r < 0.18:
            # Judith sometimes clicks Opt. Plus out of curiosity, then sees
            # "advisory required" and may navigate back
            return "Opt. Plus"
        elif self.persona.price_sensitivity > 0.70 and r < 0.55:
            # Price-sensitive Franz leans toward the cheaper Start tariff
            return "Start"
        elif self.persona.price_sensitivity > 0.65 and r < 0.70:
            return "Start"
        else:
            # Most personas gravitate toward Optimal (mid-tier)
            return "Optimal"

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def run(self) -> JourneyResult:
        """
        Run the full journey simulation.

        The simulation walks the in-scope path (10 steps). Early branching
        (coverage type, insured person) is handled with persona-weighted
        probabilities — overwhelmed personas may accidentally select
        out-of-scope paths, which cleanly exit with outcome "out_of_scope".
        """
        steps: List[StepRecord] = []
        interventions: List[Tuple[str, InterventionType, Optional[str]]] = []
        total_time = 0.0

        # ---- Step 1: Coverage type ----
        step = FunnelStep.COVERAGE_TYPE
        dwell = self._dwell_time(step)
        back_nav = self._should_back_navigate(step)
        hesitation = self.detector.compute_hesitation(step, dwell, back_nav, self.persona)

        # Overwhelmed personas (Peter) may accidentally select hospital
        if self.persona.overwhelm_tendency > 0.55 and self.rng.random() < 0.22:
            steps.append(StepRecord(
                step=step, dwell_time=dwell, hesitation_score=hesitation,
                back_navigated=back_nav, dropped=True,
                intervention_triggered=False,
                selected_option="hospital",
            ))
            return JourneyResult(
                persona_name=self.persona.name,
                persona_segment=self.persona.segment_label,
                steps=steps, total_time=dwell,
                outcome="out_of_scope", interventions=[],
                seed=self.rng.randint(0, 2 ** 31 - 1),
            )

        intervention = self.decider.decide(step, hesitation, self.persona)
        steps.append(StepRecord(
            step=step, dwell_time=dwell, hesitation_score=hesitation,
            back_navigated=back_nav, dropped=False,
            intervention_triggered=intervention is not None,
            intervention_type=intervention,
            selected_option="doctor_visits",
        ))
        total_time += dwell

        # ---- Step 2: Insured person ----
        step = FunnelStep.INSURED_PERSON
        dwell = self._dwell_time(step)
        back_nav = self._should_back_navigate(step)
        hesitation = self.detector.compute_hesitation(step, dwell, back_nav, self.persona)

        if self.persona.overwhelm_tendency > 0.65 and self.rng.random() < 0.12:
            steps.append(StepRecord(
                step=step, dwell_time=dwell, hesitation_score=hesitation,
                back_navigated=back_nav, dropped=True,
                intervention_triggered=False,
                selected_option="other_persons",
            ))
            return JourneyResult(
                persona_name=self.persona.name,
                persona_segment=self.persona.segment_label,
                steps=steps, total_time=total_time + dwell,
                outcome="out_of_scope", interventions=[],
                seed=self.rng.randint(0, 2 ** 31 - 1),
            )

        intervention = self.decider.decide(step, hesitation, self.persona)
        steps.append(StepRecord(
            step=step, dwell_time=dwell, hesitation_score=hesitation,
            back_navigated=back_nav, dropped=False,
            intervention_triggered=intervention is not None,
            intervention_type=intervention,
            selected_option="myself",
        ))
        total_time += dwell

        # ---- Remaining in-scope steps (3 → 4 → 6 → 7 → 12 → 13 → 14 → 15) ----
        remaining = [
            FunnelStep.PERSONAL_DATA,
            FunnelStep.TARIFF_SELECTION,
            FunnelStep.HEALTH_QUESTIONS,
            FunnelStep.FINAL_PRICE,
            FunnelStep.PERSONAL_DETAILS,
            FunnelStep.CONTRACT_TERMS,
            FunnelStep.PAYMENT_DETAILS,
            FunnelStep.CONFIRMATION,
        ]

        for step in remaining:
            dwell = self._dwell_time(step)
            back_nav = self._should_back_navigate(step)
            hesitation = self.detector.compute_hesitation(
                step, dwell, back_nav, self.persona,
            )

            intervention = self.decider.decide(step, hesitation, self.persona)
            triggered = intervention is not None

            if triggered:
                interventions.append((step.key, intervention, None))

            # Tariff selection logic
            selected = None
            if step == FunnelStep.TARIFF_SELECTION:
                selected = self._pick_tariff()

            drop = self._should_drop(step, hesitation)

            steps.append(StepRecord(
                step=step, dwell_time=dwell, hesitation_score=hesitation,
                back_navigated=back_nav, dropped=drop,
                intervention_triggered=triggered,
                intervention_type=intervention,
                selected_option=selected,
            ))
            total_time += dwell

            if drop:
                outcome = "intervened+dropped" if triggered else "dropped"
                return JourneyResult(
                    persona_name=self.persona.name,
                    persona_segment=self.persona.segment_label,
                    steps=steps, total_time=total_time,
                    outcome=outcome, interventions=interventions,
                    seed=self.rng.randint(0, 2 ** 31 - 1),
                )

        # Completed all steps
        had_intervention = any(r.intervention_triggered for r in steps)
        outcome = "intervened+converted" if had_intervention else "converted"

        return JourneyResult(
            persona_name=self.persona.name,
            persona_segment=self.persona.segment_label,
            steps=steps, total_time=total_time,
            outcome=outcome, interventions=interventions,
            seed=self.rng.randint(0, 2 ** 31 - 1),
        )


# ===================================================================
# 8. LLM INTERFACE
# ===================================================================


class LLMCoach:
    """
    Wraps a local Qwen2.5-7B-Instruct model for coaching message generation.

    The LLM is ONLY used to generate the coaching message text. It is never
    consulted for the decision of WHETHER to intervene or WHAT kind of
    intervention to make — those decisions belong to the Detection and
    Decision layers.

    Uses Hugging Face transformers pipeline with device_map="auto" for
    multi-GPU support on the Leonardo cluster.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.pipe = None

    def _check_model_exists(self) -> bool:
        """Verify the model directory exists and contains expected files."""
        if not os.path.exists(self.model_path):
            print(f"[ERROR] Model path does not exist: {self.model_path}")
            print("[ERROR] Set the QWEN_MODEL_PATH environment variable, e.g.:")
            print('  export QWEN_MODEL_PATH=/path/to/Qwen2.5-7B-Instruct')
            return False
        # Basic sanity: check for config.json or pytorch_model.bin
        has_config = os.path.exists(os.path.join(self.model_path, "config.json"))
        has_safetensors = any(
            f.endswith(".safetensors")
            for f in os.listdir(self.model_path)
        ) if os.path.isdir(self.model_path) else False
        if not (has_config or has_safetensors):
            print(f"[WARN] Model path exists but looks incomplete: {self.model_path}")
            print("[WARN] Expected config.json or .safetensors files.")
        return True

    def load(self):
        """Lazy-load the transformers pipeline."""
        if self.pipe is not None:
            return

        if not self._check_model_exists():
            sys.exit(1)

        print(f"[LLM] Loading model from {self.model_path} ...")
        try:
            from transformers import pipeline

            self.pipe = pipeline(
                "text-generation",
                model=self.model_path,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True,
            )
            print(f"[LLM] Model loaded. Device: {self.pipe.device}")
        except Exception as exc:
            print(f"[ERROR] Failed to load model: {exc}")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_system_prompt(persona: PersonaProfile) -> str:
        """Build the system prompt framing the coach's role."""
        advisor_note = ""
        if persona.prefers_advisor:
            advisor_note = (
                " If the customer seems to want human contact, gently mention "
                "that an advisor is available — but always present the online "
                "option as equally valid."
            )
        else:
            advisor_note = (
                " Do NOT suggest speaking to an advisor unless the customer "
                "explicitly asks — this customer prefers self-service."
            )

        return (
            f"You are a friendly, supportive UNIQA health insurance assistant. "
            f"Your name is Coach. You help customers complete their online "
            f"health insurance purchase. Your tone is warm, clear, and never "
            f"salesy. You give honest information in plain language. "
            f"Keep responses to 2-4 sentences.{advisor_note}"
        )

    @staticmethod
    def _build_user_prompt(
        persona: PersonaProfile,
        intervention_type: InterventionType,
        step: FunnelStep,
    ) -> str:
        """Build the user prompt with persona context and intervention guidance."""
        guidance = INTERVENTION_GUIDANCE[intervention_type]

        # Pick a relevant tariff detail for price steps
        tariff_note = ""
        if step.is_price_step:
            tariff_note = (
                f"\nRelevant tariffs: Start at €38.74/month (€1.27/day, €1,400 annual max) "
                f"and Optimal at €68.14/month (€2.24/day, €2,800 annual max). "
                f"Both are available to purchase online right now."
            )

        return (
            f"Customer profile: {persona.name}, age {persona.age}, "
            f"{persona.occupation} from {persona.location}. "
            f"Monthly income approximately €{persona.income_eur}. "
            f"Their self-described approach: \"{persona.typical_quote}\"\n\n"
            f"They are currently at step {step.num} of 15: \"{step.label}\".\n"
            f"{tariff_note}\n"
            f"Situation: {guidance}\n\n"
            f"Write a short, warm coaching message to {persona.name} "
            f"that helps them move forward."
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        persona: PersonaProfile,
        intervention_type: InterventionType,
        step: FunnelStep,
    ) -> Tuple[str, str]:
        """
        Generate a coaching message.

        Returns
        -------
        (full_prompt, generated_response)
            full_prompt  — the complete text sent to the LLM
            generated_response — just the assistant's reply text
        """
        self.load()

        system_prompt = self._build_system_prompt(persona)
        user_prompt = self._build_user_prompt(persona, intervention_type, step)

        # Qwen2.5 chat template
        full_prompt = (
            f"<|system|>\n{system_prompt}\n"
            f"<|user|>\n{user_prompt}\n"
            f"<|assistant|>\n"
        )

        result = self.pipe(
            full_prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=LLM_TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
        )

        generated = result[0]["generated_text"]

        # Extract only the assistant's response (everything after the last
        # <|assistant|> marker)
        if "<|assistant|>" in generated:
            parts = generated.split("<|assistant|>")
            response = parts[-1].strip()
        else:
            response = generated[len(full_prompt):].strip()

        return full_prompt, response


# ===================================================================
# 9. OUTPUT FORMATTING
# ===================================================================


def format_journey_report(result: JourneyResult) -> str:
    """Format a JourneyResult as a human-readable report string."""
    lines = []
    w = 72

    lines.append("=" * w)
    lines.append("  UNIQA Conversion Coach — Journey Simulation Report")
    lines.append("=" * w)
    lines.append(f"  Persona:      {result.persona_name}")
    lines.append(f"  Segment:      {result.persona_segment}")
    lines.append(f"  Outcome:      {result.outcome}")
    lines.append(f"  Total time:   {result.total_time:.1f} s")
    lines.append(f"  Seed:         {result.seed}")
    lines.append("-" * w)

    # Column headers
    header = (
        f"  {'#':<3} {'Step':<24} {'Dwell':>7} {'Hesit':>7} "
        f"{'Back':>5} {'Intv':>5} {'Drop':>5}"
    )
    lines.append(header)
    lines.append("-" * w)

    for rec in result.steps:
        intv = "YES" if rec.intervention_triggered else " - "
        drop = "YES" if rec.dropped else " - "
        back = "yes" if rec.back_navigated else " - "
        lines.append(
            f"  {rec.step.num:<3} {rec.step.key:<24} "
            f"{rec.dwell_time:>6.1f}s {rec.hesitation_score:>7.3f} "
            f"{back:>5} {intv:>5} {drop:>5}"
        )
        if rec.selected_option:
            lines.append(f"       → Selected: {rec.selected_option}")

    lines.append("-" * w)
    n_interv = len(result.interventions)
    lines.append(f"  Interventions triggered: {n_interv}")
    for i, (step_key, itype, _) in enumerate(result.interventions, 1):
        lines.append(f"    {i}. Step '{step_key}' → {itype.value}")

    if result.llm_prompt or result.llm_response:
        lines.append("=" * w)
        lines.append("  LLM PROMPT")
        lines.append("=" * w)
        if result.llm_prompt:
            lines.append(result.llm_prompt)
        lines.append("=" * w)
        lines.append("  LLM RESPONSE")
        lines.append("=" * w)
        if result.llm_response:
            lines.append(result.llm_response)
        else:
            lines.append("  (no response — LLM was not called)")

    lines.append("=" * w)
    return "\n".join(lines)


def write_report(result: JourneyResult) -> Path:
    """Write the journey result to a timestamped file. Returns the file path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"run_{ts}.txt"

    # Build a structured text report
    parts = []
    parts.append("UNIQA Conversion Coach — Journey Simulation Report")
    parts.append("=" * 60)
    parts.append(f"Timestamp:     {ts} UTC")
    parts.append(f"Persona:       {result.persona_name}")
    parts.append(f"Segment:       {result.persona_segment}")
    parts.append(f"Outcome:       {result.outcome}")
    parts.append(f"Total time:    {result.total_time:.1f} s")
    parts.append(f"Seed:          {result.seed}")
    parts.append("")

    parts.append("FUNNEL STEPS")
    parts.append("-" * 60)
    for rec in result.steps:
        intv_flag = "[COACH]" if rec.intervention_triggered else "       "
        drop_flag = " DROP" if rec.dropped else "     "
        back_flag = " BACK" if rec.back_navigated else "     "
        opt = f"  -> {rec.selected_option}" if rec.selected_option else ""
        parts.append(
            f"{intv_flag}{drop_flag}{back_flag} "
            f"Step {rec.step.num:>2}: {rec.step.key:<28} "
            f"dwell={rec.dwell_time:>6.1f}s  hesitation={rec.hesitation_score:.3f}"
            f"{opt}"
        )

    parts.append("")
    parts.append("INTERVENTIONS")
    parts.append("-" * 60)
    if result.interventions:
        for i, (step_key, itype, _) in enumerate(result.interventions, 1):
            parts.append(f"  {i}. Step '{step_key}' -> {itype.value}")
    else:
        parts.append("  (none)")

    if result.llm_prompt:
        parts.append("")
        parts.append("LLM PROMPT")
        parts.append("-" * 60)
        parts.append(result.llm_prompt)

    if result.llm_response:
        parts.append("")
        parts.append("LLM RESPONSE")
        parts.append("-" * 60)
        parts.append(result.llm_response)

    parts.append("")
    parts.append("=" * 60)
    parts.append("END OF REPORT")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
        f.write("\n")

    return out_path


# ===================================================================
# 10. CLI HANDLERS
# ===================================================================


def run_test(llm: LLMCoach) -> int:
    """
    Quick LLM smoke test.

    Sends a single hardcoded prompt to the LLM to verify:
      - Model loads correctly
      - Tokenizer and generation pipeline work
      - Output is coherent

    Does NOT run the full funnel simulation.
    """
    print("=" * 60)
    print("  UNIQA Conversion Coach — LLM Smoke Test")
    print("=" * 60)

    llm.load()

    system = (
        "You are a friendly, non-pushy UNIQA insurance assistant. "
        "Keep messages warm, concise, and helpful. 2-3 sentences only."
    )
    user = (
        "You are a friendly insurance assistant. A customer named Franz is "
        "hesitating at the price page. Write a short, warm coaching message "
        "in 2-3 sentences."
    )

    prompt = (
        f"<|system|>\n{system}\n"
        f"<|user|>\n{user}\n"
        f"<|assistant|>\n"
    )

    print("\n[PROMPT]")
    print(prompt)
    print("[GENERATING] ...")

    t_start = time.time()
    result = llm.pipe(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=LLM_TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        pad_token_id=llm.pipe.tokenizer.eos_token_id,
    )
    elapsed = time.time() - t_start

    generated = result[0]["generated_text"]
    if "<|assistant|>" in generated:
        parts = generated.split("<|assistant|>")
        response = parts[-1].strip()
    else:
        response = generated[len(prompt):].strip()

    print(f"\n[RESPONSE]  (generated in {elapsed:.1f}s)")
    print(response)

    # Save test result
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"test_{ts}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("UNIQA Conversion Coach — LLM Smoke Test\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp:  {ts} UTC\n")
        f.write(f"Model:      {llm.model_path}\n")
        f.write(f"Generation: {elapsed:.1f}s, {MAX_NEW_TOKENS} max tokens, "
                f"temperature={LLM_TEMPERATURE}\n\n")
        f.write("--- PROMPT ---\n")
        f.write(prompt + "\n")
        f.write("--- RESPONSE ---\n")
        f.write(response + "\n")

    print(f"\n[OK] Test result saved to {out_path}")
    return 0


def run_simulation(persona_name: str, llm: LLMCoach, seed: int = None) -> int:
    """Run the full journey simulation for a given persona."""
    # Load persona data
    print(f"[SETUP] Loading persona data from {PERSONAS_PATH}")
    if not PERSONAS_PATH.exists():
        print(f"[ERROR] Persona file not found: {PERSONAS_PATH}")
        return 1

    with open(PERSONAS_PATH, "r", encoding="utf-8") as f:
        persona_data = json.load(f)

    persona_key = PERSONA_SEGMENT_MAP[persona_name]
    persona = PersonaProfile.from_persona_data(persona_key, persona_data)

    print(f"[SETUP] Persona:  {persona.name}")
    print(f"[SETUP] Segment:   {persona.segment_label} ({persona_key})")
    print(f"[SETUP] Profile:   age {persona.age}, €{persona.income_eur}/mo, "
          f"{persona.location}")
    print(f"[SETUP] Traits:    price_sens={persona.price_sensitivity:.2f}, "
          f"overwhelm={persona.overwhelm_tendency:.2f}, "
          f"back_nav={persona.back_nav_probability:.2f}")
    print(f"[SETUP] Drop risk: {persona.primary_drop_off_step}")
    print(f"[SETUP] Concerns:")

    for c in persona.key_concerns:
        print(f"          - {c}")

    # Run simulation
    if seed is not None:
        print(f"\n[SIM] Starting journey (seed={seed}) ...\n")
    else:
        print(f"\n[SIM] Starting journey (random seed) ...\n")

    sim = JourneySimulator(persona, seed=seed)
    result = sim.run()

    # Generate LLM messages for interventions
    llm_prompt = None
    llm_response = None

    if result.interventions:
        print(f"[COACH] {len(result.interventions)} intervention(s) triggered!")
        # Use the first (usually most critical) intervention for the LLM call
        step_key, itype, _ = result.interventions[0]
        trigger_step = FunnelStep.by_key(step_key)

        if trigger_step:
            print(f"[COACH] Generating message for {persona.name} at "
                  f"'{step_key}' ({itype.value}) ...")
            llm_prompt, llm_response = llm.generate(persona, itype, trigger_step)
            # Truncate for console display
            preview = llm_response[:150] + "..." if len(llm_response) > 150 else llm_response
            print(f"[COACH] Message: {preview}")
    else:
        print("[COACH] No interventions triggered — journey ran without coaching.")

    # Write report
    result.llm_prompt = llm_prompt
    result.llm_response = llm_response
    out_path = write_report(result)
    print(f"\n[OK] Full report saved to {out_path}")

    # Print summary
    print()
    print(format_journey_report(result))
    return 0


# ===================================================================
# 11. MAIN
# ===================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UNIQA Conversion Coach — Journey Simulation & LLM Intervention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_coach.py --test                  # Quick LLM smoke test
  python run_coach.py --persona franz         # Full simulation for Franz
  python run_coach.py --persona judith        # Full simulation for Judith
  python run_coach.py --persona peter --seed 42  # Reproducible Peter run
        """,
    )
    parser.add_argument(
        "--persona",
        choices=["judith", "franz", "peter"],
        default="franz",
        help="Persona to simulate (default: franz)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick LLM smoke test — sends one hardcoded prompt and exits",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible simulation runs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve model path from env var or default
    model_path = os.environ.get("QWEN_MODEL_PATH", DEFAULT_MODEL_PATH)
    llm = LLMCoach(model_path)

    if args.test:
        return run_test(llm)

    return run_simulation(args.persona, llm, seed=args.seed)


if __name__ == "__main__":
    sys.exit(main())
