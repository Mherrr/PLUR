from __future__ import annotations

import os

_ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
_MODEL = "claude-opus-5"
# Thinking is on by default on this model and shares the max_tokens budget, so
# the cap is well above what a 150-250 word briefing needs. These are short,
# fully-specified generation tasks, so low effort keeps the demo responsive.
_MAX_TOKENS = 4000
_EFFORT = "low"


class PLURAgent:
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=_ANTHROPIC_API_KEY)
        return self._client

    def _complete(self, prompt: str) -> str | None:
        """Return the model's text, or None if it declined to answer."""
        msg = self._get_client().messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            output_config={"effort": _EFFORT},
            messages=[{"role": "user", "content": prompt}],
        )
        if msg.stop_reason == "refusal":
            return None
        # Adaptive thinking means content[0] may be a thinking block, not text.
        return next((b.text for b in msg.content if b.type == "text"), None)

    def generate_rationale(
        self,
        changes: list[dict],
        risk_before: float,
        risk_after: float,
        venue_name: str,
    ) -> str:
        if not _ANTHROPIC_API_KEY:
            return _placeholder_rationale(changes, risk_before, risk_after)

        changes_text = "\n".join(
            f"  - {c['artist']}: moved from {c['from_stage']} @ {c['from_time']} "
            f"to {c['to_stage']} @ {c['to_time']}"
            for c in changes
        ) or "  (no changes — original schedule was already near-optimal)"

        prompt = f"""You are a crowd-safety expert reviewing a schedule optimization for {venue_name}.

Changes made:
{changes_text}

Risk score: {risk_before:.3f} → {risk_after:.3f} (lower = safer).

Write a short plain-text summary (max 150 words). No markdown, no bold (**), no headers (#), no bullet characters. Just plain sentences.

Format: one line per key move. Each line starts with the artist name and move, then a dash, then WHY it helps in plain language. Separate each line with a blank line.

End with a one-line disclaimer. Keep it simple and readable."""

        try:
            text = self._complete(prompt)
            return text if text else _placeholder_rationale(changes, risk_before, risk_after)
        except Exception as e:
            return f"{_placeholder_rationale(changes, risk_before, risk_after)}\n\n[Claude unavailable: {e}]"

    def generate_safety_briefing(
        self,
        venue_name: str,
        risk_windows: list[dict],
        peak_density: float,
        schedule: list[dict] | None = None,
        amenities: list[dict] | None = None,
    ) -> str:
        if not _ANTHROPIC_API_KEY:
            return _placeholder_briefing(venue_name, peak_density)

        windows_text = "\n".join(
            f"  Stage {w['stage']}: {w['t_start']}–{w['t_end']} min into event, risk score {w['score']:.2f}"
            for w in risk_windows[:8]
        ) or "  No critical risk windows detected."

        schedule_text = ""
        if schedule:
            schedule_text = "\nSCHEDULE:\n" + "\n".join(
                f"  {s['artist']} at {s['stage']} ({s['start']}–{s['end']})"
                for s in schedule
            )

        amenity_text = ""
        if amenities:
            _LABELS = {"restroom": "Restroom", "water": "Water Station", "bar": "Bar"}
            amenity_text = "\nAMENITY PLACEMENT (current positions, user-adjustable):\n" + "\n".join(
                f"  {_LABELS.get(a.get('facility_type', ''), a.get('facility_type', 'Facility'))} "
                f"'{a.get('name', a.get('id', '?'))}': lat {a.get('lat', 0):.5f}, lon {a.get('lon', 0):.5f}"
                for a in amenities
            )

        prompt = f"""You are a crowd-safety expert producing a pre-event safety briefing for {venue_name}.

SIMULATION RESULTS:
- Peak simulated crowd density: {peak_density:.1f} people/m² (danger threshold: 6.0)
- Risk windows (stages/times exceeding safe capacity):
{windows_text}
{schedule_text}
{amenity_text}

Write a plain-text safety briefing (no markdown, no bold, no headers with #). Use these sections separated by blank lines:

EXECUTIVE SUMMARY
Two sentences on overall risk level.

RISK ASSESSMENT
Which stages and time windows are most dangerous and why. Be specific with stage names and times.

RECOMMENDATIONS
Practical actions the ops team should consider — where to focus staff, where crowd flow needs attention, timing concerns. Keep it actionable.

AMENITY PLACEMENT
Based on hotspot locations and crowd flow, suggest whether any restrooms, water stations, or bars should be repositioned to better distribute crowds and reduce congestion and wait times at those locations. Be specific about which facility and where to move it.

End with this exact line:
PLUR is a planning and decision-support prototype, not a certified life-safety system. All recommendations must be reviewed by qualified event-safety professionals.

Max 250 words. Plain text only. No bullet characters, no asterisks, no markdown."""

        try:
            text = self._complete(prompt)
            return text if text else _placeholder_briefing(venue_name, peak_density)
        except Exception as e:
            return f"{_placeholder_briefing(venue_name, peak_density)}\n\n[Claude unavailable: {e}]"


def _placeholder_rationale(changes: list[dict], risk_before: float, risk_after: float) -> str:
    pct = (risk_before - risk_after) / max(risk_before, 1e-9) * 100
    return (
        f"The optimized schedule reduces integrated crowd-crush risk by {pct:.1f}% "
        f"(score: {risk_before:.3f} → {risk_after:.3f}). "
        f"Key changes separate high-draw acts from adjacent stages and stagger egress times, "
        f"reducing crowd migration spikes between concurrent sets. "
        f"Headliners remain in their designated final slots per event requirements.\n\n"
        f"PLUR is a planning and decision-support prototype, not a certified life-safety system."
    )


def _placeholder_briefing(venue_name: str, peak_density: float) -> str:
    return (
        f"SAFETY BRIEFING — {venue_name}\n\n"
        f"Executive Summary: Crowd simulation identifies peak density of {peak_density:.1f} people/m². "
        f"Review risk windows and consider targeted interventions at flagged stages.\n\n"
        f"PLUR is a planning and decision-support prototype, not a certified life-safety system. "
        f"All recommendations must be reviewed by qualified event-safety professionals."
    )
