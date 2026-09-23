"""Shared renderers and the palette they draw from.

Colour rules followed here, from the data-visualisation method:

  * **Status colours are reserved and never carry meaning alone.** Every status
    ships a glyph and a word alongside the colour, so the state survives
    colour-blindness, greyscale printing and forced-colors mode.
  * **Similarity is magnitude**, so it gets bars with one series in a single
    blue -- not a colour ramp. Bar length already encodes the value; shading it
    too would be double-encoding.
  * **Headline numbers are stat tiles, not charts.**
  * **Text wears text tokens**, never a series colour.

The palette is the reference dark-surface instance, validated against this
app's own surface:
    node scripts/validate_palette.js "#3987e5,#d95926,#199e70" \
         --mode dark --surface "#1a1a19"      -> ALL CHECKS PASS
"""

from __future__ import annotations

import html
from typing import Any, Iterable

import streamlit as st

# --- palette ----------------------------------------------------------------

SURFACE = "#1a1a19"
PLANE = "#0d0d0d"
INK = "#ffffff"
INK_2 = "#c3c2b7"
INK_MUTED = "#898781"
HAIRLINE = "#2c2c2a"
BASELINE = "#383835"

SERIES_1 = "#3987e5"  # the single blue used for magnitude
SERIES_2 = "#d95926"
SERIES_3 = "#199e70"

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
    "neutral": INK_MUTED,
}

# glyph + word, so colour is never the only channel
STAGE_STATUS = {
    "done": ("good", "✓", "done"),
    "running": ("warning", "◐", "running"),
    "skipped": ("neutral", "—", "skipped"),
    "pending": ("neutral", "○", "pending"),
}


def _esc(value: Any) -> str:
    return html.escape(str(value))


# --- primitives -------------------------------------------------------------


def stat_tile(label: str, value: str, note: str = "") -> str:
    """A headline number. Deliberately not a chart."""
    return f"""
<div style="background:{SURFACE};border:1px solid {HAIRLINE};border-radius:8px;
            padding:14px 16px;height:100%;">
  <div style="color:{INK_MUTED};font-size:11px;letter-spacing:.06em;
              text-transform:uppercase;">{_esc(label)}</div>
  <div style="color:{INK};font-size:28px;line-height:1.15;margin-top:6px;">
    {_esc(value)}</div>
  <div style="color:{INK_2};font-size:12px;margin-top:4px;">{_esc(note)}</div>
</div>"""


def pill(status: str, glyph: str, text: str) -> str:
    colour = STATUS.get(status, INK_MUTED)
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;'
        f'border:1px solid {colour};color:{colour};border-radius:999px;'
        f'padding:2px 10px;font-size:12px;white-space:nowrap;">'
        f"<span>{glyph}</span><span>{_esc(text)}</span></span>"
    )


def stage_strip(progress) -> str:
    """The four stages with their state.

    `skipped` is a first-class state, not a gap: when the classifier routes an
    abusive or action-requiring ticket straight to a human, retrieval and
    drafting genuinely never run, and showing that is the point.
    """
    from ui.pipeline import STAGE_LABELS, STAGES

    cells = []
    for stage in STAGES:
        state = progress.status_of(stage)
        status, glyph, word = STAGE_STATUS[state]
        colour = STATUS.get(status, INK_MUTED)
        attempts = progress.attempts(stage)
        retry = (
            f'<div style="color:{STATUS["warning"]};font-size:11px;margin-top:4px;">'
            f"ran {attempts}× (retry)</div>"
            if attempts > 1
            else ""
        )
        served = (progress.state or {}).get("served_by", {}).get(
            {"classify": "classifier", "respond": "responder", "judge": "judge"}.get(
                stage, ""
            ),
            "",
        )
        model = (
            f'<div style="color:{INK_MUTED};font-size:11px;margin-top:4px;'
            f'font-family:monospace;">{_esc(served)}</div>'
            if served
            else ""
        )
        cells.append(
            f"""<div style="flex:1;min-width:0;background:{SURFACE};
                 border:1px solid {HAIRLINE};border-left:3px solid {colour};
                 border-radius:8px;padding:12px 14px;">
              <div style="color:{INK};font-size:13px;">{_esc(STAGE_LABELS[stage])}</div>
              <div style="margin-top:8px;">{pill(status, glyph, word)}</div>
              {retry}{model}
            </div>"""
        )
    return (
        '<div style="display:flex;gap:10px;align-items:stretch;">'
        + "".join(cells)
        + "</div>"
    )


def similarity_bars(chunks: Iterable[dict], floor: float) -> str:
    """Retrieval scores as magnitude, with the grounding floor marked.

    One series in one blue. The floor is drawn as a reference line because the
    single most important thing to read off this chart is which side of 0.40
    the best chunk falls on.
    """
    chunks = list(chunks)
    if not chunks:
        return (
            f'<div style="color:{INK_MUTED};font-size:13px;">'
            f"Retrieval did not run.</div>"
        )

    rows = []
    for chunk in chunks:
        sim = 1.0 - float(chunk["distance"])
        pct = max(0.0, min(1.0, sim)) * 100
        above = sim >= floor
        rows.append(
            f"""<div style="margin-bottom:10px;">
  <div style="display:flex;justify-content:space-between;font-size:12px;
              color:{INK_2};margin-bottom:4px;">
    <span style="font-family:monospace;">{_esc(chunk['source'])}</span>
    <span style="font-variant-numeric:tabular-nums;color:{INK};">{sim:.3f}</span>
  </div>
  <div style="position:relative;height:10px;background:{PLANE};
              border-radius:5px;overflow:hidden;">
    <div style="width:{pct:.2f}%;height:100%;border-radius:0 4px 4px 0;
                background:{SERIES_1};opacity:{'1' if above else '0.45'};"></div>
  </div>
</div>"""
        )

    marker = f"""<div style="position:relative;height:18px;margin-top:-4px;">
  <div style="position:absolute;left:{floor * 100:.2f}%;top:0;bottom:0;
              border-left:1px dashed {STATUS['warning']};"></div>
  <div style="position:absolute;left:calc({floor * 100:.2f}% + 6px);top:0;
              color:{STATUS['warning']};font-size:11px;">
    grounding floor {floor:.2f}</div>
</div>"""
    return "".join(rows) + marker


def judge_rubric(checks: dict[str, bool]) -> str:
    """The judge's checks as pass/fail, including the code gates."""
    if not checks:
        return (
            f'<div style="color:{INK_MUTED};font-size:13px;">'
            f"The judge did not run.</div>"
        )

    labels = {
        "addresses_ticket": "Addresses the ticket",
        "grounded_in_sources": "Every claim traceable to a source",
        "cites_sources": "Cites a source",
        "touches_sensitive": "Touches a sensitive topic",
        "above_grounding_floor": "Retrieval above the grounding floor",
        "responder_refused": "Responder declined to answer",
    }
    # For these two, True is the escalating condition rather than the passing one.
    inverted = {"touches_sensitive", "responder_refused"}

    rows = []
    for key, value in checks.items():
        ok = (not value) if key in inverted else bool(value)
        status, glyph, word = (
            ("good", "✓", "pass") if ok else ("critical", "✕", "escalate")
        )
        rows.append(
            f"""<div style="display:flex;justify-content:space-between;
                 align-items:center;gap:12px;padding:7px 0;
                 border-bottom:1px solid {HAIRLINE};">
              <span style="color:{INK_2};font-size:13px;">
                {_esc(labels.get(key, key))}</span>
              {pill(status, glyph, word)}
            </div>"""
        )
    return "".join(rows)


def trace_block(trace: Iterable[str]) -> None:
    lines = list(trace)
    if not lines:
        st.caption("No trace recorded.")
        return
    st.code("\n".join(lines), language="text")
