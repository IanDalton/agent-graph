"""Tests for :class:`ReasoningSplitter` — re-routing reasoning leaked across channels."""

from __future__ import annotations

from backend.reasoning_split import ReasoningSplitter


def _drain_thinking(s: ReasoningSplitter, *deltas: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for d in deltas:
        out += s.feed_thinking(d)
    out += s.flush()
    return out


def _joined(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Concatenate routed chunks per channel for easy assertions."""
    acc = {"thinking": "", "text": ""}
    for channel, text in pairs:
        acc[channel] += text
    return acc


def test_clean_thinking_stream_stays_thinking() -> None:
    """No tags (Anthropic/OpenAI): every thinking chunk stays on the thinking channel."""
    s = ReasoningSplitter()
    out = _drain_thinking(s, "Let me ", "consider ", "the options.")
    assert _joined(out) == {"thinking": "Let me consider the options.", "text": ""}


def test_clean_text_stream_stays_text() -> None:
    s = ReasoningSplitter()
    out = s.feed_text("The answer is 42.") + s.flush()
    assert _joined(out) == {"thinking": "", "text": "The answer is 42."}


def test_answer_trapped_in_thinking_after_close_tag() -> None:
    """The reported bug: reasoning + </think> + answer all arrive on the thinking channel."""
    s = ReasoningSplitter()
    out = _drain_thinking(s, "I should summarize.</think>Here is the **report**.")
    assert _joined(out) == {
        "thinking": "I should summarize.",
        "text": "Here is the **report**.",
    }


def test_answer_continues_on_thinking_channel_after_close() -> None:
    """Once </think> is seen in the thinking stream, later thinking deltas are the answer too."""
    s = ReasoningSplitter()
    out = _drain_thinking(s, "reasoning</think>part one ", "and part two")
    assert _joined(out) == {"thinking": "reasoning", "text": "part one and part two"}


def test_close_tag_split_across_deltas() -> None:
    """</think> straddling a delta boundary must not leak a stray '</thi' into the output."""
    s = ReasoningSplitter()
    out = _drain_thinking(s, "done</thi", "nk>the answer")
    assert _joined(out) == {"thinking": "done", "text": "the answer"}


def test_whole_block_in_text_channel() -> None:
    """<think>...</think>answer all as text: reasoning is pulled out to the thinking channel."""
    s = ReasoningSplitter()
    out = s.feed_text("<think>weighing it</think>final answer") + s.flush()
    assert _joined(out) == {"thinking": "weighing it", "text": "final answer"}


def test_open_tag_split_across_text_deltas() -> None:
    s = ReasoningSplitter()
    out = s.feed_text("<thi") + s.feed_text("nk>hidden</think>shown") + s.flush()
    assert _joined(out) == {"thinking": "hidden", "text": "shown"}


def test_lone_angle_bracket_is_literal_text() -> None:
    """A '<' that isn't a tag (e.g. '3 < 5') passes through untouched."""
    s = ReasoningSplitter()
    out = s.feed_text("3 < 5 is true") + s.flush()
    assert _joined(out) == {"thinking": "", "text": "3 < 5 is true"}


def test_trailing_partial_tag_flushed_as_text() -> None:
    """A stream ending on a real '<thin' (never completed) isn't silently dropped."""
    s = ReasoningSplitter()
    out = s.feed_text("answer<thin") + s.flush()
    assert _joined(out) == {"thinking": "", "text": "answer<thin"}


# --------------------------------------------------------------------------- #
# Empty-answer fallback (main._empty_answer_fallback): never leave the UI stuck
# in the reasoning bubble when a turn reasons but produces no answer.
# --------------------------------------------------------------------------- #
def test_empty_answer_fallback_fires_when_reasoning_but_no_answer() -> None:
    """Model cut off mid-<think> (no closing tag): all content stays on the thinking channel."""
    from backend.main import _empty_answer_fallback

    s = ReasoningSplitter()
    out = _drain_thinking(s, "<think>reasoning that never closes")
    routed = _joined(out)
    assert routed["text"] == ""  # nothing on the answer channel
    assert routed["thinking"]  # reasoning is present
    notice = _empty_answer_fallback(routed["text"], routed["thinking"])
    assert notice is not None and notice.strip()


def test_empty_answer_fallback_fires_for_unterminated_text_think() -> None:
    from backend.main import _empty_answer_fallback

    s = ReasoningSplitter()
    out = s.feed_text("<think>still reasoning") + s.flush()
    routed = _joined(out)
    assert routed["text"] == "" and routed["thinking"]
    assert _empty_answer_fallback(routed["text"], routed["thinking"]) is not None


def test_empty_answer_fallback_silent_on_normal_answer() -> None:
    """A turn that produced an answer (or nothing at all) gets no fallback notice."""
    from backend.main import _empty_answer_fallback

    assert _empty_answer_fallback("Here is the answer.", "some reasoning") is None
    assert _empty_answer_fallback("", "") is None  # tool-only / empty turn: no spurious notice
    assert _empty_answer_fallback("  ", "   ") is None  # whitespace doesn't count
