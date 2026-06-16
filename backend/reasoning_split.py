"""Repair reasoning that a model leaks across the wrong stream via literal ``<think>`` tags.

A leaf module (no project deps) so both :mod:`backend.main` and :mod:`backend.skills.subagent`
can route their thinking/text deltas through one :class:`ReasoningSplitter` without an import
cycle.

**Why this exists.** Reasoning models served over Ollama (e.g. ``qwen3``) don't always separate
their chain-of-thought into a distinct thinking part. Two failure modes show up:

* the model dumps its reasoning *and then the final answer* into the thinking stream, marking the
  boundary only with a literal ``</think>`` token — so the answer is trapped in the UI's thinking
  column and ``final_text`` comes back empty; or
* the whole ``<think>...</think>answer`` block arrives as ordinary *text*, leaking the reasoning
  into the answer bubble.

:class:`ReasoningSplitter` scans both streams for the literal ``<think>`` / ``</think>`` markers,
strips them, and re-routes each chunk so reasoning lands on the ``thinking`` channel and the answer
on the ``text`` channel. It is a **no-op** for providers that separate the two channels natively
(Anthropic/OpenAI never emit the literal tags): with no tags present, every chunk stays on the
channel it arrived on.
"""

from __future__ import annotations

_OPEN = "<think>"
_CLOSE = "</think>"


def _segments(buf: str) -> tuple[list[tuple[str, str]], str]:
    """Tokenise *buf* into ``("text"|"open"|"close", payload)`` tokens + a trailing pending tail.

    The pending tail is a suffix of *buf* that is a proper prefix of ``<think>`` / ``</think>`` —
    it might be the front half of a tag whose remainder is in the next delta, so it is held back
    and prepended on the following call rather than emitted as text (which would print a stray
    ``<thi`` to the user). Everything else is returned as tokens immediately.
    """
    tokens: list[tuple[str, str]] = []
    i, n = 0, len(buf)
    while i < n:
        lt = buf.find("<", i)
        if lt == -1:  # no more tags possible
            tokens.append(("text", buf[i:]))
            return tokens, ""
        if lt > i:
            tokens.append(("text", buf[i:lt]))
            i = lt
        rest = buf[i:]
        if rest.startswith(_OPEN):
            tokens.append(("open", ""))
            i += len(_OPEN)
        elif rest.startswith(_CLOSE):
            tokens.append(("close", ""))
            i += len(_CLOSE)
        elif _OPEN.startswith(rest) or _CLOSE.startswith(rest):
            # A partial tag at the very end of the buffer — hold it for the next delta.
            return tokens, rest
        else:
            # A lone '<' that isn't the start of a tag — ordinary text.
            tokens.append(("text", "<"))
            i += 1
    return tokens, ""


class ReasoningSplitter:
    """Stateful per-run router from model deltas to the ``thinking`` / ``text`` channels.

    Feed thinking-origin deltas to :meth:`feed_thinking` and text-origin deltas to
    :meth:`feed_text`; each returns a list of ``(channel, text)`` pairs (``channel`` is
    ``"thinking"`` or ``"text"``) with the literal tags stripped. Call :meth:`flush` once the
    stream ends to release any held-back partial-tag tail as literal text.

    The two streams are tracked independently so the no-op guarantee holds: a clean thinking
    stream (no tags) always routes to ``thinking``; a clean text stream always routes to ``text``.
    """

    def __init__(self) -> None:
        # Set once the thinking stream emits a literal </think>: from there on the model is
        # answering inside the thinking channel, so its content is the answer (route to text).
        self._thinking_closed = False
        # True while inside a <think> block that was opened within the *text* stream.
        self._text_in_think = False
        self._think_tail = ""  # held-back partial tag from the thinking stream
        self._text_tail = ""  # held-back partial tag from the text stream

    def feed_thinking(self, delta: str) -> list[tuple[str, str]]:
        """Route a chunk that arrived on the model's thinking channel."""
        out: list[tuple[str, str]] = []
        tokens, self._think_tail = _segments(self._think_tail + delta)
        for kind, text in tokens:
            if kind == "open":
                self._thinking_closed = False  # reasoning (re)opened
            elif kind == "close":
                self._thinking_closed = True  # answer follows
            elif text:
                out.append(("text" if self._thinking_closed else "thinking", text))
        return out

    def feed_text(self, delta: str) -> list[tuple[str, str]]:
        """Route a chunk that arrived on the model's text channel."""
        out: list[tuple[str, str]] = []
        tokens, self._text_tail = _segments(self._text_tail + delta)
        for kind, text in tokens:
            if kind == "open":
                self._text_in_think = True
            elif kind == "close":
                self._text_in_think = False
            elif text:
                out.append(("thinking" if self._text_in_think else "text", text))
        return out

    def flush(self) -> list[tuple[str, str]]:
        """Emit any held-back partial-tag tails as literal text (the stream ended mid-tail)."""
        out: list[tuple[str, str]] = []
        if self._think_tail:
            out.append(("text" if self._thinking_closed else "thinking", self._think_tail))
            self._think_tail = ""
        if self._text_tail:
            out.append(("thinking" if self._text_in_think else "text", self._text_tail))
            self._text_tail = ""
        return out
