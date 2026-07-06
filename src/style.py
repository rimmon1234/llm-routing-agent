"""
Style utilities for pastel-coloured CLI output.
Auto-disables colours when output is piped or terminal doesn't support them.
Uses a Catppuccin Mocha-inspired pastel palette — soft on the eyes, beautiful on dark terminals.
"""

import os
import sys

# ── Terminal detection ────────────────────────────────────

def _supports_colour() -> bool:
    """Detect if the terminal supports ANSI colour codes."""
    if not sys.stdout.isatty():
        return False
    term = os.environ.get("TERM", "").lower()
    if "dumb" in term or "emacs" in term:
        return False
    return True

_USE_COLOUR = _supports_colour()

# ── Catppuccin Mocha-inspired pastel palette ──────────────

_PINK     = (245, 194, 231)  # #F5C2E7
_MAUVE    = (203, 166, 247)  # #CBA6F7 — lavender purple
_RED      = (243, 139, 168)  # #F38BA8 — soft coral
_PEACH    = (250, 179, 135)  # #FAB387 — warm peach
_YELLOW   = (249, 226, 175)  # #F9E2AF — soft butter
_GREEN    = (166, 227, 161)  # #A6E3A1 — pastel mint
_TEAL     = (148, 226, 213)  # #94E2D5
_SKY      = (137, 220, 235)  # #89DCEB
_BLUE     = (137, 180, 250)  # #89B4FA
_LAVENDER = (180, 190, 254)  # #B4BEFE
_TEXT     = (205, 214, 244)  # #CDD6F4 — bright off-white
_SUBTEXT0 = (166, 173, 200)  # #A6ADC8 — muted gray
_SURFACE2 = (88,  91, 112)   # #585B70 — dim gray


class _Style:
    """ANSI escape codes for terminal colours and formatting."""

    def __init__(self, enabled: bool):
        self._enabled = enabled

    # ── Escape helpers ────────────────────────────────────

    def _c(self, code: str, text: str) -> str:
        """Apply an ANSI SGR code (for formatting only)."""
        return f"\033[{code}m{text}\033[0m" if self._enabled else text

    def _tc(self, r: int, g: int, b: int, text: str) -> str:
        """Apply a 24-bit truecolor foreground."""
        return f"\033[38;2;{r};{g};{b}m{text}\033[0m" if self._enabled else text

    def _btc(self, r: int, g: int, b: int, text: str) -> str:
        """Apply bold + 24-bit truecolor foreground."""
        return f"\033[1;38;2;{r};{g};{b}m{text}\033[0m" if self._enabled else text

    # ── Foreground colours (pastel truecolor) ─────────────

    def black(self, text: str) -> str:
        return self._tc(*_SURFACE2, text)

    def red(self, text: str) -> str:
        return self._tc(*_RED, text)

    def green(self, text: str) -> str:
        return self._tc(*_GREEN, text)

    def yellow(self, text: str) -> str:
        return self._tc(*_YELLOW, text)

    def blue(self, text: str) -> str:
        return self._tc(*_BLUE, text)

    def magenta(self, text: str) -> str:
        return self._tc(*_MAUVE, text)

    def cyan(self, text: str) -> str:
        return self._tc(*_TEAL, text)

    def white(self, text: str) -> str:
        return self._tc(*_TEXT, text)

    # ── Bright variants (mapped to pastel equivalents) ────

    def bright_red(self, text: str) -> str:
        return self._tc(*_RED, text)

    def bright_green(self, text: str) -> str:
        return self._tc(*_GREEN, text)

    def bright_yellow(self, text: str) -> str:
        return self._tc(*_YELLOW, text)

    def bright_blue(self, text: str) -> str:
        return self._tc(*_BLUE, text)

    def bright_magenta(self, text: str) -> str:
        return self._tc(*_MAUVE, text)

    def bright_cyan(self, text: str) -> str:
        return self._tc(*_TEAL, text)

    def bright_white(self, text: str) -> str:
        return self._tc(*_TEXT, text)

    # ── Formatting ────────────────────────────────────────

    def bold(self, text: str) -> str:
        return self._c("1", text)

    def dim(self, text: str) -> str:
        return self._c("2", text)

    def italic(self, text: str) -> str:
        return self._c("3", text)

    def underline(self, text: str) -> str:
        return self._c("4", text)

    # ── Compound helpers (pastel palette) ─────────────────

    def header(self, text: str) -> str:
        """Bold mauve — for section headings."""
        return self._btc(*_MAUVE, text)

    def sub_header(self, text: str) -> str:
        """Mauve — for sub-headings."""
        return self._tc(*_MAUVE, text)

    def label(self, text: str) -> str:
        """Bold off-white — for field labels."""
        return self._btc(*_TEXT, text)

    def value(self, text: str) -> str:
        """Off-white — for normal values."""
        return self._tc(*_TEXT, text)

    def good(self, text: str) -> str:
        """Bold pastel green — for positive metrics."""
        return self._btc(*_GREEN, text)

    def warn(self, text: str) -> str:
        """Bold pastel peach — for warnings."""
        return self._btc(*_PEACH, text)

    def error(self, text: str) -> str:
        """Bold pastel red — for errors."""
        return self._btc(*_RED, text)

    def money(self, text: str) -> str:
        """Pastel yellow — for dollar amounts."""
        return self._tc(*_YELLOW, text)

    def accent(self, text: str) -> str:
        """Pastel pink — for accents/highlights."""
        return self._tc(*_PINK, text)

    def muted(self, text: str) -> str:
        """Muted gray — for less important info."""
        return self._tc(*_SUBTEXT0, text)

    def bar(self, text: str) -> str:
        """Dim gray — for separator bars."""
        return self._tc(*_SURFACE2, text)


# Global singleton
S = _Style(_USE_COLOUR)
