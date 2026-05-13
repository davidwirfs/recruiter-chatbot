"""
Generate the OG image for the LinkedIn / social preview tile.

Output: ../static/og-card-1200x627.png  (1200x627, summary_large_image)

Design direction (from Steve Jobs consult, 2026-05-13):
  "The image IS the chatbot, frozen mid-question. Kill the marketing.
   Three things only: name top-left, logo top-right, the question center."

Why this image, not a banner: when a recruiter scrolls past the LinkedIn
Featured tile, they have ~1 second of attention. A branded banner wastes
that second on marketing. A frozen frame of the chatbot itself — same UI
they'd see if they clicked — earns the next 30 seconds by feeling like
the artifact is already in their hand.

How to regenerate (e.g. after a UI change to the live chatbot, or to swap
the example question):

  pip install pillow
  python3 scripts/generate-og-image.py

Requires: pillow, and a clean sans-serif TTF at the FONT_DIR path below.
DejaVu Sans is the default (ubiquitous on Linux); on macOS, swap to
/System/Library/Fonts/Helvetica.ttc or any installed sans-serif.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# --- canvas ---
W, H = 1200, 627  # LinkedIn 1.91:1, summary_large_image tile

# --- paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGO_PATH = PROJECT_ROOT / "static" / "blitzscale-logo.png"
OUTPUT_PATH = PROJECT_ROOT / "static" / "og-card-1200x627.png"

# --- fonts ---
# DejaVu Sans is installed on most Linux systems and inside Docker base images.
# On macOS local dev, override the paths below to a system font like Helvetica.
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_ITALIC = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"

# --- colours (match the live chatbot Apple-style palette) ---
COLOR_BG = "#ffffff"
COLOR_TEXT = "#1d1d1f"
COLOR_LINE = "#c5c5c7"
COLOR_CURSOR = "#1d1d1f"


def main() -> None:
    canvas = Image.new("RGB", (W, H), COLOR_BG)
    draw = ImageDraw.Draw(canvas)

    font_name = ImageFont.truetype(FONT_REGULAR, 26)
    font_question = ImageFont.truetype(FONT_ITALIC, 38)

    # Safe-crop band: LinkedIn Featured center-crops 1200x627 down to roughly
    # 1:1 (visible area = 627x627 centered horizontally → x=287 to x=913).
    # All critical content must sit inside this band so it survives both the
    # full 1.91:1 OG preview AND the LinkedIn Featured square crop.
    SAFE_LEFT = 290
    SAFE_RIGHT = 910

    # 1. Header — "David Wirfs" inside safe-left
    draw.text((SAFE_LEFT, 60), "David Wirfs", fill=COLOR_TEXT, font=font_name)

    # 2. Logo inside safe-right (small, NOT the headline — per Jobs)
    logo = Image.open(LOGO_PATH).convert("RGBA")
    logo_size = 48
    logo_resized = logo.resize((logo_size, logo_size), Image.LANCZOS)
    logo_x = SAFE_RIGHT - logo_size
    logo_y = 60 - 6
    canvas.paste(logo_resized, (logo_x, logo_y), logo_resized)

    # 3. The question — italic, centered (mirrors live chatbot placeholder)
    question = "What does he need from his next role?"
    qbbox = draw.textbbox((0, 0), question, font=font_question)
    qw = qbbox[2] - qbbox[0]
    qh = qbbox[3] - qbbox[1]
    qx = (W - qw) // 2
    qy = (H - qh) // 2 - 10
    draw.text((qx, qy), question, fill=COLOR_TEXT, font=font_question)

    # 4. Cursor mark right after the question — live-input cue
    cursor_x = qx + qw + 6
    draw.line(
        (cursor_x, qy + 4, cursor_x, qy + qh - 2),
        fill=COLOR_CURSOR,
        width=2,
    )

    # 5. Underline — narrow to match the safe-crop band so it stays
    # visually balanced both in the 1.91:1 preview and the LinkedIn 1:1 crop.
    underline_y = qy + qh + 24
    draw.line(
        (SAFE_LEFT, underline_y, SAFE_RIGHT, underline_y),
        fill=COLOR_LINE,
        width=2,
    )

    canvas.save(OUTPUT_PATH, "PNG", optimize=True)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Size: {W}x{H}")
    print(f"Safe-crop band: x={SAFE_LEFT} to x={SAFE_RIGHT}")


if __name__ == "__main__":
    main()
