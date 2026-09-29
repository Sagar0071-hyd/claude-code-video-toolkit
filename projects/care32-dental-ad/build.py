"""Care32 dental ad — animated overlays + end card composited onto AI footage.

Inputs (assets/):  footage.mp4 (9:16, ~28s silent), vo.mp3, music.mp3, care32-logo.jpg
Output:            out/care32-ad.mp4  (1080x1920, 30fps, 30s)

Timeline (30s):
  0.0 → 25.5  footage + persistent logo (top-center) + "BOOK NOW | 82 82 82 1409" bar (bottom)
              + "20+ Years of Experience" pop-in badge + per-shot service chips
  25.5 → 30.0 end card: logo, 20+ years, address, website, both numbers, locator line

Every overlay is a pre-rendered RGBA sprite animated per frame (spring pop, slide, pulse,
shine sweep, count-up), then piped to ffmpeg together with the mixed audio.

Usage:
  python3 build.py                      # full render
  python3 build.py --preview 2 6 27.5   # still frames at those seconds → out/prev_*.jpg
"""
import math
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
A, OUT = ROOT / "assets", ROOT / "out"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

W, H, FPS = 1080, 1920, 30
TOTAL = 30.0
CARD_START = 25.5
CARD_FADE = 0.5

DARK_BLUE = (27, 117, 188)
NAVY = (14, 58, 110)
LIGHT_BLUE = (41, 171, 226)
WHITE = (255, 255, 255)
GOLD = (255, 196, 61)
GREY = (40, 40, 40)

BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

PHONE_1 = "82 82 82 1409"
PHONE_2 = "82 82 82 1479"
ADDRESS = [
    "3rd Floor, ARD Magnum, Above ICICI Bank,",
    "Rainbow Vistas Road, Kukatpally,",
    "Hyderabad – 500018",
]
WEBSITE = "www.care32.co.in"
LOCATOR = ["2 mins Away from", "Rainbow Vistas & Marina Skies"]

# (start, end, label) — aligned to the 6 shots in the Seedance multi-shot prompt
SERVICE_CHIPS = [
    (4.3, 8.8, "Painless Root Canal"),
    (9.3, 13.8, "Dental Implants"),
    (14.3, 18.8, "Clear Aligners & Smile Makeover"),
    (19.3, 23.8, "Gentle Kids Dentistry"),
]
BADGE_WINDOW = (0.8, 4.3)

LOGO_Y = 60
BAR_H, BAR_BOTTOM = 150, 70
BAR_Y = H - BAR_H - BAR_BOTTOM


# ── easing ────────────────────────────────────────────────────────────────────
def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def prog(t, start, dur):
    return clamp((t - start) / dur)


def ease_out_cubic(p):
    return 1 - (1 - p) ** 3


def ease_in_cubic(p):
    return p ** 3


def ease_out_back(p, s=1.70158):
    p -= 1
    return 1 + (s + 1) * p ** 3 + s * p ** 2


def spring(p, damping=6.0, freq=2.2):
    """Damped spring 0 → 1 with overshoot."""
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    return 1 - math.exp(-damping * p) * math.cos(2 * math.pi * freq * p)


# ── drawing helpers ───────────────────────────────────────────────────────────
@lru_cache(None)
def font(path, size):
    return ImageFont.truetype(path, size)


def text_size(s, f):
    l, t, r, b = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), s, font=f)
    return r - l, b - t, l, t


def text_sprite(s, f, fill, pad=6):
    w, h, l, t = text_size(s, f)
    img = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((pad - l, pad - t), s, font=f, fill=fill)
    return img


def pill_sprite(s, f, fg, bg, padx=40, pady=18, alpha=255):
    w, h, l, t = text_size(s, f)
    iw, ih = w + 2 * padx, h + 2 * pady
    img = Image.new("RGBA", (iw, ih), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, iw - 1, ih - 1], radius=ih // 2, fill=bg + (alpha,))
    d.text((padx - l, pady - t), s, font=f, fill=fg)
    return img


def logo_chip(width):
    logo = Image.open(A / "care32-logo.jpg").convert("RGB")
    lw = width - 60
    logo = logo.resize((lw, round(logo.height * lw / logo.width)), Image.LANCZOS)
    chip = Image.new("RGBA", (width, logo.height + 44), (245, 245, 245, 255))
    chip.paste(logo, (30, 22))
    mask = Image.new("L", chip.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, chip.width - 1, chip.height - 1], radius=28, fill=248)
    chip.putalpha(mask)
    return chip


def with_shine(sprite, p):
    """Diagonal white highlight sweeping left→right across the sprite at progress p."""
    if p <= 0 or p >= 1:
        return sprite
    w, h = sprite.size
    xs = np.arange(w)[None, :] + np.arange(h)[:, None] * 0.5
    centre = -h + p * (w + 2 * h)
    band = np.clip(1 - np.abs(xs - centre) / 60, 0, 1) * 0.55
    arr = np.asarray(sprite).astype(np.float32)
    a = arr[..., 3:4] / 255
    arr[..., :3] = arr[..., :3] + (255 - arr[..., :3]) * band[..., None] * a
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def place(canvas, sprite, cx, cy, scale=1.0, alpha=1.0):
    """Composite sprite centred at (cx, cy) with scale and opacity."""
    if alpha <= 0.01 or scale <= 0.01:
        return
    if abs(scale - 1) > 1e-3:
        sprite = sprite.resize((max(1, round(sprite.width * scale)), max(1, round(sprite.height * scale))),
                               Image.BILINEAR)
    if alpha < 0.999:
        sprite = sprite.copy()
        sprite.putalpha(sprite.getchannel("A").point(lambda v: int(v * alpha)))
    canvas.alpha_composite(sprite, (round(cx - sprite.width / 2), round(cy - sprite.height / 2)))


# ── sprites ───────────────────────────────────────────────────────────────────
class Sprites:
    def __init__(self):
        self.logo = logo_chip(640)

        bar = Image.new("RGBA", (W - 100, BAR_H), (0, 0, 0, 0))
        ImageDraw.Draw(bar).rounded_rectangle([0, 0, bar.width - 1, BAR_H - 1], radius=34, fill=NAVY + (240,))
        self.bar = bar
        self.book_pill = pill_sprite("BOOK NOW", font(BOLD, 44), WHITE, LIGHT_BLUE, padx=28, pady=26)
        self.bar_phone = text_sprite(PHONE_1, font(BOLD, 60), WHITE)

        box = Image.new("RGBA", (780, 380), (0, 0, 0, 0))
        ImageDraw.Draw(box).rounded_rectangle([0, 0, 779, 379], radius=40, fill=DARK_BLUE + (235,),
                                              outline=GOLD, width=6)
        self.badge_box = box
        self.badge_l1 = text_sprite("YEARS OF", font(BOLD, 58), WHITE)
        self.badge_l2 = text_sprite("EXPERIENCE", font(BOLD, 58), WHITE)

        self.chips = [(s, e, pill_sprite(lbl, font(BOLD, 50), DARK_BLUE, WHITE, alpha=235))
                      for s, e, lbl in SERVICE_CHIPS]

        # end card pieces
        self.card_logo = logo_chip(900)
        exp = Image.new("RGBA", (800, 230), (0, 0, 0, 0))
        ImageDraw.Draw(exp).rounded_rectangle([0, 0, 799, 229], radius=36, fill=DARK_BLUE)
        self.card_exp_box = exp
        self.card_exp_sub = text_sprite("OF EXPERIENCE", font(BOLD, 56), WHITE)
        self.card_tag = text_sprite("Zero Pain • Same-Day Treatment", font(BOLD, 44), NAVY)
        self.card_addr = [text_sprite(line, font(REG, 42), GREY) for line in ADDRESS]
        self.card_web = text_sprite(WEBSITE, font(BOLD, 58), DARK_BLUE)
        self.card_book = pill_sprite("BOOK NOW", font(BOLD, 52), WHITE, LIGHT_BLUE, padx=45, pady=22)
        self.card_phones = [text_sprite(f"☎ {p}", font(BOLD, 66), NAVY) for p in (PHONE_1, PHONE_2)]
        self.card_loc = [text_sprite(line, font(BOLD, 60), LIGHT_BLUE) for line in LOCATOR]
        self._counts = {}

    def count(self, text, size):
        key = (text, size)
        if key not in self._counts:
            self._counts[key] = text_sprite(text, font(BOLD, size), GOLD)
        return self._counts[key]


# ── scene layers ──────────────────────────────────────────────────────────────
def draw_persistent(c, S, t):
    # logo: drop in from above with back-ease, shine sweep at 1.0s and every 7s after
    p = ease_out_back(prog(t, 0.1, 0.7))
    ly = LOGO_Y + S.logo.height / 2 - (1 - p) * 260
    shine_p = ((t - 1.0) % 7.0) / 1.1 if t >= 1.0 else 0
    place(c, with_shine(S.logo, shine_p), W / 2, ly, alpha=clamp(p * 1.5))

    # bottom bar: slide up, BOOK NOW pops then pulses, phone number slides in
    pb = ease_out_cubic(prog(t, 0.3, 0.6))
    by = BAR_Y + BAR_H / 2 + (1 - pb) * 260
    place(c, S.bar, W / 2, by, alpha=pb)
    pp = ease_out_back(prog(t, 0.7, 0.5))
    pulse = 1 + 0.06 * max(0, math.sin(2 * math.pi * 1.1 * (t - 1.2))) if t > 1.2 else 1
    place(c, S.book_pill, 74 + S.book_pill.width / 2, by, scale=pp * pulse, alpha=clamp(pp * 2))
    pt = ease_out_cubic(prog(t, 0.9, 0.5))
    phone_cx = (74 + S.book_pill.width + W - 50) / 2
    place(c, S.bar_phone, phone_cx + (1 - pt) * 80, by, alpha=pt)


def draw_badge(c, S, t):
    s, e = BADGE_WINDOW
    if not s <= t <= e:
        return
    pin = spring(prog(t, s, 0.9))
    pout = ease_in_cubic(prog(t, e - 0.35, 0.35))
    scale, alpha = pin * (1 - 0.4 * pout), 1 - pout
    cy = 1370
    place(c, S.badge_box, W / 2, cy, scale=scale, alpha=alpha)
    n = round(20 * ease_out_cubic(prog(t, s + 0.1, 0.9)))
    place(c, S.count(f"{n}+", 170), W / 2, cy - 85 * scale, scale=scale, alpha=alpha)
    for i, line in enumerate((S.badge_l1, S.badge_l2)):
        pl = ease_out_cubic(prog(t, s + 0.35 + i * 0.12, 0.4))
        place(c, line, W / 2, cy + (60 + i * 70) * scale + (1 - pl) * 30, scale=scale, alpha=alpha * pl)


def draw_chips(c, S, t):
    cy = BAR_Y - 75
    for s, e, spr in S.chips:
        if not s <= t <= e:
            continue
        pin = ease_out_back(prog(t, s, 0.5), 1.2)
        pout = ease_in_cubic(prog(t, e - 0.35, 0.35))
        x = W / 2 - (1 - pin) * (W * 0.8) + pout * (W * 0.8)
        place(c, spr, x, cy, alpha=clamp(pin * 1.4) * (1 - pout))


def draw_end_card(c, S, t):
    lt = t - CARD_START  # local time
    card = Image.new("RGBA", (W, H), WHITE + (255,))
    d = ImageDraw.Draw(card)
    grow = ease_out_cubic(prog(lt, 0.0, 0.8))
    d.rectangle([0, 0, W * grow, 24], fill=LIGHT_BLUE)
    d.rectangle([W * (1 - grow), H - 24, W, H], fill=DARK_BLUE)

    def rise(sprite, cy, start, dur=0.45, dist=40):
        p = ease_out_cubic(prog(lt, start, dur))
        place(card, sprite, W / 2, cy + (1 - p) * dist, alpha=p)

    y = 120
    pl = ease_out_back(prog(lt, 0.0, 0.6))
    place(card, with_shine(S.card_logo, prog(lt, 1.2, 1.0)), W / 2, y + S.card_logo.height / 2,
          scale=0.8 + 0.2 * pl, alpha=clamp(pl * 1.5))
    y += S.card_logo.height + 70

    pe = spring(prog(lt, 0.2, 0.8))
    ae = clamp(pe * 1.5)
    place(card, S.card_exp_box, W / 2, y + 115, scale=pe, alpha=ae)
    n = round(20 * ease_out_cubic(prog(lt, 0.25, 0.8)))
    place(card, S.count(f"{n}+ Years", 110), W / 2, y + 115 - 45 * pe, scale=pe, alpha=ae)
    place(card, S.card_exp_sub, W / 2, y + 115 + 60 * pe, scale=pe, alpha=ae)
    y += 230 + 50

    rise(S.card_tag, y + 28, 0.45)
    y += 110
    for i, spr in enumerate(S.card_addr):
        rise(spr, y + 25, 0.6 + i * 0.1)
        y += 58
    y += 40
    rise(S.card_web, y + 35, 0.95)
    y += 100

    pb = ease_out_back(prog(lt, 1.1, 0.5))
    pulse = 1 + 0.07 * max(0, math.sin(2 * math.pi * 1.2 * (lt - 1.6))) if lt > 1.6 else 1
    place(card, S.card_book, W / 2, y + 48, scale=pb * pulse, alpha=clamp(pb * 2))
    y += 130
    for i, spr in enumerate(S.card_phones):
        p = ease_out_cubic(prog(lt, 1.25 + i * 0.12, 0.45))
        place(card, spr, W / 2 + (1 - p) * (-120 if i == 0 else 120), y + 40, alpha=p)
        y += 88
    y += 50
    for i, spr in enumerate(S.card_loc):
        p = ease_out_back(prog(lt, 1.55 + i * 0.12, 0.5))
        loc = with_shine(spr, prog(lt, 2.4 + i * 0.1, 0.9))
        place(card, loc, W / 2, y + 36, scale=0.85 + 0.15 * p, alpha=clamp(p * 1.5))
        y += 72
    assert y < H - 60, f"end card overflows: y={y}"

    place(c, card, W / 2, H / 2, alpha=ease_out_cubic(prog(lt, 0.0, CARD_FADE)))


# ── render ────────────────────────────────────────────────────────────────────
def footage_frames():
    cmd = [FFMPEG, "-loglevel", "error", "-i", str(A / "footage.mp4"),
           "-vf", f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    last = None
    while True:
        buf = proc.stdout.read(W * H * 3)
        if len(buf) == W * H * 3:
            last = Image.frombuffer("RGB", (W, H), buf).convert("RGBA")
        yield last  # holds the last frame if footage runs short


def compose(S, t, bg):
    c = bg.copy()
    if t < CARD_START + CARD_FADE:
        draw_persistent(c, S, t)
        draw_badge(c, S, t)
        draw_chips(c, S, t)
    if t >= CARD_START:
        draw_end_card(c, S, t)
    return c


def build(preview_times=None):
    need = ("footage.mp4",) if preview_times else ("footage.mp4", "vo.mp3", "music.mp3")
    for f in need:
        if not (A / f).exists():
            sys.exit(f"missing assets/{f}")
    OUT.mkdir(exist_ok=True)
    S = Sprites()
    frames = footage_frames()

    if preview_times:
        wanted = {round(x * FPS) for x in preview_times}
        for i in range(max(wanted) + 1):
            bg = next(frames)
            if i in wanted:
                path = OUT / f"prev_{i / FPS:05.2f}.jpg"
                compose(S, i / FPS, bg).convert("RGB").resize((540, 960)).save(path)
                print("wrote", path)
        return

    af = (f"[1:a]adelay=400|400,apad,atrim=0:{TOTAL}[vo];"
          f"[2:a]apad,atrim=0:{TOTAL},volume=0.14,afade=t=in:d=0.8,"
          f"afade=t=out:st={TOTAL - 1.5}:d=1.5[mus];"
          "[vo][mus]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-14:TP=-1.5[aout]")
    enc = subprocess.Popen(
        [FFMPEG, "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-i", str(A / "vo.mp3"), "-i", str(A / "music.mp3"),
         "-filter_complex", af, "-map", "0:v", "-map", "[aout]", "-t", str(TOTAL),
         "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(OUT / "care32-ad.mp4")],
        stdin=subprocess.PIPE)
    for i in range(int(TOTAL * FPS)):
        enc.stdin.write(compose(S, i / FPS, next(frames)).tobytes())
    enc.stdin.close()
    if enc.wait():
        sys.exit("ffmpeg encode failed")
    print("wrote", OUT / "care32-ad.mp4")


if __name__ == "__main__":
    build([float(x) for x in sys.argv[2:]] if sys.argv[1:2] == ["--preview"] else None)
