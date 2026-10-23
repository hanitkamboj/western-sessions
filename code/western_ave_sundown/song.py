"""
song.py — the composition itself: "Western Ave Sundown"
Original West Coast / G-funk instrumental. Key: F# minor. Tempo: 93 BPM.

Structure (68 bars + tail):
  INTRO        bars 0-3      (vinyl swell, reversed crash, Rhodes/synth pad,
                              talkbox sigh, bass pickup, fill into the drop)
  VERSE 1      bars 4-19     (full groove; guitar 16ths; sparse talkbox calls)
  HOOK 1       bars 20-27    (8-bar singable motif: talkbox + strings + clap)
  VERSE 2      bars 28-43    (busier percussion, bass variation, string line)
  BRIDGE       bars 44-51    (drums out, half-time build, snare roll, riser)
  HOOK 2       bars 52-59    (full stack + synth-lead octave double)
  OUTRO        bars 60-67    (final hook tag -> decay, last F#m9 hit, tail)
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
import numpy as np

BPM = 93.0
BEAT_S = 60.0 / BPM
BAR_S = 4 * BEAT_S
SR = 48000
SWING = 0.575          # 16th-note swing ratio (0.5 = straight)
TOTAL_BARS = 68
TAIL_S = 8.0
SONG_S = TOTAL_BARS * BAR_S


def b2s(beat: float) -> float:
    return beat * BEAT_S


def swing_beat(beat: float) -> float:
    """Displace the second 16th of every 8th-note pair (classic hip-hop swing)."""
    sixteenth = round(beat * 4)
    if abs(beat * 4 - sixteenth) < 1e-6 and sixteenth % 2 == 1:
        return beat + (SWING - 0.5) * 0.5
    return beat


# ---------------------------------------------------------------- harmony
# voicings are deliberate: Rhodes in the middle, strings spread, guitar dyads high
CHORDS = {
    "F#m9":   dict(bass=30, rhodes=[54, 61, 64, 68], guitar=[54, 61], strings=[54, 61, 64, 68, 73]),
    "Dmaj9":  dict(bass=38, rhodes=[50, 57, 61, 64, 66], guitar=[50, 57], strings=[50, 57, 61, 64, 66]),
    "C#m7":   dict(bass=37, rhodes=[49, 56, 59, 64], guitar=[49, 56], strings=[49, 56, 59, 64]),
    "Bm9":    dict(bass=35, rhodes=[47, 54, 57, 61, 62], guitar=[47, 54], strings=[47, 54, 57, 61, 62]),
    "E":      dict(bass=28, rhodes=[52, 56, 59, 61, 66], guitar=[52, 59], strings=[52, 56, 59, 61, 66]),
    "A":      dict(bass=33, rhodes=[57, 61, 64, 68], guitar=[57, 64], strings=[57, 61, 64, 68]),
    "C#7sus": dict(bass=37, rhodes=[49, 54, 56, 59], guitar=[49, 56], strings=[49, 54, 56, 59]),
    "C#7":    dict(bass=37, rhodes=[49, 53, 56, 59], guitar=[49, 56], strings=[49, 53, 56, 59]),
    "G#m7b5": dict(bass=32, rhodes=[44, 50, 54, 59], guitar=[44, 50], strings=[44, 50, 54, 59]),
    "Emaj9":  dict(bass=28, rhodes=[52, 56, 59, 63, 66], guitar=[52, 59], strings=[52, 56, 59, 63, 66]),
}

PROG_VERSE = ["F#m9", "Dmaj9", "C#m7", "Bm9", "F#m9", "Dmaj9", "C#7sus", "C#7"]
PROG_HOOK = ["F#m9", "E", "Dmaj9", "C#7sus", "F#m9", "E", "Dmaj9", "C#7"]
PROG_BRIDGE = ["Bm9", "C#m7", "Dmaj9", "E", "Bm9", "C#7sus", "F#m9", "F#m9"]
PROG_OUTRO = ["F#m9", "Dmaj9", "Bm9", "C#7sus", "F#m9", "Dmaj9", "F#m9", "F#m9"]

SECTIONS = [
    ("intro", 0, 4),
    ("verse1", 4, 16),
    ("hook1", 20, 8),
    ("verse2", 28, 16),
    ("bridge", 44, 8),
    ("hook2", 52, 8),
    ("outro", 60, 8),
]


def prog_at(bar: int) -> str:
    for name, start, length in SECTIONS:
        if start <= bar < start + length:
            if name in ("verse1", "verse2"):
                return PROG_VERSE[(bar - start) % 8]
            if name in ("hook1", "hook2"):
                return PROG_HOOK[(bar - start) % 8]
            if name == "bridge":
                return PROG_BRIDGE[(bar - start) % 8]
            if name == "outro":
                return PROG_OUTRO[(bar - start) % 8]
            if name == "intro":
                return "F#m9"
    return "F#m9"


# ---------------------------------------------------------------- performance
@dataclass
class Humanizer:
    seed: int = 2026
    timing_ms: float = 4.0
    vel_pct: float = 0.05

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def time(self, beat: float, amount=1.0) -> float:
        j = self.rng.normal(0, self.timing_ms / 1000.0) * amount
        return max(0.0, b2s(swing_beat(beat)) + j)

    def vel(self, v: float) -> float:
        return float(np.clip(v * (1 + self.rng.normal(0, self.vel_pct)), 0.02, 1.0))


@dataclass
class Events:
    """schedule for every instrument in the piece"""
    kick: list = field(default_factory=list)
    snare: list = field(default_factory=list)
    clap: list = field(default_factory=list)
    rim: list = field(default_factory=list)
    hat: list = field(default_factory=list)        # (t, vel, open, pan)
    perc: list = field(default_factory=list)       # (t, vel, kind, pan)
    tom: list = field(default_factory=list)        # (t, vel, midi, pan)
    crash: list = field(default_factory=list)      # (t, vel, reverse)
    bass: list = field(default_factory=list)       # dicts
    rhodes: list = field(default_factory=list)     # (t, dur, notes, vel, pan)
    guitar: list = field(default_factory=list)     # (t, dur, notes, vel, muted, pan)
    strings: list = field(default_factory=list)    # (t, dur, notes, vel, octave_up)
    lead: list = field(default_factory=list)       # (t, dur, midi, vel, vowel, glide, bend)
    synth: list = field(default_factory=list)      # same as lead
    fx: list = field(default_factory=list)         # (t, kind, params)
    riser: list = field(default_factory=list)


# ================================================================ DRUMS
def add_groove(E: Events, H: Humanizer, bar: int, style="verse", energy=1.0,
               fill=False, variation=0):
    """style: verse | hook | halftime | breakdown"""
    t0 = bar * 4
    def hit(lst, beat, vel, *extra):
        lst.append((H.time(t0 + beat), H.vel(vel * energy)) + extra)

    if style == "verse":
        kicks = [(0, 1.0), (1.75, 0.82), (2.5, 0.92)]
        if variation % 4 == 1:
            kicks = [(0, 1.0), (1.5, 0.72), (2.5, 0.95), (3.75, 0.6)]
        if variation % 4 == 3:
            kicks = [(0, 1.0), (1.75, 0.78), (2.25, 0.7), (3.0, 0.9)]
        for b, v in kicks:
            hit(E.kick, b, v)
        hit(E.snare, 1.0, 0.95)
        hit(E.snare, 3.0, 1.0)
        hit(E.snare, 2.25, 0.18)
        hit(E.snare, 3.75, 0.24)
        if variation % 8 in (5, 7):
            hit(E.snare, 1.25, 0.16)
        # swung 16th hats, accents on the downbeats and the "&"
        for i in range(16):
            b = i * 0.25
            acc = 0.62 if i % 4 == 0 else (0.46 if i % 2 == 0 else 0.30)
            if i in (3, 11):
                acc *= 0.55
            open_ = (i in (6, 14)) and (variation % 2 == 0)
            hit(E.hat, b, acc, open_, 0.16)
        if variation % 8 in (3, 7):
            hit(E.hat, 3.5, 0.5, True, 0.16)
        if variation % 8 == 6:
            hit(E.clap, 1.0, 0.35)
            hit(E.clap, 3.0, 0.4)

    elif style == "hook":
        kicks = [(0, 1.0), (1.75, 0.88), (2.5, 1.0), (3.5, 0.82)]
        if variation % 4 == 2:
            kicks = [(0, 1.0), (0.75, 0.65), (1.75, 0.8), (2.5, 1.0), (3.5, 0.9)]
        for b, v in kicks:
            hit(E.kick, b, v)
        hit(E.snare, 1.0, 1.0)
        hit(E.snare, 3.0, 1.0)
        hit(E.clap, 1.0, 0.85)
        hit(E.clap, 3.0, 0.9)
        hit(E.snare, 2.25, 0.2)
        hit(E.snare, 3.75, 0.3)
        for i in range(16):
            b = i * 0.25
            acc = 0.68 if i % 4 == 0 else (0.5 if i % 2 == 0 else 0.34)
            open_ = i in (6, 14)
            hit(E.hat, b, acc, open_, 0.18)
        for i in range(4):
            hit(E.perc, i + 0.5, 0.5 if i % 2 else 0.62, "tamb", -0.2)
        hit(E.hat, 3.5, 0.42, True, 0.18)

    elif style == "half":
        for b, v in [(0, 1.0), (2.5, 0.88)]:
            hit(E.kick, b, v)
        hit(E.snare, 2.0, 0.9)
        hit(E.snare, 3.75, 0.2)
        for i in range(8):
            hit(E.hat, i * 0.5, 0.5 if i % 2 == 0 else 0.34, i == 6, 0.12)
        hit(E.tom, 3.0, 0.5, 45, 0.1)

    elif style == "breakdown":
        hit(E.kick, 0, 0.9)
        hit(E.rim, 1.0, 0.55)
        if variation % 2:
            hit(E.rim, 3.0, 0.45)
        for i in range(8):
            hit(E.hat, i * 0.5, 0.28 if i % 2 else 0.38, False, 0.12)

    if fill:
        # expressive tom fill on beat 4, tuned to F# minor (F#2, A2, C#3)
        seq = [(3.0, 0.7, 41), (3.25, 0.75, 45), (3.5, 0.85, 48), (3.75, 0.95, 53)]
        if variation % 2:
            seq = [(2.75, 0.5, 41), (3.0, 0.65, 45), (3.25, 0.7, 41), (3.5, 0.8, 48),
                   (3.625, 0.6, 45), (3.75, 0.9, 53), (3.875, 0.7, 48)]
        for b, v, m in seq:
            hit(E.tom, b, v, m, 0.2)
        hit(E.snare, 3.5, 0.4)


def build_drums(E: Events, H: Humanizer):
    # ---- intro
    add_groove(E, H, 1, "half", energy=0.55)
    add_groove(E, H, 2, "verse", energy=0.78, variation=0)
    add_groove(E, H, 3, "verse", energy=0.85, variation=1, fill=True)
    # ---- verse 1
    for i, bar in enumerate(range(4, 20)):
        fill = (i % 8 == 7)
        add_groove(E, H, bar, "verse", energy=1.0, variation=i, fill=fill)
    # ---- hook 1
    for i, bar in enumerate(range(20, 28)):
        add_groove(E, H, bar, "hook", energy=1.0, variation=i, fill=(i == 7))
        if i in (0, 4):
            E.crash.append((H.time(bar * 4), 0.75, False))
    # ---- verse 2 (busier)
    for i, bar in enumerate(range(28, 44)):
        add_groove(E, H, bar, "verse", energy=1.02, variation=i + 3,
                   fill=(i % 8 == 7))
        if i % 8 == 4:
            E.crash.append((H.time(bar * 4), 0.35, False))
        if i >= 4:  # shaker from the second half of verse 2
            for k in range(16):
                if k % 2 == 0 or k in (3, 7, 11, 15):
                    E.perc.append((H.time(bar * 4 + k * 0.25), H.vel(0.30 if k % 4 else 0.42),
                                   "shaker", 0.35))
    # ---- bridge
    for i, bar in enumerate(range(44, 52)):
        if i < 2:
            add_groove(E, H, bar, "breakdown", energy=0.65, variation=i)
        elif i < 4:
            add_groove(E, H, bar, "half", energy=0.8)
            for k in range(4):
                E.perc.append((H.time(bar * 4 + k), H.vel(0.35), "shaker", 0.3))
        else:
            add_groove(E, H, bar, "half", energy=0.9, variation=i)
            add_groove(E, H, bar, "breakdown", energy=0.5, variation=i)
        if i in (2, 3):
            E.crash.append((H.time(bar * 4), 0.35, False))
        if i >= 4:
            for k in range(16):
                E.perc.append((H.time(bar * 4 + k * 0.25),
                               H.vel(0.25 + 0.14 * (k % 4 == 0)), "shaker", 0.3))
    # snare roll build (bars 50-51)
    for k in range(8):
        E.snare.append((H.time(50 * 4 + k * 0.5), H.vel(0.25 + 0.03 * k)))
    for k in range(16):
        E.snare.append((H.time(51 * 4 + k * 0.25), H.vel(0.32 + 0.025 * k)))
    for k in range(8):
        E.snare.append((H.time(51 * 4 + 2 + k * 0.125), H.vel(0.55 + 0.03 * k)))
    # ---- hook 2
    for i, bar in enumerate(range(52, 60)):
        add_groove(E, H, bar, "hook", energy=1.06, variation=i + 2, fill=(i == 7))
        if i in (0, 4):
            E.crash.append((H.time(bar * 4), 0.85, False))
        for k in range(16):
            E.perc.append((H.time(bar * 4 + k * 0.25), H.vel(0.3 if k % 4 else 0.45),
                           "shaker", 0.3))
    # ---- outro
    for i, bar in enumerate(range(60, 64)):
        add_groove(E, H, bar, "hook", energy=0.95 - 0.05 * i, variation=i,
                   fill=(i == 3))
        E.crash.append((H.time(bar * 4), 0.5, False))
    # bar 64: music stops
    for b, v in [(0, 0.95), (1.75, 0.7)]:
        E.kick.append((H.time(64 * 4 + b), H.vel(v)))
    E.snare.append((H.time(64 * 4 + 1.0), H.vel(0.9)))
    E.crash.append((H.time(64 * 4), 0.6, False))
    # final downbeat of the outro "sting"
    E.kick.append((H.time(66 * 4), H.vel(0.9)))
    E.tom.append((H.time(66 * 4 + 1.5), H.vel(0.5), 41, 0.2))


# ================================================================ BASS
BASS_PATTERNS = [
    # (step16, semitone offset from root, duration in 16ths, vel, art, slide?)
    [(0, 0, 3, 0.95, "finger", False),
     (3, 0, 1, 0.55, "mute", False),
     (6, 12, 2, 0.80, "finger", False),
     (8, 0, 2, 0.85, "finger", False),
     (10, 10, 1, 0.62, "finger", False),
     (11, 12, 3, 0.82, "finger", True),
     (14, 7, 2, 0.70, "finger", False)],
    [(0, 0, 2, 0.98, "finger", False),
     (2, 0, 1, 0.60, "finger", False),
     (3, 12, 1, 0.78, "slap", False),
     (6, 10, 2, 0.72, "finger", False),
     (8, 0, 2, 0.88, "finger", False),
     (10, 12, 1, 0.70, "finger", False),
     (11, 0, 2, 0.80, "finger", True),
     (13, 3, 1, 0.55, "mute", False),
     (14, 5, 2, 0.75, "finger", False)],
    [(0, 0, 2, 0.95, "finger", False),
     (2, 0, 1, 0.55, "mute", False),
     (3, 7, 1, 0.75, "finger", False),
     (6, 10, 2, 0.80, "finger", False),
     (8, 12, 2, 0.90, "finger", False),
     (10, 10, 1, 0.65, "finger", False),
     (11, 7, 1, 0.72, "finger", False),
     (12, 5, 1, 0.60, "mute", False),
     (13, 3, 1, 0.70, "finger", False),
     (14, 0, 2, 0.85, "finger", True)],
    [(0, 0, 3, 0.95, "finger", False),
     (4, 0, 1, 0.5, "mute", False),
     (7, 12, 1, 0.72, "finger", False),
     (8, 12, 2, 0.85, "finger", False),
     (11, 10, 1, 0.65, "finger", False),
     (12, 7, 1, 0.7, "finger", False),
     (13, 5, 1, 0.7, "finger", False),
     (14, 0, 2, 0.82, "finger", True)],
]


MAJOR_CHORDS = {"Dmaj9", "E", "A", "Emaj9"}
DOM_CHORDS = {"C#7"}
SUS_CHORDS = {"C#7sus"}


def bass_offset(chord: str, off: int) -> int:
    """Keep the bassline diatonic to each chord's quality (no minor 7th over a
    major chord, no natural 3rd over a dominant/sus chord)."""
    if chord in MAJOR_CHORDS:
        return {10: 11, 3: 4, 5: 7}.get(off, off)
    if chord in DOM_CHORDS:
        return {3: 4}.get(off, off)
    if chord in SUS_CHORDS:
        return {3: 5}.get(off, off)
    return off


def add_bass_bar(E: Events, H: Humanizer, bar: int, chord: str, variant=0,
                 energy=1.0, last_bar_of_phrase=False):
    root = CHORDS[chord]["bass"]
    pat = BASS_PATTERNS[variant % len(BASS_PATTERNS)]
    t0 = bar * 4
    prev_midi = None
    for (step, off, dur16, vel, art, slide) in pat:
        beat = step / 4.0
        midi = root + bass_offset(chord, off)
        glide = None
        if slide and prev_midi is not None:
            glide = prev_midi
        elif slide:
            glide = midi - 3
        dur = dur16 / 4.0
        E.bass.append(dict(
            t=H.time(t0 + beat, 0.8),
            dur=b2s(dur) * (0.92 if art == "mute" else 1.0),
            midi=midi, vel=H.vel(vel * energy), art=art,
            glide_from=glide, glide_s=0.055 + 0.02 * (variant % 2)))
        prev_midi = midi
    # phrase-ending fill: walk up to the next chord
    if last_bar_of_phrase:
        E.bass.append(dict(t=H.time(t0 + 3.5, 0.6), dur=b2s(0.22), midi=root + 7,
                           vel=H.vel(0.6 * energy), art="finger", glide_from=root,
                           glide_s=0.05))
        E.bass.append(dict(t=H.time(t0 + 3.75, 0.6), dur=b2s(0.2), midi=root + 12,
                           vel=H.vel(0.72 * energy), art="finger", glide_from=root + 7,
                           glide_s=0.05))


def build_bass(E: Events, H: Humanizer):
    # intro pickup
    E.bass.append(dict(t=H.time(2 * 4 + 3.0), dur=b2s(0.5), midi=30, vel=0.55,
                       art="finger", glide_from=25, glide_s=0.09))
    E.bass.append(dict(t=H.time(2 * 4 + 3.5), dur=b2s(0.5), midi=30, vel=0.7,
                       art="finger", glide_from=30, glide_s=0.06))
    E.bass.append(dict(t=H.time(3 * 4), dur=b2s(0.9), midi=30, vel=0.8,
                       art="finger", glide_from=None, glide_s=0.05))
    E.bass.append(dict(t=H.time(3 * 4 + 2.5), dur=b2s(0.4), midi=42, vel=0.6,
                       art="finger", glide_from=30, glide_s=0.06))
    E.bass.append(dict(t=H.time(3 * 4 + 3.5), dur=b2s(0.45), midi=37, vel=0.7,
                       art="finger", glide_from=42, glide_s=0.06))

    # verse 1 — 16 bars, patterns evolve
    for i, bar in enumerate(range(4, 20)):
        chord = prog_at(bar)
        variant = [0, 1, 0, 2, 0, 1, 2, 3][i % 8]
        add_bass_bar(E, H, bar, chord, variant, 1.0, last_bar_of_phrase=(i % 8 == 7))

    # hook 1 — melodic, octave pops, more slides
    for i, bar in enumerate(range(20, 28)):
        chord = prog_at(bar)
        variant = [1, 0, 2, 1, 3, 0, 1, 2][i]
        add_bass_bar(E, H, bar, chord, variant, 1.05, last_bar_of_phrase=(i == 7))

    # verse 2
    for i, bar in enumerate(range(28, 44)):
        chord = prog_at(bar)
        variant = [0, 2, 1, 0, 3, 1, 0, 2][i % 8]
        add_bass_bar(E, H, bar, chord, variant, 1.02, last_bar_of_phrase=(i % 8 == 7))

    # bridge — sustained, slidey, then driving
    for i, bar in enumerate(range(44, 52)):
        chord = prog_at(bar)
        root = CHORDS[chord]["bass"]
        t0 = bar * 4
        if i < 2:
            E.bass.append(dict(t=H.time(t0, 0.7), dur=b2s(2.6), midi=root, vel=0.75,
                               art="finger", glide_from=root - 5 if i else None,
                               glide_s=0.16, vibrato=0.12))
            E.bass.append(dict(t=H.time(t0 + 3.0, 0.7), dur=b2s(0.8), midi=root + 7,
                               vel=0.55, art="finger", glide_from=root, glide_s=0.09))
        elif i < 4:
            E.bass.append(dict(t=H.time(t0, 0.7), dur=b2s(1.8), midi=root, vel=0.8,
                               art="finger", glide_from=root - 2, glide_s=0.12))
            E.bass.append(dict(t=H.time(t0 + 2.0, 0.7), dur=b2s(0.9), midi=root + 12,
                               vel=0.7, art="finger", glide_from=root, glide_s=0.08))
            E.bass.append(dict(t=H.time(t0 + 3.25, 0.7), dur=b2s(0.6), midi=root + 7,
                               vel=0.62, art="finger", glide_from=root + 12, glide_s=0.07))
        else:
            for k in range(8):
                midi = root + [0, 0, 12, 0, 7, 0, 10, 12][k]
                E.bass.append(dict(t=H.time(t0 + k * 0.5, 0.7), dur=b2s(0.42),
                                   midi=midi, vel=0.7 + 0.03 * k, art="finger",
                                   glide_from=midi - 12 if k in (2, 5) else None,
                                   glide_s=0.05))
    # hook 2
    for i, bar in enumerate(range(52, 60)):
        chord = prog_at(bar)
        variant = [1, 3, 2, 1, 0, 2, 1, 3][i]
        add_bass_bar(E, H, bar, chord, variant, 1.08, last_bar_of_phrase=(i == 7))
    # outro
    for i, bar in enumerate(range(60, 64)):
        chord = prog_at(bar)
        add_bass_bar(E, H, bar, chord, [0, 1, 2, 0][i], 0.95,
                     last_bar_of_phrase=(i == 3))
    # final notes
    E.bass.append(dict(t=H.time(64 * 4, 0.7), dur=b2s(2.0), midi=30, vel=0.85,
                       art="finger", glide_from=37, glide_s=0.10, vibrato=0.1))
    E.bass.append(dict(t=H.time(66 * 4, 0.7), dur=b2s(6.0), midi=30, vel=0.9,
                       art="finger", glide_from=None, glide_s=0.05, vibrato=0.05))


# ================================================================ KEYS
def build_rhodes(E: Events, H: Humanizer):
    # intro pad
    for bar, vel in [(0, 0.42), (1, 0.5), (2, 0.62), (3, 0.7)]:
        E.rhodes.append((H.time(bar * 4, 0.6), b2s(3.7), CHORDS["F#m9"]["rhodes"],
                         H.vel(vel), 0.02))
    # verse comping
    for i, bar in enumerate(range(4, 20)):
        ch = CHORDS[prog_at(bar)]
        t0 = bar * 4
        hits = [(0.0, 0.9, 0.78), (1.5, 0.45, 0.55), (2.5, 1.1, 0.68)]
        if i % 4 == 3:
            hits = [(0.0, 0.7, 0.7), (2.0, 0.6, 0.6), (3.25, 0.6, 0.52)]
        for b, dur, v in hits:
            E.rhodes.append((H.time(t0 + b, 0.6), b2s(dur), ch["rhodes"],
                             H.vel(v), 0.04))
    # hook — sustained, big, with pickup stabs
    for i, bar in enumerate(range(20, 28)):
        ch = CHORDS[prog_at(bar)]
        t0 = bar * 4
        E.rhodes.append((H.time(t0, 0.6), b2s(2.4), ch["rhodes"], H.vel(0.72), 0.03))
        E.rhodes.append((H.time(t0 + 2.5, 0.6), b2s(1.2), ch["rhodes"], H.vel(0.58), 0.05))
        if i % 4 == 3:
            E.rhodes.append((H.time(t0 + 3.5, 0.6), b2s(0.4), ch["rhodes"], H.vel(0.5), 0.0))
    # verse 2
    for i, bar in enumerate(range(28, 44)):
        ch = CHORDS[prog_at(bar)]
        t0 = bar * 4
        hits = [(0.0, 1.0, 0.74), (1.75, 0.45, 0.5), (2.5, 1.0, 0.66), (3.5, 0.4, 0.45)]
        if i % 8 == 5:
            hits = [(0.0, 1.4, 0.7), (2.5, 0.8, 0.62)]
        for b, dur, v in hits:
            E.rhodes.append((H.time(t0 + b, 0.6), b2s(dur), ch["rhodes"],
                             H.vel(v), 0.04))
    # bridge — sparse and spacious
    for i, bar in enumerate(range(44, 52)):
        ch = CHORDS[prog_at(bar)]
        t0 = bar * 4
        if i < 4:
            E.rhodes.append((H.time(t0, 0.8), b2s(3.8), ch["rhodes"], H.vel(0.55), 0.02))
        else:
            E.rhodes.append((H.time(t0, 0.8), b2s(2.2), ch["rhodes"], H.vel(0.62), 0.02))
            E.rhodes.append((H.time(t0 + 2.5, 0.8), b2s(1.4), ch["rhodes"], H.vel(0.48), 0.04))
    # hook 2
    for i, bar in enumerate(range(52, 60)):
        ch = CHORDS[prog_at(bar)]
        t0 = bar * 4
        E.rhodes.append((H.time(t0, 0.6), b2s(2.4), ch["rhodes"], H.vel(0.78), 0.03))
        E.rhodes.append((H.time(t0 + 2.5, 0.6), b2s(1.2), ch["rhodes"], H.vel(0.62), 0.05))
    # outro
    for i, bar in enumerate(range(60, 64)):
        ch = CHORDS[prog_at(bar)]
        E.rhodes.append((H.time(bar * 4, 0.7), b2s(3.6), ch["rhodes"], H.vel(0.6), 0.02))
    E.rhodes.append((H.time(64 * 4, 0.7), b2s(2.0), CHORDS["F#m9"]["rhodes"], H.vel(0.7), 0.0))
    E.rhodes.append((H.time(66 * 4, 0.9), b2s(12.0), CHORDS["F#m9"]["rhodes"], H.vel(0.8), 0.0))


# ================================================================ GUITAR
def build_guitar(E: Events, H: Humanizer):
    # intro: sparse muted chanks
    for bar, offs, vel in [(1, [1.5, 3.5], 0.4), (2, [0.75, 1.5, 2.75, 3.5], 0.5),
                           (3, [0.75, 1.25, 2.0, 2.75, 3.25, 3.75], 0.55)]:
        ch = CHORDS["F#m9"]
        for b in offs:
            E.guitar.append((H.time(bar * 4 + b, 1.1), b2s(0.16), ch["guitar"],
                             H.vel(vel), True, -0.22))
    # verse chank patterns — rhythm variations every 4 bars
    for i, bar in enumerate(range(4, 20)):
        ch = CHORDS[prog_at(bar)]
        t0 = bar * 4
        if i % 8 in (3, 7):      # leave space at phrase ends
            offs = [0, 1.5, 2.75]
            vel = 0.55
        elif i % 4 == 1:
            offs = [0, 0.75, 1.25, 2.0, 2.5, 3.25, 3.5, 3.75]
            vel = 0.6
        else:
            offs = [0, 0.75, 1.5, 2.25, 2.5, 3.25, 3.75]
            vel = 0.62
        for b in offs:
            acc = vel * (1.15 if abs((b * 2) % 1) < 0.1 else 0.85)
            E.guitar.append((H.time(t0 + b, 1.0), b2s(0.15), ch["guitar"],
                             H.vel(acc), True, -0.22))
    # hook: off-beat stabs (contrast with the verse's 16ths)
    for i, bar in enumerate(range(20, 28)):
        ch = CHORDS[prog_at(bar)]
        t0 = bar * 4
        for b, v in [(0.75, 0.7), (1.25, 0.55), (2.75, 0.72), (3.25, 0.6)]:
            E.guitar.append((H.time(t0 + b, 1.0), b2s(0.3), ch["guitar"],
                             H.vel(v), True, -0.22))
        if i % 4 == 0:
            E.guitar.append((H.time(t0 + 3.75, 1.0), b2s(0.25), ch["guitar"],
                             H.vel(0.5), True, -0.22))
    # verse 2: busier 16ths with open-string pedal feel
    for i, bar in enumerate(range(28, 44)):
        ch = CHORDS[prog_at(bar)]
        t0 = bar * 4
        offs = [0, 0.5, 0.75, 1.25, 1.5, 2.0, 2.25, 2.75, 3.0, 3.25, 3.75]
        for b in offs:
            acc = 0.66 * (1.2 if abs((b) % 1) < 0.1 else 0.8)
            E.guitar.append((H.time(t0 + b, 1.0), b2s(0.14), ch["guitar"],
                             H.vel(acc), True, -0.22))
    # bridge: sparse, clean dyads
    for i, bar in enumerate(range(44, 52)):
        ch = CHORDS[prog_at(bar)]
        if i < 4:
            E.guitar.append((H.time(bar * 4 + 2.5, 1.0), b2s(0.5), ch["guitar"],
                             0.45, False, -0.22))
        else:
            for b in [0.75, 1.75, 2.5, 3.25]:
                E.guitar.append((H.time(bar * 4 + b, 1.0), b2s(0.2), ch["guitar"],
                                 H.vel(0.5 + 0.03 * i), True, -0.22))
    # hook 2 + outro
    for i, bar in enumerate(range(52, 64)):
        ch = CHORDS[prog_at(bar)]
        t0 = bar * 4
        for b, v in [(0.0, 0.6), (0.75, 0.75), (1.25, 0.6), (2.0, 0.55),
                     (2.75, 0.78), (3.25, 0.62), (3.75, 0.55)]:
            E.guitar.append((H.time(t0 + b, 1.0), b2s(0.16), ch["guitar"],
                             H.vel(v), True, -0.22))
    for b in [0.75, 2.75]:
        E.guitar.append((H.time(64 * 4 + b), b2s(0.2), CHORDS["F#m9"]["guitar"], 0.45, True, -0.22))


# ================================================================ STRINGS
def build_strings(E: Events, H: Humanizer):
    # intro swell
    E.strings.append((0.0, b2s(15.5), CHORDS["F#m9"]["strings"], 0.30, False))
    # verse beds: long notes, sparse
    for start in (4, 12, 28, 36):
        for k in range(4):
            bar = start + k * 2
            ch = CHORDS[prog_at(bar)]
            E.strings.append((H.time(bar * 4, 1.4), b2s(7.6), ch["strings"],
                              H.vel(0.34 + 0.02 * k), False))
    # hook 1
    for i, bar in enumerate(range(20, 28)):
        ch = CHORDS[prog_at(bar)]
        E.strings.append((H.time(bar * 4, 1.2), b2s(3.8), ch["strings"],
                          H.vel(0.55), True))
    # verse 2 string counter-line (top voice movement)
    for i, bar in enumerate(range(28, 44, 2)):
        ch = CHORDS[prog_at(bar)]
        E.strings.append((H.time(bar * 4, 1.2), b2s(7.7), ch["strings"],
                          H.vel(0.4), False))
    # bridge: swell upward
    for i, bar in enumerate(range(44, 52)):
        ch = CHORDS[prog_at(bar)]
        if i < 4:
            E.strings.append((H.time(bar * 4, 1.2), b2s(3.9), ch["strings"],
                              H.vel(0.38), False))
        else:
            E.strings.append((H.time(bar * 4, 1.2), b2s(3.9), ch["strings"],
                              H.vel(0.45 + 0.05 * (i - 4)), True))
    # big swell into hook 2
    E.strings.append((H.time(50 * 4, 1.2), b2s(7.8), CHORDS["F#m9"]["strings"],
                      H.vel(0.6), True))
    # hook 2
    for i, bar in enumerate(range(52, 60)):
        ch = CHORDS[prog_at(bar)]
        E.strings.append((H.time(bar * 4, 1.2), b2s(3.8), ch["strings"],
                          H.vel(0.58), True))
    # outro
    for i, bar in enumerate(range(60, 64)):
        ch = CHORDS[prog_at(bar)]
        E.strings.append((H.time(bar * 4, 1.4), b2s(4.0), ch["strings"],
                          H.vel(0.5 - 0.04 * i), False))
    E.strings.append((H.time(64 * 4, 1.6), b2s(8.0), CHORDS["F#m9"]["strings"], 0.42, True))
    E.strings.append((H.time(66 * 4, 1.8), b2s(14.0), CHORDS["F#m9"]["strings"], 0.5, True))


# ================================================================ LEADS
def ev(beat, dur, midi, vel=0.7, vowel="oh", glide=False, bend=0.0, vib=0.12):
    return dict(beat=beat, dur=dur, midi=midi, vel=vel, vowel=vowel,
                glide=glide, bend=bend, vib=vib)


HOOK_MELODY = [
    # bars 1-4: the "call"
    ev(0.00, 0.50, 66, 0.80, "oh"),
    ev(0.75, 0.50, 69, 0.85, "oh", glide=True),
    ev(1.50, 1.50, 73, 0.9, "ah", vib=0.18),
    ev(4.00, 0.50, 71, 0.8, "oh"),
    ev(4.75, 0.50, 68, 0.78, "oh"),
    ev(5.50, 0.75, 64, 0.72, "oo"),
    ev(6.50, 0.75, 68, 0.82, "oh", glide=True),
    ev(8.00, 0.75, 69, 0.85, "ah", glide=True),
    ev(9.00, 0.50, 66, 0.78, "oh"),
    ev(9.50, 0.50, 64, 0.72, "oh"),
    ev(10.00, 1.25, 62, 0.8, "oo", vib=0.16),
    ev(11.50, 0.50, 66, 0.72, "oh"),
    ev(12.00, 1.00, 64, 0.78, "ah"),
    ev(13.25, 1.50, 61, 0.7, "oo", vib=0.2),
    # bars 5-8: the "response" (higher, answers itself)
    ev(16.00, 0.50, 73, 0.85, "oh"),
    ev(16.50, 0.50, 76, 0.9, "ah"),
    ev(17.00, 1.25, 78, 0.95, "ah", vib=0.22),
    ev(20.00, 0.75, 76, 0.85, "oh"),
    ev(20.75, 0.50, 73, 0.8, "oh"),
    ev(21.50, 1.00, 71, 0.8, "oo", vib=0.16),
    ev(24.00, 0.75, 69, 0.82, "ah"),
    ev(25.00, 0.50, 71, 0.8, "oh"),
    ev(25.50, 0.75, 73, 0.85, "ah"),
    ev(26.50, 0.75, 78, 0.9, "ah", vib=0.2),
    ev(28.00, 0.50, 71, 0.8, "oh"),
    ev(28.50, 0.50, 69, 0.78, "oh"),
    ev(29.00, 0.50, 68, 0.76, "oo"),
    ev(29.50, 2.00, 73, 0.88, "ah", bend=-1.0, vib=0.22),
]


def hook_events(bar0: int, transpose=0, vel_scale=1.0, octave=0):
    out = []
    for e in HOOK_MELODY:
        out.append(dict(beat=bar0 * 4 + e["beat"], dur=e["dur"],
                        midi=e["midi"] + transpose + 12 * octave,
                        vel=min(1.0, e["vel"] * vel_scale), vowel=e["vowel"],
                        glide=e["glide"], bend=e["bend"], vib=e["vib"]))
    return out


def build_leads(E: Events, H: Humanizer):
    rng = np.random.default_rng(77)

    def push(target, beat, dur, midi, vel, vowel="oh", glide=False, bend=0.0, vib=0.14):
        target.append(dict(t=H.time(beat, 0.7), dur=b2s(dur), midi=midi,
                           vel=H.vel(vel), vowel=vowel, glide=glide,
                           bend=bend, vib=vib))

    # ---- intro: the hook's opening phrase, stated quietly (the "teaser")
    push(E.lead, 1 * 4 + 1.5, 0.5, 66, 0.5, "oo")
    push(E.lead, 1 * 4 + 2.25, 0.5, 69, 0.55, "oo", glide=True)
    push(E.lead, 1 * 4 + 3.0, 1.5, 73, 0.6, "ah", glide=True, vib=0.18)
    push(E.lead, 3 * 4 + 2.0, 0.5, 73, 0.5, "ah", glide=True)
    push(E.lead, 3 * 4 + 2.75, 1.25, 69, 0.55, "oo", bend=-0.4)

    # ---- verse calls: 1-bar answers in the gaps (call & response with the groove)
    calls = {
        6: [(0.0, 0.5, 73, 0.6, "oh"), (0.75, 0.5, 69, 0.58, "oo"),
            (1.5, 1.0, 66, 0.62, "ah")],
        10: [(0.0, 0.75, 71, 0.6, "oh"), (1.0, 0.5, 68, 0.55, "oo"),
             (1.75, 1.25, 64, 0.58, "ah", -0.4)],
        14: [(0.0, 0.5, 66, 0.62, "oh"), (0.5, 0.5, 69, 0.6, "oh"),
             (1.25, 1.5, 73, 0.68, "ah")],
        18: [(0.0, 0.75, 73, 0.62, "ah"), (1.0, 0.5, 71, 0.58, "oo"),
             (1.75, 1.25, 68, 0.6, "ah", -0.5)],
    }
    for bar, notes in calls.items():
        for item in notes:
            b, d, m, v, vow = item[:5]
            bend = item[5] if len(item) > 5 else 0.0
            push(E.lead, bar * 4 + b, d, m, v, vow, glide=(b > 0), bend=bend)

    # verse 2 calls (denser, higher)
    calls2 = {
        30: [(0.0, 0.5, 66, 0.62, "oh"), (0.75, 0.5, 73, 0.65, "ah"),
             (1.5, 1.0, 71, 0.6, "oo")],
        34: [(0.0, 0.75, 69, 0.6, "ah"), (1.0, 0.5, 66, 0.55, "oh"),
             (1.75, 1.0, 61, 0.58, "oo", -0.3)],
        38: [(0.0, 0.5, 73, 0.68, "ah"), (0.5, 0.5, 76, 0.7, "ah"),
             (1.25, 1.5, 78, 0.72, "ah")],
        42: [(0.0, 0.5, 71, 0.6, "oh"), (0.75, 0.5, 68, 0.58, "oo"),
             (1.5, 1.0, 64, 0.62, "ah", -0.4)],
    }
    for bar, notes in calls2.items():
        for (b, d, m, v, vow, *rest) in notes:
            push(E.lead, bar * 4 + b, d, m, v, vow, glide=(b > 0),
                 bend=rest[0] if rest else 0.0)

    # ---- bridge: slow, expressive, half-time phrasing
    bridge_notes = [
        (44 * 4 + 0.5, 2.0, 66, 0.62, "oo", False, 0),
        (44 * 4 + 2.75, 1.5, 69, 0.6, "ah", True, -0.5),
        (45 * 4 + 0.0, 2.5, 61, 0.58, "oo", True, 0),
        (45 * 4 + 3.0, 1.0, 64, 0.55, "ah", True, 0),
        (46 * 4 + 0.5, 2.0, 66, 0.6, "oh", True, 0),
        (47 * 4 + 1.0, 2.0, 71, 0.65, "ah", True, 0),
        (48 * 4 + 0.0, 1.0, 66, 0.62, "ah", True, 0),
        (48 * 4 + 1.5, 0.75, 68, 0.6, "oh", True, 0),
        (48 * 4 + 2.5, 1.25, 69, 0.65, "ah", True, 0),
        (49 * 4 + 0.0, 1.5, 71, 0.7, "ah", True, 0),
        (50 * 4 + 0.0, 0.75, 73, 0.72, "ah", True, 0),
        (50 * 4 + 1.0, 0.75, 71, 0.7, "oh", True, 0),
        (50 * 4 + 2.0, 1.75, 69, 0.72, "ah", True, 0),
        (51 * 4 + 0.0, 0.5, 68, 0.7, "oh", True, 0),
        (51 * 4 + 0.75, 0.5, 69, 0.72, "ah", True, 0),
        (51 * 4 + 1.5, 2.0, 73, 0.78, "ah", True, 0.5),
    ]
    for (b, d, m, v, vow, gl, bend) in bridge_notes:
        push(E.lead, b, d, m, v, vow, glide=gl, bend=bend, vib=0.2)

    # ---- hook 1: the full motif (talkbox)
    for e in hook_events(20):
        push(E.lead, e["beat"], e["dur"], e["midi"], e["vel"], e["vowel"],
             e["glide"], e["bend"], e["vib"])
    # ---- hook 2: same motif, small ornaments, synth double a 12th above
    for e in hook_events(52):
        push(E.lead, e["beat"], e["dur"], e["midi"], e["vel"], e["vowel"],
             e["glide"], e["bend"], e["vib"])
    # ornaments on the last hook
    push(E.lead, 57 * 4 + 3.5, 0.5, 78, 0.7, "ah", glide=True)
    push(E.lead, 58 * 4 + 3.0, 1.5, 81, 0.75, "ah", glide=True, vib=0.25)
    # ---- outro tags: motif fragment answered, then a long final note
    push(E.lead, 60 * 4 + 0.0, 1.0, 73, 0.62, "ah", glide=True)
    push(E.lead, 61 * 4 + 0.5, 1.5, 69, 0.6, "oo", glide=True, bend=-0.5)
    push(E.lead, 62 * 4 + 0.0, 0.75, 66, 0.55, "oh")
    push(E.lead, 62 * 4 + 1.0, 1.5, 73, 0.62, "ah", glide=True)
    push(E.lead, 63 * 4 + 0.0, 3.0, 78, 0.68, "ah", bend=-0.8, vib=0.22)
    push(E.lead, 66 * 4 + 0.5, 3.0, 66, 0.55, "oo", glide=True, bend=-0.3, vib=0.2)

    # ---- synth lead: doubles the hook an octave up (hook 2), plus bridge line
    for e in hook_events(52, transpose=12, vel_scale=0.72):
        E.synth.append(dict(t=H.time(e["beat"], 0.7), dur=b2s(e["dur"]),
                            midi=e["midi"], vel=H.vel(e["vel"]), vowel="ah",
                            glide=e["glide"], bend=e["bend"], vib=e["vib"]))
    # synth sparkles in hook 1 (short answers, lower octave)
    spark = [(21 * 4 + 3.5, 0.5, 73), (23 * 4 + 3.5, 0.5, 78),
             (25 * 4 + 3.5, 0.75, 81), (27 * 4 + 2.5, 1.25, 78)]
    for (b, d, m) in spark:
        E.synth.append(dict(t=H.time(b, 0.7), dur=b2s(d), midi=m, vel=H.vel(0.5),
                            vowel="ah", glide=True, bend=0.0, vib=0.15))
    # outro synth pad note
    E.synth.append(dict(t=H.time(66 * 4, 0.8), dur=b2s(8.0), midi=78,
                        vel=0.4, vowel="ah", glide=False, bend=0.0, vib=0.2))


# ================================================================ FX
def build_fx(E: Events, H: Humanizer):
    for t, kind, params in [
        (0.0, "vinyl_start", {}),
        (0.0, "rev_crash", {"dur": 3.0, "vel": 0.8}),
        (0.0, "sub_drop", {"midi": 30, "dur": 2.0}),
        (3.0 * 4 - 1.0, "riser", {"dur": 2.2}),
        (3.75 * 4, "rev_crash", {"dur": 1.6, "vel": 0.5}),
        (19.75 * 4, "rev_crash", {"dur": 1.8, "vel": 0.6}),
        (27.5 * 4, "riser", {"dur": 2.0}),
        (43.5 * 4, "riser", {"dur": 2.4}),
        (43.5 * 4, "rev_crash", {"dur": 2.0, "vel": 0.5}),
        (51.0 * 4, "riser", {"dur": 4.0}),
        (51.0 * 4, "rev_crash", {"dur": 2.2, "vel": 0.65}),
        (59.75 * 4, "rev_crash", {"dur": 1.5, "vel": 0.4}),
        (64.0 * 4, "sub_drop", {"midi": 30, "dur": 2.2}),
        (64.0 * 4, "rev_crash", {"dur": 2.0, "vel": 0.55}),
    ]:
        E.fx.append((b2s(t), kind, params))


# ================================================================ build all
def compose():
    H = Humanizer(seed=2026, timing_ms=4.0, vel_pct=0.045)
    E = Events()
    build_drums(E, H)
    build_bass(E, H)
    build_rhodes(E, H)
    build_guitar(E, H)
    build_strings(E, H)
    build_leads(E, H)
    build_fx(E, H)
    return E, H


if __name__ == "__main__":
    E, H = compose()
    print("bars", TOTAL_BARS, "tempo", BPM, "length_s", round(SONG_S + TAIL_S, 2))
    for name in ("kick", "snare", "clap", "rim", "hat", "perc", "tom", "crash",
                 "bass", "rhodes", "guitar", "strings", "lead", "synth", "fx"):
        print(f"{name:8s}", len(getattr(E, name)))
