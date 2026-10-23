"""
render.py — renders every performance to raw stereo buses at 48 kHz / float32.
Raw buses are cached outside the delivered project (session/.cache) and are
consumed by mix.py.
"""
from __future__ import annotations
import os
import time
import numpy as np
import soundfile as sf

from . import dsp
from . import instruments as I
from . import song as S
from .dsp import SR, pan_gains

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache", "raw")


class StereoBus:
    def __init__(self, n: int):
        self.n = n
        self.x = np.zeros((n, 2), dtype=np.float32)

    def add(self, y: np.ndarray, t: float, pan: float = 0.0, gain: float = 1.0):
        i = int(round(t * SR))
        if i >= self.n:
            return
        if i < 0:
            y = y[-i:]
            i = 0
        if y.ndim == 1:
            m = min(len(y), self.n - i)
            if m <= 0:
                return
            seg = y[:m]
            l, r = pan_gains(pan)
            self.x[i:i + m, 0] += seg * (l * gain)
            self.x[i:i + m, 1] += seg * (r * gain)
        else:
            m = min(len(y), self.n - i)
            if m <= 0:
                return
            self.x[i:i + m, 0] += y[:m, 0] * gain
            self.x[i:i + m, 1] += y[:m, 1] * gain


def _save(name: str, x: np.ndarray):
    os.makedirs(RAW_DIR, exist_ok=True)
    path = os.path.join(RAW_DIR, f"{name}.wav")
    sf.write(path, x, SR, subtype="FLOAT", format="WAV")
    return path


def render_drums(E: S.Events, n: int):
    out = {}
    # ---- kick
    b = StereoBus(n)
    for i, (t, v) in enumerate(E.kick):
        b.add(I.kick(vel=v, seed=100 + i), t)
    out["kick"] = b.x
    # ---- snare (slight timbral variation per hit, like a real backbeat)
    b = StereoBus(n)
    for i, (t, v) in enumerate(E.snare):
        tone = 1.0 + 0.05 * np.sin(i * 1.7)
        bright = 1.0 + 0.10 * np.sin(i * 0.9)
        y = I.snare(vel=v, seed=300 + i, tone=tone, bright=bright)
        b.add(y, t, pan=0.02 * np.sin(i * 2.1))
    out["snare"] = b.x
    # ---- clap (stereo double for width)
    b = StereoBus(n)
    for i, (t, v) in enumerate(E.clap):
        yl = I.clap(vel=v, seed=500 + 2 * i)
        yr = I.clap(vel=v, seed=501 + 2 * i)
        b.x[:, 0] += 0
        b.add(yl, t, pan=-0.35)
        b.add(yr, t, pan=0.35)
    out["clap"] = b.x
    # ---- rim
    b = StereoBus(n)
    for i, (t, v) in enumerate(E.rim):
        b.add(I.rim(vel=v, seed=700 + i), t, pan=0.25)
    out["rim"] = b.x
    # ---- hats
    b = StereoBus(n)
    for i, (t, v, open_, pan) in enumerate(E.hat):
        length = None
        y = I.hat(vel=v, open_=bool(open_), seed=900 + i)
        b.add(y, t, pan=pan + 0.02 * np.sin(i * 0.7))
    out["hats"] = b.x
    # ---- percussion (shaker/tamb)
    b = StereoBus(n)
    for i, (t, v, kind, pan) in enumerate(E.perc):
        if kind == "shaker":
            y = I.shaker(vel=v, seed=1100 + i)
        else:
            y = I.tamb(vel=v, seed=1200 + i, length=0.28 if v > 0.5 else 0.2)
        b.add(y, t, pan=pan)
    out["perc"] = b.x
    # ---- toms
    b = StereoBus(n)
    for i, (t, v, midi, pan) in enumerate(E.tom):
        b.add(I.tom(midi=midi, vel=v, seed=1300 + i), t, pan=pan)
    out["toms"] = b.x
    # ---- crashes (decorrelated L/R for a wide image)
    b = StereoBus(n)
    for i, (t, v, reverse) in enumerate(E.crash):
        length = 3.0 if not reverse else 2.0
        yl = I.crash(reverse=reverse, length=length, seed=1400 + 2 * i, vel=v)
        yr = I.crash(reverse=reverse, length=length, seed=1401 + 2 * i, vel=v)
        b.add(yl, t, pan=-0.45)
        b.add(yr, t, pan=0.45)
    out["crash"] = b.x
    return out


def render_bass(E: S.Events, n: int):
    b = StereoBus(n)
    for i, e in enumerate(sorted(E.bass, key=lambda z: z["t"])):
        y = I.bass_note(e["midi"], e["dur"] + 0.06, vel=e["vel"], art=e["art"],
                        glide_from=e.get("glide_from"), glide_s=e.get("glide_s", 0.06),
                        seed=1600 + i, vibrato=e.get("vibrato", 0.0))
        b.add(y, e["t"])
    return b.x


def render_rhodes(E: S.Events, n: int):
    b = StereoBus(n)
    k = 0
    for (t, dur, notes, vel, pan) in E.rhodes:
        for m in notes:
            # gentle note-to-note placement, low notes left, high notes right
            p = pan + 0.045 * np.clip((m - 60) / 12.0, -1.5, 1.5)
            y = I.rhodes_note(m, dur * (0.97 + 0.02 * ((m * 7) % 5)), vel=vel,
                              seed=1800 + k)
            b.add(y, t + 0.002 * ((m * 3) % 5), pan=p, gain=0.9)
            k += 1
    return b.x


def render_guitar(E: S.Events, n: int):
    b = StereoBus(n)
    k = 0
    for (t, dur, notes, vel, muted, pan) in E.guitar:
        y = np.zeros(int((dur + 0.3) * SR), dtype=np.float32)
        for j, m in enumerate(notes):
            note = I.guitar_note(m, dur, vel=vel * (1.0 - 0.08 * j), muted=bool(muted),
                                 seed=2000 + k * 3 + j)
            L = min(len(y), len(note))
            y[:L] += note[:L]
            k += 1
        p = pan + 0.03 * ((t * 10) % 3 - 1)
        b.add(y, t, pan=p)
    return b.x


def render_strings(E: S.Events, n: int):
    b = StereoBus(n)
    for i, (t, dur, notes, vel, octave_up) in enumerate(E.strings):
        y = I.string_chord(notes, dur, vel=vel, seed=2200 + i, octave_up=bool(octave_up),
                           bright=1.0 if not octave_up else 1.08)
        b.add(y, t)
    return b.x


def _lead_events(lst):
    evs = []
    for e in sorted(lst, key=lambda z: z["t"]):
        evs.append(dict(start=e["t"], dur=e["dur"], midi=e["midi"], vel=e["vel"],
                        vowel=e.get("vowel", "oh"), glide=e.get("glide", False),
                        bend=e.get("bend", 0.0), vib=e.get("vib", 0.14),
                        glide_s=e.get("glide_s", 0.10)))
    return evs


def render_lead(E: S.Events, n: int):
    y = I.lead_phrase(_lead_events(E.lead), kind="talkbox", seed=71)
    out = np.zeros((n, 2), dtype=np.float32)
    m = min(n, len(y))
    out[:m] = y[:m]
    return out


def render_synth(E: S.Events, n: int):
    y = I.lead_phrase(_lead_events(E.synth), kind="synth", seed=73)
    out = np.zeros((n, 2), dtype=np.float32)
    m = min(n, len(y))
    out[:m] = y[:m]
    return out


def render_fx(E: S.Events, n: int):
    b = StereoBus(n)
    # full-length, very low vinyl/room texture
    length_s = n / SR
    for ch, seed in ((0, 3100), (1, 3101)):
        v = I.vinyl_bed(length_s, seed=seed, level=0.010)
        b.x[:len(v), ch] += v
    for i, (t, kind, params) in enumerate(E.fx):
        if kind == "rev_crash":
            yl = I.crash(reverse=True, length=params.get("dur", 2.0), seed=3200 + i,
                         vel=params.get("vel", 0.6))
            yr = I.crash(reverse=True, length=params.get("dur", 2.0), seed=3201 + i,
                         vel=params.get("vel", 0.6))
            b.add(yl, t, pan=-0.5)
            b.add(yr, t, pan=0.5)
        elif kind == "sub_drop":
            y = I.sub_drop(midi=params.get("midi", 30), dur_s=params.get("dur", 2.0))
            b.add(y, t)
        elif kind == "riser":
            y = I.riser(dur_s=params.get("dur", 2.0), seed=3400 + i)
            # risers widen as they climb
            b.add(y, t, pan=-0.15)
            b.add(y, t + 0.006, pan=0.15)
        elif kind == "vinyl_start":
            pass
    return b.x


BUSES = ["kick", "snare", "clap", "rim", "hats", "perc", "toms", "crash",
         "bass", "rhodes", "guitar", "strings", "lead", "synth", "fx"]


def render_all(n_bars=None, verbose=True):
    E, H = S.compose()
    total_s = (n_bars * S.BAR_S if n_bars else S.SONG_S) + S.TAIL_S
    n = int(total_s * SR)
    t_start = time.time()
    drums = render_drums(E, n)
    for name, x in drums.items():
        _save(name, x)
        if verbose:
            print(f"  rendered {name:7s} {x.shape[0]/SR:7.2f}s  peak {np.max(np.abs(x)):.3f}")
    for name, fn in [("bass", render_bass), ("rhodes", render_rhodes),
                     ("guitar", render_guitar), ("strings", render_strings),
                     ("lead", render_lead), ("synth", render_synth), ("fx", render_fx)]:
        x = fn(E, n)
        _save(name, x)
        if verbose:
            print(f"  rendered {name:7s} {x.shape[0]/SR:7.2f}s  peak {np.max(np.abs(x)):.3f}")
    if verbose:
        print(f"render complete in {time.time()-t_start:.1f}s -> {RAW_DIR}")
    return n


if __name__ == "__main__":
    import sys
    nb = int(sys.argv[1]) if len(sys.argv) > 1 else None
    render_all(nb)
