"""
engine.py — the modern session engine.

One code path renders any number of songs from a SongSpec:
  compose -> render parts (48k float) -> bus processing -> vocalise leveling
  -> mix + sends -> arrangement automation -> master -> deliverables
Each song ships as a VOCAL version and an INSTRUMENTAL version.
"""
from __future__ import annotations
import os, json, math, time, gc
import numpy as np
import soundfile as sf
from scipy import signal
from dataclasses import dataclass, field

from . import dsp
from . import inst
from . import inst2 as i2
from .dsp import SR, db

ROOT = "/home/user/modern_sessions"
SONGS = os.path.join(ROOT, "songs")
CACHE = os.path.join(ROOT, ".cache")


# ================================================================ SPEC
@dataclass
class SongSpec:
    slug: str
    title: str
    genre: str
    bpm: float
    key_name: str
    bars: int
    tail_s: float = 7.0
    swing: float = 0.56
    sections: list = field(default_factory=list)     # (name, start_bar, bars)
    chords: dict = field(default_factory=dict)       # name -> dict(bass, pcs, center)
    prog: dict = field(default_factory=dict)         # section -> [chord names]
    dynamics: dict = field(default_factory=dict)     # section -> gain
    bus_levels: dict = field(default_factory=dict)   # bus -> target rms dBFS
    master_eq: list = field(default_factory=list)
    vocal_pct: float = 6.0

    @property
    def beat_s(self):
        return 60.0 / self.bpm

    @property
    def bar_s(self):
        return 4 * self.beat_s

    @staticmethod
    def _base(name):
        return name.rstrip("0123456789")

    def chord_at(self, bar):
        for name, start, length in self.sections:
            if start <= bar < start + length:
                prog = (self.prog.get(name) or self.prog.get(self._base(name))
                        or self.prog.get("verse") or ["I"])
                return prog[(bar - start) % len(prog)]
        return (self.prog.get("verse") or ["I"])[0]

    def section_of(self, bar):
        for name, start, length in self.sections:
            if start <= bar < start + length:
                return name
        return "verse"


# ================================================================ PERFORMANCE
@dataclass
class Humanizer:
    seed: int = 7
    timing_ms: float = 4.0
    vel_pct: float = 0.05
    swing: float = 0.56

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def time(self, beat, amount=1.0, spec=None):
        s16 = round(beat * 4)
        b = beat
        if abs(beat * 4 - s16) < 1e-6 and s16 % 2 == 1:
            b = beat + (self.swing - 0.5) * 0.5
        j = self.rng.normal(0, self.timing_ms / 1000.0) * amount
        return max(0.0, b * (60.0 / (spec.bpm if spec else 90.0)) + j)

    def vel(self, v):
        return float(np.clip(v * (1 + self.rng.normal(0, self.vel_pct)), 0.03, 1.0))


# ================================================================ HARMONY
def voicing(chord, n=4, center=60, low=50, high=76):
    """Close voicing from a chord's pitch classes, near `center`, in range."""
    pcs = chord["pcs"]
    pool = [m for m in range(low, high + 1) if m % 12 in pcs]
    if not pool:
        return [center]
    pool.sort(key=lambda m: abs(m - center))
    picked = sorted(pool[:n])
    while len(picked) < n:
        picked.append(picked[-1] + 12)
    return picked


def chord_dict(bass, pcs, center=60):
    return dict(bass=bass, pcs=set(pcs), center=center)


# ================================================================ RENDER PARTS
def _events_by_voice(evs):
    groups = {}
    for e in evs:
        groups.setdefault(e.get("voice", 0), []).append(e)
    return [groups[k] for k in sorted(groups)]


def render_parts(spec, parts, to_disk=True, verbose=True):
    """parts: {bus_name: (kind, [events])}. Renders each bus separately and
    (by default) writes it to the song cache so the mixer can stream it."""
    n = int((spec.bars * spec.bar_s + spec.tail_s) * SR)

    def bus():
        return np.zeros((n, 2), dtype=np.float32)

    def add(dst, y, t, pan=0.0, gain=1.0):
        i = int(round(t * SR))
        if i >= n:
            return
        if i < 0:
            y = y[-i:]; i = 0
        m = min(len(y), n - i)
        if m <= 0:
            return
        seg = y[:m]
        if seg.ndim == 1:
            l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
            dst[i:i + m, 0] += seg * l * gain
            dst[i:i + m, 1] += seg * r * gain
        else:
            dst[i:i + m, 0] += seg[:, 0] * gain
            dst[i:i + m, 1] += seg[:, 1] * gain

    out = {}
    rawdir = os.path.join(CACHE, "raw", spec.slug)
    if to_disk:
        os.makedirs(rawdir, exist_ok=True)
    for name, (kind, evs) in parts.items():
        b = bus()
        if kind == "kick":
            for i, (t, v) in enumerate(evs):
                add(b, inst.kick(vel=v, seed=1000 + i, tune_midi=evs[i][2] if len(evs[i]) > 2 else 30.5), t)
        elif kind == "kick808":
            for i, (t, v) in enumerate(evs):
                add(b, inst.kick(vel=v * 0.9, seed=1100 + i, tune_midi=28.5), t, gain=0.8)
        elif kind == "snare":
            for i, (t, v) in enumerate(evs):
                add(b, inst.snare(vel=v, seed=1200 + i, tone=1 + 0.05 * np.sin(i), bright=1 + 0.08 * np.sin(i * 1.3)),
                    t, pan=0.02 * np.sin(i * 2.3))
        elif kind == "clap":
            for i, (t, v) in enumerate(evs):
                add(b, inst.clap(vel=v, seed=1300 + 2 * i), t, pan=-0.32)
                add(b, inst.clap(vel=v, seed=1301 + 2 * i), t, pan=0.32)
        elif kind == "rim":
            for i, (t, v) in enumerate(evs):
                add(b, inst.rim(vel=v, seed=1400 + i), t, pan=0.22)
        elif kind == "snap":
            for i, (t, v) in enumerate(evs):
                add(b, i2.snap(vel=v, seed=1500 + i), t, pan=-0.28)
                add(b, i2.snap(vel=v, seed=1550 + i), t, pan=0.30)
        elif kind == "hats":
            for i, (t, v, open_, pan) in enumerate(evs):
                add(b, inst.hat(vel=v, open_=bool(open_), seed=1600 + i,
                                length=0.5 if open_ else 0.06), t, pan=pan)
        elif kind == "perc":
            for i, (t, v, sub, pan) in enumerate(evs):
                fn = inst.shaker if sub == "shaker" else inst.tamb
                add(b, fn(vel=v, seed=1700 + i), t, pan=pan)
        elif kind == "toms":
            for i, (t, v, midi, pan) in enumerate(evs):
                add(b, inst.tom(midi=midi, vel=v, seed=1800 + i), t, pan=pan)
        elif kind == "crash":
            for i, (t, v, rev) in enumerate(evs):
                add(b, inst.crash(reverse=bool(rev), length=2.6, seed=1900 + 2 * i, vel=v), t, pan=-0.45)
                add(b, inst.crash(reverse=bool(rev), length=2.6, seed=1901 + 2 * i, vel=v), t, pan=0.45)
        elif kind == "finger":
            for e in evs:
                y = inst.bass_note(e["midi"], e["dur"], vel=e["vel"], art=e.get("art", "finger"),
                                   glide_from=e.get("glide_from"), glide_s=e.get("glide_s", 0.06),
                                   seed=e.get("seed", 2000))
                add(b, y, e["t"])
        elif kind == "sub808":
            for i, e in enumerate(evs):
                y = i2.sub808_note(e["midi"], e["dur"], vel=e["vel"],
                                   glide_from=e.get("glide_from"),
                                   glide_s=e.get("glide_s", 0.09), seed=2100 + i,
                                   drive=e.get("drive", 1.6))
                add(b, y, e["t"])
        elif kind == "rhodes":
            for i, e in enumerate(evs):
                for j, m in enumerate(e["notes"]):
                    y = inst.rhodes_note(m, e["dur"], vel=e["vel"], seed=2200 + i * 5 + j)
                    add(b, y, e["t"] + 0.002 * j, pan=e.get("pan", 0) + 0.04 * (m - 60) / 12)
        elif kind == "wurli":
            for i, e in enumerate(evs):
                for j, m in enumerate(e["notes"]):
                    y = i2.wurli_note(m, e["dur"], vel=e["vel"], seed=2300 + i * 5 + j)
                    add(b, y, e["t"] + 0.002 * j, pan=e.get("pan", 0) + 0.03 * (m - 60) / 12)
        elif kind == "clav":
            for i, e in enumerate(evs):
                for j, m in enumerate(e["notes"]):
                    y = i2.clav_note(m, e["dur"], vel=e["vel"], seed=2400 + i * 5 + j,
                                     wah=e.get("wah", 1500))
                    add(b, y, e["t"] + 0.001 * j, pan=e.get("pan", 0))
        elif kind == "organ":
            for i, e in enumerate(evs):
                for j, m in enumerate(e["notes"]):
                    y = i2.organ_note(m, e["dur"], vel=e["vel"], seed=2500 + i * 5 + j)
                    add(b, y, e["t"] + 0.003 * j, pan=e.get("pan", 0))
        elif kind == "vibes":
            for i, e in enumerate(evs):
                for j, m in enumerate(e["notes"]):
                    y = i2.vibes_note(m, e["dur"], vel=e["vel"], seed=2600 + i * 5 + j)
                    add(b, y, e["t"] + 0.004 * j, pan=e.get("pan", 0))
        elif kind == "guitar":
            for i, e in enumerate(evs):
                y = np.zeros(int((e["dur"] + 0.3) * SR), dtype=np.float32)
                for j, m in enumerate(e["notes"]):
                    note = inst.guitar_note(m, e["dur"], vel=e["vel"] * (1 - 0.05 * j),
                                            muted=bool(e.get("muted", True)), seed=2700 + i * 3 + j)
                    L = min(len(y), len(note)); y[:L] += note[:L]
                add(b, y, e["t"], pan=e.get("pan", 0))
        elif kind == "strings":
            for i, e in enumerate(evs):
                y = inst.string_chord(e["notes"], e["dur"], vel=e["vel"], seed=2800 + i,
                                      octave_up=bool(e.get("octave", False)))
                add(b, y, e["t"])
        elif kind == "pad":
            for i, e in enumerate(evs):
                y = i2.pad_chord(e["notes"], e["dur"], vel=e["vel"], seed=2900 + i,
                                 cutoff=e.get("cutoff", 2600), attack=e.get("attack", 0.9),
                                 release=e.get("release", 1.2))
                add(b, y, e["t"])
        elif kind == "brass":
            for i, e in enumerate(evs):
                y = i2.brass_chord(e["notes"], e["dur"], vel=e["vel"], seed=3000 + i,
                                   bright=e.get("bright", 1.0))
                add(b, y, e["t"])
        elif kind == "bell":
            for i, e in enumerate(evs):
                for j, m in enumerate(e["notes"] if isinstance(e["notes"], list) else [e["midi"]]):
                    y = i2.bell_note(m, e["dur"], vel=e["vel"], seed=3100 + i * 4 + j)
                    add(b, y, e["t"] + 0.003 * j, pan=e.get("pan", 0))
        elif kind == "flute":
            for i, e in enumerate(evs):
                add(b, i2.flute_note(e["midi"], e["dur"], vel=e["vel"], seed=3200 + i),
                    e["t"], pan=e.get("pan", 0.1))
        elif kind == "talkbox":
            evs2 = [dict(start=e["t"], dur=e["dur"], midi=e["midi"], vel=e["vel"],
                         vowel=e.get("vowel", "oh"), glide=e.get("glide", False),
                         bend=e.get("bend", 0.0), vib=e.get("vib", 0.14),
                         glide_s=e.get("glide_s", 0.1)) for e in sorted(evs, key=lambda z: z["t"])]
            y = inst.lead_phrase(evs2, kind="talkbox", seed=3300)
            add(b, y, 0.0)
        elif kind == "synth":
            evs2 = [dict(start=e["t"], dur=e["dur"], midi=e["midi"], vel=e["vel"],
                         vowel=e.get("vowel", "ah"), glide=e.get("glide", True),
                         bend=e.get("bend", 0.0), vib=e.get("vib", 0.15),
                         glide_s=e.get("glide_s", 0.1)) for e in sorted(evs, key=lambda z: z["t"])]
            y = inst.lead_phrase(evs2, kind="synth", seed=3400)
            add(b, y, 0.0)
        elif kind == "vox":
            groups = _events_by_voice(evs)
            y = i2.vox_choir(groups, seed=3500)
            add(b, y, 0.0)
        elif kind == "fx":
            for i, e in enumerate(evs):
                t = e["t"]
                if e["type"] == "riser":
                    add(b, inst.riser(dur_s=e.get("dur", 2.0), seed=3600 + i), t, pan=-0.1)
                    add(b, inst.riser(dur_s=e.get("dur", 2.0), seed=3601 + i), t + 0.006, pan=0.1)
                elif e["type"] == "rev_crash":
                    add(b, inst.crash(reverse=True, length=e.get("dur", 1.8), seed=3700 + 2 * i,
                                      vel=e.get("vel", 0.5)), t, pan=-0.45)
                    add(b, inst.crash(reverse=True, length=e.get("dur", 1.8), seed=3701 + 2 * i,
                                      vel=e.get("vel", 0.5)), t, pan=0.45)
                elif e["type"] == "sub":
                    add(b, inst.sub_drop(midi=e.get("midi", 30), dur_s=e.get("dur", 1.6)), t)
                elif e["type"] == "vinyl":
                    v = inst.vinyl_bed(e.get("dur", n / SR), seed=3800 + i, level=e.get("level", 0.010))
                    add(b, v, t)
        if to_disk:
            sf.write(os.path.join(rawdir, f"{name}.wav"), b, SR, subtype="FLOAT", format="WAV")
            if verbose:
                print(f"    {name:8s} peak {np.max(np.abs(b)):.2f}  "
                      f"{b.nbytes/1e6:.0f} MB")
            del b
        else:
            out[name] = b
    return (rawdir if to_disk else out), n


# ================================================================ MIX
DEFAULT_BUS = dict(hp=40, eq=[], sat=(1.0, 0.0), comp=(-22, 2.5, 12, 140),
                   rev=0.0, delay=0.0, width=1.0, target=-24.0, duck=0.0)

BUS_DEFAULTS = {
    "kick":    dict(hp=26, target=-15.5, eq=[("peak", 55, 1.0, 0.9), ("peak", 240, -3.0, 1.0), ("peak", 3800, 3.0, 1.1)],
                    sat=(1.35, 0.3), comp=(-18, 3.2, 12, 110)),
    "snare":   dict(hp=120, target=-19.5, eq=[("peak", 200, 1.0, 1.0), ("peak", 420, -1.5, 1.2), ("peak", 4200, 2.2, 0.9)],
                    sat=(1.22, 0.28), comp=(-20, 3.6, 6, 120), rev=0.18),
    "clap":    dict(hp=420, target=-24.0, eq=[("peak", 1100, 1.6, 1.1), ("peak", 3200, 1.0, 1.0)],
                    sat=(1.15, 0.25), comp=(-22, 3.0, 4, 110), rev=0.22, width=1.15),
    "rim":     dict(hp=600, target=-27.0, eq=[("peak", 2600, 2.0, 1.0)], sat=(1.1, 0.2),
                    comp=(-22, 2.5, 4, 90), rev=0.20),
    "snap":    dict(hp=700, target=-24.5, eq=[("peak", 2800, 2.0, 1.0)], sat=(1.1, 0.2),
                    comp=(-22, 2.6, 4, 90), rev=0.22, width=1.2),
    "hats":    dict(hp=700, target=-27.5, eq=[("peak", 2200, -1.2, 1.0), ("high_shelf", 10500, 1.0, 0.7)],
                    sat=(1.05, 0.12), comp=(-24, 2.6, 3, 70), rev=0.05),
    "perc":    dict(hp=900, target=-29.0, eq=[("peak", 6000, 1.2, 0.9)], sat=(1.05, 0.12),
                    comp=(-24, 2.4, 5, 90), rev=0.12, width=1.1),
    "toms":    dict(hp=45, target=-24.0, eq=[("peak", 120, 1.0, 0.9), ("peak", 2400, 0.8, 1.0)],
                    sat=(1.3, 0.28), comp=(-20, 2.8, 8, 140), rev=0.18),
    "crash":   dict(hp=380, target=-33.0, eq=[("high_shelf", 8000, 1.0, 0.7)], sat=(1.05, 0.1),
                    comp=(-26, 2.0, 10, 200), rev=0.28, width=1.2),
    "bass":    dict(hp=30, target=-16.5, eq=[("peak", 88, 1.4, 1.0), ("peak", 300, -2.0, 1.1), ("peak", 800, 1.2, 1.0)],
                    sat=(1.35, 0.35), comp=(-20, 3.0, 18, 150), duck=2.5),
    "sub808":  dict(hp=24, target=-15.5, eq=[("peak", 60, 1.0, 1.0), ("peak", 260, -2.5, 1.0)],
                    sat=(1.5, 0.4), comp=(-18, 3.5, 10, 160), duck=3.5),
    "rhodes":  dict(hp=115, target=-23.5, eq=[("peak", 330, -2.4, 1.0), ("peak", 1900, 1.4, 0.8)],
                    sat=(1.12, 0.22), comp=(-22, 2.0, 20, 180), rev=0.2, width=1.12, duck=1.2),
    "wurli":   dict(hp=110, target=-24.0, eq=[("peak", 400, -2.0, 1.0), ("peak", 1700, 1.6, 0.9)],
                    sat=(1.18, 0.25), comp=(-22, 2.2, 18, 170), rev=0.2, width=1.1, duck=1.0),
    "clav":    dict(hp=190, target=-26.0, eq=[("peak", 900, -1.5, 1.0), ("peak", 2400, 2.2, 0.9)],
                    sat=(1.2, 0.3), comp=(-20, 3.5, 5, 90), rev=0.1, width=1.05),
    "organ":   dict(hp=140, target=-27.5, eq=[("peak", 700, 1.0, 0.9)], sat=(1.15, 0.2),
                    comp=(-22, 2.0, 15, 160), rev=0.18, width=1.15),
    "vibes":   dict(hp=200, target=-25.0, eq=[("peak", 1200, -1.0, 1.0), ("high_shelf", 7000, 0.8, 0.7)],
                    sat=(1.05, 0.1), comp=(-22, 2.2, 15, 180), rev=0.28, delay=0.12, width=1.15),
    "guitar":  dict(hp=170, target=-26.0, eq=[("peak", 430, -1.6, 1.0), ("peak", 1900, 2.8, 0.9)],
                    sat=(1.25, 0.3), comp=(-20, 4.0, 4, 90), rev=0.1, width=1.1),
    "strings": dict(hp=95, target=-27.5, eq=[("peak", 400, -2.2, 1.0), ("peak", 2400, 1.2, 0.8)],
                    sat=(1.08, 0.15), comp=(-20, 1.8, 45, 260), rev=0.32, width=1.18),
    "pad":     dict(hp=100, target=-28.0, eq=[("peak", 450, -1.5, 1.0), ("peak", 2500, 1.0, 0.8)],
                    sat=(1.06, 0.12), comp=(-22, 1.7, 60, 300), rev=0.34, width=1.25),
    "brass":   dict(hp=150, target=-23.5, eq=[("peak", 800, 1.5, 1.0), ("peak", 2600, 1.5, 0.9)],
                    sat=(1.25, 0.3), comp=(-20, 3.0, 6, 110), rev=0.18, width=1.15),
    "bell":    dict(hp=400, target=-27.0, eq=[("high_shelf", 6000, 1.0, 0.7)], sat=(1.05, 0.1),
                    comp=(-24, 2.2, 6, 120), rev=0.3, delay=0.18, width=1.2),
    "flute":   dict(hp=350, target=-27.0, eq=[("peak", 1400, 1.2, 0.9)], sat=(1.05, 0.1),
                    comp=(-22, 2.0, 15, 160), rev=0.26, delay=0.1, width=1.1),
    "talkbox": dict(hp=200, target=-21.0, eq=[("peak", 260, -1.5, 1.0), ("peak", 2300, 3.0, 0.9), ("high_shelf", 6500, 1.0, 0.7)],
                    sat=(1.15, 0.25), comp=(-18, 3.0, 12, 110), rev=0.18, delay=0.16, width=1.12),
    "synth":   dict(hp=220, target=-26.5, eq=[("peak", 2800, 1.2, 0.9)], sat=(1.1, 0.2),
                    comp=(-22, 2.5, 15, 140), rev=0.24, delay=0.18, width=1.2),
    "fx":      dict(hp=35, target=-31.0, eq=[], sat=(1.0, 0.0), comp=(-24, 1.8, 20, 200),
                    rev=0.2, width=1.1),
    # the vocalise bus is special-cased: its level is set by percentage
    "vox":     dict(hp=140, target=None, eq=[("peak", 300, -1.0, 1.0), ("peak", 2600, 1.8, 0.9),
                                            ("high_shelf", 9000, 0.6, 0.7)],
                    sat=(1.12, 0.18), comp=(-20, 2.4, 12, 140), rev=0.26, delay=0.12,
                    width=1.1, special="vox"),
}


def bus_cfg(name, spec):
    cfg = dict(DEFAULT_BUS)
    cfg.update(BUS_DEFAULTS.get(name, {}))
    if spec.bus_levels.get(name):
        cfg["target"] = spec.bus_levels[name]
    return cfg


def _apply_eq(x, bands):
    for kind, f, g, q in bands:
        if kind == "peak":
            x = dsp.peak(x, f, g, q)
        elif kind == "low_shelf":
            x = dsp.low_shelf(x, f, g, q)
        elif kind == "high_shelf":
            x = dsp.high_shelf(x, f, g, q)
    return x


def process_bus(name, x, cfg, key_env):
    y = x
    if cfg["hp"] > 0:
        y = np.stack([dsp.hp(y[:, 0], cfg["hp"]), dsp.hp(y[:, 1], cfg["hp"])], axis=1)
    if cfg["eq"]:
        y = _apply_eq(y, cfg["eq"])
    drive, mix = cfg["sat"]
    if drive > 1.0:
        y = np.stack([dsp.saturate(y[:, 0], drive, 0.02, mix),
                      dsp.saturate(y[:, 1], drive, 0.02, mix)], axis=1)
    thr, ratio, atk, rel = cfg["comp"]
    y = dsp.compressor(y, thr_db=thr, ratio=ratio, attack_ms=atk, release_ms=rel,
                       knee_db=6.0)
    if name == "rhodes":
        y = dsp.tremolo_stereo(y, rate=5.1, depth=0.13)
        y = dsp.chorus(y, rate=0.26, depth_ms=6.0, mix=0.15)
    elif name == "wurli":
        y = dsp.tremolo_stereo(y, rate=5.8, depth=0.2)
        y = dsp.chorus(y, rate=0.21, depth_ms=4.0, mix=0.12)
    elif name == "organ":
        y = dsp.chorus(y, rate=6.2, depth_ms=2.2, mix=0.3)
    elif name == "clav":
        y = dsp.chorus(y, rate=0.17, depth_ms=2.5, mix=0.06)
    elif name in ("strings", "pad"):
        y = dsp.chorus(y, rate=0.12, depth_ms=4.5, mix=0.12)
    elif name == "vox":
        y = dsp.chorus(y, rate=0.19, depth_ms=3.2, mix=0.10)
    if cfg["width"] != 1.0:
        y = dsp.ms_width(y, cfg["width"])
    if cfg["duck"] > 0 and key_env is not None:
        g = (1.0 - (1.0 - 10 ** (-cfg["duck"] / 20.0)) * key_env)[:, None]
        y = y * g
    return y.astype(np.float32)


def mix_song(spec, raw, verbose=True):
    """raw: path to the per-song raw directory (streamed, memory-safe)."""
    if isinstance(raw, str):
        names = sorted(f[:-4] for f in os.listdir(raw) if f.endswith(".wav"))
        n = sf.info(os.path.join(raw, names[0] + ".wav")).frames
        load = lambda nm: sf.read(os.path.join(raw, nm + ".wav"), dtype="float32")[0]
    else:
        names = list(raw)
        n = raw[names[0]].shape[0]
        load = lambda nm: raw[nm]

    def kenvelope():
        key = "kick" if "kick" in names else ("kick808" if "kick808" in names else None)
        if key is None:
            return None
        k = load(key)
        env = np.max(np.abs(k), axis=1)
        env = dsp._smooth_env(env, 0.004); env = dsp._smooth_env(env, 0.11)
        e = (env / (np.max(env) + 1e-9)).astype(np.float32)
        del k, env
        return e

    key_env = kenvelope()
    mix = np.zeros((n, 2), dtype=np.float32)
    rev_send = np.zeros((n, 2), dtype=np.float32)
    dly_send = np.zeros((n, 2), dtype=np.float32)
    vox_bus = None
    for name in names:
        cfg = bus_cfg(name, spec)
        x = load(name)
        p = process_bus(name, x, cfg, key_env)
        del x
        act = dsp.active_rms_db(p)
        if act < -100:
            del p
            continue
        if name == "vox":
            vox_bus = (p, act)
            continue
        tgt = cfg["target"]
        if tgt is not None:
            p *= db(tgt - act)
        mix += p
        if cfg["rev"] > 0:
            rev_send += p * cfg["rev"]
        if cfg["delay"] > 0:
            dly_send += p * cfg["delay"]
        if verbose:
            print(f"  {name:8s} in {act:6.1f} dBFS -> fader {tgt - act:+6.1f} dB")
        del p

    vox_gain_db = None
    if vox_bus is not None:
        vox, vox_act = vox_bus
        m = np.max(np.abs(vox), axis=1) > db(-55)
        if np.any(m):
            vox_rms = 20 * np.log10(np.sqrt(np.mean(vox[m].astype(np.float64) ** 2)) + 1e-12)
            bed_rms = 20 * np.log10(np.sqrt(np.mean(mix[m].astype(np.float64) ** 2)) + 1e-12)
            ratio_db = 20 * np.log10(spec.vocal_pct / (100.0 - spec.vocal_pct))
            vox_gain_db = ratio_db - (vox_rms - bed_rms)
            vox = vox * db(vox_gain_db)
        mix_with_vox = mix + vox
    else:
        vox = None
        mix_with_vox = mix

    ir = dsp.make_ir(rt60=2.0 if "trap" in spec.genre else 2.3,
                     predelay_ms=20, bright=0.6, seed=13)
    rev = dsp.convolve_stereo(rev_send, ir)[:n]
    del rev_send
    rev = np.stack([dsp.hp(rev[:, 0], 240), dsp.hp(rev[:, 1], 240)], axis=1)
    rev = np.stack([dsp.lp(rev[:, 0], 7800), dsp.lp(rev[:, 1], 7800)], axis=1)
    rev *= db(-15.5)
    dly = dsp.pingpong_delay(dly_send, delay_ms=3 * 60000 / spec.bpm / 4, fb=0.3, mix=1.0)
    del dly_send
    dly *= db(-19.0)

    curve = section_curve(spec, n)[:, None]
    inst = (mix + rev + dly) * curve
    share_pct = None
    if vox is not None:
        full = (mix_with_vox + rev + dly) * curve
        diff = full - inst                      # vocal bus + its reverb / delay
        act = np.max(np.abs(diff), axis=1) > db(-60)
        if np.any(act):
            a1 = np.sqrt(np.mean(diff[act].astype(np.float64) ** 2))
            a2 = np.sqrt(np.mean(inst[act].astype(np.float64) ** 2))
            g = (spec.vocal_pct / (100.0 - spec.vocal_pct)) * (a2 / max(a1, 1e-9))
            if abs(g - 1.0) > 0.005:
                diff *= g
                full = inst + diff
                a1 *= g
                if vox_gain_db is not None:
                    vox_gain_db += 20 * np.log10(g)
            share_pct = float(100 * a1 / (a1 + a2))
        del diff
    else:
        full = inst.copy()
    if verbose:
        print(f"  mix {dsp.lufs_integrated(full):.1f} LUFS | instrumental "
              f"{dsp.lufs_integrated(inst):.1f} LUFS | vocal "
              f"{(share_pct if share_pct else 0):.2f}% ({vox_gain_db or 0:+.1f} dB)")
    del mix, mix_with_vox, rev, dly
    if vox is not None:
        del vox
    gc.collect()
    return dict(mix=full, instrumental=inst, vocal_share_pct=share_pct,
                vox_gain_db=vox_gain_db, n=n)


def section_curve(spec, n):
    c = np.ones(n, dtype=np.float32)
    for name, start, length in spec.sections:
        g = spec.dynamics.get(name, 1.0)
        i0 = max(0, min(n, int(start * spec.bar_s * SR)))
        i1 = max(0, min(n, int((start + length) * spec.bar_s * SR)))
        if i1 > i0:
            c[i0:i1] = g
    k = int(0.05 * SR)
    return np.convolve(c, np.ones(k, dtype=np.float32) / k, mode="same").astype(np.float32)


# ================================================================ MASTER
def master(x, spec, target_lufs=-9.2, verbose=True):
    y = dsp.compressor(x, thr_db=-17.0, ratio=1.7, attack_ms=28, release_ms=240,
                       knee_db=8.0, makeup_db=1.0)
    y = np.stack([dsp.tape_sat(y[:, 0], 1.1, 0.16), dsp.tape_sat(y[:, 1], 1.1, 0.16)], axis=1)
    y = np.stack([dsp.hp(y[:, 0], 22, order=4), dsp.hp(y[:, 1], 22, order=4)], axis=1)
    bands = spec.master_eq or [("peak", 160, 1.8, 1.1), ("peak", 420, -1.0, 0.9),
                               ("peak", 1500, 1.6, 0.6), ("peak", 2400, 1.0, 0.8),
                               ("high_shelf", 8500, 0.8, 0.7)]
    y = _apply_eq(y, bands)
    y = dsp.mono_below_freq(y, 95)
    y = dsp.ms_width(y, 1.06)
    y = dsp.compressor(y, thr_db=-14.0, ratio=1.35, attack_ms=38, release_ms=300,
                       knee_db=10.0)
    y /= (np.max(np.abs(y)) + 1e-9) / db(-6.0)
    gain = 12.0          # sensible starting drive for a loud urban master
    lim = None
    hist = []
    best = None
    for it in range(8):
        lim = dsp.limit(y * db(gain), ceiling_db=-1.3, lookahead_ms=3.2, release_ms=140)
        lufs = dsp.lufs_integrated(lim)
        tp = dsp.true_peak_db(lim)
        trim = (-1.02 - tp) if tp > -1.02 else 0.0
        after = lufs + trim
        hist.append((round(gain, 1), round(lufs, 2), round(after, 2)))
        if best is None or abs(after - target_lufs) < best[0]:
            best = (abs(after - target_lufs), gain)
        if abs(after - target_lufs) < 0.08:
            break
        gain = float(np.clip(gain + (target_lufs - after) * 0.9, 0, 30))
    gain = best[1]
    lim = dsp.limit(y * db(gain), ceiling_db=-1.3, lookahead_ms=3.2, release_ms=140)
    if verbose:
        print(f"  limiter: drive {gain:.1f} dB | {hist[-3:]}")
    fi = int(0.004 * SR)
    lim[:fi] *= np.linspace(0, 1, fi)[:, None]
    fo = int(1.2 * SR)
    if np.max(np.abs(lim[-fo:])) > 0.002:
        lim[-fo:] *= np.linspace(1, 0, fo)[:, None] ** 1.5
    tp = dsp.true_peak_db(lim)
    if tp > -1.02:
        lim *= db(-1.02 - tp)
    if verbose:
        print(f"  master: {dsp.lufs_integrated(lim):.2f} LUFS | "
              f"{dsp.true_peak_db(lim):.2f} dBTP | crest {dsp.crest_db(lim):.2f} dB | "
              f"corr {dsp.stereo_correlation(lim):.2f}")
    return lim.astype(np.float32)


# ================================================================ RUN
def run_song(spec, build_fn, write_audio=True, verbose=True):
    t0 = time.time()
    parts = build_fn(spec)
    if verbose:
        print(f"[{spec.slug}] composing: " + ", ".join(f"{k}:{len(v[1])}" for k, v in parts.items()))
    raw, n = render_parts(spec, parts, to_disk=True, verbose=verbose)
    if verbose:
        print(f"[{spec.slug}] rendered {n/SR:.1f}s in {time.time()-t0:.0f}s")
    res = mix_song(spec, raw, verbose=verbose)
    return res, n, parts


def save_song(spec, res, n, parts, formats=("wav", "flac", "mp3"), verbose=True):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    outdir = os.path.join(SONGS, spec.slug)
    for sub in ("master", "analysis"):
        os.makedirs(os.path.join(outdir, sub), exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    tag = "24bit_48kHz"
    names = {}
    if verbose:
        print(f"[{spec.slug}] mastering vocal")
    mv = master(res["mix"], spec, target_lufs=-9.2, verbose=verbose)
    v_metrics = dict(lufs=round(dsp.lufs_integrated(mv), 2),
                     tp=round(dsp.true_peak_db(mv), 2),
                     crest=round(dsp.crest_db(mv), 2),
                     corr=round(dsp.stereo_correlation(mv), 3),
                     bands={k: round(v, 1) for k, v in dsp.band_energy(mv).items()},
                     sections={name: round(dsp.lufs_integrated(
                         mv[int(a * spec.bar_s * SR):int((a + b) * spec.bar_s * SR)]), 2)
                         for name, a, b in spec.sections})
    base = f"{spec.slug.upper()}_Western_Sessions"
    paths = dict(
        vocal_wav=os.path.join(outdir, "master", f"{base}_VOCAL_{tag}_MASTER.wav"),
        inst_wav=os.path.join(outdir, "master", f"{base}_INSTRUMENTAL_{tag}_MASTER.wav"),
        vocal_mp3=os.path.join(outdir, "master", f"{base}_VOCAL_320.mp3"),
        inst_mp3=os.path.join(outdir, "master", f"{base}_INSTRUMENTAL_320.mp3"),
    )
    sf.write(paths["vocal_wav"], mv, SR, subtype="PCM_24", format="WAV")
    del mv
    res["mix"] = None
    gc.collect()
    if verbose:
        print(f"[{spec.slug}] mastering instrumental")
    mi = master(res["instrumental"], spec, target_lufs=-9.2, verbose=verbose)
    i_metrics = dict(lufs=round(dsp.lufs_integrated(mi), 2),
                     tp=round(dsp.true_peak_db(mi), 2),
                     crest=round(dsp.crest_db(mi), 2))
    if "flac" in formats:
        sf.write(paths["inst_wav"].replace(".wav", ".flac"), mi, SR, subtype="PCM_24", format="FLAC")
    else:
        sf.write(paths["inst_wav"], mi, SR, subtype="PCM_24", format="WAV")

    # ---- plot (instrumental is still in memory)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(12, 5.4), sharex=True)
    k = 480
    m = len(mi) // k
    pk = np.max(np.abs(mi[:m * k].reshape(m, k, 2)), axis=(1, 2))
    ax[0].fill_between(np.arange(m) * k / SR, -pk, pk, color="#264653", linewidth=0)
    ax[0].set_title(f"{spec.title} — Orion Music & Hanit Kamboj  |  {spec.genre}  |  "
                    f"{spec.key_name}, {spec.bpm:g} BPM  |  {i_metrics['lufs']:.1f} LUFS",
                    fontsize=10, loc="left")
    ax[0].set_ylim(-1.02, 1.02)
    for name, start, length in spec.sections:
        a, b = start * spec.bar_s, (start + length) * spec.bar_s
        ax[0].axvline(a, color="#999", lw=0.5)
        ax[0].text((a + b) / 2, 0.9, name.upper(), ha="center", fontsize=6.5)
    yk = dsp.k_weight(mi.astype(np.float64))
    win, hop = 3 * SR, int(0.5 * SR)
    vals = [-0.691 + 10 * np.log10(np.mean(yk[i:i + win] ** 2) + 1e-20)
            for i in range(0, len(yk) - win, hop)]
    ax[1].plot((np.arange(len(vals)) * hop + win / 2) / SR, vals, color="#b8860b", lw=1.1)
    ax[1].set_ylabel("LUFS (3 s)")
    ax[1].set_xlabel("time (s)")
    ax[1].grid(alpha=0.2)
    ax[0].grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "analysis", "waveform_loudness.png"), dpi=130)
    plt.close(fig)
    del mi
    gc.collect()

    share = res.get("vocal_share_pct")
    metrics = dict(
        title=spec.title, artist="Orion Music & Hanit Kamboj", genre=spec.genre,
        key=spec.key_name, bpm=spec.bpm, bars=spec.bars,
        duration_s=round(n / SR, 2),
        vocal_lufs=v_metrics["lufs"], instrumental_lufs=i_metrics["lufs"],
        vocal_true_peak=v_metrics["tp"], instrumental_true_peak=i_metrics["tp"],
        vocal_crest=v_metrics["crest"], instrumental_crest=i_metrics["crest"],
        stereo_correlation=v_metrics["corr"],
        vocal_share_pct_of_rms=round(share, 2) if share else None,
        vocal_level_db_below_mix=round(res.get("vox_gain_db") or 0.0, 1),
        bands=v_metrics["bands"], sections=v_metrics["sections"],
    )
    with open(os.path.join(outdir, "analysis", "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    if verbose:
        print(f"[{spec.slug}] vocal share {share:.1f}% | written {outdir}")
    import shutil
    rawdir = os.path.join(CACHE, "raw", spec.slug)
    if os.path.isdir(rawdir):
        shutil.rmtree(rawdir, ignore_errors=True)
    return paths, metrics


def export_midi(spec, parts, verbose=True):
    import mido
    from mido import Message, MidiFile, MidiTrack, MetaMessage
    TPQ = 960
    mid_dir = os.path.join(SONGS, spec.slug, "session")
    os.makedirs(mid_dir, exist_ok=True)
    drum_map = dict(kick=36, kick808=36, snare=38, clap=39, rim=37, snap=39,
                    hats=42, perc=70, toms=45, crash=49)
    full = MidiFile(ticks_per_beat=TPQ)
    head = MidiTrack()
    head.append(MetaMessage("track_name", name=spec.title, time=0))
    head.append(MetaMessage("set_tempo", tempo=mido.bpm2tempo(spec.bpm), time=0))
    head.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    head.append(MetaMessage("marker", text="ORION MUSIC & HANIT KAMB0J", time=0))
    for name, start, length in spec.sections:
        head.append(MetaMessage("marker", text=f"{name.upper()} @ bar {start+1}", time=0))
    full.tracks.append(head)

    def mk(name, events, ch, is_drum=False, prog=None):
        tr = MidiTrack()
        tr.append(MetaMessage("track_name", name=name, time=0))
        if prog is not None and not is_drum:
            tr.append(Message("program_change", program=prog, channel=ch, time=0))
        ev = []
        for (beat, dur, midi, vel) in events:
            on = int(round(beat * TPQ)); off = int(round((beat + max(dur, 0.02)) * TPQ))
            ev.append((on, 1, int(midi), int(np.clip(vel * 127, 1, 127))))
            ev.append((off, 0, int(midi), 0))
        ev.sort(key=lambda z: (z[0], z[1]))
        last = 0
        for tick, is_on, m, v in ev:
            dt = max(0, tick - last); last = tick
            if is_on:
                tr.append(Message("note_on", note=m, velocity=v, channel=9 if is_drum else ch, time=dt))
            else:
                tr.append(Message("note_off", note=m, velocity=0, channel=9 if is_drum else ch, time=dt))
        full.tracks.append(tr)
        return tr

    used = []
    ch = 0
    for name, (kind, evs) in parts.items():
        if not evs or kind == "fx":
            continue
        if kind in drum_map:
            notes = [(e[0] / spec.beat_s, 0.25, drum_map[kind] + (0 if kind != "hats" else 0), e[1])
                     for e in evs]
            mk(name, notes, 9, is_drum=True)
            continue
        if kind == "vox":
            for vi in sorted({e.get("voice", 0) for e in evs}):
                sel = [e for e in evs if e.get("voice", 0) == vi]
                notes = [(e["t"] / spec.beat_s, e["dur"] / spec.beat_s, e["midi"], e["vel"]) for e in sel]
                mk(f"vox_{vi}", notes, 7 + vi, prog=53)
            continue
        ch = 0 if ch > 15 else ch
        notes = []
        for e in evs:
            if not isinstance(e, dict):
                continue
            if "notes" in e:
                for m in e["notes"]:
                    notes.append((e["t"] / spec.beat_s, e["dur"] / spec.beat_s, m, e["vel"]))
            else:
                notes.append((e["t"] / spec.beat_s, e["dur"] / spec.beat_s, e["midi"], e["vel"]))
        prog = dict(finger=33, sub808=38, rhodes=4, wurli=4, clav=7, organ=16,
                    vibes=11, guitar=27, strings=48, pad=89, brass=61, bell=10,
                    flute=73, talkbox=85, synth=81).get(kind)
        mk(name, notes, ch, prog=prog)
        ch += 1
    path = os.path.join(mid_dir, f"{spec.slug}_FULL.mid")
    full.save(path)
    with open(os.path.join(mid_dir, "arrangement.json"), "w") as f:
        json.dump(dict(title=spec.title, genre=spec.genre, key=spec.key_name, bpm=spec.bpm,
                       swing=spec.swing, bars=spec.bars,
                       sections=[dict(name=n_, start_bar=s + 1, bars=l,
                                      start_s=round(s * spec.bar_s, 2),
                                      end_s=round((s + l) * spec.bar_s, 2))
                                 for n_, s, l in spec.sections]), f, indent=2)
    if verbose:
        print(f"[{spec.slug}] midi -> {path}")
    return path
