"""
instruments.py — original instrument models for the session.
All voices are synthesized from oscillators / noise / filters, tuned and
articulated per note. No copyrighted samples or presets are used.
"""
from __future__ import annotations
import numpy as np
from scipy import signal
from .dsp import (SR, TWO_PI, db, midi_hz, osc_saw, osc_square, osc_sine, noise,
                  env_exp, fade_edges, lp, hp, bp, peak, saturate, polyblep)


# =========================================================== DRUMS
def kick(vel=1.0, tune_midi=30.5, seed=1, sr=SR):
    """Deep, saturated boom-bap kick with a tuned body and beater click."""
    n = int(0.62 * sr)
    t = np.arange(n) / sr
    f0 = midi_hz(tune_midi)
    # pitch sweep: ~3.1x at the transient down to the fundamental in 40 ms
    f = f0 * (1.0 + 2.1 * np.exp(-t / 0.028))
    ph = TWO_PI * np.cumsum(f) / sr
    body = np.sin(ph) * np.exp(-t / 0.24)
    body += 0.32 * np.sin(2 * ph) * np.exp(-t / 0.11)
    body += 0.10 * np.sin(3 * ph) * np.exp(-t / 0.05)
    # beater click
    click = bp(noise(n, seed), 2400, 10000, sr) * np.exp(-t / 0.0042)
    click *= 0.34
    # sub weight / note-consistent low end
    sub = np.sin(TWO_PI * f0 * 1.0 * t) * np.exp(-t / 0.30) * 0.32
    x = body + sub + click
    x = saturate(x, 1.9, asym=0.06)
    x = lp(x, 5200, sr)
    # tighten the decay tail so kicks stay out of the bass's way
    x *= np.clip((0.55 - t) / 0.07, 0.0, 1.0)
    x = fade_edges(x, 0.3, 8)
    return (x / (np.max(np.abs(x)) + 1e-9) * 0.92 * vel).astype(np.float32)


def snare(vel=1.0, seed=3, sr=SR, tone=1.0, bright=1.0):
    """Crisp, dry hip-hop snare: shell tone + band-split noise."""
    n = int(0.36 * sr)
    t = np.arange(n) / sr
    f = 196 * (1 + 0.18 * np.exp(-t / 0.012))
    shell = (np.sin(TWO_PI * np.cumsum(f) / sr) * np.exp(-t / 0.085) * 0.55
             + np.sin(2 * np.pi * 342 * t) * np.exp(-t / 0.05) * 0.28)
    nz = noise(n, seed)
    lo = bp(nz, 260, 1500, sr) * np.exp(-t / 0.095)
    hi = bp(nz, 1400, 11000, sr) * np.exp(-t / 0.135)
    crack = bp(noise(n, seed + 1), 3200, 12000, sr) * np.exp(-t / 0.055)
    noise_part = (lo * 0.75 + hi * 0.85 * bright + crack * 0.35 * bright)
    noise_part = hp(noise_part, 200, sr)
    x = shell * tone + noise_part * 0.85
    x = saturate(x, 1.35)
    x = peak(x, 3800, 1.5, 0.8, sr)
    x *= np.clip((0.34 - t) / 0.05, 0, 1)
    x = fade_edges(x, 0.2, 4)
    return (x / (np.max(np.abs(x)) + 1e-9) * 0.9 * vel).astype(np.float32)


def clap(vel=1.0, seed=5, sr=SR, width=1.0):
    """Layered hand clap, used to widen the backbeat."""
    n = int(0.34 * sr)
    t = np.arange(n) / sr
    x = np.zeros(n)
    offsets = [0.0, 0.010, 0.021, 0.033]
    for i, o in enumerate(offsets):
        k = int(o * sr)
        burst = bp(noise(n - k, seed + i), 900, 4800, sr) * np.exp(-t[:n - k] / 0.018)
        x[k:] += burst * (0.85 - 0.16 * i)
    tail = bp(noise(n, seed + 9), 1000, 6000, sr) * np.exp(-t / 0.11)
    x += tail * 0.42
    x = peak(x, 1200, 2.5, 1.0, sr)
    x = saturate(x, 1.2)
    x = fade_edges(x, 0.15, 3)
    return (x / (np.max(np.abs(x)) + 1e-9) * 0.75 * vel).astype(np.float32)


def rim(vel=1.0, seed=11, sr=SR):
    n = int(0.14 * sr)
    t = np.arange(n) / sr
    x = (np.sin(TWO_PI * 780 * t) * np.exp(-t / 0.018)
         + 0.6 * np.sin(TWO_PI * 1650 * t) * np.exp(-t / 0.011))
    x += bp(noise(n, seed), 2000, 8000, sr) * np.exp(-t / 0.006) * 0.7
    x = saturate(x, 1.2)
    x = fade_edges(x, 0.2, 3)
    return (x / (np.max(np.abs(x)) + 1e-9) * 0.8 * vel).astype(np.float32)


def hat(vel=1.0, open_=False, seed=13, sr=SR, length=None):
    """808-style metallic hat from six inharmonic square partials + noise."""
    base = 205.3
    ratios = [1.0, 1.482, 1.800, 2.548, 2.630, 3.897]
    dur = length if length is not None else (0.32 if open_ else 0.085)
    n = int(dur * sr)
    t = np.arange(n) / sr
    x = np.zeros(n)
    for i, r in enumerate(ratios):
        f = base * r * (1.0 + 0.001 * i)
        x += 0.5 * np.sign(np.sin(TWO_PI * f * t + i * 1.7))
    x = hp(x, 6000, sr, order=3)
    nz = hp(noise(n, seed), 7000, sr, order=2)
    decay = 0.20 if open_ else 0.028
    env = np.exp(-t / decay) * (1 - np.exp(-t / 0.0015))
    x = x * 0.55 + nz * 0.45
    x *= env
    x = fade_edges(x, 0.2, 5)
    x = peak(x, 9000, 1.0, 0.8, sr)
    return (x / (np.max(np.abs(x)) + 1e-9) * 0.62 * vel).astype(np.float32)


def tamb(vel=1.0, seed=17, sr=SR, length=0.20):
    n = int(length * sr)
    t = np.arange(n) / sr
    nz = noise(n, seed)
    j = bp(nz, 4500, 13000, sr) * np.exp(-t / 0.055)
    j += bp(noise(n, seed + 2), 7000, 16000, sr) * np.exp(-t / 0.03) * 0.6
    env = (1 - np.exp(-t / 0.0015))
    x = j * env
    x = hp(x, 4000, sr)
    x = fade_edges(x, 0.2, 3)
    return (x / (np.max(np.abs(x)) + 1e-9) * 0.5 * vel).astype(np.float32)


def shaker(vel=1.0, seed=19, sr=SR):
    n = int(0.11 * sr)
    t = np.arange(n) / sr
    nz = bp(noise(n, seed), 5500, 12000, sr)
    env = np.exp(-t / 0.028) * (1 - np.exp(-t / 0.004))
    x = nz * env
    x = hp(x, 5000, sr)
    return (x / (np.max(np.abs(x)) + 1e-9) * 0.42 * vel).astype(np.float32)


def tom(midi=45, vel=1.0, seed=23, sr=SR):
    n = int(0.55 * sr)
    t = np.arange(n) / sr
    f0 = midi_hz(midi)
    f = f0 * (1 + 0.55 * np.exp(-t / 0.05))
    ph = TWO_PI * np.cumsum(f) / sr
    x = np.sin(ph) * np.exp(-t / 0.22)
    x += 0.18 * np.sin(2 * ph) * np.exp(-t / 0.09)
    x += bp(noise(n, seed), 400, 4000, sr) * np.exp(-t / 0.01) * 0.25
    x = saturate(x, 1.3)
    x = fade_edges(x, 0.3, 10)
    return (x / (np.max(np.abs(x)) + 1e-9) * 0.75 * vel).astype(np.float32)


def crash(reverse=False, length=3.2, seed=29, sr=SR, vel=1.0):
    n = int(length * sr)
    t = np.arange(n) / sr
    x = np.zeros(n)
    rng = np.random.default_rng(seed)
    for i in range(9):
        f = 320 * (1.6 ** i) * (1 + 0.02 * rng.standard_normal())
        x += 0.25 * np.sign(np.sin(TWO_PI * f * t + rng.random() * 6.28))
    x = hp(x, 3500, sr, order=2)
    nz = hp(noise(n, seed + 3), 4000, sr, order=2)
    x = x * 0.4 + nz * 0.6
    env = np.exp(-t / (length * 0.28))
    x *= env
    if reverse:
        x = x[::-1].copy()
        x *= np.linspace(0, 1, n) ** 1.6
    x = fade_edges(x, 0.5, 30)
    return (x / (np.max(np.abs(x)) + 1e-9) * 0.55 * vel).astype(np.float32)


# =========================================================== BASS
def bass_note(midi, dur_s, vel=0.8, art="finger", glide_from=None,
              glide_s=0.065, sr=SR, seed=41, vibrato=0.0):
    """Electric fingerstyle bass model: plucked comb-filtered harmonic stack,
    per-partial damping, finger transient, optional slide/portamento."""
    n = max(int((dur_s + 0.16) * sr), 256)
    t = np.arange(n) / sr
    f0 = midi_hz(midi)
    # pitch contour (portamento from a previous note)
    if glide_from is not None:
        f_start = midi_hz(glide_from)
        gl = np.clip(t / glide_s, 0, 1)
        gl = gl * gl * (3 - 2 * gl)
        f_t = f_start * (f0 / f_start) ** gl
    else:
        f_t = np.full(n, f0)
    if vibrato > 0:
        onset = np.clip((t - 0.18) / 0.2, 0, 1)
        f_t = f_t * 2 ** (vibrato * onset * np.sin(TWO_PI * 5.1 * t) / 12)
    phase = TWO_PI * np.cumsum(f_t) / sr

    nh = int(min(48, max(6, 5200 / f0)))
    x = np.zeros(n)
    # two pluck/pickup position combs give the characteristic hollow electric tone
    for k in range(1, nh + 1):
        a = abs(np.sin(np.pi * k * 0.19)) * abs(np.sin(np.pi * k * 0.53) + 0.35)
        a = 0.85 * a / (k ** 1.05)
        if art == "slap":
            damp = 0.85 + 0.20 * k ** 1.35
        elif art == "mute":
            damp = 5.0 + 0.9 * k
        else:
            damp = 0.45 + 0.068 * k ** 1.40
        x += a * np.exp(-t * damp) * np.sin(k * phase + 0.13 * k)
    # body / sustain: low harmonics hold longer, add a warm fundamental bed
    hold = {"mute": 0.0, "slap": 0.10}.get(art, 0.42)
    bed = (np.sin(phase) * 0.5 + np.sin(2 * phase) * 0.16) * np.exp(-t / 0.9)
    x = x + bed * hold
    # amp envelope
    att = 0.004 if art != "mute" else 0.002
    env = (1 - np.exp(-t / att)) * np.exp(-t / (0.55 if art != "mute" else 0.10))
    gate = np.clip((dur_s + 0.10 - t) / 0.09, 0.0, 1.0)
    x *= env * gate
    # finger/pick noise
    nz = bp(noise(n, seed), 350, 2600 if art != "slap" else 4200, sr)
    x += nz * np.exp(-t / 0.008) * 0.045 * (1.0 + 1.4 * (art == "slap"))
    # sub thump
    x += np.sin(TWO_PI * (f0 * 0.5) * t) * np.exp(-t / 0.045) * 0.10
    x = saturate(x, 1.25)
    x = lp(x, 950 if art == "mute" else (3600 if art == "slap" else 3300), sr)
    x = hp(x, 32, sr)
    x = fade_edges(x, 0.4, 8)
    return (x / (np.max(np.abs(x)) + 1e-9) * 0.85 * vel).astype(np.float32)


# =========================================================== RHODES
def rhodes_note(midi, dur_s, vel=0.8, sr=SR, seed=51, pan_seed=None):
    """FM Rhodes: bell/tine attack over a warm bark, tremolo-ready."""
    n = max(int((dur_s + 0.5) * sr), 512)
    t = np.arange(n) / sr
    f = midi_hz(midi)
    phase = TWO_PI * f * t
    I1 = (0.28 + 1.15 * vel) * np.exp(-t / 0.24) + 0.10
    I2 = (0.35 + 1.05 * vel) * np.exp(-t / 0.035)
    # small inharmonic chorus per note for a real "tine + pickup" shimmer
    det = 1.0018
    y = (np.sin(phase + I1 * np.sin(phase) + I2 * np.sin(14.0 * phase))
         + 0.42 * np.sin(det * phase + 0.8 * I1 * np.sin(det * phase)
                         + 0.7 * I2 * np.sin(14.0 * det * phase + 1.1)))
    # wooden bark on harder notes
    y += 0.10 * vel * np.sin(2 * phase) * np.exp(-t / 0.09)
    env = (1 - np.exp(-t / 0.0025)) * np.exp(-t / (1.15 - 0.35 * vel))
    gate = np.clip((dur_s + 0.45 - t) / 0.40, 0, 1) ** 1.4
    body = y * env * gate
    # key-off / damper thump
    click = bp(noise(n, seed), 1500, 6500, sr) * np.exp(-t / 0.004) * 0.012
    out = body * 0.22 + click
    out = lp(out, 5200, sr)
    out = saturate(out, 1.1)
    return (out / (np.max(np.abs(out)) + 1e-9) * 0.5 * (0.35 + 0.65 * vel)).astype(np.float32)


# =========================================================== GUITAR
def guitar_note(midi, dur_s, vel=0.8, muted=True, sr=SR, seed=61):
    """Plucked-string guitar model (damped funk chank or clean stab)."""
    n = max(int((min(dur_s, 1.2) + 0.25) * sr), 256)
    t = np.arange(n) / sr
    f0 = midi_hz(midi)
    nh = int(min(22, max(4, 4200 / f0)))
    x = np.zeros(n)
    for k in range(1, nh + 1):
        a = abs(np.sin(np.pi * k * 0.28)) / (k ** 1.1)
        damp = (18.0 if muted else 2.0) + (1.4 if muted else 0.25) * k ** 1.3
        det = 1.0 + 0.00022 * k * k  # string stiffness
        x += a * np.exp(-t * damp) * np.sin(TWO_PI * f0 * k * det * t + 0.4 * k)
    env = (1 - np.exp(-t / 0.0018)) * np.exp(-t / (0.055 if muted else 0.7))
    if muted:
        env *= np.clip((min(dur_s, 0.14) + 0.015 - t) / 0.02, 0.0, 1.0)
    else:
        env *= np.clip((dur_s + 0.18 - t) / 0.18, 0.0, 1.0)
    x *= env
    # pick noise
    pick = bp(noise(n, seed), 900, 6500, sr) * np.exp(-t / 0.006) * (0.10 if muted else 0.06)
    x += pick
    # palm thump
    x += np.sin(TWO_PI * f0 * 0.5 * t) * np.exp(-t / 0.02) * (0.08 if muted else 0.03)
    x = lp(x, 2400 if muted else 4300, sr)
    x = saturate(x, 1.15)
    x = fade_edges(x, 0.3, 6)
    return (x / (np.max(np.abs(x)) + 1e-9) * 0.55 * vel).astype(np.float32)


# =========================================================== LEADS
VOWELS = {
    "oo": [(320, 5.0, 1.00), (780, 5.5, 0.72), (2250, 6.5, 0.42), (3150, 7.0, 0.20)],
    "oh": [(450, 5.0, 1.00), (830, 5.5, 0.78), (2480, 6.0, 0.46), (3300, 7.0, 0.22)],
    "ah": [(730, 4.5, 1.00), (1100, 5.0, 0.80), (2450, 6.0, 0.52), (3450, 7.0, 0.24)],
    "eh": [(530, 5.0, 1.00), (1840, 5.5, 0.72), (2500, 6.5, 0.44), (3400, 7.0, 0.20)],
    "ee": [(300, 5.0, 1.00), (2150, 5.5, 0.68), (2900, 6.5, 0.40), (3600, 7.0, 0.20)],
}


def _formant_filter(x, vowel, sr=SR):
    y = np.zeros_like(x)
    for fc, q, g in VOWELS[vowel]:
        y += bp(x, fc * 2 ** (-1.1 / q), fc * 2 ** (1.1 / q), sr, order=2) * g
    # low body + high "buzz" horn formant so the talkbox cuts
    y += lp(x, 320, sr) * 0.40
    y += bp(x, 3300, 7200, sr) * 0.28
    return y


def lead_phrase(events, kind="talkbox", sr=SR, seed=71):
    """Render a monophonic lead phrase with legato glides.

    events: dicts with start (s), dur (s), midi, vel, vowel, glide (bool),
            glide_s, bend (semitones at end), vib (depth semitones)
    Returns stereo (slight doubling/chorus for width).
    """
    if not events:
        return np.zeros((1, 2), dtype=np.float32)
    end = max(e["start"] + e["dur"] for e in events) + 0.6
    n = int(end * sr)
    freq = np.zeros(n)
    amp = np.zeros(n)
    formant = np.zeros(n, dtype=np.int16)
    vib_depth = np.zeros(n)
    bend = np.zeros(n)
    vowel_ids = {v: i + 1 for i, v in enumerate(VOWELS)}
    events = sorted(events, key=lambda e: e["start"])
    prev_end = None
    prev_midi = None
    for ev in events:
        i0 = int(ev["start"] * sr)
        i1 = int((ev["start"] + ev["dur"]) * sr)
        i1 = min(i1, n)
        length = i1 - i0
        if length <= 4:
            continue
        tt = np.arange(length) / sr
        target = midi_hz(ev["midi"])
        if ev.get("glide") and prev_midi is not None:
            gs = ev.get("glide_s", 0.09)
            g = np.clip(tt / gs, 0, 1)
            g = g * g * (3 - 2 * g)
            f_here = midi_hz(prev_midi) * (target / midi_hz(prev_midi)) ** g
        else:
            f_here = np.full(length, target)
        # scoop into notes from just below (expressive, singer-like)
        if ev.get("scoop", True) and not ev.get("glide"):
            sc = np.clip(tt / 0.05, 0, 1)
            f_here = f_here * 2 ** (-0.45 * (1 - sc) / 12)
        if ev.get("bend"):
            bend_amt = ev["bend"]
            bs = np.clip((tt - length / sr * 0.65) / 0.18, 0, 1)
            f_here = f_here * 2 ** (bend_amt * bs / 12)
        # vibrato (delayed onset)
        vib = ev.get("vib", 0.14)
        vo = np.clip((tt - 0.16) / 0.22, 0, 1)
        f_here = f_here * 2 ** (vib * vo * np.sin(2 * np.pi * 5.3 * tt + 0.3) / 12)
        freq[i0:i1] = f_here
        # amp envelope
        a = 0.020 if (ev.get("glide") or prev_end == ev["start"]) else 0.012
        e = (1 - np.exp(-tt / a)) * np.exp(-tt / (0.9 + ev["dur"]))
        rel = np.clip((ev["dur"] - tt) / 0.06, 0, 1) ** 0.8
        amp[i0:i1] = e * rel * ev["vel"]
        formant[i0:i1] = vowel_ids.get(ev.get("vowel", "oh"), 2)
        prev_end = ev["start"] + ev["dur"]
        prev_midi = ev["midi"]
    # fill holes smoothly
    idx = np.where(amp > 1e-6)[0]
    if len(idx) == 0:
        return np.zeros((n, 2), dtype=np.float32)
    # carrier
    phase = TWO_PI * np.cumsum(freq) / sr
    if kind == "talkbox":
        src = 0.7 * osc_saw(np.maximum(freq, 1), n, sr) + 0.35 * osc_square(
            np.maximum(freq, 1), n, sr, pulse=0.42)
        src += 0.06 * noise(n, seed) * (amp > 0)
        src = saturate(src, 2.2, asym=0.08, mix=0.65)
        # process each contiguous vowel segment through its formant bank
        out = np.zeros(n, dtype=np.float32)
        change = np.flatnonzero(np.diff(formant) != 0) + 1
        edges = np.concatenate([[0], change, [n]])
        bounds = [(int(edges[i]), int(edges[i + 1]), int(formant[edges[i]]))
                  for i in range(len(edges) - 1)]
        inv = {v: k for k, v in vowel_ids.items()}
        for s, e, v in bounds:
            if v == 0:
                continue
            seg = src[s:e]
            # 8 ms crossfades between vowel segments
            f = _formant_filter(seg, inv.get(int(v), "oh"), sr)
            if e - s > 96:
                f[:48] *= np.linspace(0, 1, 48)
                f[-48:] *= np.linspace(1, 0, 48)
            out[s:e] += f
        out = saturate(out, 1.5, mix=0.5)
        out = lp(out, 10500, sr)
        out *= amp
        out = hp(out, 160, sr)
        # doubler for width
        d = int(0.011 * sr)
        left = out.copy()
        right = np.zeros_like(out)
        right[d:] = out[:-d]
        right = hp(right, 250, sr)
        st = np.stack([left * 0.95 + right * 0.16, right * 0.95 + left * 0.16], axis=1)
    else:  # analog synth lead
        uni = osc_saw(np.maximum(freq, 1), n, sr, unison=3, detune_cents=11, seed=seed)
        sub = 0.25 * osc_square(np.maximum(freq, 1) * 0.5, n, sr, pulse=0.5)
        raw = uni + sub
        raw = lp(raw, 3200, sr, order=2)
        raw = peak(raw, 1800, 1.5, 0.9, sr)
        out = saturate(raw, 1.7, mix=0.55)
        out *= amp
        out = hp(out, 140, sr)
        st = np.stack([out * 0.98, np.roll(out, 11) * 0.98], axis=1)
    st = fade_edges(st, 1.0, 20)
    m = np.max(np.abs(st)) + 1e-9
    return (st / m * 0.62).astype(np.float32)


def _tv_lowpass(x, cutoff, sr=SR, block=128, order=2):
    y = np.empty_like(x)
    for i in range(0, len(x), block):
        fc = float(np.clip(cutoff[min(i, len(cutoff) - 1)], 80, sr * 0.45))
        sos = signal.butter(order, fc, btype="low", fs=sr, output="sos")
        y[i:i + block] = signal.sosfilt(sos, x[i:i + block])
    return y


# =========================================================== STRINGS
def string_chord(notes, dur_s, vel=0.6, sr=SR, seed=81, attack=0.5,
                 release=0.9, octave_up=False, bright=1.0):
    """Small string section: detuned saw per voice, per-voice panning,
    slow bow attack, delayed vibrato."""
    n = int((dur_s + release) * sr)
    t = np.arange(n) / sr
    out = np.zeros((n, 2), dtype=np.float32)
    rng = np.random.default_rng(seed)
    notes = list(notes)
    if octave_up:
        notes = notes + [m + 12 for m in notes]
    for vi, m in enumerate(notes):
        f0 = midi_hz(m)
        voice = np.zeros(n)
        for u in range(5):
            cents = (u - 2) * 4.2 + rng.standard_normal() * 1.5
            f = f0 * 2 ** (cents / 1200)
            src = osc_saw(f, n, sr, phase0=rng.random())
            # bow vibrato, delayed
            src = src  # phase modulation added below via resampling is overkill; use chorus
            voice += src
        voice /= 5
        # slow attack + expression swell, then release
        e = (1 - np.exp(-t / max(attack, 0.05)))
        e *= 0.75 + 0.25 * np.exp(-t / 2.0)
        e *= np.clip((dur_s + release - t) / release, 0, 1) ** 1.2
        # slight delayed vibrato via amplitude movement (keeps it organic without PM cost)
        vib = 1.0 + 0.05 * np.sin(TWO_PI * 4.8 * t + vi) * np.clip((t - 0.4) / 0.5, 0, 1)
        voice *= e * vib * vel * (0.9 if m > 72 else 1.0)
        p = (vi / max(1, len(notes) - 1)) * 2 - 1
        pan = 0.45 * p
        l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        out[:, 0] += (voice * l).astype(np.float32)
        out[:, 1] += (voice * r).astype(np.float32)
    out = lp(out, 2400 * bright, sr)
    out = peak(out, 350, 1.2, 1.0, sr)
    out = hp(out, 90, sr)
    m = np.max(np.abs(out)) + 1e-9
    out = out / m * 0.42
    return out.astype(np.float32)


# =========================================================== FX
def riser(dur_s=2.0, sr=SR, seed=91, bright_end=9000):
    n = int(dur_s * sr)
    t = np.arange(n) / sr
    nz = noise(n, seed)
    env = (t / dur_s) ** 2.2
    # crossfade three band-passes to simulate an opening filter
    a = bp(nz, 300, 1200, sr) * (1 - t / dur_s)
    b = bp(nz, 900, 4000, sr) * np.sin(np.pi * np.clip(t / dur_s, 0, 1))
    c = bp(nz, 3000, 13000, sr) * (t / dur_s) ** 1.5
    x = (a + b + c) * env
    x += 0.20 * np.sin(TWO_PI * np.cumsum(np.linspace(180, 900, n)) / sr) * env
    x = fade_edges(x, 10, 5)
    m = np.max(np.abs(x)) + 1e-9
    return (x / m * 0.4).astype(np.float32)


def sub_drop(midi=30, dur_s=1.2, sr=SR, vel=1.0):
    n = int(dur_s * sr)
    t = np.arange(n) / sr
    f = midi_hz(midi) * (1 + 0.8 * np.exp(-t / 0.06))
    x = np.sin(TWO_PI * np.cumsum(f) / sr) * np.exp(-t / (dur_s * 0.45))
    x = saturate(x, 1.4)
    x = fade_edges(x, 1, 30)
    return (x / (np.max(np.abs(x)) + 1e-9) * 0.35 * vel).astype(np.float32)


def vinyl_bed(dur_s, sr=SR, seed=99, level=0.012):
    """Very low level surface texture: room air + sparse crackle, no sample used."""
    n = int(dur_s * sr)
    rng = np.random.default_rng(seed)
    air = lp(rng.standard_normal(n), 900, sr) * 0.6
    air += hp(rng.standard_normal(n), 3000, sr) * 0.05
    clicks = np.zeros(n)
    n_clicks = int(dur_s * 260)
    pos = rng.integers(0, n, n_clicks)
    clicks[pos] = rng.standard_normal(n_clicks) * rng.uniform(0.1, 1.0, n_clicks)
    clicks = bp(clicks, 900, 6000, sr) * 0.8
    x = air * 0.25 + clicks
    x = hp(x, 120, sr)
    return (x / (np.max(np.abs(x)) + 1e-9) * level).astype(np.float32)
