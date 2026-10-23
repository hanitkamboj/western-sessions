"""
inst2.py — second instrument bank for the modern sessions.

New voices (all synthesized from first principles):
  vibes, wurlitzer, clavinet, organ, brass section, bell/pluck, 808 sub,
  pad, flute, finger-snap — plus the wordless VOCALISE engine
  (vowel formant voices with legato, vibrato, breath, choir stacking).
"""
from __future__ import annotations
import numpy as np
from scipy import signal
from .dsp import (SR, TWO_PI, db, midi_hz, osc_saw, osc_square, osc_sine, noise,
                  env_exp, fade_edges, lp, hp, bp, peak, saturate, polyblep)


# =========================================================== KEYBOARDS
def vibes_note(midi, dur_s, vel=0.7, sr=SR, seed=101, motor=5.2):
    """FM vibraphone: metallic attack, long ringing bar, motor tremolo."""
    n = int((dur_s + 1.6) * sr)
    t = np.arange(n) / sr
    f = midi_hz(midi)
    ph = TWO_PI * f * t
    I = (0.55 + 0.9 * vel) * np.exp(-t / 0.045) + 0.06
    y = np.sin(ph + I * np.sin(4.0 * ph + 0.4))
    y += 0.30 * np.sin(2.01 * ph) * np.exp(-t / 0.7)
    y += 0.10 * np.sin(5.97 * ph) * np.exp(-t / 0.12)
    env = (1 - np.exp(-t / 0.0015)) * np.exp(-t / (1.15 - 0.25 * vel))
    gate = np.clip((dur_s + 1.4 - t) / 0.5, 0, 1) ** 1.4
    # motor: amplitude tremolo, slight and warm
    trem = 1.0 - 0.14 * (0.5 + 0.5 * np.sin(TWO_PI * motor * t))
    y = y * env * gate * trem
    y += bp(noise(n, seed), 3000, 9000, sr) * np.exp(-t / 0.004) * 0.035
    y = lp(y, 6500, sr)
    y = fade_edges(y, 0.5, 30)
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.42 * vel).astype(np.float32)


def wurli_note(midi, dur_s, vel=0.7, sr=SR, seed=111):
    """Wurlitzer: reed bark, softer and rounder than the Rhodes."""
    n = int((dur_s + 0.45) * sr)
    t = np.arange(n) / sr
    f = midi_hz(midi)
    ph = TWO_PI * f * t
    I = (0.9 + 1.5 * vel) * np.exp(-t / 0.05) + 0.25
    y = np.sin(ph + I * np.sin(ph)) + 0.25 * np.sin(3 * ph) * np.exp(-t / 0.12)
    y += 0.12 * np.sin(2 * ph + 0.6)
    env = (1 - np.exp(-t / 0.003)) * np.exp(-t / (0.85 - 0.2 * vel))
    gate = np.clip((dur_s + 0.40 - t) / 0.35, 0, 1) ** 1.3
    y = y * env * gate
    y += bp(noise(n, seed), 1200, 5000, sr) * np.exp(-t / 0.003) * 0.02
    y = lp(y, 4200, sr)
    y = saturate(y, 1.15, mix=0.3)
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.45 * vel).astype(np.float32)


def clav_note(midi, dur_s, vel=0.8, sr=SR, seed=121, wah=1500.0):
    """Clavinet: bright plucked string through a fixed wah band-pass."""
    n = int((min(dur_s, 0.6) + 0.18) * sr)
    t = np.arange(n) / sr
    f0 = midi_hz(midi)
    nh = int(min(30, max(6, 9000 / f0)))
    x = np.zeros(n)
    for k in range(1, nh + 1):
        a = abs(np.sin(np.pi * k * 0.14)) / (k ** 0.85)
        damp = 16.0 + 1.1 * k ** 1.25
        x += a * np.exp(-t * damp) * np.sin(TWO_PI * f0 * k * (1 + 0.00028 * k * k) * t)
    env = (1 - np.exp(-t / 0.0012)) * np.exp(-t / 0.16)
    env *= np.clip((min(dur_s, 0.6) + 0.1 - t) / 0.08, 0, 1)
    x *= env
    x += bp(noise(n, seed), 2000, 12000, sr) * np.exp(-t / 0.004) * 0.12
    x = bp(x, wah * 0.45, min(wah * 2.4, 12000), sr, order=2)
    x = peak(x, wah, 4.0, 1.6, sr)
    x = saturate(x, 1.5, mix=0.5)
    x = fade_edges(x, 0.2, 5)
    return (x / (np.max(np.abs(x)) + 1e-9) * 0.45 * vel).astype(np.float32)


def organ_note(midi, dur_s, vel=0.6, sr=SR, seed=131, bars=(1.0, 2.0, 3.0, 4.0, 6.0)):
    """Drawbar organ tone with key click and gentle chorus-ready drift."""
    n = int((dur_s + 0.3) * sr)
    t = np.arange(n) / sr
    f = midi_hz(midi)
    x = np.zeros(n)
    weights = [1.0, 0.62, 0.45, 0.30, 0.18, 0.12]
    for w, b in zip(weights, bars):
        x += w * np.sin(TWO_PI * f * b * t + 0.3 * b)
    # key click
    click = bp(noise(n, seed), 1500, 8000, sr) * np.exp(-t / 0.0035) * 0.25
    env = (1 - np.exp(-t / 0.012)) * np.clip((dur_s + 0.25 - t) / 0.25, 0, 1) ** 1.1
    x = (x + click) * env
    x = saturate(x, 1.3, mix=0.35)
    x = lp(x, 7000, sr)
    return (x / (np.max(np.abs(x)) + 1e-9) * 0.34 * vel).astype(np.float32)


# =========================================================== HORNS / PADS
def brass_chord(notes, dur_s, vel=0.7, sr=SR, seed=141, bright=1.0):
    """Small horn section: saw ensemble with scoop, filter swell and vibrato."""
    n = int((dur_s + 0.45) * sr)
    t = np.arange(n) / sr
    out = np.zeros((n, 2), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for vi, m in enumerate(notes):
        f0 = midi_hz(m)
        voice = np.zeros(n)
        for u in range(3):
            cents = (u - 1) * 7.0 + rng.standard_normal() * 2.0
            f = f0 * 2 ** (cents / 1200)
            scoop = 2 ** (-0.5 * np.exp(-t / 0.035) / 12)  # fall into pitch
            ph = TWO_PI * np.cumsum(f * scoop) / sr
            voice += 2 * (ph % 1.0) - 1.0 - polyblep(ph, f / sr)
        voice /= 3
        # filter envelope: open on the attack, settle back
        cut = 1400 * bright + 3200 * bright * np.exp(-t / 0.14)
        voice = np.array([np.tanh(v * 1.4) for v in np.zeros(1)]) if False else voice
        voice = _blocks_lp(voice, cut, sr)
        vib = 1 + 0.006 * np.sin(TWO_PI * 5.4 * t + vi) * np.clip((t - 0.15) / 0.2, 0, 1)
        env = (1 - np.exp(-t / 0.02)) * np.clip((dur_s + 0.4 - t) / 0.4, 0, 1) ** 1.2
        voice *= env * vib * vel * 0.35
        pan = (vi / max(1, len(notes) - 1)) * 1.2 - 0.6
        l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        out[:, 0] += (voice * l).astype(np.float32)
        out[:, 1] += (voice * r).astype(np.float32)
    out = peak(out, 900, 2.0, 1.0, sr)
    out = saturate(out, 1.2, mix=0.3)
    m = np.max(np.abs(out)) + 1e-9
    return (out / m * 0.55).astype(np.float32)


def _blocks_lp(x, cutoff, sr=SR, block=128, order=2):
    y = np.empty_like(x)
    for i in range(0, len(x), block):
        fc = float(np.clip(cutoff[min(i, len(cutoff) - 1)], 80, sr * 0.45))
        sos = signal.butter(order, fc, btype="low", fs=sr, output="sos")
        y[i:i + block] = signal.sosfilt(sos, x[i:i + block])
    return y


def pad_chord(notes, dur_s, vel=0.55, sr=SR, seed=151, attack=0.9, release=1.2,
              cutoff=2600, detune=9.0):
    n = int((dur_s + release) * sr)
    t = np.arange(n) / sr
    out = np.zeros((n, 2), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for vi, m in enumerate(notes):
        f0 = midi_hz(m)
        voice = np.zeros(n)
        for u in range(4):
            cents = (u - 1.5) * detune + rng.standard_normal() * 1.5
            f = f0 * 2 ** (cents / 1200)
            ph = TWO_PI * f * t + rng.random() * 6.28
            voice += (2 * ((ph / TWO_PI) % 1.0) - 1.0)
        voice /= 4
        env = (1 - np.exp(-t / attack)) * np.clip((dur_s + release - t) / release, 0, 1) ** 1.3
        voice *= env * vel * 0.30
        pan = (vi / max(1, len(notes) - 1)) * 1.4 - 0.7
        l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        out[:, 0] += (voice * l).astype(np.float32)
        out[:, 1] += (voice * r).astype(np.float32)
    out = lp(out, cutoff, sr)
    out = np.stack([dsp_hp(out[:, 0], 120, sr), dsp_hp(out[:, 1], 120, sr)], axis=1)
    m = np.max(np.abs(out)) + 1e-9
    return (out / m * 0.40).astype(np.float32)


def dsp_hp(x, fc, sr=SR):
    return hp(x, fc, sr)


def bell_note(midi, dur_s=1.2, vel=0.7, sr=SR, seed=161, ratio=3.01):
    """Bright FM bell/pluck used for trap melodies."""
    n = int((dur_s + 0.35) * sr)
    t = np.arange(n) / sr
    f = midi_hz(midi)
    ph = TWO_PI * f * t
    I = (1.2 + 2.2 * vel) * np.exp(-t / 0.16) + 0.10
    y = np.sin(ph + I * np.sin(ratio * ph))
    y += 0.22 * np.sin(2 * ph) * np.exp(-t / 0.3)
    env = (1 - np.exp(-t / 0.002)) * np.exp(-t / (0.55 + 0.2 * vel))
    gate = np.clip((dur_s + 0.3 - t) / 0.25, 0, 1) ** 1.2
    y = y * env * gate
    y = lp(y, 9000, sr)
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.33 * vel).astype(np.float32)


def flute_note(midi, dur_s, vel=0.6, sr=SR, seed=171):
    n = int((dur_s + 0.3) * sr)
    t = np.arange(n) / sr
    f = midi_hz(midi)
    vib = 1 + 0.009 * np.sin(TWO_PI * 5.0 * t) * np.clip((t - 0.15) / 0.25, 0, 1)
    ph = TWO_PI * np.cumsum(f * vib) / sr
    tone = np.sin(ph) + 0.22 * np.sin(2 * ph) + 0.06 * np.sin(3 * ph)
    breath = bp(noise(n, seed), 1800, 7000, sr) * 0.9
    env = (1 - np.exp(-t / 0.035)) * np.clip((dur_s + 0.25 - t) / 0.25, 0, 1) ** 1.2
    y = (tone + breath * 0.16) * env
    y = peak(y, 1200, 2.5, 0.9, sr)
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.30 * vel).astype(np.float32)


def snap(vel=1.0, sr=SR, seed=181):
    n = int(0.12 * sr)
    t = np.arange(n) / sr
    x = bp(noise(n, seed), 1400, 7000, sr) * np.exp(-t / 0.012)
    x += bp(noise(n, seed + 1), 2500, 9000, sr) * np.exp(-t / 0.005) * 0.7
    x = peak(x, 2600, 3.0, 1.2, sr)
    x = fade_edges(x, 0.2, 4)
    return (x / (np.max(np.abs(x)) + 1e-9) * 0.6 * vel).astype(np.float32)


# =========================================================== 808
def sub808_note(midi, dur_s, vel=0.9, glide_from=None, glide_s=0.09, sr=SR,
                drive=1.6, tail=0.5):
    """808 sub: pure sine, pitch glide, long decay, gentle saturation."""
    n = int((dur_s + tail) * sr)
    t = np.arange(n) / sr
    f0 = midi_hz(midi)
    if glide_from is not None:
        fs0 = midi_hz(glide_from)
        g = np.clip(t / max(glide_s, 1e-3), 0, 1)
        g = g * g * (3 - 2 * g)
        f = fs0 * (f0 / fs0) ** g
    else:
        f = np.full(n, f0) * (1 + 0.35 * np.exp(-t / 0.02))
    ph = TWO_PI * np.cumsum(f) / sr
    decay = 0.9 + 0.9 * dur_s
    y = np.sin(ph) * np.exp(-t / decay)
    y += 0.12 * np.sin(2 * ph) * np.exp(-t / 0.16)
    env = (1 - np.exp(-t / 0.004)) * np.clip((dur_s + tail - t) / tail, 0, 1) ** 1.1
    y *= env
    click = bp(noise(n, 191), 1500, 6000, sr) * np.exp(-t / 0.004) * 0.06
    y = y + click
    y = np.tanh(y * drive) / np.tanh(drive)
    y = lp(y, 1800, sr)
    y = hp(y, 22, sr)
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.92 * vel).astype(np.float32)


# =========================================================== VOCALISE
VOWELS = {
    # (center Hz, Q, gain) x 4 formants  — tuned by ear for a neutral pop voice
    "oo":  [(300, 5.5, 1.00), (790, 6.5, 0.66), (2300, 8.0, 0.30), (3250, 9.0, 0.14)],
    "oh":  [(440, 5.0, 1.00), (820, 6.0, 0.72), (2450, 7.0, 0.38), (3350, 8.0, 0.16)],
    "ah":  [(760, 4.6, 1.00), (1140, 5.4, 0.78), (2500, 6.6, 0.42), (3500, 8.0, 0.18)],
    "eh":  [(540, 5.0, 1.00), (1750, 5.8, 0.62), (2480, 7.0, 0.36), (3400, 8.5, 0.15)],
    "ee":  [(290, 5.2, 1.00), (2180, 6.0, 0.55), (2920, 7.5, 0.32), (3700, 9.0, 0.15)],
    "mmm": [(270, 4.0, 1.00), (1050, 9.0, 0.14), (2400, 11.0, 0.06), (3200, 12.0, 0.03)],
    "doo": [(315, 5.2, 1.00), (860, 6.2, 0.60), (2350, 8.0, 0.28), (3300, 9.0, 0.13)],
}
CONSONANT = {"d": 1.0, "h": 0.5, "": 0.0}


def _glottal(n, f_curve, sr, seed, breath=0.10, jitter=0.004):
    """Band-limited glottal-ish source: soft saw with jitter + breath."""
    rng = np.random.default_rng(seed)
    # small random pitch jitter for a human edge (low-passed noise)
    j = signal.lfilter([0.02], [1, -0.98], rng.standard_normal(n)) * jitter
    f = f_curve * (1 + j)
    ph = np.cumsum(f) / sr
    saw = 2 * (ph % 1.0) - 1.0 - polyblep(ph, f / sr)
    # soften upper harmonics like a vocal fold pulse
    saw = lp(saw, 5200, sr, order=2)
    br = bp(rng.standard_normal(n), 900, 6800, sr) * breath
    return (saw + br).astype(np.float32)


def _formants(x, vowel, sr=SR, scale=1.0, shift=1.0):
    y = np.zeros_like(x)
    for fc, q, g in VOWELS[vowel]:
        c = fc * scale
        y += bp(x, max(c * 2 ** (-1.2 / q), 60), min(c * 2 ** (1.2 / q), sr * 0.45),
                sr, order=2) * g
    y += lp(x, 260, sr) * 0.30          # body / chest
    y += bp(x, 3200 * shift, 7600, sr) * 0.10   # air / presence
    return y


def vox_note(midi, dur_s, vel=0.6, vowel="oo", sr=SR, seed=201, legato_from=None,
             legato_s=0.07, bend=0.0, vib=0.13, consonant="", breath=0.10,
             scoop=True, attack=0.035, release=0.10, detune_cents=0.0,
             formant_scale=1.0):
    """One wordless vocal note. Returns mono float32 at performance level."""
    f0 = midi_hz(midi)
    n = int((dur_s + release + 0.05) * sr)
    t = np.arange(n) / sr
    # pitch contour
    if legato_from is not None:
        f_start = midi_hz(legato_from)
        g = np.clip(t / max(legato_s, 1e-3), 0, 1)
        g = g * g * (3 - 2 * g)
        f = f_start * (f0 / f_start) ** g
    elif scoop:
        s = np.clip(t / 0.05, 0, 1)
        f = f0 * 2 ** (-0.5 * (1 - s) / 12)
    else:
        f = np.full(n, f0)
    if bend:
        b = np.clip((t - dur_s * 0.7) / 0.18, 0, 1)
        f = f * 2 ** (bend * b / 12)
    vo = np.clip((t - 0.12) / 0.20, 0, 1)
    f = f * 2 ** (vib * vo * np.sin(TWO_PI * 5.2 * t + seed % 7) / 12)
    f = f * 2 ** (detune_cents / 1200.0)
    src = _glottal(n, f, sr, seed, breath=breath)
    # vowel filter with a small time-varying formant drift (singer movement)
    drift = 1.0 + 0.012 * np.sin(TWO_PI * 0.7 * t + 1.3)
    # process in 4 blocks with crossfades for the drift
    nseg = 4
    y = np.zeros(n, dtype=np.float32)
    for i in range(nseg):
        a = int(i * n / nseg); b = int((i + 1) * n / nseg)
        seg = _formants(src[a:b], vowel, sr, scale=formant_scale * drift[(a + b) // 2])
        w = np.ones(b - a, dtype=np.float32)
        xf = min(240, (b - a) // 2)
        if i > 0:
            w[:xf] = np.linspace(0, 1, xf)
        if i < nseg - 1:
            w[-xf:] = np.linspace(1, 0, xf)
        y[a:b] += seg * w
    # amplitude envelope
    att = attack
    env = (1 - np.exp(-t / att))
    env *= np.clip((dur_s + release - t) / release, 0, 1) ** 1.1
    # consonant transient
    if consonant in ("d", "b", "g"):
        burst = bp(noise(n, seed + 1), 700, 4500, sr) * np.exp(-t / 0.010) * 0.22
        y += burst
    elif consonant == "h":
        y += bp(noise(n, seed + 2), 1200, 7000, sr) * np.exp(-t / 0.045) * 0.10
    y = y * env * vel
    y = saturate(y, 1.25, mix=0.28)
    y = lp(y, 11000, sr)
    y = hp(y, 110, sr)
    y = fade_edges(y, 1.5, 8)
    return y.astype(np.float32)


def vox_voice(events, sr=SR, seed=301, spread=0.0):
    """Render a monophonic vocal line (list of event dicts) with legato.
    Returns stereo with slight double-track width."""
    if not events:
        return np.zeros((1, 2), dtype=np.float32)
    end = max(e["t"] + e["dur"] for e in events) + 1.6
    n = int(end * sr)
    mono = np.zeros(n, dtype=np.float32)
    evs = sorted(events, key=lambda e: e["t"])
    prev = None
    for i, e in enumerate(evs):
        legato = e.get("legato", False) and prev is not None
        y = vox_note(e["midi"], e["dur"], vel=e["vel"], vowel=e.get("vowel", "oo"),
                     sr=sr, seed=seed + i * 17,
                     legato_from=(prev["midi"] if legato and "glide_to" in e and e["glide_to"] else None),
                     legato_s=e.get("legato_s", 0.075),
                     bend=e.get("bend", 0.0), vib=e.get("vib", 0.13),
                     consonant=e.get("consonant", ""), breath=e.get("breath", 0.10),
                     attack=e.get("attack", 0.035), release=e.get("release", 0.12),
                     detune_cents=e.get("detune", 0.0),
                     formant_scale=e.get("formant_scale", 1.0))
        i0 = int(e["t"] * sr)
        m = min(len(y), n - i0)
        mono[i0:i0 + m] += y[:m]
        prev = e
    if spread > 0:
        d = int(spread * sr)
        left = mono.copy()
        right = np.concatenate([np.zeros(d, dtype=np.float32), mono[:-d]]) if d > 0 else mono
        st = np.stack([left * 0.95 + right * 0.12, right * 0.95 + left * 0.12], axis=1)
    else:
        st = np.stack([mono, mono], axis=1)
    return st.astype(np.float32)


def vox_choir(events_by_voice, sr=SR, seed=401, pan_spread=0.55):
    """Stack 2-4 independent vocal lines (harmonies) with panning."""
    if not events_by_voice:
        return np.zeros((1, 2), dtype=np.float32)
    end = max(max(e["t"] + e["dur"] for e in evs) + 1.8 for evs in events_by_voice if evs)
    n = int(end * sr)
    out = np.zeros((n, 2), dtype=np.float32)
    k = len(events_by_voice)
    for vi, evs in enumerate(events_by_voice):
        st = vox_voice(evs, sr=sr, seed=seed + vi * 97, spread=0.009)
        pan = pan_spread * ((vi / max(1, k - 1)) * 2 - 1) if k > 1 else 0.0
        l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        m = min(len(st), n)
        out[:m, 0] += st[:m, 0] * l
        out[:m, 1] += st[:m, 1] * r
    m = np.max(np.abs(out)) + 1e-9
    return (out / m * 0.6).astype(np.float32)
