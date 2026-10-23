"""
dsp.py — signal-processing toolbox for the Western Ave Sundown session.
Everything is synthesized from first principles (no sample libraries), rendered
at 48 kHz / 32-bit float internally, delivered as 24-bit PCM.
"""
from __future__ import annotations
import numpy as np
from scipy import signal
import scipy.ndimage as ndi

SR = 48000
TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------- basics
def tvec(n: int, sr: int = SR) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / sr


def silence(n: int) -> np.ndarray:
    return np.zeros(max(0, int(n)), dtype=np.float32)


def db(x: float) -> float:
    return 10.0 ** (x / 20.0)


def midi_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


def polyblep(phase: np.ndarray, dt: float) -> np.ndarray:
    """PolyBLEP residual for band-limited saw/square edges."""
    p = phase % 1.0
    dt = np.broadcast_to(np.asarray(dt, dtype=np.float64), p.shape)
    out = np.zeros_like(p)
    m = p < dt
    x = p[m] / dt[m]
    out[m] = x + x - x * x - 1.0
    m2 = p > 1.0 - dt
    x2 = (p[m2] - 1.0) / dt[m2]
    out[m2] = x2 * x2 + x2 + x2 + 1.0
    return out


def osc_saw(freq, n: int, sr: int = SR, phase0: float = 0.0,
            unison: int = 1, detune_cents: float = 0.0, seed: int = 0) -> np.ndarray:
    """Band-limited saw; unison voices spread symmetrically around the center pitch."""
    if np.isscalar(freq):
        f = np.full(n, float(freq))
    else:
        f = np.asarray(freq, dtype=np.float64)
    dt = f / sr
    phase = np.cumsum(dt) + phase0
    out = np.zeros(n)
    rng = np.random.default_rng(seed)
    for u in range(unison):
        if unison == 1:
            cents = 0.0
        else:
            cents = detune_cents * (2.0 * u / (unison - 1) - 1.0)
        drift = 1.0 + cents / 1200.0
        p = phase * drift + rng.random()
        s = 2.0 * (p % 1.0) - 1.0 - polyblep(p, dt * drift)
        out += s
    return out / unison


def osc_square(freq, n: int, sr: int = SR, phase0: float = 0.0,
               pulse: float = 0.5) -> np.ndarray:
    if np.isscalar(freq):
        f = np.full(n, float(freq))
    else:
        f = np.asarray(freq, dtype=np.float64)
    dt = f / sr
    phase = np.cumsum(dt) + phase0
    p = phase % 1.0
    s = np.where(p < pulse, 1.0, -1.0)
    s += polyblep(phase, dt) - polyblep(phase + (1.0 - pulse), dt)
    return s


def osc_sine(freq, n: int, sr: int = SR, phase0: float = 0.0) -> np.ndarray:
    if np.isscalar(freq):
        ph = TWO_PI * float(freq) * np.arange(n) / sr + phase0
    else:
        ph = TWO_PI * np.cumsum(np.asarray(freq, dtype=np.float64)) / sr + phase0
    return np.sin(ph)


def noise(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)


# ---------------------------------------------------------------- envelopes
def env_exp(n: int, attack_s: float, decay_s: float, sr: int = SR,
            curve: float = 1.0, sustain_level: float = 0.0) -> np.ndarray:
    t = tvec(n, sr)
    a = max(attack_s, 1.0 / sr)
    e = np.exp(-np.maximum(t - a, 0.0) / max(decay_s, 1e-4))
    e = e * (1.0 - np.exp(-t / a))
    if curve != 1.0:
        e = e ** curve
    if sustain_level > 0:
        e = sustain_level + (1.0 - sustain_level) * e
    return e


def env_adsr(n: int, a: float, d: float, s: float, r: float,
             sr: int = SR) -> np.ndarray:
    """Vectorized ADSR; r is the release time appended after the sustain span."""
    t = tvec(n, sr)
    total = n / sr
    e = np.zeros(n)
    # attack
    ma = t < a
    e[ma] = t[ma] / max(a, 1e-5)
    # decay
    md = (t >= a) & (t < a + d)
    if a + d > a:
        e[md] = 1.0 - (1.0 - s) * ((t[md] - a) / max(d, 1e-5))
    # sustain
    ms = (t >= a + d) & (t < max(total - r, a + d))
    e[ms] = s
    # release
    mr = t >= max(total - r, a + d)
    start = max(total - r, a + d)
    e[mr] = s * np.clip(1.0 - (t[mr] - start) / max(r, 1e-5), 0.0, 1.0)
    return np.clip(e, 0.0, 1.0)


def fade_edges(x: np.ndarray, ms_in: float = 2.0, ms_out: float = 6.0,
               sr: int = SR) -> np.ndarray:
    n = len(x)
    a = min(n, max(1, int(ms_in * sr / 1000)))
    b = min(n, max(1, int(ms_out * sr / 1000)))
    fi = np.linspace(0, 1, a, dtype=x.dtype) ** 0.7
    fo = np.linspace(1, 0, b, dtype=x.dtype) ** 1.2
    if x.ndim == 2:
        fi = fi[:, None]
        fo = fo[:, None]
    x[:a] *= fi
    x[-b:] *= fo
    return x


# ---------------------------------------------------------------- filters
def lp(x, fc, sr=SR, order=2, q=0.707):
    fc = min(max(fc, 10.0), sr * 0.49)
    sos = signal.butter(order, fc, btype="low", fs=sr, output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)


def hp(x, fc, sr=SR, order=2, q=0.707):
    fc = min(max(fc, 5.0), sr * 0.49)
    sos = signal.butter(order, fc, btype="high", fs=sr, output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)


def bp(x, lo, hi, sr=SR, order=2):
    lo = max(lo, 5.0)
    hi = min(hi, sr * 0.49)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    sos = signal.butter(order, [lo, hi], btype="band", fs=sr, output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)


def peak(x, fc, gain_db, q=1.0, sr=SR):
    A = db(gain_db)
    w0 = TWO_PI * fc / sr
    alpha = np.sin(w0) / (2 * q)
    b = [1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A]
    a = [1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A]
    return signal.lfilter(b, a, x).astype(np.float32)


def low_shelf(x, fc, gain_db, q=0.707, sr=SR):
    A = db(gain_db)
    w0 = TWO_PI * fc / sr
    alpha = np.sin(w0) / (2 * q)
    c = np.cos(w0)
    sq = 2 * np.sqrt(A) * alpha
    b = [A * ((A + 1) - (A - 1) * c + sq),
         2 * A * ((A - 1) - (A + 1) * c),
         A * ((A + 1) - (A - 1) * c - sq)]
    a = [(A + 1) + (A - 1) * c + sq,
         -2 * ((A - 1) + (A + 1) * c),
         (A + 1) + (A - 1) * c - sq]
    return signal.lfilter(b, a, x).astype(np.float32)


def high_shelf(x, fc, gain_db, q=0.707, sr=SR):
    A = db(gain_db)
    w0 = TWO_PI * fc / sr
    alpha = np.sin(w0) / (2 * q)
    c = np.cos(w0)
    sq = 2 * np.sqrt(A) * alpha
    b = [A * ((A + 1) + (A - 1) * c + sq),
         -2 * A * ((A - 1) + (A + 1) * c),
         A * ((A + 1) + (A - 1) * c - sq)]
    a = [(A + 1) - (A - 1) * c + sq,
         2 * ((A - 1) - (A + 1) * c),
         (A + 1) - (A - 1) * c - sq]
    return signal.lfilter(b, a, x).astype(np.float32)


def time_varying_lp(x, cutoff_curve, sr=SR, block=256, order=2):
    """Piecewise-constant low-pass sweep (click-free enough for musical FX)."""
    y = np.empty_like(x)
    n = len(x)
    for i in range(0, n, block):
        fc = float(cutoff_curve[min(i, len(cutoff_curve) - 1)])
        seg = x[i:i + block]
        sos = signal.butter(order, min(max(fc, 30), sr * 0.49), btype="low",
                            fs=sr, output="sos")
        zi = signal.sosfilt_zi(sos) * 0.0
        y[i:i + block] = signal.sosfilt(sos, seg, zi=zi)
    return y


def time_varying_bp(x, center_curve, bandwidth_oct=1.6, sr=SR, block=256, order=2):
    y = np.empty_like(x)
    n = len(x)
    for i in range(0, n, block):
        fc = float(center_curve[min(i, len(center_curve) - 1)])
        fc = min(max(fc, 60), sr * 0.45)
        lo = max(fc * 2 ** (-bandwidth_oct / 2), 20)
        hi = min(fc * 2 ** (bandwidth_oct / 2), sr * 0.49)
        seg = x[i:i + block]
        sos = signal.butter(max(1, order - 1), [lo, hi], btype="band",
                            fs=sr, output="sos")
        y[i:i + block] = signal.sosfilt(sos, seg)
    return y


# ---------------------------------------------------------------- character
def saturate(x, drive=1.5, asym=0.0, mix=1.0):
    d = max(drive, 1e-6)
    if asym:
        y = np.tanh(d * x + asym) - np.tanh(asym)
    else:
        y = np.tanh(d * x)
    y = y / np.tanh(d)
    return ((1.0 - mix) * x + mix * y).astype(np.float32)


def tape_sat(x, drive=1.2, hf_soften=0.25):
    """Soft odd-harmonic drive with gentle high-loss, like a pushed 1/4-inch."""
    y = np.tanh(drive * x) / np.tanh(drive)
    if hf_soften > 0:
        y = (1 - hf_soften) * y + hf_soften * lp(y, 6500)
    return y.astype(np.float32)


def dc_block(x):
    return signal.lfilter([1, -1], [1, -0.9995], x).astype(np.float32)


# ---------------------------------------------------------------- stereo
def pan_gains(pan: float):
    """constant-power pan, pan in [-1, 1]"""
    p = np.clip(pan, -1, 1)
    ang = (p + 1) * np.pi / 4
    return np.cos(ang), np.sin(ang)


def to_stereo(x, pan=0.0, width=1.0):
    l, r = pan_gains(pan)
    if x.ndim == 2:
        return x.astype(np.float32)
    y = np.stack([x * l, x * r], axis=1).astype(np.float32)
    if width != 1.0:
        y = widen(y, width)
    return y


def widen(x, width=1.2):
    m = 0.5 * (x[:, 0] + x[:, 1])
    s = 0.5 * (x[:, 0] - x[:, 1]) * width
    return np.stack([m + s, m - s], axis=1).astype(np.float32)


def mono_below(x, fc=120, sr=SR):
    """Collapse low frequency content to mono."""
    low = np.stack([lp(x[:, 0], fc, sr), lp(x[:, 1], fc, sr)], axis=1)
    mid = (low[:, 0] + low[:, 1]) * 0.5
    high = x - low
    return np.stack([high[:, 0] + mid, high[:, 1] + mid], axis=1).astype(np.float32)


def stereo_delay(x, delay_samples: int, fb=0.3, mix=0.25, pingpong=True):
    """Simple feedback delay with damped repeats."""
    n = len(x)
    d = max(1, int(delay_samples))
    y = np.zeros_like(x)
    # process in blocks via lfilter on each channel with a feedback comb
    b = np.zeros(d + 1); b[0] = 1.0
    a = np.zeros(d + 1); a[0] = 1.0
    l_ch = signal.lfilter(b, [1.0] + [0.0] * (d - 1) + [-fb], x[:, 0])
    r_in = x[:, 1] if pingpong else x[:, 1]
    r_ch = signal.lfilter(b, [1.0] + [0.0] * (d - 1) + [-fb],
                          np.roll(l_ch, 0) * 0 + x[:, 1])
    wet = np.stack([l_ch, r_ch], axis=1)
    wet[:, 0] = lp(wet[:, 0], 3800)
    wet[:, 1] = lp(wet[:, 1], 3200)
    return ((1 - mix) * x + mix * wet).astype(np.float32)


def haas(x, ms=12.0, sr=SR, side="right"):
    d = int(ms * sr / 1000)
    y = np.array(x, copy=True)
    if side == "right":
        y[d:, 1] = x[:-d, 1]
    else:
        y[d:, 0] = x[:-d, 0]
    return y


# ---------------------------------------------------------------- reverb
def make_ir(sr=SR, rt60=2.1, predelay_ms=18, bright=0.55, seed=7, size=1.0):
    """Stereo plate-ish impulse: decorrelated noise with frequency-dependent decay."""
    n = int(sr * rt60 * 1.15)
    rng = np.random.default_rng(seed)
    tt = np.arange(n) / sr
    # high frequencies decay faster than lows (air absorption)
    lo = rng.standard_normal(n) * np.exp(-tt / (rt60 / 3.2))
    hi = rng.standard_normal(n) * np.exp(-tt / (rt60 / 6.5))
    lo = lp(lo, 2200, sr)
    hi = hp(hi, 2200, sr)
    base = lo * 1.0 + hi * bright
    # sparse early reflections
    early = np.zeros(n)
    taps = [0.007, 0.011, 0.017, 0.023, 0.031, 0.043, 0.057] 
    for i, tp in enumerate(taps):
        idx = int(tp * sr * size)
        if idx < n:
            early[idx] += (0.55 ** i) * (1 if i % 2 == 0 else -1)
    early = lp(early, 6000, sr) * 0.5
    out = []
    for ch in range(2):
        nz = rng.standard_normal(n) * np.exp(-tt / (rt60 / 3.6))
        nz = lp(nz, 5200, sr) * 0.6 + nz * 0.4
        conv = base * (0.85 if ch == 0 else 0.88) + nz * 0.35 + early * (1.0 if ch == 0 else 0.9)
        conv[: int(predelay_ms * sr / 1000)] = 0.0
        out.append(conv)
    ir = np.stack(out, axis=1)
    ir /= np.max(np.abs(ir)) + 1e-9
    return ir.astype(np.float32)


def convolve_stereo(x, ir):
    """FFT convolution of a stereo bus with a stereo IR (tail included)."""
    import scipy.signal as sg
    out = np.zeros((len(x) + len(ir) - 1, 2), dtype=np.float32)
    for ch in range(2):
        out[:, ch] = sg.oaconvolve(x[:, ch].astype(np.float32),
                                   ir[:, ch].astype(np.float32), mode="full")
    return out


# ---------------------------------------------------------------- dynamics
def _smooth_env(x, tau_s, sr=SR):
    a = np.exp(-1.0 / (max(tau_s, 1e-5) * sr))
    return signal.lfilter([1 - a], [1, -a], np.abs(x))


def compressor(x, thr_db=-18.0, ratio=2.5, attack_ms=25.0, release_ms=180.0,
               knee_db=6.0, makeup_db=0.0, sr=SR, sc=None):
    """Feed-forward compressor with soft knee; stereo-linked via max(|L|,|R|)."""
    mono_sc = np.max(np.abs(x), axis=1) if sc is None else np.abs(sc)
    env = _smooth_env(mono_sc, attack_ms / 1000.0, sr)
    env = _smooth_env(env, release_ms / 1000.0, sr)
    level = 20 * np.log10(np.maximum(env, 1e-7))
    over = level - thr_db
    # soft knee curve
    gain_db = np.zeros_like(level)
    lo = over < -knee_db / 2
    hi = over > knee_db / 2
    mid = ~(lo | hi)
    gain_db[hi] = -(over[hi]) * (1 - 1 / ratio)
    gain_db[mid] = -(1 - 1 / ratio) * ((over[mid] + knee_db / 2) ** 2) / (2 * knee_db)
    g = (10 ** ((gain_db + makeup_db) / 20.0)).astype(np.float32)
    # smooth gain movement to avoid zipper
    g = _smooth_env(g, 0.002, sr) * 0  + signal.lfilter([0.08], [1, -0.92], g)
    g = np.clip(g, 1e-4, 8.0).astype(np.float32)
    return (x * g[:, None]).astype(np.float32)


def os_peak_per_sample(x, os=2, chunk=1 << 20, sr=SR):
    """Per-sample max of an os-times oversampled peak detector (memory-light)."""
    n = len(x)
    out = np.zeros(n, dtype=np.float32)
    pad = 64
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        a = max(0, start - pad)
        b = min(n, end + pad)
        up = signal.resample_poly(x[a:b], os, 1, axis=0)
        base = (start - a) * os
        seg = up[base:base + (end - start) * os]
        if seg.shape[0] < (end - start) * os:
            seg = np.pad(seg, ((0, (end - start) * os - seg.shape[0]), (0, 0)))
        out[start:end] = np.max(np.abs(seg), axis=1).reshape(end - start, os).max(axis=1)
    return out


def limit(x, ceiling_db=-1.0, lookahead_ms=3.0, release_ms=120.0, sr=SR,
          soft_clip=True):
    """True-peak look-ahead limiter (2x oversampled detection, delayed gain)."""
    ceiling = db(ceiling_db)
    n = len(x)
    la = max(2, int(lookahead_ms * sr / 1000))
    pk = np.maximum(os_peak_per_sample(x, os=2), np.max(np.abs(x), axis=1))
    w = ndi.maximum_filter1d(pk, size=la + 1, mode="nearest")
    shift = la // 2
    w_future = np.concatenate([w[shift:], np.full(shift, w[-1])])
    req = np.minimum(1.0, ceiling / np.maximum(w_future, 1e-9))
    att = 1.0 / max(1, int(0.35 * la))
    g = signal.lfilter([att], [1, -(1 - att)], req)
    rel = np.exp(-1.0 / (release_ms / 1000.0 * sr))
    g = signal.lfilter([1 - rel], [1, -rel], g)
    g = np.minimum(g, req)
    g_app = np.concatenate([np.ones(la), g[:-la]])
    y = (x * g_app[:, None]).astype(np.float32)
    if soft_clip:
        knee = 0.93 * ceiling
        a = np.abs(y)
        over = a > knee
        if np.any(over):
            sign = np.sign(y)
            soft = knee + (ceiling - knee) * np.tanh((a - knee) / (ceiling - knee))
            y = np.where(over, sign * soft, y).astype(np.float32)
    return y.astype(np.float32)


# ---------------------------------------------------------------- loudness
def k_weight(x, sr=SR):
    if sr != 48000:
        raise ValueError("K-weighting coefficients are for 48 kHz")
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, -1.99004745483398, 0.99007225036621]
    y = signal.lfilter(b1, a1, x, axis=0)
    y = signal.lfilter(b2, a2, y, axis=0)
    return y


def lufs_integrated(x, sr=SR):
    if x.ndim == 1:
        x = x[:, None]
    y = k_weight(x.astype(np.float64), sr)
    block = int(0.4 * sr)
    step = int(0.1 * sr)
    n = len(y)
    if n < block:
        return -np.inf
    starts = np.arange(0, n - block + 1, step)
    ms = np.array([np.mean(y[s:s + block] ** 2, axis=0) for s in starts])
    # channel weights: L/R = 1
    pw = ms.sum(axis=1)
    with np.errstate(divide="ignore"):
        lk = -0.691 + 10 * np.log10(np.maximum(pw, 1e-20))
    gated = lk[lk > -70.0]
    if len(gated) == 0:
        return -np.inf
    rel = np.mean(10 ** (gated / 10))
    thresh = -0.691 + 10 * np.log10(rel) - 10.0
    gated2 = gated[gated > thresh]
    return float(-0.691 + 10 * np.log10(np.mean(10 ** (gated2 / 10))))


def true_peak_db(x, sr=SR, os=4):
    pk = os_peak_per_sample(x, os=os)
    return float(20 * np.log10(np.max(pk) + 1e-12))


def sample_peak_db(x):
    return float(20 * np.log10(np.max(np.abs(x)) + 1e-12))


def crest_db(x):
    rms = np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12)
    return float(20 * np.log10(np.max(np.abs(x)) / rms))


def stereo_correlation(x):
    l = x[:, 0].astype(np.float64); r = x[:, 1].astype(np.float64)
    return float(np.corrcoef(l, r)[0, 1])


def band_energy(x, sr=SR, bands=((20, 60), (60, 120), (120, 250), (250, 500),
                                  (500, 1000), (1000, 2000), (2000, 4000),
                                  (4000, 8000), (8000, 16000), (16000, 20000))):
    mono = x.mean(axis=1) if x.ndim == 2 else x
    f, p = signal.welch(mono.astype(np.float64), sr, nperseg=8192)
    total = np.sum(p) + 1e-20
    out = {}
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        out[f"{lo}-{hi}"] = float(10 * np.log10(np.sum(p[m]) / total + 1e-20))
    return out


# ---------------------------------------------------------------- modulation
def tremolo_stereo(x, rate=5.2, depth=0.16, sr=SR, phase_offset=180.0):
    n = len(x)
    t = np.arange(n) / sr
    base = 1.0 - depth / 2.0
    lfo_l = base + depth / 2.0 * np.sin(TWO_PI * rate * t)
    lfo_r = base + depth / 2.0 * np.sin(TWO_PI * rate * t + np.deg2rad(phase_offset))
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] * lfo_l
    y[:, 1] = x[:, 1] * lfo_r
    return y.astype(np.float32)


def chorus(x, rate=0.32, depth_ms=7.0, mix=0.22, sr=SR):
    """Two modulated delay taps (different LFO phase per channel)."""
    n = len(x)
    t = np.arange(n) / sr
    out = np.empty_like(x)
    for ch in range(x.shape[1]):
        ph = 0.0 if ch == 0 else np.pi * 0.7
        mod = (depth_ms * sr / 1000.0) * (0.5 + 0.5 * np.sin(TWO_PI * rate * t + ph))
        delay = 0.012 * sr + mod
        idx = np.arange(n) - delay
        wet = np.interp(idx, np.arange(n), x[:, ch], left=0.0, right=0.0)
        out[:, ch] = (1 - mix) * x[:, ch] + mix * wet
    return out.astype(np.float32)


def comb_feedback(x, d, fb):
    """y[n] = x[n] + fb*y[n-d] in O(n) (block-recursive, no long lfilter)."""
    y = np.array(x, dtype=np.float32, copy=True)
    n = len(y)
    for i in range(d, n, d):
        m = min(d, n - i)
        y[i:i + m] += fb * y[i - d:i - d + m]
    return y


def pingpong_delay(x, delay_ms=242.0, fb=0.30, mix=0.22, sr=SR, damp=True):
    """Feedback delay with alternating emphasis and damped repeats."""
    d = max(1, int(delay_ms * sr / 1000.0))
    wl = comb_feedback(x[:, 0], d, fb)
    wr = comb_feedback(x[:, 1], d, fb)
    half = d // 2
    if half:
        wr = np.concatenate([np.zeros(half, dtype=np.float32), wr[:-half]])
    wet = np.stack([wl * 0.9 + wr * 0.35, wr * 0.9 + wl * 0.35], axis=1)
    if damp:
        wet = np.stack([lp(wet[:, 0], 3600), lp(wet[:, 1], 3200)], axis=1)
    return ((1 - mix) * x + mix * wet).astype(np.float32)


def tv_lp_stateful(x, curve, sr=SR, block=256, order=2, end_s=None):
    """Time-varying low-pass with continuous filter state (no block clicks)."""
    y = np.array(x, dtype=np.float32, copy=True)
    n = len(y)
    if end_s is not None:
        n = min(n, int(end_s * sr))
    for ch in range(y.shape[1]):
        zi = None
        for i in range(0, n, block):
            fc = float(np.clip(curve[min(i, len(curve) - 1)], 40.0, sr * 0.45))
            sos = signal.butter(order, fc, btype="low", fs=sr, output="sos")
            if zi is None:
                zi = np.zeros((sos.shape[0], 2), dtype=np.float64)
            seg, zi = signal.sosfilt(sos, y[i:i + block, ch], zi=zi)
            y[i:i + block, ch] = seg
    return y


def active_rms_db(x, floor_db=-45.0):
    """RMS of the audible (non-silent) parts of a stereo file, in dBFS."""
    mono = np.max(np.abs(x), axis=1)
    thr = 10 ** (floor_db / 20.0)
    m = mono > thr
    if not np.any(m):
        return -120.0
    v = x[m]
    return 20 * np.log10(np.sqrt(np.mean(v.astype(np.float64) ** 2)) + 1e-12)


def ms_width(x, amount=1.0, sr=SR, mono_below=0.0):
    if amount == 1.0 and mono_below <= 0:
        return x
    m = 0.5 * (x[:, 0] + x[:, 1])
    s = 0.5 * (x[:, 0] - x[:, 1]) * amount
    y = np.stack([m + s, m - s], axis=1)
    if mono_below > 0:
        y = mono_below_freq(y, mono_below, sr)
    return y.astype(np.float32)


def mono_below_freq(x, fc, sr=SR):
    low = np.stack([lp(x[:, 0], fc, sr), lp(x[:, 1], fc, sr)], axis=1)
    mid = (low[:, 0] + low[:, 1]) * 0.5
    return np.stack([x[:, 0] - low[:, 0] + mid, x[:, 1] - low[:, 1] + mid], axis=1).astype(np.float32)
