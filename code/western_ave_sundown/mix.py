"""
mix.py — bus processing, sends, master chain, deliverables.

Signal flow
  raw performance buses (48k float)
    -> optional intro filter automation
    -> corrective EQ -> saturation -> bus compression -> width/chorus/tremolo
    -> fader -> master sum + reverb send + delay send
  master sum -> mix-bus compression -> mastering EQ -> mono bass -> limiter
    -> 24-bit/48k WAV master (+ 320 kbps preview, stems, analysis)
"""
from __future__ import annotations
import os
import json
import numpy as np
import soundfile as sf
from scipy import signal

from . import dsp
from . import song as S
from .dsp import SR, db

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
RAW = os.path.join(PROJECT, ".cache", "raw")
CACHE = os.path.join(PROJECT, ".cache")
MASTER = os.path.join(PROJECT, "master")
STEMS = os.path.join(PROJECT, "stems")
ANALYSIS = os.path.join(PROJECT, "analysis")

# ---------------------------------------------------------------------------
# Bus configuration. target_rms is the audible-level aim (dBFS, active signal);
# the fader is computed against the raw bus so the mix is level-consistent.
# eq: (type, freq, gain_db, q)
# ---------------------------------------------------------------------------
BUS_CONFIG = {
    "kick": dict(
        target_rms=-15.5, hp=26, eq=[("peak", 54, 1.2, 0.9), ("peak", 88, 1.2, 1.0), ("peak", 245, -3.0, 1.0),
                                     ("peak", 3800, 3.2, 1.1), ("high_shelf", 7200, 0.8, 0.7)],
        sat=(1.35, 0.30), comp=(-18, 3.2, 12, 120), rev=0.0, delay=0.0,
        width=1.0, duck=0.0, intro_duck=0.0),
    "snare": dict(
        target_rms=-19.0, hp=115, eq=[("peak", 195, 1.2, 1.0), ("peak", 420, -1.2, 1.2),
                                      ("peak", 4200, 2.2, 0.9), ("high_shelf", 9000, 0.8, 0.7)],
        sat=(1.22, 0.28), comp=(-20, 3.6, 6, 130), rev=0.16, delay=0.0,
        width=1.06, duck=0.0, intro_duck=0.0),
    "clap": dict(
        target_rms=-24.0, hp=420, eq=[("peak", 1100, 1.8, 1.1), ("peak", 3200, 1.0, 1.0)],
        sat=(1.15, 0.25), comp=(-22, 3.0, 4, 120), rev=0.20, delay=0.0,
        width=1.15, duck=0.0, intro_duck=0.0),
    "rim": dict(
        target_rms=-27.0, hp=600, eq=[("peak", 2600, 2.0, 1.0)],
        sat=(1.1, 0.2), comp=(-22, 2.5, 4, 90), rev=0.18, delay=0.0,
        width=1.0, duck=0.0, intro_duck=0.0),
    "hats": dict(
        target_rms=-27.5, hp=650, eq=[("peak", 2200, -1.5, 1.0), ("high_shelf", 10500, 1.2, 0.7)],
        sat=(1.05, 0.15), comp=(-24, 2.6, 3, 70), rev=0.05, delay=0.0,
        width=1.05, duck=0.0, intro_duck=0.0),
    "perc": dict(
        target_rms=-29.0, hp=900, eq=[("peak", 6000, 1.2, 0.9)],
        sat=(1.05, 0.12), comp=(-24, 2.4, 5, 90), rev=0.12, delay=0.0,
        width=1.10, duck=0.0, intro_duck=0.0),
    "toms": dict(
        target_rms=-24.0, hp=45, eq=[("peak", 120, 1.0, 0.9), ("peak", 2400, 0.8, 1.0)],
        sat=(1.3, 0.28), comp=(-20, 2.8, 8, 140), rev=0.16, delay=0.0,
        width=1.02, duck=0.0, intro_duck=0.0),
    "crash": dict(
        target_rms=-33.0, hp=380, eq=[("high_shelf", 8000, 1.0, 0.7)],
        sat=(1.05, 0.1), comp=(-26, 2.0, 10, 200), rev=0.28, delay=0.0,
        width=1.20, duck=0.0, intro_duck=0.0),
    "bass": dict(
        target_rms=-16.5, hp=30, eq=[("peak", 88, 1.6, 1.0), ("peak", 290, -2.2, 1.1),
                                     ("peak", 720, 1.8, 1.0), ("peak", 1250, 1.4, 0.9), ("peak", 2600, 1.0, 0.9),
                                     ("low_shelf", 55, 1.0, 0.7)],
        sat=(1.35, 0.35), comp=(-20, 3.0, 18, 150), rev=0.0, delay=0.0,
        width=1.0, duck=2.8, intro_duck=0.0),
    "rhodes": dict(
        target_rms=-23.5, hp=115, eq=[("peak", 330, -2.6, 1.0), ("peak", 1900, 1.8, 0.8),
                                      ("high_shelf", 7000, 0.6, 0.7)],
        sat=(1.12, 0.22), comp=(-22, 2.0, 20, 180), rev=0.20, delay=0.0,
        width=1.12, duck=1.2, intro_duck=1.5),
    "guitar": dict(
        target_rms=-25.6, hp=170, eq=[("peak", 430, -1.6, 1.0), ("peak", 1900, 2.8, 0.9), ("peak", 3800, 1.2, 1.0),
                                      ("peak", 5200, 1.0, 1.0)],
        sat=(1.25, 0.30), comp=(-20, 4.0, 4, 90), rev=0.10, delay=0.0,
        width=1.10, duck=0.0, intro_duck=1.5),
    "strings": dict(
        target_rms=-27.2, hp=95, eq=[("peak", 400, -2.2, 1.0), ("peak", 2400, 1.2, 0.8)],
        sat=(1.08, 0.15), comp=(-20, 1.8, 45, 260), rev=0.30, delay=0.0,
        width=1.18, duck=0.0, intro_duck=2.0),
    "lead": dict(
        target_rms=-20.6, hp=180, eq=[("peak", 260, -1.8, 1.0), ("peak", 2300, 3.0, 0.9), ("high_shelf", 6500, 1.2, 0.7),
                                      ("peak", 4800, 0.8, 1.0)],
        sat=(1.18, 0.25), comp=(-18, 3.0, 12, 110), rev=0.17, delay=0.17,
        width=1.14, duck=0.0, intro_duck=1.0),
    "synth": dict(
        target_rms=-26.4, hp=220, eq=[("peak", 2800, 1.2, 0.9)],
        sat=(1.1, 0.2), comp=(-22, 2.5, 15, 140), rev=0.24, delay=0.20,
        width=1.2, duck=0.0, intro_duck=0.0),
    "fx": dict(
        target_rms=-30.0, hp=35, eq=[], sat=(1.0, 0.0), comp=(-24, 1.8, 20, 200),
        rev=0.22, delay=0.0, width=1.1, duck=0.0, intro_duck=0.0),
}

# reverb-return / delay-return processing
REVERB_RETURN_DB = -15.5
DELAY_RETURN_DB = -18.0


def _eq(x, bands):
    y = x
    for kind, f, g, q in bands:
        if kind == "peak":
            y = dsp.peak(y, f, g, q)
        elif kind == "low_shelf":
            y = dsp.low_shelf(y, f, g, q)
        elif kind == "high_shelf":
            y = dsp.high_shelf(y, f, g, q)
    return y


def kick_envelope(kick, attack_ms=4.0, release_ms=110.0):
    mono = np.max(np.abs(kick), axis=1)
    env = dsp._smooth_env(mono, attack_ms / 1000.0)
    env = dsp._smooth_env(env, release_ms / 1000.0)
    return (env / (np.max(env) + 1e-9)).astype(np.float32)


def process_bus(name, x, cfg, key_env=None, intro_len=0):
    y = x
    # intro filter automation (music "opens up" as the drums drop)
    if name in ("rhodes", "guitar", "strings", "lead") and intro_len > 0:
        n_intro = int(intro_len * SR)
        t = np.arange(n_intro) / SR
        # logarithmic opening 700 Hz -> 19 kHz across the 4-bar intro
        curve = 700 * (19000 / 700) ** np.clip(t / (n_intro / SR), 0, 1) ** 0.75
        curve = np.minimum(curve, 19000)
        seg = dsp.tv_lp_stateful(y[:n_intro], curve, end_s=n_intro / SR)
        # crossfade back to dry over the last 0.7 s
        xf = int(0.7 * SR)
        w = np.ones(n_intro, dtype=np.float32)
        w[-xf:] = np.linspace(1, 0, xf)
        y = y.copy()
        y[:n_intro] = seg * w[:, None] + y[:n_intro] * (1 - w[:, None])
    # corrective EQ
    if cfg["hp"] > 0:
        y = np.stack([dsp.hp(y[:, 0], cfg["hp"], SR, order=2),
                      dsp.hp(y[:, 1], cfg["hp"], SR, order=2)], axis=1)
    if cfg["eq"]:
        y = _eq(y, cfg["eq"])
    # saturation
    drive, mix = cfg["sat"]
    if drive > 1.0:
        y = np.stack([dsp.saturate(y[:, 0], drive, 0.02, mix),
                      dsp.saturate(y[:, 1], drive, 0.02, mix)], axis=1)
    # compression
    thr, ratio, atk, rel = cfg["comp"]
    y = dsp.compressor(y, thr_db=thr, ratio=ratio, attack_ms=atk, release_ms=rel,
                       knee_db=6.0, makeup_db=0.0)
    # character
    if name == "rhodes":
        y = dsp.tremolo_stereo(y, rate=5.15, depth=0.14)
        y = dsp.chorus(y, rate=0.27, depth_ms=6.0, mix=0.16)
    elif name == "guitar":
        y = dsp.chorus(y, rate=0.19, depth_ms=3.0, mix=0.08)
    elif name == "strings":
        y = dsp.chorus(y, rate=0.13, depth_ms=4.0, mix=0.10)
    # stereo width
    if cfg["width"] != 1.0:
        y = dsp.ms_width(y, cfg["width"])
    # sidechain duck from the kick
    if cfg.get("duck", 0) > 0 and key_env is not None:
        depth = cfg["duck"]
        g = (1.0 - (1.0 - 10 ** (-depth / 20.0)) * key_env)[:, None]
        y = y * g
    # fader: match the audible level to the target
    act = dsp.active_rms_db(y)
    if act < -100.0:            # silent bus in this render window
        return np.zeros_like(y), 0.0, act
    gain_db = cfg["target_rms"] - act
    y = y * db(gain_db)
    return y.astype(np.float32), gain_db, act



BAR_S = 4 * 60.0 / S.BPM


def section_dynamics(n):
    """Arrangement-level level automation: quiet breakdown, hotter hooks.
    Applied post-glue, pre-master, so the loudness moves with the arrangement."""
    c = np.ones(n, dtype=np.float32)
    def put(bar0, bar1, g0, g1=None):
        i0 = max(0, min(n, int(bar0 * BAR_S * SR)))
        i1 = max(0, min(n, int(bar1 * BAR_S * SR)))
        if i1 <= i0:
            return
        g1 = g0 if g1 is None else g1
        c[i0:i1] = np.linspace(g0, g1, i1 - i0, dtype=np.float32)
    put(0, 2, 0.80, 0.84)        # intro: filtered, quiet
    put(2, 4, 0.86, 0.96)        # build into the drop
    put(4, 20, 1.00)             # verse 1
    put(20, 28, 1.06)            # hook 1
    put(28, 44, 1.00)            # verse 2
    put(44, 46, 0.66)            # breakdown
    put(46, 48, 0.70)
    put(48, 50, 0.76)
    put(50, 52, 0.82, 1.00)      # build back up
    put(52, 60, 1.07)            # hook 2
    put(60, 64, 0.93)            # outro
    put(64, 68, 0.88, 0.80)
    # 60 ms smoothing so section edges breathe instead of clicking
    k = int(0.06 * SR)
    c = np.convolve(c, np.ones(k, dtype=np.float32) / k, mode="same").astype(np.float32)
    return c


def render_premix(intro_len=0.0, verbose=True, save=True, save_stems=False):
    """Sum all processed buses + reverb/delay returns; optionally cache the
    un-mastered mix for fast mastering iterations."""
    import time, resource
    names = [n for n in BUS_CONFIG if os.path.exists(os.path.join(RAW, f"{n}.wav"))]
    if not names:
        raise FileNotFoundError("no raw buses; run studio.render first")
    n = min(sf.info(os.path.join(RAW, f"{nm}.wav")).frames for nm in names)
    if verbose:
        print(f"mixing {len(names)} buses, {n/SR:.2f}s @ {SR} Hz")
    t0 = time.time()
    kick_raw = sf.read(os.path.join(RAW, "kick.wav"), dtype="float32")[0][:n]
    key_env = kick_envelope(kick_raw)
    del kick_raw
    mix = np.zeros((n, 2), dtype=np.float32)
    rev_send = np.zeros((n, 2), dtype=np.float32)
    dly_send = np.zeros((n, 2), dtype=np.float32)
    gains = {}
    for name in names:
        t1 = time.time()
        cfg = BUS_CONFIG[name]
        x = sf.read(os.path.join(RAW, f"{name}.wav"), dtype="float32")[0][:n]
        y, gain_db, act = process_bus(name, x, cfg, key_env, intro_len)
        gains[name] = dict(fader_db=round(float(gain_db), 2),
                           active_rms_in=round(float(act), 2))
        mix += y
        if cfg["rev"] > 0:
            rev_send += y * cfg["rev"]
        if cfg["delay"] > 0:
            dly_send += y * cfg["delay"]
        if save_stems:
            os.makedirs(CACHE, exist_ok=True)
            sf.write(os.path.join(CACHE, f"stem_{name}.wav"), y, SR,
                     subtype="FLOAT", format="WAV")
        if verbose:
            print(f"  {name:8s} fader {gain_db:+6.1f} dB  in {act:6.1f} dBFS  "
                  f"peak {np.max(np.abs(y)):5.2f}  {time.time()-t1:4.1f}s  RSS "
                  f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB")
        del x, y
    ir = dsp.make_ir(rt60=2.3, predelay_ms=22, bright=0.6, seed=11)
    if verbose:
        print("  convolving reverb send...")
    rev = dsp.convolve_stereo(rev_send, ir)[:n]
    del rev_send
    rev = np.stack([dsp.hp(rev[:, 0], 240, SR), dsp.hp(rev[:, 1], 240, SR)], axis=1)
    rev = np.stack([dsp.lp(rev[:, 0], 7800), dsp.lp(rev[:, 1], 7800)], axis=1)
    rev *= db(REVERB_RETURN_DB)
    dly = dsp.pingpong_delay(dly_send, delay_ms=3 * 60000 / S.BPM / 4, fb=0.32, mix=1.0)
    del dly_send
    dly *= db(DELAY_RETURN_DB)
    if save_stems:
        returns = rev + dly
        sf.write(os.path.join(CACHE, "return_fx.wav"), returns, SR,
                 subtype="FLOAT", format="WAV")
        del returns
    mix += rev
    mix += dly
    del rev, dly
    if verbose:
        print(f"  premix: {dsp.lufs_integrated(mix):.2f} LUFS, crest {dsp.crest_db(mix):.2f} dB, "
              f"peak {dsp.sample_peak_db(mix):.2f} dBFS, {time.time()-t0:.1f}s")
    if save:
        os.makedirs(CACHE, exist_ok=True)
        sf.write(os.path.join(CACHE, "premix_raw.wav"), mix, SR, subtype="FLOAT", format="WAV")
    # arrangement dynamics
    mix *= section_dynamics(len(mix))[:, None]
    if verbose:
        print(f"  with section automation: {dsp.lufs_integrated(mix):.2f} LUFS")
    if save:
        sf.write(os.path.join(CACHE, "premix.wav"), mix, SR, subtype="FLOAT", format="WAV")
    return mix, gains


def master_premix(pre, target_lufs=-9.0, verbose=True, out_wav=None):
    """Mix-bus glue + mastering chain + loudness/peak targeting + write."""
    import time
    t0 = time.time()
    mix = dsp.compressor(pre, thr_db=-17.0, ratio=1.7, attack_ms=28, release_ms=240,
                         knee_db=8.0, makeup_db=1.2)
    del pre
    mix = np.stack([dsp.tape_sat(mix[:, 0], 1.12, 0.18),
                    dsp.tape_sat(mix[:, 1], 1.12, 0.18)], axis=1)
    pre_m = np.stack([dsp.hp(mix[:, 0], 22, SR, order=4),
                      dsp.hp(mix[:, 1], 22, SR, order=4)], axis=1)
    del mix
    pre_m = np.stack([dsp.low_shelf(pre_m[:, 0], 58, 0.7, 0.65),
                      dsp.low_shelf(pre_m[:, 1], 58, 0.7, 0.65)], axis=1)
    pre_m = _eq(pre_m, [("peak", 45, -1.5, 1.0),      # tighten the sub
                        ("peak", 160, 2.4, 1.1),      # fill the low-mid notch
                        ("peak", 380, -1.0, 0.9),     # declutter boxiness
                        ("peak", 500, -1.2, 1.0),
                        ("peak", 720, 0.8, 0.8),
                        ("peak", 1400, 2.0, 0.55),    # broad presence lift
                        ("peak", 2200, 1.5, 0.8),
                        ("high_shelf", 8000, 1.0, 0.7)])
    pre_m = dsp.mono_below_freq(pre_m, 95)
    pre_m = dsp.ms_width(pre_m, 1.06)
    pre_m = dsp.compressor(pre_m, thr_db=-14.0, ratio=1.35, attack_ms=38, release_ms=300,
                           knee_db=10.0, makeup_db=0.0)
    pre_m /= (np.max(np.abs(pre_m)) + 1e-9) / db(-6.0)
    gain = 0.0
    limited = None
    for it in range(6):
        t1 = time.time()
        limited = dsp.limit(pre_m * db(gain), ceiling_db=-1.2, lookahead_ms=3.2,
                            release_ms=140)
        lufs = dsp.lufs_integrated(limited, SR)
        tp = dsp.true_peak_db(limited, SR)
        if verbose:
            print(f"  limiter pass {it}: pre-gain {gain:+.2f} dB -> {lufs:.2f} LUFS, "
                  f"TP {tp:+.2f} dBTP, peak {dsp.sample_peak_db(limited):+.2f} dBFS "
                  f"({time.time()-t1:.1f}s)")
        if abs(lufs - target_lufs) < 0.12:
            break
        gain += (target_lufs - lufs) * 0.9
        gain = float(np.clip(gain, -6.0, 24.0))
    del pre_m
    # ---- edge fades BEFORE the final true-peak trim
    fi = int(0.004 * SR)
    limited[:fi] *= np.linspace(0, 1, fi)[:, None]
    fo = int(1.2 * SR)
    if np.max(np.abs(limited[-fo:])) > 0.0015:
        limited[-fo:] *= np.linspace(1, 0, fo)[:, None] ** 1.5
    # ---- true-peak trim to spec (global, exact)
    tp = dsp.true_peak_db(limited, SR)
    if tp > -1.02:
        trim = -1.02 - tp
        limited *= db(trim)
        if verbose:
            print(f"  true-peak trim {trim:+.2f} dB -> {dsp.true_peak_db(limited, SR):+.2f} dBTP")
    master = limited.astype(np.float32)
    tp = dsp.true_peak_db(master, SR)
    lufs = dsp.lufs_integrated(master, SR)
    if verbose:
        print(f"  MASTER: {lufs:.2f} LUFS | true peak {tp:+.2f} dBTP | "
              f"sample peak {dsp.sample_peak_db(master):+.2f} dBFS | "
              f"crest {dsp.crest_db(master):.2f} dB | corr {dsp.stereo_correlation(master):.2f} | "
              f"{time.time()-t0:.1f}s")
    os.makedirs(MASTER, exist_ok=True)
    if out_wav is None:
        out_wav = os.path.join(MASTER, "Western_Ave_Sundown_24bit_48kHz_MASTER.wav")
    sf.write(out_wav, master, SR, subtype="PCM_24", format="WAV")
    return dict(master=master, gains=None, lufs=lufs, true_peak=tp, n=len(master),
                pre_gain=gain, wav=out_wav)


def build_master(intro_len=0.0, target_lufs=-9.0, out_wav=None, verbose=True):
    mix, gains = render_premix(intro_len=intro_len, verbose=verbose, save_stems=True)
    res = master_premix(mix, target_lufs=target_lufs, verbose=verbose, out_wav=out_wav)
    res["gains"] = gains
    return res


if __name__ == "__main__":
    import sys
    if "--remaster" in sys.argv:
        pre, _ = sf.read(os.path.join(CACHE, "premix.wav"), dtype="float32")
        master_premix(pre, target_lufs=float(sys.argv[sys.argv.index("--remaster") + 1])
                      if len(sys.argv) > sys.argv.index("--remaster") + 1 else -8.9)
    else:
        build_master(intro_len=4 * S.BAR_S)
