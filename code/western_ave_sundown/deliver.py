"""
deliver.py — encodes preview audio, grouped stems, MIDI session and analysis.

Deliverables follow the spec:
  master/Western_Ave_Sundown_MASTER_24bit_48kHz.wav
  master/Western_Ave_Sundown_preview_320kbps.mp3
  stems/01_DRUMS_320kbps.mp3 ... 08_FX_REVERB_320kbps.mp3
  session/*.mid + per-instrument MIDI
  analysis/*.png + mix_report.md
"""
from __future__ import annotations
import os
import json
import subprocess
import numpy as np
import soundfile as sf

from . import dsp
from . import song as S
from .dsp import SR
from .mix import BUS_CONFIG, CACHE, MASTER, STEMS, ANALYSIS, PROJECT, section_dynamics

import imageio_ffmpeg
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

MASTER_WAV = os.path.join(MASTER, "Western_Ave_Sundown_24bit_48kHz_MASTER.wav")
PREVIEW_MP3 = os.path.join(MASTER, "Western_Ave_Sundown_preview_320kbps.mp3")

STEM_GROUPS = {
    "01_DRUMS": ["kick", "snare", "clap", "rim", "hats", "perc", "toms", "crash"],
    "02_BASS": ["bass"],
    "03_RHODES": ["rhodes"],
    "04_GUITAR": ["guitar"],
    "05_STRINGS": ["strings"],
    "06_LEAD_TALKBOX": ["lead"],
    "07_SYNTH_LEAD": ["synth"],
    "08_FX_REVERB": ["fx", "__return__"],
}

META = dict(
    title="Western Ave Sundown",
    artist="Orion Music & Hanit Kamboj",
    album="Western Ave Sundown — Single",
    genre="West Coast Hip-Hop / G-Funk",
    date="2026",
    comment="Original instrumental. Composed, performed and mixed in-session. "
            "Key F# minor, 93 BPM. 24-bit/48 kHz master.",
)


def ff(*args):
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args]
    subprocess.run(cmd, check=True)


def encode_mp3(src, dst, bitrate="320k", meta=True):
    args = ["-i", src, "-codec:a", "libmp3lame", "-b:a", bitrate]
    if meta:
        for k, v in META.items():
            args += ["-metadata", f"{k}={v}"]
    args += [dst]
    ff(*args)


def _group_peak(group, buses, n, curve):
    acc = np.zeros((n, 2), dtype=np.float32)
    for b in buses:
        path = os.path.join(CACHE, "return_fx.wav" if b == "__return__" else f"stem_{b}.wav")
        y = sf.read(path, dtype="float32")[0]
        acc[:len(y)] += y[:n]
    acc *= curve
    pk = float(np.max(np.abs(acc)))
    del acc
    return pk


def write_group_stems(verbose=True):
    """Print grouped stems with a COMMON headroom gain so nothing clips.
    The same gain is applied to every stem, so the balance is preserved."""
    os.makedirs(STEMS, exist_ok=True)
    n = sf.info(os.path.join(CACHE, "premix.wav")).frames
    curve = section_dynamics(n)[:, None]
    peaks = {g: _group_peak(g, b, n, curve) for g, b in STEM_GROUPS.items()}
    worst = max(peaks.values())
    # leave 1 dB of true-peak-ish headroom on the loudest group
    stem_gain_db = min(0.0, -1.0 - 20 * np.log10(worst + 1e-9))
    g = 10 ** (stem_gain_db / 20)
    written = []
    tmpdir = os.path.join(CACHE, "tmp")
    os.makedirs(tmpdir, exist_ok=True)
    for group, buses in STEM_GROUPS.items():
        acc = np.zeros((n, 2), dtype=np.float32)
        for b in buses:
            path = os.path.join(CACHE, "return_fx.wav" if b == "__return__" else f"stem_{b}.wav")
            y = sf.read(path, dtype="float32")[0]
            acc[:len(y)] += y[:n]
        acc *= curve * g
        tmp = os.path.join(tmpdir, f"{group}.wav")
        sf.write(tmp, acc, SR, subtype="PCM_24", format="WAV")
        dst = os.path.join(STEMS, f"{group}_320kbps.mp3")
        encode_mp3(tmp, dst, "320k")
        written.append((group, os.path.getsize(dst) / 1e6, float(np.max(np.abs(acc)))))
        if verbose:
            print(f"  {group:18s} {os.path.getsize(dst)/1e6:5.1f} MB  peak {np.max(np.abs(acc)):.2f}")
        del acc
    with open(os.path.join(STEMS, "README.txt"), "w") as f:
        f.write(
            "Western Ave Sundown — stem groups (Orion Music & Hanit Kamboj)\n"
            "==============================================================\n\n"
            f"All stems carry a common gain of {stem_gain_db:+.2f} dB so that the loudest\n"
            "transient peaks at about -1 dBFS. Apply the inverse gain at the fader\n"
            "to reconstruct the un-mastered mix. Stems are post bus-processing and\n"
            "post arrangement automation, pre master-bus limiting.\n\n"
            "08_FX_REVERB contains the FX bus plus the shared reverb and delay returns,\n"
            "so the eight stems sum to the full mix.\n\n"
            "Delivery format: 48 kHz stereo, 320 kbps MP3 (session renders 24-bit/48 kHz\n"
            "WAV stems individually — see studio/render.py).\n")
    return written, stem_gain_db


def null_test():
    """Do the delivered stems reconstruct the mix? (post-automation premix)"""
    n = sf.info(os.path.join(CACHE, "premix.wav")).frames
    premix = sf.read(os.path.join(CACHE, "premix.wav"), dtype="float32")[0]
    acc = np.zeros((n, 2), dtype=np.float32)
    curve = section_dynamics(n)[:, None]
    for group, buses in STEM_GROUPS.items():
        for b in buses:
            path = os.path.join(CACHE, "return_fx.wav" if b == "__return__" else f"stem_{b}.wav")
            y = sf.read(path, dtype="float32")[0]
            acc[:len(y)] += y[:n] * curve
    diff = premix - acc
    rms_sum = np.sqrt(np.mean(premix.astype(np.float64) ** 2))
    rms_diff = np.sqrt(np.mean(diff.astype(np.float64) ** 2))
    return dict(null_db=float(20 * np.log10((rms_diff + 1e-12) / (rms_sum + 1e-12))),
                peak_diff=float(np.max(np.abs(diff))))


# ---------------------------------------------------------------- MIDI
def export_midi():
    import mido
    from mido import Message, MidiFile, MidiTrack, MetaMessage
    os.makedirs(os.path.join(PROJECT, "session", "midi"), exist_ok=True)
    TPQ = 960

    def tempo_track():
        t = MidiTrack()
        t.append(MetaMessage("track_name", name="Western Ave Sundown", time=0))
        t.append(MetaMessage("set_tempo", tempo=mido.bpm2tempo(S.BPM), time=0))
        t.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))
        t.append(MetaMessage("key_signature", key="F#m", time=0))
        t.append(MetaMessage("marker", text="ORION MUSIC & HANIT KAMB0J", time=0))
        for name, start, length in S.SECTIONS:
            t.append(MetaMessage("marker", text=f"{name.upper()} bar {start+1}", time=0))
        return t

    def notes_to_track(name, notes, program, channel, drum=False):
        """notes: list of (start_beat, dur_beats, midi, vel)"""
        t = MidiTrack()
        t.append(MetaMessage("track_name", name=name, time=0))
        if not drum:
            t.append(Message("program_change", program=program, channel=channel, time=0))
        ev = []
        for (b, d, m, v) in notes:
            on = int(round(b * TPQ)); off = int(round((b + max(d, 0.02)) * TPQ))
            vel = int(np.clip(v * 127, 1, 127))
            ev.append((on, 1, m, vel))
            ev.append((off, 0, m, 0))
        ev.sort(key=lambda x: (x[0], x[1]))
        last = 0
        for (tick, is_on, m, vel) in ev:
            dt = max(0, tick - last)
            last = tick
            if is_on:
                t.append(Message("note_on", note=int(m), velocity=vel,
                                 channel=9 if drum else channel, time=dt))
            else:
                t.append(Message("note_off", note=int(m), velocity=0,
                                 channel=9 if drum else channel, time=dt))
        return t

    E, H = S.compose()
    beat = lambda t: t / S.BEAT_S  # noqa: E731

    drum_notes = []
    for t, v in E.kick:
        drum_notes.append((beat(t), 0.25, 36, v))
    for t, v in E.snare:
        drum_notes.append((beat(t), 0.25, 38, v))
    for t, v in E.clap:
        drum_notes.append((beat(t), 0.25, 39, v))
    for t, v in E.rim:
        drum_notes.append((beat(t), 0.2, 37, v))
    for t, v, open_, pan in E.hat:
        drum_notes.append((beat(t), 0.2, 46 if open_ else 42, v))
    for t, v, kind, pan in E.perc:
        drum_notes.append((beat(t), 0.2, 54 if kind == "tamb" else 70, v))
    for t, v, midi, pan in E.tom:
        drum_notes.append((beat(t), 0.25, int(midi) - 5, v))
    for t, v, rev in E.crash:
        drum_notes.append((beat(t), 2.0, 49, v))
    bass_notes = [(beat(e["t"]), e["dur"] / S.BEAT_S, e["midi"], e["vel"]) for e in E.bass]
    rh_notes = []
    for (t, dur, notes, vel, pan) in E.rhodes:
        for m in notes:
            rh_notes.append((beat(t), dur / S.BEAT_S, m, vel))
    gt_notes = []
    for (t, dur, notes, vel, muted, pan) in E.guitar:
        for m in notes:
            gt_notes.append((beat(t), dur / S.BEAT_S, m, vel))
    st_notes = []
    for (t, dur, notes, vel, oct_up) in E.strings:
        for m in notes:
            st_notes.append((beat(t), dur / S.BEAT_S, m, vel))
            if oct_up:
                st_notes.append((beat(t), dur / S.BEAT_S, m + 12, vel * 0.8))
    lead_notes = [(beat(e["t"]), e["dur"] / S.BEAT_S, e["midi"], e["vel"]) for e in E.lead]
    synth_notes = [(beat(e["t"]), e["dur"] / S.BEAT_S, e["midi"], e["vel"]) for e in E.synth]

    tracks = {
        "drums": (drum_notes, 0, 9),
        "bass": (bass_notes, 33, 1),
        "rhodes": (rh_notes, 4, 2),
        "guitar": (gt_notes, 28, 3),
        "strings": (st_notes, 48, 4),
        "lead_talkbox": (lead_notes, 85, 5),
        "synth_lead": (synth_notes, 81, 6),
    }
    full = MidiFile(ticks_per_beat=TPQ)
    full.tracks.append(tempo_track())
    for name, (notes, prog, ch) in tracks.items():
        full.tracks.append(notes_to_track(name, notes, prog, ch, drum=(ch == 9)))
        single = MidiFile(ticks_per_beat=TPQ)
        single.tracks.append(tempo_track())
        single.tracks.append(notes_to_track(name, notes, prog, ch, drum=(ch == 9)))
        single.save(os.path.join(PROJECT, "session", "midi", f"{name}.mid"))
    out = os.path.join(PROJECT, "session", "Western_Ave_Sundown_FULL.mid")
    full.save(out)
    # arrangement map
    arr = [dict(section=s, start_bar=st + 1, bars=ln,
                start_s=round(st * S.BAR_S, 2), end_s=round((st + ln) * S.BAR_S, 2))
           for s, st, ln in S.SECTIONS]
    with open(os.path.join(PROJECT, "session", "arrangement.json"), "w") as f:
        json.dump(dict(title=META["title"], artist=META["artist"], key="F# minor",
                       tempo_bpm=S.BPM, swing_ratio=S.SWING, bars=S.TOTAL_BARS,
                       duration_s=round(S.SONG_S + S.TAIL_S, 2), sections=arr,
                       progressions=dict(verse=S.PROG_VERSE, hook=S.PROG_HOOK,
                                         bridge=S.PROG_BRIDGE, outro=S.PROG_OUTRO)),
                  f, indent=2)
    return out


# ---------------------------------------------------------------- analysis
def analyse():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    os.makedirs(ANALYSIS, exist_ok=True)
    x, sr = sf.read(MASTER_WAV, dtype="float32")
    n = len(x)
    t = np.arange(n) / sr
    bar_s = S.BAR_S
    sections = [(name, st * bar_s, (st + ln) * bar_s) for name, st, ln in S.SECTIONS]

    # ---- 1. waveform + loudness
    k = 480   # samples per pixel column
    m = n // k
    xr = x[:m * k].reshape(m, k, 2)
    pk = np.max(np.abs(xr), axis=(1, 2))
    tp = np.arange(m) * k / sr
    short_l, short_t = _short_term(x, sr)
    fig, ax = plt.subplots(2, 1, figsize=(13, 6.2), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.15, 1]))
    ax[0].fill_between(tp, -pk, pk, color="#1f3b57", linewidth=0)
    ax[0].set_ylim(-1.05, 1.05)
    ax[0].set_ylabel("amplitude")
    ax[0].set_title("Western Ave Sundown — Orion Music & Hanit Kamboj  |  "
                    f"master {dsp.lufs_integrated(x):.1f} LUFS, {dsp.true_peak_db(x):.2f} dBTP",
                    fontsize=11, loc="left")
    ax[1].plot(short_t, short_l, color="#b8860b", linewidth=1.2)
    ax[1].axhline(dsp.lufs_integrated(x), color="#444", linestyle="--", linewidth=0.8)
    ax[1].set_ylabel("LUFS (3 s short-term)")
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylim(-22, -5)
    for name, a, b in sections:
        for axis in ax:
            axis.axvline(a, color="#999", linewidth=0.5, alpha=0.6)
        ax[0].text((a + b) / 2, 0.92, name.upper(), ha="center", va="top",
                   fontsize=7.5, color="#111")
    ax[1].set_xlim(0, n / sr)
    for axis in ax:
        axis.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS, "waveform_loudness.png"), dpi=140)
    plt.close(fig)

    # ---- 2. spectrogram
    from scipy import signal as sg
    mono = x.mean(axis=1)
    f, tt, Sxx = sg.spectrogram(mono, sr, nperseg=4096, noverlap=3072)
    Sdb = 10 * np.log10(Sxx + 1e-12)
    # compress time axis for a readable image
    step = max(1, len(tt) // 1400)
    fsel = f >= 24
    fig, ax = plt.subplots(figsize=(13, 5.2))
    mesh = ax.pcolormesh(tt[::step], f[fsel], Sdb[fsel][:, ::step], shading="auto",
                         cmap="magma", vmin=Sdb.max() - 78, vmax=Sdb.max() - 8)
    ax.set_yscale("log")
    ax.set_ylim(30, 18000)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:,.0f}"))
    ax.set_title("Master spectrogram — arrangement in time / frequency", fontsize=11, loc="left")
    for name, a, b in sections:
        ax.axvline(a, color="white", linewidth=0.6, alpha=0.5)
        ax.text((a + b) / 2, 15000, name.upper(), color="white", fontsize=7,
                ha="center", va="top")
    fig.colorbar(mesh, ax=ax, pad=0.01, label="dB")
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS, "spectrogram.png"), dpi=140)
    plt.close(fig)

    # ---- 3. spectrum (1/3 octave) with target corridor
    cents = 100 * np.log2(f / 1000.0 + 1e-12)
    P = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))) ** 2 / len(mono)
    ff = np.fft.rfftfreq(len(mono), 1 / sr)
    centers = [31.5 * 2 ** (i / 3) for i in range(0, 28)]
    vals = []
    for fc in centers:
        m = (ff >= fc / 2 ** (1 / 6)) & (ff < fc * 2 ** (1 / 6))
        vals.append(10 * np.log10(np.sum(P[m]) + 1e-20))
    vals = np.array(vals)
    vals -= 10 * np.log10(np.sum(P) + 1e-20)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.semilogx(centers, vals, "o-", color="#b8860b", linewidth=1.6, markersize=3)
    ax.set_xlim(28, 18000)
    ax.set_ylim(-42, -2)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("band energy (dB rel. total)")
    ax.set_title("Master 1/3-octave balance", fontsize=11, loc="left")
    ax.grid(which="both", alpha=0.2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:,.0f}"))
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS, "spectrum.png"), dpi=140)
    plt.close(fig)

    # ---- 4. stem balance in the hook
    rows = []
    a, b = int(20 * bar_s * sr), int(28 * bar_s * sr)
    for name, buses in STEM_GROUPS.items():
        acc = None
        for bname in buses:
            p = os.path.join(CACHE, "return_fx.wav" if bname == "__return__" else f"stem_{bname}.wav")
            y = sf.read(p, dtype="float32")[0][a:b]
            acc = y if acc is None else acc + y
        rms = 20 * np.log10(np.sqrt(np.mean(acc.astype(np.float64) ** 2)) + 1e-12)
        rows.append((name, rms))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.barh([r[0].replace("_", " ").title() for r in rows], [r[1] for r in rows],
            color="#1f3b57")
    ax.set_xlabel("RMS in hook (dBFS)")
    ax.set_title("Stem balance — bar 21-28 (hook 1)", fontsize=11, loc="left")
    ax.set_xlim(-42, -12)
    ax.grid(axis="x", alpha=0.2)
    for i, (name, v) in enumerate(rows):
        ax.text(v + 0.4, i, f"{v:.1f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ANALYSIS, "stem_balance.png"), dpi=140)
    plt.close(fig)

    # ---- metrics
    metrics = dict(
        duration_s=round(n / sr, 2),
        sample_rate=sr,
        bit_depth=24,
        channels=2,
        integrated_lufs=round(dsp.lufs_integrated(x), 2),
        true_peak_dbtp=round(dsp.true_peak_db(x), 2),
        sample_peak_dbfs=round(dsp.sample_peak_db(x), 2),
        crest_db=round(dsp.crest_db(x), 2),
        lra_lu=round(_lra(x, sr), 2),
        stereo_correlation=round(dsp.stereo_correlation(x), 3),
        bands={k: round(v, 1) for k, v in dsp.band_energy(x).items()},
        section_loudness={name: round(dsp.lufs_integrated(x[int(a * sr):int(b * sr)]), 2)
                          for name, a, b in sections},
        spectral_centroid_hz=round(float(np.sum(ff * P) / np.sum(P)), 1),
    )
    with open(os.path.join(ANALYSIS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def _short_term(x, sr, win=3.0, hop=0.25):
    y = dsp.k_weight(x.astype(np.float64), sr)
    w = int(win * sr); h = int(hop * sr)
    vals, times = [], []
    for i in range(0, len(y) - w + 1, h):
        pw = np.mean(y[i:i + w] ** 2)
        vals.append(-0.691 + 10 * np.log10(pw + 1e-20))
        times.append((i + w / 2) / sr)
    return np.array(vals), np.array(times)


def _lra(x, sr):
    vals, _ = _short_term(x, sr, win=3.0, hop=1.0)
    vals = vals[vals > -70]
    rel = np.mean(10 ** (vals / 10))
    th = -0.691 + 10 * np.log10(rel) - 20
    v = vals[vals > th]
    return float(np.percentile(v, 95) - np.percentile(v, 10))


if __name__ == "__main__":
    print("encoding preview mp3...")
    encode_mp3(MASTER_WAV, PREVIEW_MP3, "320k")
    print("writing stems...")
    stats, gain = write_group_stems()
    print(f"  common stem gain {gain:+.2f} dB")
    print("null test:", null_test())
    print("midi...")
    print(" ", export_midi())
    print("analysis...")
    print(" ", analyse())
