"""
songs.py — repertoire for the modern sessions.

Each song is an original composition: chords, groove, bass language, comping,
melody, wordless vocalise placement and section dynamics.
"""
from __future__ import annotations
import numpy as np
from .engine import SongSpec, Humanizer, voicing, chord_dict, SONGS
from . import dsp

PC = dict(C=0, Cs=1, D=2, Eb=3, E=4, F=5, Fs=6, G=7, Ab=8, A=9, Bb=10, B=11)


def P(parts, name, kind):
    if name not in parts:
        parts[name] = (kind, [])
    return parts[name][1]


def diatonic_shift(midi, key_pcs, steps):
    """Move `steps` scale degrees (negative = down) inside a key."""
    pcs = sorted(key_pcs)
    # build a 3-octave ladder of the scale
    ladder = [m for m in range(36, 97) if m % 12 in pcs]
    i = min(range(len(ladder)), key=lambda k: abs(ladder[k] - midi))
    j = int(np.clip(i + steps, 0, len(ladder) - 1))
    return ladder[j]


# =====================================================================
# A. PAPER LANTERNS — modern boom-bap soul, 87 BPM, D minor
# =====================================================================
A = SongSpec(
    slug="paper_lanterns", title="Paper Lanterns",
    genre="boom-bap soul / instrumental hip-hop",
    bpm=87, key_name="D minor", bars=68, tail_s=7.0, swing=0.56,
    sections=[("intro", 0, 4), ("verse1", 4, 16), ("hook1", 20, 8),
              ("verse2", 28, 16), ("bridge", 44, 8), ("hook2", 52, 8),
              ("outro", 60, 8)],
    chords={
        "Dm9":   chord_dict(38, {2, 5, 9, 0, 4}, 62),
        "Bbmaj7": chord_dict(34, {10, 2, 5, 9}, 58),
        "Gm9":   chord_dict(31, {7, 10, 2, 5, 9}, 58),
        "A7b9":  chord_dict(33, {9, 1, 4, 7, 10}, 57),
        "Fmaj9": chord_dict(41, {5, 9, 0, 4, 7}, 60),
        "Em7b5": chord_dict(40, {4, 7, 10, 2}, 55),
    },
    prog={
        "intro": ["Dm9", "Dm9", "Bbmaj7", "A7b9"],
        "verse": ["Dm9", "Bbmaj7", "Gm9", "A7b9", "Dm9", "Bbmaj7", "Em7b5", "A7b9"],
        "hook": ["Dm9", "Bbmaj7", "Gm9", "A7b9", "Dm9", "Fmaj9", "Em7b5", "A7b9"],
        "bridge": ["Bbmaj7", "Gm9", "Em7b5", "A7b9", "Dm9", "Bbmaj7", "Em7b5", "A7b9"],
        "outro": ["Dm9", "Bbmaj7", "Gm9", "A7b9", "Dm9", "Fmaj9", "Dm9", "Dm9"],
    },
    dynamics=dict(intro=0.78, verse1=1.0, hook1=1.06, verse2=1.01, bridge=0.68,
                  hook2=1.07, outro=0.9),
    master_eq=[("peak", 150, 2.0, 1.1), ("peak", 420, -1.2, 0.9),
               ("peak", 1450, 1.8, 0.6), ("peak", 2500, 0.8, 0.8),
               ("high_shelf", 8500, 1.0, 0.7)],
    vocal_pct=6.0,
)

A_MODE = {2, 4, 5, 7, 9, 10, 0}   # D natural minor


def build_A(spec: SongSpec):
    H = Humanizer(seed=101, timing_ms=4.5, vel_pct=0.05, swing=spec.swing)
    parts = {}
    T = lambda b, f=1.0: H.time(b, f, spec)  # noqa: E731

    def kick(t, v): P(parts, "kick", "kick").append((t, H.vel(v)))
    def snare(t, v): P(parts, "snare", "snare").append((t, H.vel(v)))
    def rim(t, v): P(parts, "rim", "rim").append((t, H.vel(v)))
    def hat(t, v, opn=False, pan=0.16): P(parts, "hats", "hats").append((t, H.vel(v), opn, pan))
    def perc(t, v, sub="shaker", pan=0.3): P(parts, "perc", "perc").append((t, H.vel(v), sub, pan))
    def crash(t, v, rev=False): P(parts, "crash", "crash").append((t, H.vel(v), rev))

    def bass(t, midi, dur, v, art="finger", glide=None, gs=0.06):
        P(parts, "bass", "finger").append(dict(t=t, midi=midi, dur=dur, vel=H.vel(v),
                                              art=art, glide_from=glide, glide_s=gs))

    def keys(t, dur, name, v, bus="rhodes"):
        ch = spec.chords[name]
        notes = voicing(ch, n=4, center=ch["center"])
        P(parts, bus, "rhodes" if bus == "rhodes" else bus).append(
            dict(t=t, dur=dur, notes=notes, vel=H.vel(v), pan=0.02))

    def vibes(t, notes, dur, v):
        P(parts, "vibes", "vibes").append(dict(t=t, dur=dur, notes=notes, vel=H.vel(v), pan=-0.1))

    def gtr(t, dur, name, v, muted=True, pan=-0.24):
        ch = spec.chords[name]
        cand = sorted([m for m in range(52, 80) if m % 12 in ch["pcs"]],
                      key=lambda m: abs(m - 66))
        nts = [cand[0]]
        for m in cand[1:]:
            if all(abs(m - q) >= 3 for q in nts):
                nts.append(m)
            if len(nts) == 2:
                break
        nts = sorted(nts)
        P(parts, "guitar", "guitar").append(dict(t=t, dur=dur, notes=nts, vel=H.vel(v),
                                                 muted=muted, pan=pan))

    def strings(t, dur, name, v):
        ch = spec.chords[name]
        P(parts, "strings", "strings").append(
            dict(t=t, dur=dur, notes=voicing(ch, n=5, center=59, low=48, high=74),
                 vel=H.vel(v)))

    # ---------------------------------------------------------- drums
    def boom(t0, bars_variant, energy=1.0, fill=False, hats16=True, shaker=False):
        kicks = [[(0, 1.0), (1.75, 0.82), (2.5, 0.92)],
                 [(0, 1.0), (1.5, 0.72), (2.5, 0.95), (3.75, 0.58)],
                 [(0, 1.0), (1.75, 0.8), (2.25, 0.68), (3.0, 0.9)],
                 [(0, 1.0), (2.5, 1.0), (3.25, 0.6)]][bars_variant % 4]
        for b, v in kicks:
            kick(T(t0 + b), v * energy)
        snare(T(t0 + 1), 0.95 * energy)
        snare(T(t0 + 3), 1.0 * energy)
        for b, v in [(2.25, 0.17), (3.75, 0.24), (1.25, 0.12)]:
            snare(T(t0 + b), v * energy)
        if bars_variant % 4 == 2:
            rim(T(t0 + 3.75), 0.35 * energy)
        for i in range(16):
            b = i * 0.25
            acc = 0.62 if i % 4 == 0 else (0.44 if i % 2 == 0 else 0.28)
            hat(T(t0 + b), acc * energy, opn=(i in (6, 14) and bars_variant % 2 == 0))
        if shaker:
            for i in range(16):
                if i % 2 == 0 or i in (3, 7, 11, 15):
                    perc(T(t0 + i * 0.25), (0.32 if i % 4 == 0 else 0.24) * energy)
        if fill:
            for b, v, m in [(3.0, 0.5, 45), (3.25, 0.6, 48), (3.5, 0.72, 50),
                            (3.75, 0.85, 53)]:
                P(parts, "toms", "toms").append((T(t0 + b), H.vel(v * energy), m, 0.2))

    # intro: filtered keys, no drums except a soft rim pattern from bar 2
    for bar in range(0, 4):
        name = spec.chord_at(bar)
        keys(T(bar * 4), spec.beat_s * 3.6, name, 0.42 + 0.06 * bar)
        if bar >= 2:
            vibes(T(bar * 4 + 2.0), [voicing(spec.chords[name], 1, 74)[0]], 1.4, 0.34)
    rim(T(1 * 4 + 2), 0.3); rim(T(3 * 4 + 2), 0.34)
    for bar in (2, 3):
        for i in range(8):
            hat(T(bar * 4 + i * 0.5), 0.2, pan=0.1)
    crash(T(3 * 4 + 3.5), 0.55, rev=True)
    bass(T(3 * 4 + 2.5), 38, spec.beat_s * 0.5, 0.5, glide=43, gs=0.08)
    bass(T(3 * 4 + 3.0), 38, spec.beat_s * 0.9, 0.62, glide=38, gs=0.05)

    # ---------------------------------------------------------- verse 1
    for i, bar in enumerate(range(4, 20)):
        name = spec.chord_at(bar)
        boom(bar * 4, i, energy=0.96 + 0.02 * min(i, 4), fill=(i % 8 == 7),
             shaker=(i >= 8))
        keys(T(bar * 4), spec.beat_s * 1.4, name, 0.72)
        keys(T(bar * 4 + 2.5), spec.beat_s * 1.0, name, 0.6)
        if i % 4 == 1:
            keys(T(bar * 4 + 1.5), spec.beat_s * 0.4, name, 0.45)
        # vibes answer every other bar
        if i % 4 == 2:
            mel = [[72, 69], [70, 67]][(i // 4) % 2]
            vibes(T(bar * 4 + 3.0), mel, 1.1, 0.4)
        # guitar chank on the "and" of 2 and 4
        for b, v in [(1.5, 0.5), (2.75, 0.42), (3.5, 0.5)]:
            gtr(T(bar * 4 + b), 0.16, name, v)
        # bass: chord-tone line, new shape every 2 bars
        root = spec.chords[name]["bass"]
        pat = [[(0, 0, 2.6, 0.95), (2.5, 12, 1.0, 0.75), (3.5, 7, 0.5, 0.7)],
               [(0, 0, 1.8, 0.98), (2.0, 7, 0.8, 0.75), (3.0, 10, 0.8, 0.7), (3.75, 12, 0.3, 0.6)],
               [(0, 0, 2.2, 0.94), (2.5, 12, 0.7, 0.8), (3.25, 10, 0.5, 0.68), (3.75, 7, 0.3, 0.6)],
               [(0, 0, 1.2, 0.96), (1.5, 7, 0.6, 0.7), (2.0, 12, 0.8, 0.8), (3.5, 3, 0.5, 0.62)]][i % 4]
        for (b, off, d, v) in pat:
            midi = root + off
            if name in ("Bbmaj7", "Fmaj9") and off == 10:
                midi = root + 11
            if name in ("A7b9",) and off == 3:
                midi = root + 4
            bass(T(bar * 4 + b), midi, spec.beat_s * d, v,
                 glide=(midi - 2 if b > 0 and i % 2 else None))
        if i % 8 == 7:
            strings(T(bar * 4), spec.bar_s * 0.95, name, 0.28)

    # ---------------------------------------------------------- hook 1
    hook_mel = [
        (0.0, 0.75, 69, 0.78, "ah"), (0.75, 0.5, 72, 0.8, "ah"),
        (1.5, 1.5, 74, 0.85, "ah"), (3.0, 0.5, 72, 0.7, "oo"),
        (4.0, 1.0, 70, 0.78, "oh"), (5.0, 0.5, 69, 0.72, "oo"),
        (6.0, 1.75, 67, 0.8, "ah"), (8.0, 0.75, 65, 0.74, "oh"),
        (8.75, 0.5, 69, 0.76, "ah"), (9.5, 1.5, 72, 0.82, "ah"),
        (11.5, 0.5, 70, 0.7, "oo"), (12.0, 1.25, 69, 0.78, "ah"),
        (13.5, 0.5, 67, 0.7, "oo"), (14.0, 1.75, 62, 0.72, "ah"),
    ]
    for i, bar in enumerate(range(20, 28)):
        name = spec.chord_at(bar)
        boom(bar * 4, i + 1, energy=1.02, fill=(i == 7), shaker=True)
        keys(T(bar * 4), spec.beat_s * 2.4, name, 0.8)
        keys(T(bar * 4 + 2.5), spec.beat_s * 1.2, name, 0.66)
        strings(T(bar * 4), spec.bar_s * 0.98, name, 0.4)
        for b, v in [(0.75, 0.62), (1.25, 0.5), (2.75, 0.64), (3.25, 0.52)]:
            gtr(T(bar * 4 + b), 0.3, name, v, muted=True)
        # bass hook: bounce with octave pops
        root = spec.chords[name]["bass"]
        for (b, off, d, v) in [(0, 0, 1.6, 1.0), (1.75, 12, 0.7, 0.8), (2.5, 7, 0.5, 0.78),
                               (3.0, 12, 0.5, 0.74), (3.5, 10, 0.5, 0.7)]:
            midi = root + off
            if name in ("Bbmaj7", "Fmaj9") and off == 10:
                midi = root + 11
            bass(T(bar * 4 + b), midi, spec.beat_s * d, v, glide=(midi - 2 if b > 0 else None))
        if i in (0, 4):
            crash(T(bar * 4), 0.7)
    # hook melody on vibes + talkbox-free lead: vibes double
    for i, (b, d, m, v, vow) in enumerate(hook_mel):
        bar = 20 + int(b // 4)
        off = b % 4
        t = T(bar * 4 + off)
        vibes(t, [m], d * spec.beat_s, v * 0.9)
    # ---- VOCALISE: harmonised "ah" on the hook melody (3 voices)
    for (b, d, m, v, vow) in hook_mel:
        bar = 20 + int(b // 4)
        off = b % 4
        t = T(bar * 4 + off)
        for vi, steps in enumerate((0, -2, -4)):
            P(parts, "vox", "vox").append(dict(
                t=t + 0.012 * vi, dur=d * spec.beat_s * 0.95, midi=diatonic_shift(m, A_MODE, steps),
                vel=v * (0.72 - 0.12 * vi), vowel=vow, voice=vi, consonant="h" if vi == 0 else ""))

    # ---------------------------------------------------------- verse 2
    for i, bar in enumerate(range(28, 44)):
        name = spec.chord_at(bar)
        boom(bar * 4, i + 2, energy=1.0, fill=(i % 8 == 7), shaker=True)
        keys(T(bar * 4), spec.beat_s * 1.2, name, 0.74)
        keys(T(bar * 4 + 1.75), spec.beat_s * 0.6, name, 0.55)
        keys(T(bar * 4 + 2.5), spec.beat_s * 1.0, name, 0.64)
        for b, v in [(1.5, 0.52), (2.25, 0.42), (3.25, 0.55), (3.75, 0.4)]:
            gtr(T(bar * 4 + b), 0.15, name, v)
        if i % 2 == 0:
            mel = [[74, 72], [70, 69], [67, 65], [69, 72]][(i // 2) % 4]
            vibes(T(bar * 4 + 3.25), mel, 0.9, 0.36)
        root = spec.chords[name]["bass"]
        pat = [[(0, 0, 1.3, 0.98), (1.5, 7, 0.6, 0.72), (2.0, 12, 0.8, 0.82), (3.25, 10, 0.6, 0.68)],
               [(0, 0, 2.0, 0.96), (2.5, 12, 1.0, 0.78), (3.5, 7, 0.5, 0.7)],
               [(0, 0, 1.0, 0.95), (1.0, 0, 0.6, 0.6), (1.75, 12, 0.5, 0.78), (2.5, 10, 0.5, 0.7),
                (3.0, 7, 0.5, 0.72), (3.5, 5, 0.5, 0.6)],
               [(0, 0, 1.6, 0.97), (2.0, 3, 0.5, 0.66), (2.75, 7, 0.5, 0.74), (3.25, 10, 0.5, 0.7)]][i % 4]
        for (b, off, d, v) in pat:
            midi = root + off
            if name in ("Bbmaj7", "Fmaj9") and off == 10:
                midi = root + 11
            if name == "A7b9" and off == 3:
                midi = root + 4
            bass(T(bar * 4 + b), midi, spec.beat_s * d, v, glide=(midi - 2 if b > 0 else None))
        if i % 8 == 4:
            strings(T(bar * 4), spec.bar_s * 0.98, name, 0.3)
    # verse vocalise: soft "mmm" beds at phrase starts
    for bar in (29, 37):
        name = spec.chord_at(bar)
        ch = spec.chords[name]
        for vi, (m, v) in enumerate(zip(voicing(ch, 3, center=60, low=52, high=70),
                                        (0.34, 0.28, 0.24))):
            P(parts, "vox", "vox").append(dict(
                t=T(bar * 4) + 0.02 * vi, dur=spec.bar_s * 1.8, midi=m, vel=v,
                vowel="mmm", voice=vi, consonant=""))

    # ---------------------------------------------------------- bridge
    for i, bar in enumerate(range(44, 52)):
        name = spec.chord_at(bar)
        if i < 4:
            kick(T(bar * 4), 0.85)
            snare(T(bar * 4 + 2), 0.7)
            for k in range(4):
                hat(T(bar * 4 + k), 0.24)
        else:
            boom(bar * 4, i, energy=0.85, fill=(i == 7), shaker=True)
        keys(T(bar * 4), spec.beat_s * 3.7, name, 0.55)
        strings(T(bar * 4), spec.bar_s * 0.98, name, 0.3 + 0.02 * i)
        root = spec.chords[name]["bass"]
        bass(T(bar * 4), root, spec.beat_s * 2.2, 0.72, glide=(root - 2 if i else None), gs=0.12)
        bass(T(bar * 4 + 2.75), root + 7, spec.beat_s * 0.9, 0.55)
        if i == 7:
            for b, v, m in [(3.0, 0.5, 45), (3.25, 0.6, 48), (3.5, 0.72, 50), (3.75, 0.9, 53)]:
                P(parts, "toms", "toms").append((T(bar * 4 + b), H.vel(v), m, 0.2))
    # bridge vocalise: sustained choir pad
    for i, bar in enumerate(range(44, 52, 2)):
        name = spec.chord_at(bar)
        for vi, m in enumerate(voicing(spec.chords[name], 3, center=59, low=50, high=68)):
            P(parts, "vox", "vox").append(dict(
                t=T(bar * 4) + 0.03 * vi, dur=spec.bar_s * 1.9, midi=m,
                vel=0.30 - 0.04 * vi, vowel="oo", voice=vi, consonant=""))

    # ---------------------------------------------------------- hook 2
    for i, bar in enumerate(range(52, 60)):
        name = spec.chord_at(bar)
        boom(bar * 4, i, energy=1.04, fill=(i == 7), shaker=True)
        keys(T(bar * 4), spec.beat_s * 2.4, name, 0.82)
        keys(T(bar * 4 + 2.5), spec.beat_s * 1.2, name, 0.68)
        strings(T(bar * 4), spec.bar_s * 0.98, name, 0.44)
        for b, v in [(0.75, 0.6), (1.25, 0.5), (2.75, 0.62), (3.25, 0.52)]:
            gtr(T(bar * 4 + b), 0.3, name, v)
        root = spec.chords[name]["bass"]
        for (b, off, d, v) in [(0, 0, 1.6, 1.0), (1.75, 12, 0.7, 0.82), (2.5, 7, 0.5, 0.8),
                               (3.0, 12, 0.5, 0.76), (3.5, 10, 0.5, 0.72)]:
            midi = root + off
            if name in ("Bbmaj7", "Fmaj9") and off == 10:
                midi = root + 11
            bass(T(bar * 4 + b), midi, spec.beat_s * d, v, glide=(midi - 2 if b > 0 else None))
        if i in (0, 4):
            crash(T(bar * 4), 0.72)
    for (b, d, m, v, vow) in hook_mel:
        bar = 52 + int(b // 4)
        off = b % 4
        t = T(bar * 4 + off)
        vibes(t, [m], d * spec.beat_s, v)
        for vi, steps in enumerate((0, -2, -4)):
            P(parts, "vox", "vox").append(dict(
                t=t + 0.012 * vi, dur=d * spec.beat_s * 0.95,
                midi=diatonic_shift(m, A_MODE, steps), vel=v * (0.74 - 0.12 * vi),
                vowel=vow, voice=vi, consonant="h" if vi == 0 else ""))
    # counter-melody (flute) in the second half of hook 2
    for bar in (56, 57, 58, 59):
        P(parts, "flute", "flute").append(dict(t=T(bar * 4 + 2.5), dur=spec.beat_s * 1.2,
                                               midi=[81, 79, 77, 74][bar - 56], vel=0.42, pan=0.18))

    # ---------------------------------------------------------- outro
    for i, bar in enumerate(range(60, 64)):
        name = spec.chord_at(bar)
        boom(bar * 4, i, energy=0.92 - 0.03 * i, fill=(i == 3), shaker=True)
        keys(T(bar * 4), spec.beat_s * 2.6, name, 0.68 - 0.04 * i)
        strings(T(bar * 4), spec.bar_s * 1.0, name, 0.36)
        root = spec.chords[name]["bass"]
        bass(T(bar * 4), root, spec.beat_s * 1.8, 0.8, glide=(root + 2 if i == 0 else None))
        bass(T(bar * 4 + 2.5), root + 7, spec.beat_s * 0.8, 0.6)
    # last chord rings
    for i, m in enumerate(voicing(spec.chords["Dm9"], 5, center=60, low=50, high=76)):
        keys(T(64 * 4) + 0.004 * i, spec.beat_s * 8.0, "Dm9", 0.7)
    strings(T(64 * 4), spec.bar_s * 3.0, "Dm9", 0.45)
    bass(T(64 * 4), 38, spec.beat_s * 6.0, 0.85)
    kick(T(64 * 4), 0.9); crash(T(64 * 4), 0.6)
    for vi, m in enumerate(voicing(spec.chords["Dm9"], 3, center=59, low=50, high=70)):
        P(parts, "vox", "vox").append(dict(t=T(64 * 4) + 0.03 * vi, dur=spec.beat_s * 9.0,
                                           midi=m, vel=0.3 - 0.05 * vi, vowel="mmm",
                                           voice=vi, consonant=""))
    # vinyl texture + transitions
    P(parts, "fx", "fx").append(dict(t=0.0, type="vinyl", dur=spec.bars * spec.bar_s,
                                     level=0.008))
    for t, typ, kw in [(3 * 4 + 2, "rev_crash", dict(dur=2.0, vel=0.5)),
                       (19 * 4 + 3, "rev_crash", dict(dur=1.6, vel=0.45)),
                       (43 * 4 + 2, "riser", dict(dur=2.2)),
                       (51 * 4 + 2, "riser", dict(dur=2.2)),
                       (59 * 4 + 2.5, "rev_crash", dict(dur=1.4, vel=0.4))]:
        P(parts, "fx", "fx").append(dict(t=T(t), type=typ, **kw))
    return parts


# =====================================================================
# B. AFTERGLOW PROTOCOL — melodic trap / trap-soul, 144 BPM, F minor
# =====================================================================
B = SongSpec(
    slug="afterglow_protocol", title="Afterglow Protocol",
    genre="melodic trap / trap-soul",
    bpm=144, key_name="F minor", bars=80, tail_s=7.0, swing=0.52,
    sections=[("intro", 0, 8), ("verse1", 8, 16), ("hook1", 24, 8),
              ("verse2", 32, 16), ("bridge", 48, 8), ("hook2", 56, 8),
              ("outro", 64, 16)],
    chords={
        "Fm9":    chord_dict(29, {5, 8, 0, 3, 7}, 60),
        "Dbmaj9": chord_dict(37, {1, 5, 8, 0, 3}, 58),
        "Abmaj7": chord_dict(32, {8, 0, 3, 7}, 60),
        "Eb6":    chord_dict(39, {3, 7, 10, 0, 5}, 59),
        "Bbm9":   chord_dict(34, {10, 1, 5, 8, 0}, 58),
    },
    prog={
        "intro": ["Fm9", "Fm9", "Dbmaj9", "Dbmaj9", "Abmaj7", "Abmaj7", "Eb6", "Eb6"],
        "verse": ["Fm9", "Fm9", "Dbmaj9", "Dbmaj9", "Abmaj7", "Eb6", "Bbm9", "Eb6"],
        "hook": ["Fm9", "Dbmaj9", "Abmaj7", "Eb6", "Fm9", "Dbmaj9", "Bbm9", "Eb6"],
        "bridge": ["Dbmaj9", "Abmaj7", "Dbmaj9", "Eb6", "Dbmaj9", "Abmaj7", "Eb6", "Eb6"],
        "outro": ["Fm9", "Dbmaj9", "Abmaj7", "Eb6", "Fm9", "Dbmaj9", "Fm9", "Fm9"],
    },
    dynamics=dict(intro=0.74, verse1=0.97, hook1=1.06, verse2=0.98, bridge=0.7,
                  hook2=1.07, outro=0.88),
    master_eq=[("peak", 55, -0.8, 1.0), ("peak", 180, 1.6, 1.0),
               ("peak", 700, -1.2, 0.9), ("peak", 2200, 1.8, 0.7),
               ("high_shelf", 9000, 1.2, 0.7)],
    vocal_pct=5.5,
)

B_MODE = {5, 7, 8, 10, 0, 1, 3}   # F natural minor


def build_B(spec: SongSpec):
    H = Humanizer(seed=202, timing_ms=3.2, vel_pct=0.045, swing=spec.swing)
    parts = {}
    T = lambda b, f=0.9: H.time(b, f, spec)  # noqa: E731

    def kick(t, v): P(parts, "kick808", "kick808").append((t, H.vel(v)))
    def snare(t, v): P(parts, "snare", "snare").append((t, H.vel(v)))
    def clap(t, v): P(parts, "clap", "clap").append((t, H.vel(v)))
    def hat(t, v, opn=False, pan=0.14): P(parts, "hats", "hats").append((t, H.vel(v), opn, pan))
    def crash(t, v, rev=False): P(parts, "crash", "crash").append((t, H.vel(v), rev))
    def sub(t, midi, dur, v, glide=None, gs=0.09):
        P(parts, "sub808", "sub808").append(dict(t=t, midi=midi, dur=dur, vel=H.vel(v),
                                                 glide_from=glide, glide_s=gs))
    def bell(t, notes, dur, v, pan=0.0):
        P(parts, "bell", "bell").append(dict(t=t, dur=dur, notes=notes, vel=H.vel(v), pan=pan))
    def pad(t, name, dur, v, cutoff=2300):
        ch = spec.chords[name]
        P(parts, "pad", "pad").append(dict(t=t, dur=dur,
                                           notes=voicing(ch, 5, ch["center"], 45, 74),
                                           vel=H.vel(v), cutoff=cutoff, attack=0.7))
    def keys(t, name, dur, v):
        ch = spec.chords[name]
        P(parts, "rhodes", "rhodes").append(dict(t=t, dur=dur,
                                                 notes=voicing(ch, 4, ch["center"]), vel=H.vel(v)))

    # ------------------------------------------------------------ hats
    def trap_hats(t0, energy=1.0, roll_bar=False, triplets=False):
        for i in range(16):
            b = i * 0.25
            acc = 0.55 if i % 4 == 0 else (0.4 if i % 2 == 0 else 0.28)
            hat(T(t0 + b), acc * energy)
        if triplets:
            for i in range(6):
                hat(T(t0 + 2 + i / 6.0), 0.26 * energy)
        if roll_bar:
            for i in range(16):
                hat(T(t0 + 3 + i / 16.0), (0.22 + 0.03 * i) * energy)
            for i in range(8):
                hat(T(t0 + 3.5 + i / 16.0), (0.3 + 0.04 * i) * energy)

    def trap_drums(t0, variant, energy=1.0, fill=False, rolls=False):
        kicks = [[(0, 1.0), (2.5, 0.85)],
                 [(0, 1.0), (1.75, 0.7), (2.75, 0.82)],
                 [(0, 1.0), (3.0, 0.75)],
                 [(0, 1.0), (1.5, 0.6), (2.5, 0.9)]][variant % 4]
        for b, v in kicks:
            kick(T(t0 + b), v * energy)
        snare(T(t0 + 2), 0.95 * energy)       # half-time backbeat
        clap(T(t0 + 2), 0.8 * energy)
        if variant % 4 == 1:
            snare(T(t0 + 3.75), 0.22 * energy)
        trap_hats(t0, energy, roll_bar=rolls, triplets=(variant % 4 == 2))
        if fill:
            for b, v, m in [(3.0, 0.4, 45), (3.25, 0.5, 48), (3.5, 0.62, 50), (3.75, 0.8, 53)]:
                P(parts, "toms", "toms").append((T(t0 + b), H.vel(v * energy), m, 0.18))

    # ------------------------------------------------------------ intro
    for i, bar in enumerate(range(0, 8)):
        name = spec.chord_at(bar)
        pad(T(bar * 4), name, spec.bar_s * 1.05, 0.34 + 0.02 * i)
        if i >= 2:
            keys(T(bar * 4), name, spec.beat_s * 3.2, 0.4)
        if i >= 4:
            mel = [[77, 80, 84], [75, 79, 82], [72, 75, 79], [70, 74, 77]][(bar - 4) % 4]
            for j, m in enumerate(mel):
                bell(T(bar * 4 + j * 0.5), [m], 0.9, 0.34, pan=-0.2 + 0.2 * j)
    for bar in (4, 5, 6, 7):
        sub(T(bar * 4), 29, spec.beat_s * 3.4, 0.6)
    crash(T(7 * 4 + 3.0), 0.5, rev=True)
    P(parts, "fx", "fx").append(dict(t=T(7 * 4 + 2), type="riser", dur=spec.bar_s * 1.9))

    # ------------------------------------------------------------ verse 1
    melody_v = [
        (0.0, 0.5, 72), (0.75, 0.5, 75), (1.5, 1.0, 77), (3.0, 0.5, 75),
        (4.0, 0.75, 72), (5.0, 0.5, 70), (5.75, 1.25, 68), (7.5, 0.5, 67),
        (8.0, 0.5, 65), (8.5, 0.5, 68), (9.5, 1.0, 72), (11.0, 0.75, 70),
        (12.0, 0.5, 72), (12.75, 0.5, 75), (13.5, 1.5, 77), (15.5, 0.5, 80),
    ]
    for i, bar in enumerate(range(8, 24)):
        name = spec.chord_at(bar)
        trap_drums(bar * 4, i, energy=0.95 + 0.01 * min(i, 5), fill=(i % 8 == 7),
                   rolls=(i % 4 == 3))
        pad(T(bar * 4), name, spec.bar_s * 1.02, 0.28)
        if i % 2 == 0:
            keys(T(bar * 4), name, spec.beat_s * 2.0, 0.42)
        root = spec.chords[name]["bass"]
        # 808: long notes with glides, re-articulated on the kick grid
        for (b, off, d, v, gl) in [(0, 0, 2.0, 0.95, None), (2.5, 0, 1.2, 0.8, None),
                                   (3.75, 7, 0.25, 0.55, 0)]:
            midi = root + off
            sub(T(bar * 4 + b), midi, spec.beat_s * d, v,
                glide=root + gl if gl is not None else None)
        if i % 8 == 7:
            sub(T(bar * 4 + 3.5), root + 5, spec.beat_s * 0.4, 0.6, glide=root)
        # bell melody (first 8 bars), then sparse answers
        if i < 8:
            for (b, d, m) in melody_v:
                if int(b // 4) == i:
                    bell(T(bar * 4 + (b % 4)), [m], d * spec.beat_s, 0.5, pan=0.05)
        else:
            if i % 4 == 1:
                for j, m in enumerate([77, 75, 72]):
                    bell(T(bar * 4 + 2.5 + j * 0.5), [m], 0.8, 0.38, pan=-0.15)
        if i % 8 == 4:
            crash(T(bar * 4), 0.5)
    # verse vocalise: airy "ooh" bed, two entrances
    for bar in (12, 20):
        name = spec.chord_at(bar)
        for vi, m in enumerate(voicing(spec.chords[name], 3, 62, 52, 72)):
            P(parts, "vox", "vox").append(dict(t=T(bar * 4) + 0.02 * vi,
                                               dur=spec.bar_s * 2.2, midi=m,
                                               vel=0.3 - 0.04 * vi, vowel="oo", voice=vi,
                                               consonant=""))

    # ------------------------------------------------------------ hook 1
    hook_bells = [
        (0.0, 0.5, 77), (0.5, 0.5, 80), (1.0, 1.5, 84), (2.75, 0.5, 82),
        (3.5, 1.0, 80), (4.0, 0.5, 79), (4.75, 0.5, 79), (5.25, 1.25, 75),
        (6.5, 0.5, 72), (7.0, 1.0, 75), (8.0, 0.5, 72), (8.5, 0.5, 75),
        (9.0, 1.5, 80), (10.75, 0.5, 79), (11.5, 1.0, 77), (12.5, 0.5, 75),
        (13.0, 0.75, 72), (14.0, 2.0, 79),
    ]
    for i, bar in enumerate(range(24, 32)):
        name = spec.chord_at(bar)
        trap_drums(bar * 4, i + 1, energy=1.03, fill=(i == 7), rolls=(i % 2 == 1))
        pad(T(bar * 4), name, spec.bar_s * 1.02, 0.36)
        keys(T(bar * 4), name, spec.beat_s * 2.2, 0.5)
        root = spec.chords[name]["bass"]
        for (b, off, d, v, gl) in [(0, 0, 1.8, 1.0, None), (2.0, 0, 0.9, 0.85, 7),
                                   (3.0, 0, 0.8, 0.8, None)]:
            sub(T(bar * 4 + b), root + off, spec.beat_s * d, v,
                glide=root + gl if gl is not None else None)
        if i in (0, 4):
            crash(T(bar * 4), 0.7)
    for (b, d, m) in hook_bells:
        bar = 24 + int(b // 4)
        bell(T(bar * 4 + (b % 4)), [m], d * spec.beat_s, 0.6, pan=-0.1)
    # hook vocalise: wordless "ah" line (2 voices: melody + 3rd)
    for (b, d, m) in hook_bells:
        bar = 24 + int(b // 4)
        t = T(bar * 4 + (b % 4))
        for vi, steps in enumerate((0, -2)):
            P(parts, "vox", "vox").append(dict(
                t=t + 0.01 * vi, dur=d * spec.beat_s * 0.9,
                midi=diatonic_shift(m, B_MODE, steps), vel=0.5 - 0.08 * vi,
                vowel="ah" if vi == 0 else "oo", voice=vi,
                consonant="h" if vi == 0 else ""))

    # ------------------------------------------------------------ verse 2
    for i, bar in enumerate(range(32, 48)):
        name = spec.chord_at(bar)
        trap_drums(bar * 4, i + 2, energy=1.0, fill=(i % 8 == 7), rolls=(i % 4 == 3))
        pad(T(bar * 4), name, spec.bar_s * 1.02, 0.3)
        if i % 4 in (0, 2):
            keys(T(bar * 4), name, spec.beat_s * 1.6, 0.44)
        root = spec.chords[name]["bass"]
        for (b, off, d, v, gl) in [(0, 0, 1.4, 0.98, None), (1.75, 3, 0.5, 0.65, None),
                                   (2.5, 0, 1.0, 0.86, None), (3.5, 7, 0.5, 0.6, 0)]:
            midi = root + off
            sub(T(bar * 4 + b), midi, spec.beat_s * d, v,
                glide=root + gl if gl is not None else None)
        if i % 4 == 2:
            for j, m in enumerate([84, 82, 80, 77]):
                bell(T(bar * 4 + 2.0 + j * 0.5), [m], 0.7, 0.4, pan=0.1 - 0.05 * j)
        if i % 8 == 6:
            for vi, m in enumerate(voicing(spec.chords[name], 3, 64, 54, 74)):
                P(parts, "vox", "vox").append(dict(t=T(bar * 4) + 0.03 * vi,
                                                   dur=spec.bar_s * 1.6, midi=m,
                                                   vel=0.28 - 0.03 * vi, vowel="mmm",
                                                   voice=vi, consonant=""))

    # ------------------------------------------------------------ bridge
    for i, bar in enumerate(range(48, 56)):
        name = spec.chord_at(bar)
        pad(T(bar * 4), name, spec.bar_s * 1.05, 0.42, cutoff=1800)
        if i < 4:
            kick(T(bar * 4), 0.8)
            for k in range(8):
                hat(T(bar * 4 + k * 0.5), 0.3)
            sub(T(bar * 4), spec.chords[name]["bass"], spec.beat_s * 3.4, 0.7)
        else:
            trap_drums(bar * 4, i, energy=0.9, rolls=True)
            sub(T(bar * 4), spec.chords[name]["bass"], spec.beat_s * 2.0, 0.8)
            sub(T(bar * 4 + 2.5), spec.chords[name]["bass"] + 7, spec.beat_s * 1.0, 0.6,
                glide=spec.chords[name]["bass"])
        for vi, m in enumerate(voicing(spec.chords[name], 3, 62, 50, 72)):
            P(parts, "vox", "vox").append(dict(t=T(bar * 4) + 0.04 * vi,
                                               dur=spec.bar_s * 1.9,
                                               midi=diatonic_shift(m, B_MODE, -1 if i % 2 else 0),
                                               vel=0.34 - 0.05 * vi, vowel="oo",
                                               voice=vi, consonant=""))
    # build into hook 2
    for k in range(8):
        snare(T(55 * 4 + k * 0.5), 0.28 + 0.04 * k)
    for k in range(16):
        snare(T(55 * 4 + 2 + k * 0.125), 0.3 + 0.03 * k)
    P(parts, "fx", "fx").append(dict(t=T(55 * 4 - 4), type="riser", dur=spec.bar_s * 2.2))

    # ------------------------------------------------------------ hook 2
    for i, bar in enumerate(range(56, 64)):
        name = spec.chord_at(bar)
        trap_drums(bar * 4, i + 3, energy=1.06, fill=(i == 7), rolls=(i % 2 == 1))
        pad(T(bar * 4), name, spec.bar_s * 1.02, 0.4)
        keys(T(bar * 4), name, spec.beat_s * 2.2, 0.54)
        root = spec.chords[name]["bass"]
        for (b, off, d, v, gl) in [(0, 0, 1.8, 1.0, None), (2.0, 0, 0.9, 0.86, 7),
                                   (3.0, 0, 0.8, 0.82, None)]:
            sub(T(bar * 4 + b), root + off, spec.beat_s * d, v,
                glide=root + gl if gl is not None else None)
        if i in (0, 4):
            crash(T(bar * 4), 0.75)
    for (b, d, m) in hook_bells:
        bar = 56 + int(b // 4)
        t = T(bar * 4 + (b % 4))
        bell(t, [m], d * spec.beat_s, 0.62, pan=-0.1)
        for vi, steps in enumerate((0, -2, -4)):
            P(parts, "vox", "vox").append(dict(
                t=t + 0.01 * vi, dur=d * spec.beat_s * 0.9,
                midi=diatonic_shift(m, B_MODE, steps), vel=0.5 - 0.07 * vi,
                vowel="ah" if vi == 0 else "oo", voice=vi,
                consonant="h" if vi == 0 else ""))
    # counter-line: flute in the second half
    for j, bar in enumerate(range(60, 64)):
        for k, m in enumerate([[84, 82, 80], [79, 77, 75], [77, 75, 72], [75, 72, 68]][j]):
            P(parts, "flute", "flute").append(dict(t=T(bar * 4 + 2.0 + k * 0.75),
                                                   dur=spec.beat_s * 0.7, midi=m,
                                                   vel=0.36, pan=0.2))

    # ------------------------------------------------------------ outro
    for i, bar in enumerate(range(64, 72)):
        name = spec.chord_at(bar)
        pad(T(bar * 4), name, spec.bar_s * 1.05, 0.4 - 0.01 * i, cutoff=2000)
        if i < 4:
            trap_drums(bar * 4, i, energy=0.9 - 0.05 * i, rolls=(i == 3))
            sub(T(bar * 4), spec.chords[name]["bass"], spec.beat_s * 2.2, 0.8)
        for j, m in enumerate(voicing(spec.chords[name], 3, 74, 66, 86)):
            if i % 2 == 0:
                bell(T(bar * 4 + j * 0.5), [m], 1.2, 0.3, pan=-0.2 + 0.2 * j)
    # final hit
    for j, m in enumerate(voicing(spec.chords["Fm9"], 5, 60, 48, 78)):
        keys(T(72 * 4), "Fm9", spec.beat_s * 8.0, 0.6)
        P(parts, "pad", "pad").append(dict(t=T(72 * 4), dur=spec.beat_s * 8.0,
                                           notes=voicing(spec.chords["Fm9"], 5, 62, 46, 78),
                                           vel=0.36, cutoff=2200, attack=0.3))
        break
    sub(T(72 * 4), 29, spec.beat_s * 7.0, 0.85)
    kick(T(72 * 4), 0.9); crash(T(72 * 4), 0.65)
    for vi, m in enumerate(voicing(spec.chords["Fm9"], 3, 62, 50, 72)):
        P(parts, "vox", "vox").append(dict(t=T(72 * 4) + 0.04 * vi, dur=spec.beat_s * 8.0,
                                           midi=m, vel=0.3 - 0.05 * vi, vowel="oo",
                                           voice=vi, consonant=""))
    P(parts, "fx", "fx").append(dict(t=0.0, type="vinyl", dur=spec.bars * spec.bar_s,
                                     level=0.006))
    for t, typ, kw in [(23 * 4 + 3, "rev_crash", dict(dur=1.6, vel=0.5)),
                       (31 * 4 + 2.5, "rev_crash", dict(dur=1.4, vel=0.45)),
                       (47 * 4 + 2, "rev_crash", dict(dur=1.8, vel=0.5)),
                       (63 * 4 + 2.5, "rev_crash", dict(dur=1.6, vel=0.45))]:
        P(parts, "fx", "fx").append(dict(t=T(t), type=typ, **kw))
    return parts


REGISTRY = {
    "paper_lanterns": (A, build_A),
    "afterglow_protocol": (B, build_B),
}
