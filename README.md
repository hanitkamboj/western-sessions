# Western Sessions — Orion Music & Hanit Kamboj

Original instrumentals, written, performed and mixed in-session. **No samples, no
borrowed compositions, no loops** — every note comes from a synthesis + performance
engine in `code/`.

Each track is delivered twice:
* **VOCAL** — the same arrangement with a wordless vocalise layer (vowel-formant
  voices, legato, vibrato, breath). The layer is level-matched by subtraction so it
  measures 5.5-6.5% of the mix energy — an atmosphere, never a lead vocal.
* **INSTRUMENTAL** — identical mix without that layer.

24-bit / 48 kHz stereo WAV masters, plus 320 kbps MP3 previews, MIDI sessions and
loudness analysis for every track.

| Track | Genre | Key · BPM | Length | Master | Vocalise | Files |
|---|---|---|---|---|---|---|
| Afterglow Protocol | melodic trap / trap-soul | F minor · 144 | 2:20 | -9.28 LUFS | 5.5% | 5 |
| Concrete Lullaby | drill | G minor · 142 | 2:07 | -9.26 LUFS | 5.5% | 4 |
| Golden Hour Lagos | afro-fusion / afrobeats | A minor · 108 | 2:47 | -9.28 LUFS | 6.0% | 4 |
| Lowrider Neon | modern West Coast / G-funk bounce | F# minor · 98 | 2:53 | -9.27 LUFS | 6.0% | 4 |
| Paper Lanterns | boom-bap soul / instrumental hip-hop | D minor · 87 | 3:14 | -9.33 LUFS | 6.0% | 5 |
| Rain On Vinyl | lo-fi jazz-hop | C minor · 82 | 3:15 | -9.27 LUFS | 6.0% | 5 |
| Velvet Static | neo-soul / future R&B | Eb major · 84 | 3:08 | -9.25 LUFS | 6.5% | 5 |
| Western Ave Sundown | — | — · — | 3:03 | — LUFS | —% | 0 |

## Folders
`tracks/<song>/master` audio · `tracks/<song>/session` MIDI + arrangement JSON ·
`tracks/<song>/analysis` metrics and loudness plot · `code/` the whole engine ·
`PRODUCTION.md` method and known limitations.

## Honest limitations
* Instruments are physically modelled, not sampled performances.
* Masters are loud (-10 to -9 LUFS integrated); streaming normalises to ≈ -14 LUFS.
* No Dolby Atmos ADM was produced (no compatible renderer/QC workflow here).
