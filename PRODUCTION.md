## How these were made
Every sound is synthesized from first principles in Python (numpy/scipy) — no sample
libraries, no loops, no borrowed compositions. The engine renders each performance
to a 48 kHz float bus, mixes with per-bus EQ/saturation/compression, sends to a
convolution reverb and tempo delay, applies arrangement automation, then masters
(glue comp → tone EQ → mid/side → true-peak limiter) to 24-bit/48 kHz.

- `code/modern_sessions/` — SongSpec engine, instrument bank, songs, queue runner
- `code/western_ave_sundown/` — the original G-funk session

### Vocalise versions
The VOCAL masters add a wordless vocal layer (vowel-formant synthesis with legato,
vibrato, breath and consonant transients). It is level-matched by isolation
(subtract the instrumental from the mix) so it always sits at the requested 5–7%
energy share — never a lead vocal.

### Known limitations
- Instruments are physically-modelled approximations, not sampled performances.
- Masters are loud (-10 to -9 LUFS integrated) to suit urban reference listening;
  streaming platforms will normalise them down to about -14 LUFS.
- No Dolby Atmos ADM was produced (no compatible renderer/QC workflow in this
  environment). Everything is stem-based and immersive-ready if remixed elsewhere.
