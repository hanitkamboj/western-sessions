import sys, os, json, time
sys.path.insert(0, "/home/user/modern_sessions")
from studio import engine, songs

def run(slug, formats=("wav","flac","mp3")):
    spec, build = songs.REGISTRY[slug]
    t0 = time.time()
    res, n, parts = engine.run_song(spec, build)
    paths, metrics = engine.save_song(spec, res, n, parts, formats=formats)
    engine.export_midi(spec, parts)
    print(f"[{slug}] done in {time.time()-t0:.0f}s")
    print(json.dumps(metrics, indent=1)[:1200])
    return metrics

if __name__ == "__main__":
    for slug in sys.argv[1:]:
        run(slug)
