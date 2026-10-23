import sys, os, time, json, subprocess, shutil
sys.path.insert(0, "/home/user/modern_sessions")
LOG = "/tmp/queue.log"
STATUS = "/home/user/modern_sessions/queue_status.json"

def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

def publish(slug):
    """copy deliverable audio into the git staging repo (lossless in .cache)"""
    repo = "/home/user/.cache/gh_repo/tracks"
    src = f"/home/user/modern_sessions/songs/{slug}"
    if not os.path.isdir(src):
        return
    dst = os.path.join(repo, slug)
    os.makedirs(dst, exist_ok=True)
    for sub in ("master", "analysis", "session"):
        if os.path.isdir(os.path.join(src, sub)):
            shutil.copytree(os.path.join(src, sub), os.path.join(dst, sub),
                            dirs_exist_ok=True)
    # keep the workspace light: lossless lives only in the git staging repo
    for f in os.listdir(os.path.join(src, "master")):
        if f.endswith((".wav", ".flac")):
            os.remove(os.path.join(src, "master", f))

if __name__ == "__main__":
    slugs = sys.argv[1:]
    st = {"started": time.time(), "jobs": []}
    for slug in slugs:
        rec = {"slug": slug, "state": "running", "t0": time.time()}
        st["jobs"].append(rec)
        log(f"START {slug}")
        try:
            r = subprocess.run([sys.executable, "-u", "run.py", slug],
                               cwd="/home/user/modern_sessions",
                               capture_output=True, text=True, timeout=900)
            rec["state"] = "ok" if r.returncode == 0 else f"fail:{r.returncode}"
            open(f"/tmp/{slug}.log", "w").write(r.stdout[-8000:] + "\n" + r.stderr[-2000:])
        except subprocess.TimeoutExpired:
            rec["state"] = "timeout"
        rec["secs"] = round(time.time() - rec["t0"], 1)
        publish(slug)
        log(f"DONE {slug} {rec['state']} {rec['secs']}s")
        json.dump(st, open(STATUS, "w"), indent=1)
    st["finished"] = time.time()
    json.dump(st, open(STATUS, "w"), indent=1)
