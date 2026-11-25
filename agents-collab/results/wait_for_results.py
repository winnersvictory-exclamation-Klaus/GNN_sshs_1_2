"""Wait for a result directory whose visible files have settled."""

from pathlib import Path
import json
import time

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT.parent / "analysis"
INTERVAL = 10
IDLE_LIMIT = 30 * 60


def snapshot():
    pending = {}
    for directory in sorted(ROOT.iterdir()):
        if directory.name.startswith(".") or not directory.is_dir():
            continue
        if (ANALYSIS / directory.name / "findings.md").is_file():
            continue
        files = {}
        for path in directory.rglob("*"):
            relative = path.relative_to(directory)
            if any(part.startswith(".") for part in relative.parts):
                continue
            try:
                if path.is_file():
                    info = path.stat()
                    files[relative] = (info.st_size, info.st_mtime_ns)
            except FileNotFoundError:
                continue
        pending[directory.name] = files
    return pending


def main():
    previous = None
    deadline = time.monotonic() + IDLE_LIMIT
    while True:
        current = snapshot()
        if previous is not None:
            ready = [name for name, files in current.items()
                     if files and files == previous.get(name)]
            if ready:
                print(json.dumps({"ready": ready}))
                return
            if current != previous:
                deadline = time.monotonic() + IDLE_LIMIT
        previous = current
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print("No new results for 30 minutes.")
            return
        time.sleep(min(INTERVAL, remaining))


if __name__ == "__main__":
    main()
