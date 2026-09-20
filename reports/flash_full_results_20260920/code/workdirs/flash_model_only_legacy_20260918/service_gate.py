"""Cross-process gate for one shared provisioned API account."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import time


@contextmanager
def service_gate(path=None, interval=75):
    if not path:
        yield
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        lock.seek(0)
        previous = lock.read().strip()
        last = json.loads(previous)['started'] if previous else 0
        time.sleep(max(0, interval - (time.time() - last)))
        lock.seek(0); lock.truncate()
        json.dump({'started': time.time()}, lock); lock.flush()
        # Hold throughout the HTTP request: one in-flight request for this key.
        yield
