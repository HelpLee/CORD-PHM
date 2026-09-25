"""Copy the current NPZ generation, verifying every copy with SHA256."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import shutil


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def archive(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        before = digest(path)
        shutil.copy2(path, target)
        if digest(target) != before or digest(path) != before:
            raise RuntimeError(f'Archive mismatch or changing source: {path}')
        records.append(dict(path=path.relative_to(source).as_posix(), bytes=path.stat().st_size, sha256=before))
        print(f'Archived {path.name}', flush=True)
    (destination / 'archive_manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    return destination


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1] / 'data_phm'
    print(archive(root / 'processed_health_tokens', root / ('archive_health_tokens_' + datetime.now().strftime('%Y%m%d_%H%M%S'))))
