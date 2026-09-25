"""Preserve the preprocessing source tree next to the generated engine NPZ."""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

def main():
    source=Path(__file__).resolve().parent
    target=source.parent/'data_phm/processed_health_tokens/engine'
    target.mkdir(parents=True,exist_ok=True)
    hashes={}
    excluded = {
        'datasets/fuelcell', 'datasets/_archive_battery_cycle_trajectory_20260824',
        'cnn_lstm_attention_baselines', 'xgboost_baselines', '__pycache__',
    }
    with zipfile.ZipFile(target/'preprocessing_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(source.rglob('*')):
            relative = p.relative_to(source).as_posix()
            if any(relative == item or relative.startswith(item + '/') for item in excluded):
                continue
            if p.is_file() and p.suffix in {'.py','.md'}:
                key=str(p.relative_to(source.parent)).replace('\\','/')
                payload=p.read_bytes();hashes[key]=hashlib.sha256(payload).hexdigest();z.writestr(key,payload)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    (target/'preprocessing_sources.json').write_text(json.dumps(dict(git_head=commit,
        note='Snapshot includes existing local modifications; per-file hashes identify exact working sources',
        files=hashes),indent=2))
    print('Packaged',len(hashes),'source/documentation files')

if __name__=='__main__':main()
