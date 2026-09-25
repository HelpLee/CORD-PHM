"""Verified, isolated extension of capped controls, preserving full training state."""
import hashlib
import json
import shutil
import torch
from config import continuation_source


def validate_protocol(old, new):
    # Metadata/settings have a new experiment namespace; objective and numerical
    # protocol must match exactly, including initialized normalization constants.
    ignored = {'fingerprint', 'settings', 'max_epochs', 'device', 'torch_version'}
    for key in (set(old) | set(new)) - ignored:
        if old.get(key) != new.get(key): raise RuntimeError('Continuation protocol drift: '+key)
    active = new['settings']['domains']
    oldcfg, newcfg = old['settings'], new['settings']
    if tuple(oldcfg['domains']) != tuple(active): raise RuntimeError('Domain mismatch')
    if oldcfg['aggregation'] != newcfg['aggregation']: raise RuntimeError('Aggregation mismatch')
    for domain in active:
        oldrho = oldcfg.get('reconstruction_routing', {}).get(domain, 1.) * oldcfg['shared_reconstruction_fraction']
        newrho = newcfg['reconstruction_routing'][domain] * newcfg['shared_reconstruction_fraction']
        if oldrho != newrho: raise RuntimeError('Routing mismatch: '+domain)
    if newcfg['bearing_floor']: raise RuntimeError('Cannot enable a new method in a continued control')
    if old['max_epochs'] >= new['max_epochs']: raise RuntimeError('Not a ceiling extension')


def import_checkpoint(t, arm, protocol, fingerprint):
    source = continuation_source(arm)
    target = t.OUT/'last.pt'
    if source is None or target.exists(): return
    old = json.loads((source/'protocol.json').read_text())
    validate_protocol(old, protocol)
    status = json.loads((source/'training/status.json').read_text())
    if status.get('reason') != 'max_epochs': raise RuntimeError('Only capped controls may be extended')
    ck = torch.load(source/'training/last.pt', map_location='cpu', weights_only=False)
    immutable = {k:v for k,v in old.items() if k not in ('fingerprint','device','torch_version')}
    original_hash = hashlib.sha256(json.dumps(immutable,sort_keys=True).encode()).hexdigest()
    if ck['fingerprint'] != original_hash or old['fingerprint'] != original_hash:
        raise RuntimeError('Original fingerprint mismatch')
    if ck['epoch'] != old['max_epochs'] or ck['stale'] >= old['patience']:
        raise RuntimeError('Original checkpoint is not eligible for cap extension')
    # Best encoder is epoch499, not last.pt(epoch500). Preserve both separately.
    shutil.copy2(source/'training/encoder.pt', t.OUT/'encoder.pt')
    t.write(t.BASE/'continuation_provenance.json', dict(source=str(source),
        old_fingerprint=original_hash, new_fingerprint=fingerprint, epoch=ck['epoch'],
        best_epoch=ck['best_epoch'], stale=ck['stale'], old_max_epochs=old['max_epochs'],
        new_max_epochs=protocol['max_epochs'], preserved='model,optimizer,RNG,best,stale,history'))
    ck['fingerprint'] = fingerprint
    t.save(target, ck)
    t.write(t.OUT/'history.json', ck['history'])
    print('IMPORTED_CONTINUATION',arm,'from',ck['epoch'],'best',ck['best_epoch'],flush=True)
