"""Verify real checkpoint equivalence, data paths, freezing and RUL forward."""
import importlib.util
import sys
import torch
import run_experiment as run
import downstream_interval_val200 as down
from adapter_encoder import MillingAdapterEncoder

torch.set_num_threads(4)
source, converted = run.convert_checkpoint()
sys.path.insert(0,str(run.UPSTREAM_PACKAGE/'code'))
spec=importlib.util.spec_from_file_location('joint_upstream_adapter',run.UPSTREAM_PACKAGE/'code/joint_model.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
upstream=module.JointModel().encoder.eval()
upstream.load_state_dict(source,strict=True)
encoder=MillingAdapterEncoder(channels=3,dropout=.05).eval()
encoder.load_state_dict(converted,strict=True)
assert len(converted)==58
store=run.load_store()
norm=store.normalize(('C1','C4'))
seq=down.full_sequences(store,('C6',))[:1]
values=tuple(torch.as_tensor(v) for v in store.transform(seq.reshape(-1),norm))
with torch.no_grad():
    expected,local=upstream('milling',*values)
    actual=encoder(*values)
    torch.testing.assert_close(expected,actual['snapshot'],atol=0,rtol=0)
    torch.testing.assert_close(local,actual['local_hidden'],atol=0,rtol=0)
model=down.ToolAdapterTCNModel().eval()
model.encoder.load_state_dict(converted,strict=True)
with torch.no_grad():
    pred=model(*(v.reshape(1,20,*v.shape[1:]) for v in values))
assert pred.shape==(1,) and torch.isfinite(pred).all()
for stage in (1,2):
    down.configure_partial(model,stage)
    for name,p in model.encoder.named_parameters():
        expected_trainable=stage==2 and name.startswith(('transformer.layers.1.','final_norm.','adapters.1.'))
        assert p.requires_grad==expected_trainable,name
for p in model.parameters():p.requires_grad_(True)
opt=down.optimizer_for_full(model)
assert len(opt.param_groups)==2
assert opt.param_groups[0]['lr']==.001 and opt.param_groups[1]['lr']==.0001
split_counts={}
for f in (.1,.2,1.0):
    down.LABEL_FRACTION=f
    train,val,counts=run.interval_split(store)
    split_counts[str(f)]={'train':len(train),'validation':len(val),'test':len(down.full_sequences(store,('C6',)))}
result=dict(passed=True,checkpoint=str(run.JOINT),strict_load_tensors=len(converted),
            exact_upstream_forward=True,real_sequence_shape=[1,20,3,64,26],
            partial_freezing_correct=True,full_lr_correct=True,splits=split_counts,
            note='Verification used currently available checkpoint; launch reconverts final best after upstream completion.')
run.write(run.PACKAGE/'verification.json',result)
print(result)
