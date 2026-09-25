"""Check original initialization, identity, routing, masks and gradients."""
import importlib.util
import json
from pathlib import Path
import torch
from joint_model import JointModel, DOMAINS

base=Path(__file__).resolve().parents[1]
reference=base.parent/'joint_upstream500_seed42/code/joint_model.py'
spec=importlib.util.spec_from_file_location('reference_joint',reference)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
torch.set_num_threads(4)
torch.manual_seed(42);original=module.JointModel();original_rng=torch.get_rng_state()
torch.manual_seed(42);model=JointModel()
assert torch.equal(original_rng,torch.get_rng_state())
assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.state_dict().items())
assert sum(p.numel() for p in model.encoder.adapters.parameters())==29520
for d in DOMAINS:
    c={'bearing':8,'battery':1,'milling':3}[d]
    x=torch.randn(2,c,64,26);g=torch.randn(2,c,26)
    cm=torch.ones(2,c,dtype=torch.bool);tm=torch.ones(2,c,64,dtype=torch.bool)
    tm[:,:,-3:]=False
    if c>1: cm[:,-1]=False
    for training in (False,True):
        original.train(training);model.train(training)
        torch.manual_seed(123);a=original.encoder(d,x,g,cm,tm)
        torch.manual_seed(123);b=model.encoder(d,x,g,cm,tm)
        for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb,rtol=0,atol=0)
    model.zero_grad(set_to_none=True)
    loss=model.reconstruction(d,x,g,cm,tm,torch.ones_like(x,dtype=torch.bool),fixed=99)
    loss.backward()
    assert torch.isfinite(loss)
    for other in DOMAINS:
        grads=[p.grad for p in model.encoder.adapters[other].parameters()]
        if other==d:
            assert all(v is not None and torch.isfinite(v).all() for v in grads)
            assert any(v.abs().sum()>0 for v in grads)
        else:assert all(v is None for v in grads)
result=dict(passed=True,original_initialization_identical=True,rng_identical=True,
            zero_adapter_outputs_exact=True,domain_routing_gradients=True,parameters_added=29520)
(base/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
