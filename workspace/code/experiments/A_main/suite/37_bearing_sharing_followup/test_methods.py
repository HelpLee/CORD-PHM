"""CPU mathematical tests and single-domain invariance; no data needed."""
import torch
from methods import bearing_floor, route
from config import ARMS, settings


def main():
    b = torch.tensor([2., 0.]); d = torch.tensor([-1., 3.])
    got, info = bearing_floor(d,b,3)
    torch.testing.assert_close(got, torch.tensor([2/3,3.]))
    assert info['triggered'] and abs(info['dot_after']-4/3)<1e-6
    # Already feasible and zero gradient are unchanged.
    for vec, ref in ((torch.tensor([1.,3.]),b),(d,torch.zeros(2))):
        actual,_ = bearing_floor(vec,ref,3); assert torch.equal(actual,vec)
    for seed in range(20):
        torch.manual_seed(seed); b=torch.randn(71); d=torch.randn(71)
        result,_=bearing_floor(d,b,3)
        assert float(b@result) >= float(b@b)/3 - 1e-4
        correction=result-d
        orthogonal=correction-(correction@b)/(b@b)*b
        assert orthogonal.norm()<1e-5
        single,_=bearing_floor(b,b,1); assert torch.equal(single,b)
    private={0:torch.randn(5)}; rec=torch.randn(9); dyn=torch.randn(9); values=object()
    r, g, p, v=route(rec,dyn,private,values,.1)
    torch.testing.assert_close(r,rec*.1); assert g is dyn and p is private and v is values
    for arm in ARMS:
        cfg=settings(arm); assert cfg['reconstruction_routing']['milling']==1.
        assert cfg['reconstruction_routing']['battery']==(1. if arm.endswith('_component') else .3)
    from continuation import validate_protocol
    from copy import deepcopy
    old=dict(arm='single_battery_conditional',max_epochs=500,patience=30,
             settings=settings('single_battery_conditional'),component_scales={'a':1.})
    new=deepcopy(old);new['max_epochs']=2000
    validate_protocol(old,new)
    for field,value in [('patience',31),('component_scales',{'a':2.})]:
        invalid=deepcopy(new);invalid[field]=value
        try: validate_protocol(old,invalid)
        except RuntimeError: pass
        else: raise AssertionError('Protocol drift not rejected: '+field)
    invalid=deepcopy(new);invalid['settings']['reconstruction_routing']['battery']=.1
    try: validate_protocol(old,invalid)
    except RuntimeError: pass
    else: raise AssertionError('Routing drift not rejected')
    print('E37_MATH_TESTS_PASSED')


if __name__ == '__main__': main()
