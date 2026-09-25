import copy
import torch
from config import settings
from routing import Router

def build(method):
    cfg=settings('joint_'+method)
    r=Router(cfg);named=[('encoder.transformer.layers.0.weight',torch.zeros(2)),('encoder.transformer.layers.1.weight',torch.zeros(2))]
    r.initialize(named,(0,1),cfg['domains']);return r

def main():
    for method in ('cosine','learned','layerwise'):
        r=build(method);g=torch.ones(4);ref=torch.tensor([1.,1.,-1.,-1.])
        for step in range(100):out=r.apply('bearing',g,ref,step)
        assert torch.isfinite(out).all() and out.min()>=.05 and out.max()<=1.
        if method=='layerwise':assert out[0]>.5 and out[-1]<.5
        clone=build(method);clone.load_state_dict(copy.deepcopy(r.state_dict()))
        torch.testing.assert_close(r.apply('bearing',g,ref,101),clone.apply('bearing',g,ref,101),rtol=0,atol=0)
        before=r.theta['bearing'].clone();r.apply('bearing',g,ref,102,False)
        assert torch.equal(before,r.theta['bearing'])
        r=build(method)
        for step in range(100):up=r.apply('bearing',g,g,step);down=r.apply('battery',g,-g,step)
        assert float(up.mean())>float(down.mean())
    print('ROUTING_DIRECTION_BOUNDS_AND_RESUME_PASSED')

if __name__=='__main__':main()
