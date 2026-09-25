import hashlib
import numpy as np
import torch

def st(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def hh(weights):
    return hashlib.sha256(b''.join(v.numpy().tobytes() for k,v in sorted(weights.items()))).hexdigest()

@torch.no_grad()
def score(model,gpu,sequences,masks,y):
    model.eval()
    predictions=[]
    for start in range(0,len(sequences),8):
        seq=torch.as_tensor(sequences[start:start+8],device='cuda')
        mask=torch.as_tensor(masks[start:start+8],device='cuda')
        predictions.extend(model(*(v[seq] for v in gpu),mask).float().cpu().tolist())
    pred=np.asarray(predictions,dtype=np.float64)
    target=np.asarray(y[sequences[:,-1]],dtype=np.float64)
    err=pred-target
    return dict(rmse=float(np.sqrt(np.mean(err**2))),mae=float(np.mean(abs(err))),
                r2=float(1-np.sum(err**2)/np.sum((target-target.mean())**2)),
                bias=float(err.mean()),n=len(target)),pred,target
