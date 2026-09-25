import argparse
import subprocess
import sys
from config import DOMAINS,HERE,METHODS

def run(*args):subprocess.run([sys.executable,'-u',*args],cwd=HERE,check=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);args=p.parse_args()
    if args.phase=='baseline':
        for domain in DOMAINS:
            run('downstream.py','--phase','baseline','--model','scratch','--domain',domain,'--fraction','1.','--smoke-train')
            run('downstream.py','--phase','baseline','--model','e37_joint','--domain',domain,'--fraction','1.','--verify-only')
    elif args.phase=='adaptive':
        run('test_routing.py')
        for method in METHODS:
            run('train.py','--arm','joint_'+method,'--smoke','--resume')
        run('train.py','--arm','single_bearing_layerwise','--smoke','--resume')
        for domain in DOMAINS:
            run('downstream.py','--phase','adaptive','--model','smoke_joint_layerwise','--domain',domain,'--fraction','.1','--verify-only')
    else:
        for mode in ('partial','l2sp1','l2sp2'):
            run('downstream.py','--phase','bearing10','--model','e37_joint','--domain','bearing','--fraction','.1','--mode',mode,'--smoke-train')
    print('SMOKE_PASSED',args.phase,flush=True)

if __name__=='__main__':main()
