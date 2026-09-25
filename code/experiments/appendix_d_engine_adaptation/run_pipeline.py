"""Verify, calibrate, then run downstream."""
import argparse, os, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--gpu',default='0');args=parser.parse_args();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(args.gpu)
subprocess.run([sys.executable,str(HERE/'experiment.py'),'--verify-only'],cwd=HERE,env=env,check=True)
subprocess.run([sys.executable,str(HERE/'experiment.py')],cwd=HERE,env=env,check=True)
