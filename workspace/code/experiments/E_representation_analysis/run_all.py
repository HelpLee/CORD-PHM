"""Run E1--E4 frozen-representation analyses without encoder training."""
import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', help='cpu or cuda')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    for domain in ('bearing', 'battery', 'milling'):
        for variant in ('single_domain', 'three_domain_no_adapter', 'three_domain_adapter'):
            command = [sys.executable, str(HERE / 'extract_representations.py'),
                       '--domain', domain, '--variant', variant, '--device', args.device]
            if args.verify_only:
                command.append('--verify-only')
            subprocess.run(command, check=True, cwd=HERE)
    if not args.verify_only:
        subprocess.run([sys.executable, str(HERE / 'E1_health_trajectory/analyze.py')],
                       check=True, cwd=HERE)
        subprocess.run([sys.executable, str(HERE / 'E2_cross_domain_degradation_axis_transfer/analyze.py')],
                       check=True, cwd=HERE)
        subprocess.run([sys.executable, str(HERE / 'E3_layerwise_health_information_gain/analyze.py'),
                        '--device', args.device], check=True, cwd=HERE)
        subprocess.run([sys.executable, str(HERE / 'E4_knn_health_consistency/analyze.py'),
                        '--device', args.device, '--k', '5'], check=True, cwd=HERE)


if __name__ == '__main__':
    main()
