"""Khôi phục bộ nộp đã chốt và checkpoint; không huấn luyện lại."""
import argparse
from pathlib import Path
from src.final_submission import prepare_runtime,restore_checkpoints

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',type=Path)
    p.add_argument('--checkpoints',action='store_true')
    p.add_argument('--demo-only',action='store_true')
    a=p.parse_args()
    root=restore_checkpoints(a.destination,a.demo_only) if a.checkpoints or a.demo_only else prepare_runtime(a.destination)
    print(root)
if __name__=='__main__':main()
