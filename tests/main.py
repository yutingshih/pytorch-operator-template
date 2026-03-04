import argparse

import torch

import kernels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, help="Length of vector to be computed")
    return parser.parse_args()


def main():
    args = parse_args()
    in1 = torch.randn(args.N, device="cuda")
    in2 = torch.randn(args.N, device="cuda")
    out = kernels.vadd(in1, in2)

    print(f"in1:\n{in1}")
    print(f"in2:\n{in2}")
    print(f"out:\n{out}")


if __name__ == "__main__":
    main()
