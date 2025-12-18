import numpy as np
import argparse

# python convert_for_plotting.py --file your_input.npy --out fixed_output.npy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to input .npy file")
    parser.add_argument("--out", required=True, help="Path to save output .npy file")
    args = parser.parse_args()

    # Load the data
    data = np.load(args.file)

    # 1. Select only the first channel (index 0) to drop the second channel
    # 2. Reshape to (1, X, Y, 1)
    # Input shape: (X, Y, 2) -> Intermediate: (X, Y) -> Output: (1, X, Y, 1)
    new_data = data[:, :, 0].reshape(1, data.shape[0], data.shape[1])

    np.save(args.out, new_data)
    print(f"Saved {args.out} with shape {new_data.shape}")

if __name__ == "__main__":
    main()