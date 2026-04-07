#!/usr/bin/env python3
"""Convert a JSON data file to hex files for $readmemh.

Usage: json-to-hex.py <input.json> <output_dir>

For each entry in the JSON, writes a .hex file to output_dir.
Scalar entries (data is a plain number) produce a single-line hex file.
Memory entries (data is a list) produce one hex value per line.

JSON format:
{
  "arg0": {
    "data": 5,
    "format": {"numeric_type": "bitnum", "is_signed": false, "width": 32}
  },
  "mem0": {
    "data": [1, 2, 3],
    "format": {"numeric_type": "bitnum", "is_signed": false, "width": 32}
  }
}
"""

import json
import math
import os
import sys


def to_hex(value, width, is_signed):
    """Convert an integer value to a zero-padded hex string."""
    hex_digits = math.ceil(width / 4)
    if is_signed and value < 0:
        # Two's complement for negative values.
        value = (1 << width) + value
    return f"{value:0{hex_digits}x}"


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.json> <output_dir>", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    with open(input_path) as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for name, entry in data.items():
        fmt = entry["format"]
        width = fmt["width"]
        is_signed = fmt.get("is_signed", False)
        is_scalar = not isinstance(entry["data"], list)

        hex_path = os.path.join(output_dir, f"{name}.hex")
        with open(hex_path, "w") as f:
            if is_scalar:
                f.write(to_hex(entry["data"], width, is_signed) + "\n")
            else:
                for val in entry["data"]:
                    f.write(to_hex(val, width, is_signed) + "\n")


if __name__ == "__main__":
    main()
