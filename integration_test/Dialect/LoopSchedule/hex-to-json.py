#!/usr/bin/env python3
"""Convert hex dump files back to JSON data format.

Usage: hex-to-json.py <hex_dir> <reference.json> <output.json>

Reads hex files from hex_dir and uses reference.json for format metadata.
For each entry in reference.json, looks for {name}_out.hex (output dump) or
{name}.hex in hex_dir. Scalars (data is a plain number) produce a single value;
memories (data is a list) produce one value per line.

Writes a JSON file in the same format with updated data values.
"""

import json
import math
import os
import sys


def from_hex(hex_str, width, is_signed):
    """Convert a hex string to an integer value."""
    value = int(hex_str.strip(), 16)
    if is_signed and value >= (1 << (width - 1)):
        value -= 1 << width
    return value


def main():
    if len(sys.argv) != 4:
        print(
            f"Usage: {sys.argv[0]} <hex_dir> <reference.json> <output.json>",
            file=sys.stderr,
        )
        sys.exit(1)

    hex_dir = sys.argv[1]
    ref_path = sys.argv[2]
    output_path = sys.argv[3]

    with open(ref_path) as f:
        ref_data = json.load(f)

    result = {}

    for name, entry in ref_data.items():
        fmt = entry["format"]
        width = fmt["width"]
        is_signed = fmt.get("is_signed", False)
        is_scalar = not isinstance(entry["data"], list)

        # Look for output dump file, then fall back to input file.
        out_hex = os.path.join(hex_dir, f"{name}_out.hex")
        in_hex = os.path.join(hex_dir, f"{name}.hex")
        hex_path = out_hex if os.path.exists(out_hex) else in_hex

        if not os.path.exists(hex_path):
            # Entry not found in dumps, skip.
            continue

        with open(hex_path) as f:
            lines = [line.strip() for line in f if line.strip()]

        new_entry = {"format": fmt}
        if is_scalar:
            new_entry["data"] = from_hex(lines[0], width, is_signed)
        else:
            new_entry["data"] = [from_hex(line, width, is_signed) for line in lines]

        result[name] = new_entry

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
