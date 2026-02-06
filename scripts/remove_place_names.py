#!/usr/bin/env python3
import argparse
import shutil
import sys
import re

def strip_names(text):
    out = []
    in_section = False
    for line in text.splitlines(keepends=True):
        if line.strip().upper().startswith("NODE_COORD_SECTION"):
            in_section = True
            out.append(line)
            continue
        if in_section:
            if line.strip().upper().startswith("EOF"):
                in_section = False
                out.append(line)
                continue
            # only process lines that start with an index number
            if re.match(r'^\s*\d+\s+', line):
                parts = line.split()
                if len(parts) >= 3 and parts[0].isdigit():
                    # keep only index, x, y
                    line = f"{parts[0]} {parts[1]} {parts[2]}\n"
        out.append(line)
    return "".join(out)

def main():
    p = argparse.ArgumentParser(description="Remove quoted place names from TSP NODE_COORD_SECTION lines.")
    p.add_argument("input", help="input .tsp file")
    p.add_argument("-o", "--output", help="output file (default: overwrite with .bak saved)", default=None)
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        src = f.read()

    res = strip_names(src)

    outpath = args.output or args.input
    if outpath == args.input:
        shutil.copy2(args.input, f"{args.input}.bak")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(res)

if __name__ == "__main__":
    main()