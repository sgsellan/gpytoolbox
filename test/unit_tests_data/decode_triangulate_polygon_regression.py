"""Turns the block test_regression prints on a platform with no stored
triangulations into that platform's stored .npz files.

    python test/unit_tests_data/decode_triangulate_polygon_regression.py pasted.txt

The file is the test's output, markers included; anything outside them is
ignored. The platform the meshes belong to is read from the marker, so the
files land under the right names whatever machine this is run on.
"""
import base64
import io
import os
import re
import sys

import numpy as np

_BEGIN = re.compile(r"-----\s*BEGIN triangulate_polygon regression (\S+)\s*-----")
_END = re.compile(r"-----\s*END triangulate_polygon regression (\S+)\s*-----")


def parse(text):
    platform, meshes, name = None, {}, None
    for line in text.splitlines():
        begin = _BEGIN.search(line)
        if begin:
            platform, name = begin.group(1), None
            continue
        if _END.search(line):
            break
        if platform is None:
            continue
        if line.strip() and not line.startswith(" "):
            name = line.strip()
            meshes[name] = []
        elif name is not None:
            meshes[name].append(line.strip())
    return platform, {n: "".join(parts) for n, parts in meshes.items()}


def main(path):
    platform, meshes = parse(open(path).read())
    if platform is None:
        sys.exit("no 'BEGIN triangulate_polygon regression' marker in " + path)
    if not meshes:
        sys.exit("no meshes between the markers in " + path)
    out = os.path.dirname(os.path.abspath(__file__))
    for name, blob in sorted(meshes.items()):
        with np.load(io.BytesIO(base64.b64decode(blob))) as data:
            V, F = data["V"], data["F"]
        dest = os.path.join(out, f"triangulate_polygon_{name}_{platform}.npz")
        np.savez(dest, V=V, F=F, platform=platform)
        print(f"{os.path.basename(dest)}: {V.shape[0]} vertices, {F.shape[0]} faces")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
