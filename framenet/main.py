"""
@author: <seantrott@icsi.berkeley.edu>

Initializes FrameNetBuilder and FrameNet objects.
"""

import sys
from os.path import join

from framenet.builder import FramenetBuilder


def main(data_path):
    frame_path = join(data_path, "frame")
    relation_path = join(data_path, "frRelation.xml")
    lu_path = join(data_path, "lu")
    fnb = FramenetBuilder(frame_path, relation_path, lu_path)
    fn = fnb.read()  # fnb.read()
    fn.build_relations()
    fn.build_typesystem()
    return fn, fnb


if __name__ == "__main__":
    fn, fnb = main(sys.argv[1])

    fnb.build_lus_for_frame("Motion", fn)
