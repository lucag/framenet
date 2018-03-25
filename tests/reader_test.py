#
from pprint import pprint

from framenet.data.reader import Tree


def to_json(h, t):
    return h, list(t)

def test_unstack():
    T = Tree
    tree = T('s', [T('t'),
                   T('ann_1', [T('layer_11', [T('label_111'),
                                              T('label_112'),
                                              T('label_113'),
                                              T('label_114')]),
                               T('layer_12', [T('label_121'),
                                              T('label_122')])]),
                   T('ann_2', [T('layer_21', [T('label_211'),
                                              T('label_212')]),
                               T('layer_22', [T('label_221'),
                                              T('label_222')])])])

    pprint(tree.fold(to_json))

    print()

    t2 = tree.flatten(levels=1)
    pprint(t2.fold(to_json))

    print()

    t3 = tree.flatten(levels=3)
    pprint(t3.fold(to_json))

    t4 = tree.flatten(levels=4)
    pprint(t4.fold(to_json))

    assert len(t2) == 2
    assert len(t2.tail[0]) == 3
