# import pytest
import framenet.ecg as ecg
from framenet.ecg.generation import TestTree, unstack


def test_type():
    a  = ecg.Type('a',  [])
    b1 = ecg.Type('b1', [a])
    b2 = ecg.Type('b2', [a])
    c  = ecg.Type('c',  [a, b1])
    d  = ecg.Type('d',  [c])

    assert a.subtypeof(a)
    assert b1.subtypeof(a)
    assert b2.subtypeof(a)
    assert not a.subtypeof(b1)
    assert not a.subtypeof(b2)
    assert c.subtypeof(a)
    assert c.subtypeof(b1)
    assert not c.subtypeof(b2)
    assert d.subtypeof(a)
    assert d.subtypeof(b1)
    assert not d.subtypeof(b2)

    d_as = list(ecg.ancestors(d))
    bs   = [t in d_as for t in (a, b1, c, d)]
    assert all(bs), 'ancerstors: %s' % d_as


def test_unstack():
    tt = TestTree
    tree = tt('s',
           [tt('t'),
            tt('ann_1',
               [tt('layer_11',
                   [tt('label_111'),
                    tt('label_112'),
                    tt('label_113'),
                    tt('label_114')]),
                tt('layer_12',
                   [tt('label_121'),
                    tt('label_122')])]),
            tt('ann_2',
               [tt('layer_21',
                   [tt('label_211'),
                    tt('label_212')]),
                tt('layer_22',
                   [tt('label_221'),
                    tt('label_222')])])])

    rs = unstack(1, tree)
    assert len(rs) == 10
    assert len(rs[0]) == 5
