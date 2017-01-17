# import pytest
import framenet.ecg as ecg
from framenet.data.annotation import Graph, Pattern, annoset_for, get_frame_df, cols, Link, Node
from framenet.ecg.generation import TestTree, unstack
from framenet.util import cata


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
    assert all(bs), 'ancestors: %s' % d_as


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


def test_pattern1():
    vertices = ext, target, obj, dep = ['Ext', 'Target', 'Obj', 'Dep']
    edges    = ((ext, [target]),
                (target, [obj]),
                (obj, [dep]),
                (dep, [dep]))

    g = Graph(vertices=vertices, edges=edges)
    p = Pattern(g)

    assert p.match(['Ext', 'Target', 'Obj', 'Dep', 'Dep'])
    assert p.match(['Ext', 'Target', 'Obj', 'Dep'])

    assert not p.match(['Ext', 'Target', 'Obj', 'Obj', 'Dep'])
    assert not p.match(['Target', 'Obj', 'Dep', 'Dep'])


def test_pattern2():
    vertices = ext, t, dep = ('Ext', 'Target', 'Dep')
    ext_dep = Pattern(Graph(vertices=vertices, edges=((ext, [t]), (t, [dep]), (dep, [dep]))))

    assert ext_dep.match(['Ext', 'Target', 'Dep', 'Dep'])


def test_pattern3():
    vertices = ext, t, dep = ('Ext', 'Target', 'Dep')
    ext_dep = Pattern(Graph(vertices=vertices, edges=((ext, [t]), (t, [dep]), (dep, [dep, None]), (None, [None]))))

    assert ext_dep.match(['Ext', 'Target', 'Dep', 'Dep', 'Dep', None])


def test_annoset_for():
    cm_aset = annoset_for('Cause_motion')
    assert len(cm_aset) == 823


def test_get_frame_df(frame_name='Cause_motion', count=10788):
    cm_df = get_frame_df(frame_name)
    assert len(cm_df) == 10788, 'Test falied: %d' % len(cm_df)


def test_cols():
    assert cols(Link(Node('id',  'fe',  'gf',  True ),
                     Node('id2', 'fe2', 'gf2', False))) == [
        'source_id', 'source_FE', 'source_GF', 'source_core',
        'target_id', 'target_FE', 'target_GF', 'target_core'
    ]


def test_cata():
    TT = TestTree
    t  = TT(1, [TT(2), TT(3)])

    def f(items, children):
        return TestTree(items + 1, children)

    assert len(list(cata(f, t).children())) == 2

