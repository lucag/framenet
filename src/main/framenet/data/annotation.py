import gzip, os, re

import operator              as op
import xml.etree.ElementTree as et
import pandas                as pd

from io                      import StringIO
from collections             import defaultdict, namedtuple, Callable
from glob                    import iglob
from os.path                 import join
from pickle                  import dump, load
from pprint                  import pprint
from typing                  import NamedTuple, List, Tuple, Set, Mapping, Dict, Union, Any

from framenet.ui.flowdiagram import flowdiagram
from framenet.ui.html        import b, make_table_with_sentences
from IPython.display         import HTML
from functools               import reduce
from framenet.builder        import build
from framenet.ecg.generation import unstack_all, root_for, base_for, FN
from framenet.util           import (flatmap, iget, curry, memoize,
                                     groupby, aget, reduceby, isnan,
                                     groupwise, singleton, flatten, merge, compose, identity)


# TODO: this is a duplicate 1
def lu_sents():
    """Content of `sentence` tag in all LUs"""
    lu_roots = map(root_for['lu'], iglob(join(base_for['lu'], '*.xml')))
    return flatten(r.findall('.//fn:sentence', FN) for r in lu_roots)


# TODO: this is a duplicate 2
def ft_sents():
    """Content of `sentence` tag in all full text files"""
    ft_roots = map(root_for['fulltext'], iglob(join(base_for['fulltext'], '*.xml')))
    return flatten(r.findall('.//fn:sentence', FN) for r in ft_roots)


def lus_for(frame_name):
    """LUs for `frame` as a pd.DataFrame."""
    cs    = ('annotationSet.ID', 'annotationSet.LU')
    lu    = aget('ID', 'lu')
    fn, _ = build()
    frame = fn.get_frame(frame_name)
    return pd.DataFrame([dict(zip(cs, lu(ann))) for ann in frame.annotations])


def core_FE_for(frame_name):
    fn, _ = build()
    frame = fn.get_frame(frame_name)
    return set(int(e.ID) for e in frame.elements if e.coreType == 'Core')


@memoize
def annoset_for(frame_name: str) -> Set[int]:
    """Returns annotation IDs for frame `frame_name`.
    """
    fn, fnb = build()
    frame   = fn.get_frame(frame_name)
    fnb.build_lus_for_frame(frame_name, fn)
    return set(int(ann.ID) for ann in frame.annotations)


def roots_for(elem_type: str):
    if elem_type not in root_for:
        raise ValueError('element type should be one of %s' % ', '.join(root_for.keys()))

    return map(root_for[elem_type], iglob(join(base_for['lu'], '*.xml')))


# TODO: duplicate of lu_sents()
def sentences_for(roots: List[et.Element]):
    """Gather all <sentence> elements"""
    return flatten(r.findall('.//fn:sentence', FN) for r in roots)


def records_for(frame_name: str) -> List[dict]:
    """Return records for `frame` by creating them or retrieving from the current directory.
    """
    # Note: FT stuff is not used right now.
    # lu_roots, ft_roots = map(roots_for,     ('lu', 'fulltext'))
    # lu_sents, ft_sents = map(sentences_for, (lu_roots, ft_roots))

    lu_recs  = lambda: get_object('lu_recs', unstack_all(lu_sents))
    aset     = annoset_for(frame_name)
    fn, _    = build()
    frame    = fn.get_frame(frame_name)

    @curry
    def crit(aset_ids, d):
        return d['annotationSet.ID'] in aset_ids

    # Only pick records that contain annotations for the specific frame
    return get_object('%s_recs' % frame, lambda: filter(crit(aset), lu_recs()))

# Ad-hoc function. TODO: make into a general one.

def make_groups(records) -> List[List[List[Dict]]]:
    """Group by sentence.ID and label.start, hierarchically; return just the groups."""

    itypes = {k: i for i, k in enumerate(('CNI', 'INI', 'DNI'))}

    def default_start(_, d):
        #         print('>> default_start called!')
        k1, k2 = iget('label.itype', 'label.feID', default=100)(d)
        return 9999 + itypes.get(k1, 100) + 10 * k2

    def label_start(d):
        s = d.get('label.start')
        return default_start(None, d) if isnan(s) else s

    sentence_id                   = iget('sentence.ID')
    grouped_by_sentence           = groupby(sentence_id, records)
    grouped_by_sentence_and_start = ((k, groupby(label_start, g)) for k, g in grouped_by_sentence)

    return to_list(grouped_by_sentence_and_start, levels=2, with_keys=False)


fields = iget('layer.name', 'label.name', 'label.itype', 'label.coreFE', default=None)

def pivot(m: Dict, fields=fields) -> List[Tuple[str, Any]]:
    a, b, c, d_ = fields(m)
    d           = False if isnan(d_) else True
    if not isnan(c):
        return [(a, b), ('GF', c), ('PT', None)] + ([('core', d)] if d else [])
    elif not isnan(a) and not isnan(b):
        return [(a, b)] + ([('core', d)] if d else [])
    else:
        return []


def name(obj):
    return obj.__class__.__name__

class Layer(tuple):
    __slots__ = ()
    __str__ = __repr__  = lambda self: f"{name(self)}({', '.join(f'{k}: {v.fget(self)}' for k, v in self.properties())})"
    properties          = lambda self: ((k, v) for k, v in self.__class__.__dict__.items() if type(v) is property)


class Valence(Layer):
    __slots__ = ()

    def __new__(cls, gf, fe, core, pt=None): return tuple.__new__(cls, (gf, fe, core, pt))

    GF, FE, core, PT = (property(op.itemgetter(i)) for i in range(4))


class Target(Layer):
    __slots__ = ()

    def __new__(cls, v): return tuple.__new__(cls, (v,))

    verb = property(op.itemgetter(0))


layer_names = ('GF', 'PT', 'FE', 'Target', 'CNI', 'DNI', 'INI')

def to_layers(dss: List[List[Dict]], layer_names=layer_names) -> List[Dict]:
    """Return layers in a seq of a seq of dictionaries (i.e., a single group)."""

    def p(ds):
        return merge(dict(), [i for d in ds for i in pivot(d)
                              if d['layer.name'] in layer_names and d['layer.rank'] == 1])

    try:
        ls = [p(ds) for ds in dss if p(ds)]
        return ls
    except ValueError:
        pprint('Problem with group')
        pprint(dss)
        raise


# def groups_for(frame_name: str) -> List[dict]:
#     """Retrieve all layers for the given `frame_name`.
#     """
#     by_sentence_id = groupby(iget('sentence.ID', default=-1))
#     by_label_start = groupby(iget('label.start', default=9999))
#     records        = records_for(frame_name)
#
#     def gf_and_target(group):
#         def target(key):
#             return to_lu[values['annotationSet.ID']] if key == 'Target' else None
#         _, values = group
#         to_lu = lu_set_for(frame_name)
#         return compose(iget('GF', 'Target', default=to_lu), iget(1))(group)
#
#     def gf_or_tgt(group):
#         """Turn a group into a pattern. Example:
#         >>> g1 = [(0,  {'FE': 'Agent', 'GF': 'Ext', 'PT': 'NP'}),
#         ...       (2,  {'Target': 'Target'}),
#         ...       (13, {'FE': 'Theme', 'GF': 'Obj', 'PT': 'NP'}),
#         ...       (29, {'FE': 'Path',  'GF': 'Dep', 'PT': 'PP'}),
#         ...       (50, {'FE': 'Goal',  'GF': 'Dep', 'PT': 'PP'})]
#         >>> list(map(gf_or_tgt, g1))
#         ['Ext', 'Target', 'Obj', 'Dep', 'Dep']
#         """
#         f, s = gf_and_target(group)
#         return f or s
#
#     groups = ((k, by_label_start(g)) for k, g in by_sentence_id(records))
#
#     # TODO
#     return to_list(xxx, levels=2)


def to_list(group, levels=1, with_keys=True):
    """Turns a grouping (returned by groupby) into a list of (key, group) pairs, recursively."""
    if levels > 0:
        if with_keys:
            return [(k, to_list(g, levels - 1, with_keys)) for k, g in group]
        else:
            return [to_list(g, levels - 1, with_keys) for _, g in group]
    else:
        return list(group)

@memoize
def lu_set_for(frame_name: str):
    fn, _ = build()
    frame = fn.get_frame(frame_name)
    items = [aget('ID', 'lu')(ann) for ann in frame.annotations]
    d     = defaultdict(set)
    for ann_id, lu in items: d[ann_id].add(lu)
    return d


def get_lu_df(base='.'):
    """Return LU records as a pd.DataFrame."""

    LU_PICKLE      = 'lu.pkl'
    lu_pickle_path = os.path.join(base, LU_PICKLE)
    if os.access(lu_pickle_path, os.R_OK):
        # Read lu_df back in
        lu_df = pd.read_pickle(lu_pickle_path)
        return lu_df
    else:
        # Save to a file in the current directory
        lu_df = pd.DataFrame(list(unstack_all(lu_sents())))
        lu_df.to_pickle(lu_pickle_path)
        return lu_df


def get_frame_df(frame_name, base_dir='.'):
    """Get pd.DataFrame for `frame_name`, normalize by adding Target LUs using the above."""

    lu_df       = get_lu_df(base_dir)
    aset_ids    = annoset_for(frame_name)
    core_fe_ids = core_FE_for(frame_name)
    selected    = lu_df.loc[lu_df['annotationSet.ID'].isin(aset_ids)]
    merged      = pd.merge(selected, lus_for(frame_name), how='outer', on='annotationSet.ID')
    merged.loc[
        merged['label.name'] == 'Target',
        'label.name'
    ] = merged['annotationSet.LU']
    merged.loc[
        merged['label.feID'].isin(core_fe_ids),
        'label.coreFE'
    ] = True
    return merged


def get_object(name, it, path='.'):
    """Pickle an object for the current directory or cache it using the iterable `it`.
    """
    fname = join(path, '%s.pkl.gz' % name)
    if not os.access(fname, os.R_OK):
        print('Writing pickle for', name)
        obj = list(it() if callable(it) else it)
        with gzip.open(fname, mode='w+b') as sout:
            dump(obj, sout)
        return obj
    else:
        print('Reading pickle for', name)
        with gzip.open(fname, mode='rb') as sin:
            return load(sin)


fields      = iget('layer.name', 'label.name', 'label.itype', 'label.coreFE', default=None)
fe_gf       = iget('FE', 'GF', 'core', default=None)

Node        = namedtuple('Node', 'id FE GF core')
Node2       = namedtuple('Node', 'id FE GF core text')
Link        = namedtuple('Link', 'source target')

def to_node(group):
    """Make a single node"""

    ls     = to_layers(group)
    # pprint(ls)
    fe_gfs = [fe_gf(l) for l in ls if fe_gf(l)[0] and fe_gf(l)[1]]
    # print('-' * 72)
    # pprint(fe_gfs)
    try:
        return [Node('%d:%s:%s' % (i, fe, gf)
                     , str(fe)
                     , str(gf)
                     , bool(core)
                     )
                for i, (fe, gf, core) in enumerate(fe_gfs)]
    except ValueError:
        pprint(fe_gfs)
        pprint(ls)
        raise


def to_json_node(group):
    ls     = to_layers(group)
    fe_gfs = [fe_gf(l) for l in ls if fe_gf(l)[0]]
    try:
        return [{'id': '%d:%s' % (i, fe),
                 'FE': str(fe),
                 'GF': str(gf),
                 'core': bool(core),
                 'text': text(group)}
                for i, (fe, gf, core) in enumerate(fe_gfs)]
    except ValueError:
        pprint(fe_gfs)
        pprint(ls)
        raise


def cols(link: Link) -> List[str]:
    return ['%s_%s' % (st, k)
            for st, n in zip(('source', 'target'), link)
            for k in n._asdict().keys()]


@curry
def links(noncore, from_nodes):
    """Generate links between pairs of nodes, removing _NI GFs."""

    if noncore or all(n.core for n in from_nodes):
        return [Link(s, t) for s, t in groupwise(2, from_nodes)]
    else:
        return []


def write_records(sout, groups, noncore=False, sep='\t'):

    def to_rec(link_and_count, sentence):
        link, cnt = link_and_count
        return  ( link.source.id, link.source.FE, link.source.GF, link.source.core
                , link.target.id, link.target.FE, link.target.GF, link.target.core
                , cnt
                , sentence
                )

    i            = 0
    count        = lambda ys, _: ys + 1
    nodes, sents = [to_node(g) for g in groups], [text(g) for g in groups]
    lnk_cnt_txt  = zip(reduceby(identity, 0, count, flatmap(links(noncore), nodes)), sents)
    # pprint(link_and_counts)
    sjoin = sep.join
    for i, (lc, txt) in enumerate(lnk_cnt_txt):
        if i == 0:
            # prepend header
            print(sjoin(cols(lc[0]) + ['count', 'text']), file=sout)
        print(sjoin(str(lc) for lc in to_rec(lc, txt)), file=sout)

    print('Written %d records.' % i)


def unique(it, key=None):
    """Unique elements in it according to key. Typical usage:
    >>> unique((1, 2, 3, 4), key = lambda x: x)
    [1, 2, 3, 4]
    >>> unique((1, 2, 3, 4), key = lambda x: x % 2)
    [1, 2]"""

    def step(ys, x):
        xs, ks = ys
        k = key(x)
        if k in ks:
            return xs, ks
        else:
            return xs + [x], ks | {k}

    key = key or identity

    return reduce(step, it, ([], set()))[0]


def to_json(groups, noncore=False):
    count           = lambda ys, _: ys + 1
    nodes           = [to_node(g) for g in groups]
    link_and_counts = list(reduceby(identity, 0, count, flatmap(links(noncore), nodes)))
    json_nodes      = unique(flatten(nodes))
    json_links      = [{'source': json_nodes.index(l.source)
                        , 'target': json_nodes.index(l.target)
                        , 'value': c }
                       for l, c in link_and_counts]
    return {'nodes': [dict(n._asdict()) for n in json_nodes], 'links': json_links}


def write_csv(fname, groups, noncore=False, base='.', sep='\t'):
    path = os.path.join(base, '%s.csv' % fname)
    with open(path, 'w+') as sout:
        write_records(sout, groups, noncore, sep)


# Helpers for `Patterns`: should go somewhere else.

def rec(ns):
    return tuple('%s: %s' % (n.GF, n.FE) for n in ns)


gf_fe_tgt = iget('GF', 'FE', 'Target', default='')

def gf_or_(group):
    """Turn a group into a pattern."""
    gf, fe, target = gf_fe_tgt(group)
    return '%s: %s' % (b(gf), fe) if not target else 'v'


gf_fe_pt_tgt = iget('GF', 'FE', 'PT', 'Target', default='')

def gf_pt_or_(group):
    """Turn a group into a pattern."""
    # print(group)
    gf, fe, pt, target = gf_fe_pt_tgt(group)
    return '%s: %s (%s)' % (b(gf), fe, pt) if not target else 'v'


gf_tgt = compose(iget('GF', 'Target', default=None))

def gf_or_target(layer):
    """Turn a layer into a pattern."""
    f, s = gf_tgt(layer)
    return f or s


@curry
def match(matcher, layer):
    #  pprint(layer)
    return matcher.match(map(gf_or_target, layer))


class Groups:
    """Groups for a specific frame."""

    def __init__(self, frame_name, base_dir='.'):
        self.data_frame = get_frame_df(frame_name, base_dir=base_dir)
        self.groups     = make_groups(self.data_frame.to_dict(orient='records'))


def text(group):
    return group[0][0]['text.contents']


class Patterns:
    """Patterns for a group record."""

    def __init__(self, groups):
        if isinstance(groups, Groups):
            self.groups = groups.groups
        else:
            self.groups = groups

    def select(self, pattern_matcher):
        matches = match(pattern_matcher)
        gs = [g for g in self.groups if matches(to_layers(g))]
        return Patterns(gs)

    def diagram(self, noncore=True):
        with StringIO() as sout:
            write_records(sout, self.groups, noncore)
            return flowdiagram(sout.getvalue())

    def display(self, pattern_matcher=None, negative=False, min_count=0):
        """Display patterns and (optionally) sentences in an HTML table."""

        res   = (lambda b: not b) if negative and pattern_matcher else (lambda b: b)
        pred  = (lambda _: res(True)) if not pattern_matcher else compose(res, match(pattern_matcher))

        # nodes = [to_node(g) for g in self.groups]

        gts   = [(tuple(gf_pt_or_(l) for l in to_layers(g)), text(g))
                 for g in self.groups if pred(to_layers(g))]

        p_to_ss  = defaultdict(list)
        for i, (k, v) in enumerate(gts): p_to_ss[k].append(v)

        p_ss = [(p, ss) for p, ss in sorted(list(p_to_ss.items()), key=lambda p: len(p[1]), reverse=True)
                if len(ss) > min_count]

        return HTML(make_table_with_sentences(p_ss))


if __name__ == '__main__':
    import doctest

    doctest.testmod()