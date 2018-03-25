# data.__init__.py

import os, re

#@formatter:off
from lxml               import etree as et, objectify
from itertools          import chain
from typing             import Any, Callable, Dict, Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar
from multimethods       import Default, MultiMethod
from framenet.builder   import URI
from framenet.util      import Struct, compose, curry, flatmap, flatten, iget, memoize
#@formatter:on

A, B = map(TypeVar, ('A', 'B'))

class Tree(Generic[A]):
    __repr__ = __str__ = lambda self: '%s with %d children' % (self.__class__.__name__, len(list(self.tail)))

    def __init__(self, head: A, tail: List['Tree[A]'] = None):
        self._head, self._tail = head, tail or []

    @property
    def head(self) -> A:
        return self._head

    @property
    def tail(self) -> List['Tree[A]']:
        return self._tail

    def fold(self, f: Callable[[A, Iterable[B]], B]) -> B:
        """A catamorphism:
        data Tree a = Node a [Tree a] deriving (Show)

        cata :: (a -> [b] -> b) -> Tree a -> b
        cata f (Node root children) = f root (map (cata f) children)
        """
        return f(self.head, (t.fold(f) for t in self.tail))

    def make(self, head: A, tail: List['Tree[A]']):
        return Tree(head, tail)

    def flatten(self, levels=1) -> 'Tree[A]':
        def fltn(xss):
            return flatten(xss) if xss and type(xss[0]) is list else xss

        if levels == 0:
            return self
        else:
            flattened = [t.flatten(levels - 1) for t in self.tail]
            return self.make(head=fltn([self.head] + [t.head for t in flattened]), tail=flattened)

    # def traverse(self, depth=0) -> Iterator[A]:
    #     yield depth, self.head()
    #     for c in self.tail():
    #         yield from c.traverse(depth + 1)


def make_config(base=os.getenv('FN_HOME')):
    path = os.path.join
    ns = {'fn': 'http://framenet.icsi.berkeley.edu'}
    return Struct(frame         = (path(base, 'frameIndex'),    ('.//fn:frame',             ns)),
                  lu            = (path(base, 'luIndex'),       ('.//fn:lu',                ns)),
                  relations     = (path(base, 'frRelation'),    ('.//fn:frameRelationType', ns)),
                  sem_types     = (path(base, 'semTypes'),      ('.//fn:semType',           ns)),
                  fulltext      = path(base, 'fulltext'),
                  frame_by_name = lambda frame_name: path(base, 'frame', f'{frame_name}'),
                  lu_by_id      = lambda lu_id: path(base, 'lu', f'lu{lu_id}'),
                  ns            = ns,
                  base          = base)


def to_json(head, tail):
    json = head.copy()
    children = list(tail)
    if children:
        json.update({'#children': children})
    return json


def et_root(path):
    fname = '%s.xml' % path if not path.endswith('.xml') else path
    return objectify.parse(fname).getroot()


class FnReader:
    def __init__(self, config=make_config()):
        self.config = config
        self.idx    = Struct(frame=None, lu=None)

    # TODO  remove this, and have al the methods return an et.Element
    # TODO  (or the equivalent from lxml)
    # noinspection PyUnresolvedReferences,PyTypeChecker
    class DictEtTree(Tree):
        def __init__(self, root: et.Element, ns: dict):
            self.root = root
            self.ns = ns
            head, tail = self._make(root, ns)
            super().__init__(head, tail)

        def _make(self, element: et.Element, ns: dict) -> Tuple[A, List['Tree[A]']]:
            tag         = strip_ns(element.tag)
            contents    = dict(typify(item) for item in element.items())
            text        = element.text.strip() or None if element.text else None
            if text:
                if contents:
                    # if contents has other stuff, just add a tagged text element
                    contents.update({'#text': text})
                else:
                    # else, content IS the text itself
                    contents = text

            head = {tag: contents}
            tail = [FnReader.DictEtTree(e, ns) for e in iter(element)]
            return head, tail

        @property
        def element(self) -> et.Element:
            return self.root

        def find(self, xpath) -> Optional[et.Element]:
            return self.root.find(xpath, self.ns)

        def findall(self, xpath) -> Sequence['FnReader.DictEtTree']:
            return [FnReader.DictEtTree(e, self.ns) for e in self.root.findall(xpath, self.ns)]

    class NsElement:
        """An Element-Namespace pair, to avoid having to specify the latter in all the fin/findall calls.
        """
        def __init__(self, element, ns):
            self.element, self.ns = element, ns

        @property
        def attrib(self):
            return self.element.attrib

        @property
        def tag(self):
            return self.element.tag

        def findall(self, xpath):
            ns = self.ns
            return (FnReader.NsElement(e, ns) for e in self.element.findall(xpath, ns))

        def find(self, xpath):
            ns = self.ns
            return (FnReader.NsElement(e, ns) for e in self.element.find(xpath, ns))

    def et(self, path) -> 'FnReader.NsElement':
        #     *start, last = path
        fname = '%s.xml' % path if not path.endswith('.xml') else path
        # return FnReader.DictEtTree(et.parse(fname).getroot(), self.config.ns)
        return FnReader.NsElement(objectify.parse(fname).getroot(), self.config.ns)

    def make_tree(self, element, ns=None):
        return FnReader.DictEtTree(element, ns or self.config.ns)

    kinds = F, L = 'frame', 'lu'

    @property
    def frame_index(self):
        return self.index(FnReader.F)

    @property
    def lu_index(self):
        return self.index(FnReader.L)

    def index(self, kind):
        assert kind in self.kinds

        idx = self.idx[kind]
        if not idx:
            path, (xpath, _) = self.config[kind]
            id_and_name = iget('ID', 'name')
            attributes  = (f.element.attrib for f in self.et(path).findall(xpath))
            by_id, by_name = {}, {}
            for a in attributes:
                i, n = id_and_name(a)
                by_id[i] = by_name[n] = a
            idx = Struct(by_id = by_id, by_name = by_name)
        return idx

    def annotations_for_lu(self, lu_id) -> Sequence['FnReader.DictEtTree']:
        lu = self.lu_by_id(lu_id)
        return lu.findall('.//fn:annotationSet')

    def annotations_for_frame(self, frame):
        lu_ids = self.lu_ids_for_frame(frame)
        return flatten(self.annotations_for_lu(lu_id) for lu_id in lu_ids)

    def doc(self, file: str):
        path = os.path.join
        abs_file = path(self.config.base, file) if not os.path.isabs(file) else file
        return self.et(abs_file)

    def lu_ids_for_frame(self, frame: str) -> Sequence[str]:
        return [lu['ID'] for lu in self.lus() if frame in lu['frameName']]

    def layers_for_ann(self, ann_id):
        pass

    @property
    def frames(self):
        return self.frame_index.by_id.values()

    def frame_by_id(self, frame_id):
        return self.frame_by_name(self.frame_index.by_id[frame_id]['name'])

    def frame_by_name(self, frame_name: str) -> 'FnReader.DictEtTree':
        return self.et(self.config.frame_by_name(frame_name))

    @property
    def lus(self):
        """LU Index Elements, really."""
        return self.lu_index.by_id.values()

    def lu_by_id(self, lu_id) -> DictEtTree:
        return self.et(self.config.lu_by_id(lu_id))

    def lu_by_name(self, lu_name):
        lu_ids = (lu.get('ID') for lu in self.lus() if lu_name in lu.get('name'))
        return [self.lu_by_id(lu_id) for lu_id in lu_ids]

    @property
    def relations(self):
        path, (xpath, _) = self.config.relations
        return self.et(path).findall(xpath)

    @property
    def sem_types(self):
        path, (xpath, _) = self.config.sem_types
        return self.et(path).findall(xpath)

@curry
def _map(f, xs):
    return map(f, xs)


@curry
def _iter(uri, frame, tag):
    """An ElementTree iterator. Usage:
    >>> ee = list(it(frame_root('Intentionally_act'), 'frame'))
    >>> it(ee[0], 'FE')
    <_elementtree._element_iterator object at 0x10ebf1240>
    >>> list(it(ee[0], 'FE'))[2]
    [<Element '{http://framenet.icsi.berkeley.edu}FE' at 0x10ec64638>,\
     <Element '{http://framenet.icsi.berkeley.edu}FE' at 0x10ec64778>]
    """
    return frame.iter('{%s}%s' % (uri, tag))


# An ET iterator that does away with the stupid NS thing in ET itself
it = _iter(URI)


def strip_ns(s: str) -> str:
    """Remove namespace specification from string `s`."""
    return s[s.rindex('}') + 1:]


def maybe_int(item):
    """Make the value an int if the key contains ID."""
    k, v = item
    print('k:', k, 'v:', v)
    return k, (int(v) if 'ID' in k else v)


istrue  = re.compile('true', re.IGNORECASE)

isfalse = re.compile('false', re.IGNORECASE)

def typify(item) -> Tuple[str, Any]:
    """Guess type of value and try to enforce it."""
    k, v = item
    if v.isnumeric():
        return k, int(v)
    elif istrue.match(v):
        return k, True
    elif isfalse.match(v):
        return k, False
    else:
        return k, v


@curry
def qualify(tag, item):
    # pprint(item)
    k, v = item
    return '%s.%s' % (strip_ns(tag), k), v


def dispatch_items(et_elem: et.Element) -> str:
    return strip_ns(et_elem.tag)


# This is it!
items = MultiMethod('items', dispatch_items)
items.__doc__ = """Return items for a specific XML <element ...>. See also `dispatch_items."""


@items.method('label')
def label_items(elem: et.Element, exclude=re.compile('cBy|.*[cC]olor')):
    """exclude: Exclude these from <label ...>
    """
    # span = [('span', iget('start', 'end', default=NaN)(elem.attrib))]
    # ii = span + [(k, v) for k, v in elem.items() if not exclude.match(k)]
    ii = [(k, v) for k, v in elem.items() if not exclude.match(k)]
    # print('label_items: ii:', ii)
    return [qualify(elem.tag, typify(i)) for i in ii]


@items.method('annotationSet')
def aset_items(elem: et.Element, exclude=re.compile('cDate')):
    """exclude: Exclude these from <annotationSet ...>
    """
    trans = compose(qualify(elem.tag), typify)
    return [trans(i) for i in elem.items() if not exclude.match(i[0])]


@items.method('text')
def text_items(elem: et.Element):
    return [qualify(elem.tag, ('contents', elem.text.strip()))]


@items.method('definition')
def text_items(elem: et.Element):
    return [qualify(elem.tag, ('contents', elem.text.strip()))]


@items.method(Default)
def default_items(elem: et.Element):
    trans = compose(qualify(elem.tag), typify)
    return [trans(i) for i in elem.items()]


class EtTree(Tree):
    def __init__(self, root: et.Element) -> None:
        assert isinstance(et_root, et.Element)
        self.et_root = et_root
        self._items  = items(et_root)

    def head(self):
        return self._items

    def tail(self):
        return [EtTree(c) for c in iter(self.et_root)]

    def __repr__(self):
        ii   = ['%s: %s' % (a, v) for a, v in self.head()]
        tag  = strip_ns(self.et_root.tag)
        rest = [c for c in self.tail()]
        return '%s %s' % (tag, ii) if not rest else '%s %s\n  %s' % (tag, ii, rest)

    __str__ = __repr__


@curry
def unstack(min_depth: int, tree: Tree) -> List[List[A]]:
    """Unstack tree at `tree`.
    """
    @curry
    def unstack_one(d: int, t: Tree) -> List[List[A]]:
        cs = t.tail()
        if not cs:
            return [[t.head()]]
        elif not cs[0].tail() and d < min_depth:
            return [[t.head()] + vs for vs in unstack_many(d + 1, cs)]
        else:
            vss = flatmap(unstack_one(d + 1), cs)
            return [[t.head()] + vs for vs in vss]

    # Note to self: this is actually a cross product!
    @curry
    def unstack_many(d: int, ts: List[A]) -> List[List[A]]:
        if not ts:
            return [[]]
        else:
            t, *rest = ts
            vss      = flatmap(unstack_one(d + 1), rest)
            return [vs1 + vs2 for vs1 in unstack_one(d + 1, t) for vs2 in vss]

    return unstack_one(0, tree)


def unstack_one(element):
    """Unstack the whole XML subtree at `element`."""
    return [dict(flatten(vs)) for vs in unstack(2, EtTree(element))]


def unstack_all(elements):
    return flatmap(unstack_one, elements)


@memoize
def _frame_element_relations(root: EtTree) -> List[Dict]:
    """Builds the entire table off of the root XML element."""
    rtypes  = map(EtTree, it(root, 'frameRelationType'))
    cap     = lambda s: s[0].upper() + s[1:]
    items   = lambda fer, fr, rt: chain(fer.value(),
                                        [('relationType', rt.get('name'))],
                                        [('relation%s' % cap(k), v) for k, v in fr.value()])
    return [dict(map(maybe_int, items(fer, fr, rt)))
            for rt in rtypes for fr in rt.tail() for fer in fr.head()]


# # @memoize
# def frame_element_relations(xml_fname='frRelation', as_dataframe=False):
#     loader = root_for['.']
#     if as_dataframe:
#         return pd.DataFrame(_frame_element_relations(loader(xml_fname)))
#     else:
#         return _frame_element_relations(loader(xml_fname))



@curry
def frame_element(root):
    """Builds a single frame/fe record, off of the root XML element."""
    attrs  = 'ID coreType name'.split()
    pairs  = lambda f, fe: chain(
        [(k, v) for k, v in fe.value() if k in attrs],
        [('frameID', f.get('ID'))])
    fs     = it(root, 'frame')
    return [dict(map(maybe_int, pairs(f, fe)))
            for f in fs for fe in f if fe.tag.endswith('FE')]


# @curry
# def fe_relations_for(fn, frame):
#     ancestors = fn.typesystem.get_ancestors
#     fe_rels   = frame_element_relations(as_dataframe=True)
#     ancs      = [fr.ID for fr in ancestors(frame)]
#     crit      = fe_rels['relationSubID'].map(lambda i: i in ancs)
#     cols      = ('relationSubFrameName', 'subFEName',
#                  'relationType',
#                  'relationSuperFrameName', 'superFEName')
#     # noinspection PyUnresolvedReferences
#     return fe_rels.loc[crit, cols]