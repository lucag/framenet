"""
In beta. Attempting to formulate procedures to hypothesize/build
ECG constructions from valence data. So far, there are a couple of methods:

1) Directly from valence patterns --> one valence pattern generates one HypothesizedConstruction
    * hypothesize_construction_from_pattern({PATTERN}, {N=index in total list})
2) From an entire frame, using individual valences --> build custom "valence patterns" from compatible valence units.
    * CXNS = collapse_valences_to_cxns({FRAME}, {FILTER=boolean})
    * for cxn in CXNS: hypothesize_construction_from_collapsed_pattern(CXNS, {N=index})

Ideally, we'll want a third way:
3) Top-down processing, fitting valence patterns to the existing grammar.

"""

# @formatter:off
import os, string, operator, re

import xml.etree.ElementTree   as et
import pandas                  as pd

from pprint                    import pprint
from abc                       import abstractmethod, ABC
from typing                    import List, Callable, Sequence, Any, Generic, Iterable, Iterator, TypeVar
from collections               import OrderedDict
from glob                      import glob, iglob
from itertools                 import chain
from os.path                   import join
from framenet.ecg              import Construction, Constituent
from framenet.example.scripts  import get_valence_patterns, invert_preps
from framenet.lexical_unit     import ValencePattern
from framenet.util             import curry, flatmap, memoize, unique, flatten, invert, iget, getitems, juxt, compose
from multimethods              import MultiMethod, Default
from multipledispatch          import dispatch


# This test is for verbs only

# Requires a "built" lu - e.g., one with valence patterns, et

# Takes in a mapping of prepositions onto the types of FEs they occur with.
# {'PP[in]': ['Area', 'Goal', etc.]}
# Could potentially be generalized to other POS.

# Generates preps by their frame.

# Generates prep cxn for that frame ("To-Goal")

# Takes as input a list of ROLES, which it creates PP constructions based on.

# Could probably be generalized into other things than entities
# Based on an input frame, builds tokens sub of {Frame}Type
# Each token is an lu from frame and sub-frames
# role_name should be the role you want to modify in parent frame
from numpy import NaN


def get_schemas(fn):
    """Returns list of ECG schemas from FrameNet frames."""
    return generate_schemas_for_frames(fn.frames)


def get_cxns(fn, fnb, frame="Motion", role="Manner", pos="V"):
    """ Returns dictionary of types/tokens, valence cxns, and prepositions for a frame. """
    return build_cxns_for_frame(frame, fn, fnb, role, pos)

    # DEMO: SCHEMAS
    # schemas = utils.generate_schemas_for_frames(fn.frames)
    # Write these to a file

    # DEMO: CONSTRUCTIONS
    # total = build_cxns_for_frame("Motion", fn, fnb, "Manner", "V")

    # You can then write the values from the total dictionary into files:
    # * cxns_all: all valences converted 1-1 to cxns
    # * cxns_collapsed: valences collapsed into smaller set
    # * tokens: tokens created from these frames ("swarm.v", etc.)
    # * types: type-cxns ("Fluidic_motionType") created from these frames
    # * pp: PP constructions that are specific to the frame (e.g., Instrument-PP)
    # * prep_types: General prepositional type constructions used for frame (E.g., "Instrument-Prep")
    # * prepositions: Preposition constructions that are used for that frame (E.g. "With-Preposition"[subcase of Instrument-Prep])
    # NOTE: The last three (pp, prep_types, and prepositions) are necessary for the collapsed cxns, since this filters by PP-type,
    # and collapses two valence units if they are PPs mapping onto the same FE.

    # DEMO: PREPOSITION CONSTRUCTIONS (distinct from build_cxns_for_frame, above)
    # prepositions = utils.build_prepositions(fn)


@curry
def et_loader(base, path):
    #     *start, last = path
    fname = '%s.xml' % path if not path.endswith('.xml') else path
    return et.parse(join(base, fname)).getroot()


# TODO: This stuff needs to be better organized
URI          = 'http://framenet.icsi.berkeley.edu'
base_dir     = os.getenv('FN_HOME')
dir_names    = '.', 'frame', 'lu', 'fulltext'
base_for     = OrderedDict((d, join(base_dir, d)) for d in dir_names)
root_for     = {k: et_loader(v) for k, v in base_for.items()}


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


T = TypeVar('T')
class Tree(ABC, Generic[T]):
    @abstractmethod
    def children(self) -> List[T]: return []

    @abstractmethod
    def value(self) -> T: return None

    __repr__ = __str__ = lambda self: '%s with %d children' % (self.__class__.__name__, len(self.children()))

    def traverse(self, depth=0) -> Iterator[T]:
        yield depth, self.value()
        for c in self.children():
            yield from c.traverse(depth + 1)


class TestTree(Tree):
    """A Test tree. Usage:
    >>> TT = TestTree
    >>> t  = TT(1, [TT(2), TT(3)])
    >>> def f(items, children): return TestTree(items + 1, children)
    >>> cata(f, t)
    TestTree: 2 [TestTree: 3, TestTree: 4]
    """
    def __init__(self, value, children=None):
        self._value, self._children = value, children or []

    def value(self):
        return self._value

    def children(self):
        return self._children


def strip_ns(s):
    """Remove namespace spec. from string `s`."""
    return s[s.rindex('}') + 1:]


def maybe_int(item):
    """Make the value an int if the key contains ID.
    """
    k, v = item
    print('k:', k, 'v:', v)
    return k, (int(v) if 'ID' in k else v)


def typify(item):
    """Guess type of value and try to enforce it."""
    k, v = item
    try:
        return k, int(v)
    except ValueError:
        return k, v


@curry
def qualify(tag, item):
    # pprint(item)
    k, v = item
    return '%s_%s' % (strip_ns(tag), k), v


def dispatch_items(et_elem: et.Element) -> str:
    return strip_ns(et_elem.tag)


# This is it!
items = MultiMethod('items', dispatch_items)
items.__doc__ = """Return items for a specific XML <element ...>. See also `dispatch_items."""

# Exclude these from <label ...>
# label_exclude = re.compile('start|end|cBy|.*[cC]olor')
label_exclude = re.compile('cBy|.*[cC]olor')


@items.method('label')
def label_items(elem: et.Element, exclude=label_exclude):
    # span = [('span', iget('start', 'end', default=NaN)(elem.attrib))]
    # ii = span + [(k, v) for k, v in elem.items() if not exclude.match(k)]
    ii = [(k, v) for k, v in elem.items() if not exclude.match(k)]
    # print('label_items: ii:', ii)
    return [qualify(elem.tag, typify(i)) for i in ii]


aset_exclude = re.compile('cDate')


@items.method('annotationSet')
def aset_items(elem: et.Element, exclude=aset_exclude):
    trans = compose(qualify(elem.tag), typify)
    return [trans(i) for i in elem.items() if not exclude.match(i[0])]


@items.method('text')
def text_items(elem: et.Element):
    return [qualify(elem.tag, ('contents', elem.text.strip()))]


@items.method(Default)
def default_items(elem: et.Element):
    trans = compose(qualify(elem.tag), typify)
    return [trans(i) for i in elem.items()]


class EtTree(Tree):
    def __init__(self, et_root: et.Element) -> None:
        assert isinstance(et_root, et.Element)
        self.et_root = et_root
        self._items  = items(et_root)

    def value(self):
        return self._items

    def children(self):
        return [EtTree(c) for c in iter(self.et_root)]

    def __repr__(self):
        ii   = ['%s: %s' % (a, v) for a, v in self.value()]
        tag  = strip_ns(self.et_root.tag)
        rest = [c for c in self.children()]
        return '%s %s' % (tag, ii) if not rest else '%s %s\n  %s' % (tag, ii, rest)

    __str__ = __repr__


@curry
def unstack(min_depth: int, tree: Tree) -> List[List[T]]:
    """Unstack tree at `tree`.
    """
    @curry
    def unstack_one(d: int, t: Tree) -> List[List[T]]:
        cs = t.children()
        if not cs:
            return [[t.value()]]
        elif not cs[0].children() and d < min_depth:
            return [[t.value()] + vs for vs in unstack_many(d + 1, cs)]
        else:
            vss = flatmap(unstack_one(d + 1), cs)
            return [[t.value()] + vs for vs in vss]

    # Note to self: this is actually a cross product!
    @curry
    def unstack_many(d: int, ts: List[T]) -> List[List[T]]:
        if not ts:
            return [[]]
        else:
            t, *rest = ts
            vss      = flatmap(unstack_one(d + 1), rest)
            return [vs1 + vs2 for vs1 in unstack_one(d + 1, t) for vs2 in vss]

    return unstack_one(0, tree)


@memoize
def _frame_element_relations(root):
    """Builds the entire table off of the root XML element."""
    rtypes  = it(root, 'frameRelationType')
    cap     = lambda s: s[0].upper() + s[1:]
    items   = lambda fer, fr, rt: chain(fer.value(),
                                        [('relationType', rt.get('name'))],
                                        [('relation%s' % cap(k), v) for k, v in fr.value()])
    return [dict(map(maybe_int, items(fer, fr, rt)))
            for rt in rtypes for fr in rt for fer in fr]


@memoize
def frame_element_relations(xml_fname='frRelation', as_dataframe=False):
    loader = root_for['.']
    if as_dataframe:
        return pd.DataFrame(_frame_element_relations(loader(xml_fname)))
    else:
        return _frame_element_relations(loader(xml_fname))


def frames(path=base_for['frame']):
    """Return all frames.
    """
    loader   = root_for[path]
    et_roots = (loader(fname) for fname in iglob('%s/*.xml' % path))

    return flatmap(frame_element, et_roots)


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


@curry
def fe_relations_for(fn, frame):
    ancestors = fn.typesystem.get_ancestors
    fe_rels   = frame_element_relations(as_dataframe=True)
    ancs      = [fr.ID for fr in ancestors(frame)]
    crit      = fe_rels['relationSubID'].map(lambda i: i in ancs)
    cols      = ('relationSubFrameName', 'subFEName',
                 'relationType',
                 'relationSuperFrameName', 'superFEName')
    # noinspection PyUnresolvedReferences
    return fe_rels.loc[crit, cols]


def inherited_elements(src_frames):
    return unique(flatten(p.elements for p in src_frames))


def format_schema(fn, frame):
    # semtype_to_frame = dict(Physical_object="Entity", Artifact="Artifact", Living_thing="Biological_entity")
    keywords = ['construction', 'situation', 'map', 'form', 'feature',
                'constraints', 'type', 'constituents', 'meaning']
    name = frame.name
    uses = []
    # inh_es = inherited_elements(fn.typesystem.get_ancestors(frame))
    inh_es = inherited_elements(frame.get_parents(fn))
    fe_to_ancestors = invert((e.frame_name, e.name) for e in inh_es)

    for relation in frame.relations:
        if relation.relation_type == "Uses":
            uses = [r for r in relation.related_frames]
            # print(relation.related_frames)
    causes = []
    for relation in frame.relations:
        if relation.relation_type == "Is Causative of":
            causes = [r for r in relation.related_frames]

    def maybe_ns(s):
        """Just in case a FE is just like an ECG kayword."""
        return 'fn_%s' % s if s in keywords else s

    def as_role(frame_elt):
        """Return an ECG representation of a FE."""
        fe_name = frame_elt.name.lower()
        if frame_elt.semtype:
            return '%s: @%s' % (maybe_ns(fe_name), frame_elt.semtype.name.lower())
        else:
            return maybe_ns(fe_name)

    def also_defined_in(frame_elt):
        return [a for a in fe_to_ancestors[frame_elt.name] if a != name]

    elements = [(as_role(fe), also_defined_in(fe)) for fe in frame.elements]
    final    = 'schema {} \n'.format(name)
    parents  = ', '.join(frame.parents) if frame.parents else None

    if parents:
        final += '    subcase of {} \n'.format(parents)
    for use in uses:
        final += '    evokes {} as {} \n'.format(use.name, use.name.lower())
    for cause in causes:
        final += '    evokes {} as {} \n'.format(cause.name, cause.name.lower())

    final += '    roles \n'

    for role, defd_in in sorted(elements, key=lambda p: (-len(p[1]), p[0])):
        row = "// %s (inherited from %s)" % (role, ', '.join(defd_in)) if defd_in else role
        final += '       %s\n' % row

    if len(frame.fe_relations) > 0:
        final += "     constraints \n"

    def as_ecg_constraint(relation):
        f1, f2 = map(maybe_ns, (relation.fe1.lower(), relation.fe2.lower()))
        if relation.name == 'Using':
            f1 = relation.superFrame.lower() + "." + f1
        elif relation.name == "Causative_of":
            f2 = relation.subFrame.lower() + "." + f2
        r = '       {} ⟷ {}'.format(f1, f2)
        if relation.name != 'Inheritance' or f1 != f2:
            return r
        else:
            return '       // inherited from %s: %s' % (relation.superFrame, r.lstrip())

    return final + '\n'.join(as_ecg_constraint(rel) for rel in frame.fe_relations)
# @formatter:on


def generate_schemas_for_frames(frames):
    return [format_schema(frame) + "\n\n" for frame in frames]


def format_valence_verb_cxn(valence_pattern, n):
    final = ""
    name = valence_pattern.frame + "_pattern{}".format(n)
    final += "construction {} \n".format(name)
    final += "     subcase of ArgumentStructure\n"
    final += "	   constructional\n"
    final += "		constituents\n"
    final += "		v: Verb\n"  # HACK
    for v in valence_pattern.valenceUnits:
        if v.gf != "Ext":
            pt = v.pt.replace("[", "-").replace("]", "")
            total = v.total
            ommission_prob = (total / valence_pattern.total)
            if pt in ['INI', 'DNI', 'CNI']:
                final += "		{}: {} [{}, .9]\n".format(pt.lower(), pt, ommission_prob)
            else:
                final += "		{}: {} [{}, .9]\n".format(pt.lower(), pt, ommission_prob)
    final += "	   meaning: {}\n".format(valence_pattern.frame)
    final += "		constraints\n"
    final += "		self.m <--> v.m\n"  # HACK
    for v in valence_pattern.valenceUnits:
        if v.gf == "Ext":  # HACK
            final += "		ed.profiledParticipant <--> self.m.{}\n".format(v.fe)
        else:
            pt = v.pt.replace("[", "-").replace("]", "")
            # if pt.split("-")[0] == "PP":
            #	constituent = "{}-PP".format(v.fe)
            #	final += "		self.m.{} <--> {}.m\n".format(v.fe, constituent.lower())
            #	final += "		self.m.Theme <--> {}.m.Trajector\n".format(constituent.lower())
            # else:
            final += "		self.m.{} <--> {}.m\n".format(v.fe, pt.lower())
    return final


def generate_cxns_for_lu(lu):
    returned = ""
    i = 1
    for realization in lu.valences:
        for pattern in realization.valencePatterns:
            returned += format_valence_verb_cxn(pattern, i) + "\n\n"
            i += 1
    return returned


def generate_cxns_from_patterns(patterns, collapsed=True):
    returned = ""
    i = 1
    for pattern in patterns:
        if collapsed:
            returned += from_collapsed_pattern(pattern, i).format_to_cxn() + "\n\n"
        else:
            returned += from_pattern(pattern, i).format_to_cxn() + "\n\n"
        i += 1
    return returned


def generate_preps_from_types(types, fn):
    returned = ""
    preps = sorted([lu for lu in fn.lexemes_to_frames.keys() if lu.split(".")[1] == "prep"])
    for k, v in types.value():
        meaning = None
        supers = ["{}-Preposition".format(supertype) for supertype in v]
        name = k.split("[")[1].replace("]", "")
        lu = "{}.prep".format(name)
        parents = ", ".join(supers)
        if lu in preps:
            frames = fn.get_frames_from_lu(lu)
            for frame in frames:
                meaning = frame.name
                prep = format_prep(name, parents, meaning)
                returned += prep + "\n\n"
        else:
            prep = format_prep(name, parents, meaning)
            returned += prep + "\n\n"
    return returned


def format_prep(name, parents, meaning=None):
    prep = "construction {}-Preposition-{}\n".format(name, str(meaning))
    prep += "    subcase of {}\n".format(parents)
    prep += "    form\n"
    prep += "      constraints\n"
    prep += "        self.f.orth <-- \"{}\"\n".format(name)
    if meaning:
        prep += "     meaning: {}".format(meaning)
    return prep


def build_prepositions(fn):
    returned = ""
    preps = sorted([lu for lu in fn.lexemes_to_frames.keys() if lu.split(".")[1] == "prep"])
    for prep in preps:
        frames = fn.get_frames_from_lu(prep)
        for frame in frames:
            orth, frame_name = prep.split(".")[0], frame.name
            if len(orth.split(" ")) <= 1:
                returned += build_preposition(orth, frame_name) + "\n\n"
    return returned


def build_preposition(orth, frame_name):
    returned = "construction {}-{}\n".format(orth, frame_name)
    returned += "    subcase of Preposition\n"
    returned += "    form\n"
    returned += "    constraints\n"
    returned += "      self.f.orth <-- \"{}\"\n".format(orth)
    returned += "    meaning: {}".format(frame_name)
    return returned


def generate_pps_from_roles(roles):
    returned = ""
    for role in roles:
        pp = ""
        pp += "construction {}-PP\n".format(role)
        pp += "  subcase of PP\n"
        pp += "  constructional\n"
        pp += "    constituents\n"
        pp += "      prep: {}-Preposition".format(role)
        returned += pp + "\n\n"
    return returned


def generate_general_preps_from_roles(roles):
    returned = ""
    for role in roles:
        pp = ""
        pp += "general construction {}-Preposition\n".format(role)
        pp += "	 subcase of Preposition\n"
        returned += pp + "\n\n"
    return returned


def generate_tokens(entity_frame, fn, role_name, pos):
    lus = gather_lexicalUnits(entity_frame, fn)
    returned = ""
    # seen = []
    for lu in lus:
        lexeme = lu.name.split(".")[0]
        if lu.pos == pos:  # and lexeme not in seen:
            returned += "{} :: {}Type :: self.m.{} <-- \"{}\"".format(lexeme, lu.frame_name, role_name, lexeme)
            if lu.semtype:
                # This will only work for entities, Events will likely need a different role name
                pass
                # returned += " :: self.m.Type_fn <-- @{}".format(lu.semtype.name)
            returned += "\n"
            # seen.append(lexeme)
    return returned


def generate_types(parent_frame, fn, role_name, pos_type):
    returned = ""
    types = gather_types(parent_frame, fn)
    returned += format_type_cxn(parent_frame, pos_type, role_name)
    # for frame in parent_frame.children:
    #	returned += format_type_cxn(fn.get_frame(frame), "{}Type".format(parent_frame.name), role_name)
    for frame in types:
        returned += format_type_cxn(frame, "{}Type".format(frame.parents[0]), role_name)
    return returned


def gather_types(parent_frame, fn):
    all_types = []
    for frame in parent_frame.children:
        actual = fn.get_frame(frame)
        all_types.append(actual)
        all_types += gather_types(actual, fn)
    return all_types


def format_type_cxn(frame, parent_cxn, role_name):
    returned = "construction {}Type\n".format(frame.name)
    returned += "     subcase of {}\n".format(parent_cxn)
    returned += "     meaning: {}\n".format(frame.name)
    returned += "       constraints\n"
    returned += "         self.m.{} <-- \"*\"\n".format(role_name)
    returned += "\n"
    return returned


def gather_lexicalUnits(parent, fn):
    lus = list(parent.lexicalUnits)

    for frame in parent.children:
        actual = fn.get_frame(frame)
        lus += gather_lexicalUnits(actual, fn)
    return lus


if __name__ == '__main__':
    import doctest
    doctest.testmod()

event_elements = ['Time', 'Place', 'Duration']


def from_pattern(valence_pattern, n=1):
    hypothesis = Construction(valence_pattern.frame, n=n)
    hypothesis.add_annotations(valence_pattern.annotations)
    total = sum(i.total for i in valence_pattern.valenceUnits)
    for unit in valence_pattern.valenceUnits:
        # if unit.pt not in ["2nd", "pp[because of]"]:
        probabilities = [1.0, .9]  # Is this right? These are for doing direct fit for valence patterns, so maybe
        pt = unit.pt.replace("[", "-").replace("]", "")
        constituent = Constituent(pt, unit.fe, unit.gf, probabilities)
        hypothesis.add_constituent(constituent)
    return hypothesis


def from_collapsed_pattern(valence_pattern, n=1):
    hypothesis = Construction(valence_pattern.frame, n=n)
    total = sum(i.total for i in valence_pattern.valenceUnits)
    for unit in valence_pattern.valenceUnits:
        # if unit.pt not in ["2nd", "pp[because of]"]:
        ommission_prob = round((unit.total / valence_pattern.total), 3)
        if ommission_prob <= 0:
            ommission_prob = 0.001
        probabilities = [ommission_prob, .9]
        pt = unit.pt.replace("[", "-").replace("]", "")
        constituent = Constituent(pt, unit.fe, unit.gf, probabilities)
        hypothesis.add_constituent(constituent)
    return hypothesis


def collapse_with_seed(initial_pattern, other_list, frame):
    for i in other_list:
        if (i not in initial_pattern.valenceUnits
                and frame.get_element(i.fe).coreType == "Core"
                and i.pt not in ['INI', 'DNI', 'CNI']):
            add = True
            for j in initial_pattern.valenceUnits:
                base_element, element = frame.get_element(i.fe), frame.get_element(j.fe)
                if not frame.compatible_elements(base_element, element):
                    add = False
                if i.pt == j.pt or i.fe == j.fe:
                    add = False
                if i.fe in event_elements:
                    add = False
            if add:
                initial_pattern.add_valenceUnit(i)
    initial_pattern.total = sum(i.total for i in initial_pattern.valenceUnits)
    return initial_pattern


def filter_collapsed_patterns(collapsed_patterns):
    new_list = []
    for g in collapsed_patterns:
        if g not in new_list:
            new_list.append(g)
    return new_list


def collapse_valences_to_cxns(frame, filter=True):
    all_patterns = []
    s = [valence for valence in frame.individual_valences if valence.lexeme.split(".")[1] == "v"]
    if filter:
        s = filter_by_pp(s)
    by_total = sorted(s, key=lambda valence: valence.total, reverse=True)
    for i in by_total:
        initial_pattern = ValencePattern(frame.name, 0, None)
        if i.pt in ['INI', 'DNI', 'CNI']:
            continue
        initial_pattern.add_valenceUnit(i)
        all_patterns.append(collapse_with_seed(initial_pattern, by_total, frame))
    return filter_collapsed_patterns(all_patterns)


def filter_by_pp(valences):
    """ Should return a reduced list with valence PT changed to more general PT, e.g. "Area-PP". """
    second = []
    for i in valences:
        new = i.clone()
        if i.pt.split("[")[0] == "PP":
            new.pt = "{}-PP".format(i.fe)
        if new not in second:
            second.append(new)
        else:
            second[second.index(new)].total += new.total
            second[second.index(new)].add_annotations(new.annotations)
    return second


def build_cxns_for_frame(frame_name, fn, fnb, role_name, pos, filter_value=False):
    """
    Takes in:
    -frame_name, e.g. "Motion"
    -FrameNet object (fn)
    -FrameNetBuilder object (fnb)
    -"filter_value" boolean: determines if you want to filter valence patterns
    -role_name: role to modify in types/tokens
    -pos: lexical unit POS to create tokens for (e.g., "V")

    TO DO: add PP constructions?

    Returns:
    -tokens
    -types
    -VP valences (non-collapsed)
    -VP valences (collapsed)
    -VP constructions (non-collapsed)
    -vP constructions (collapsed)
    """

    pos_to_type = dict(V="LexicalVerbType",
                       N="NounType")

    fnb.build_lus_for_frame(frame_name, fn)
    frame          = fn.get_frame(frame_name)
    tokens         = generate_tokens(frame, fn, role_name, pos)
    # types  = utils.generate_types(frame, fn, role_name, pos_to_type[pos])

    valence_patterns   = get_valence_patterns(frame)
    collapsed_valences = collapse_valences_to_cxns(frame)

    cxns_all       = generate_cxns_from_patterns(valence_patterns, collapsed=False)
    cxns_collapsed = generate_cxns_from_patterns(collapsed_valences)

    roles          = [v.fe for v in frame.individual_valences if v.pt.split("[")[0].lower() == "pp"]
    types          = invert_preps(frame.individual_valences)
    pp             = generate_pps_from_roles(roles)
    prep_types     = generate_general_preps_from_roles(roles)
    prepositions   = generate_preps_from_types(types, fn)

    returned = dict(tokens=tokens,
                    types=types,
                    valence_patterns=valence_patterns,
                    collapsed_valences=collapsed_valences,
                    cxns_all=cxns_all,
                    cxns_collapsed=cxns_collapsed,
                    pp=pp,
                    prep_types=prep_types,
                    prepositions=prepositions)
    return returned