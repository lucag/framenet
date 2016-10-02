"""A module for ECG-related stuff.
"""

import os
import xml.etree.ElementTree as et
from pprint import pprint

import pandas                as pd

from itertools        import chain

from framenet.hypothesize_constructions import from_collapsed_pattern, from_pattern
from .example.scripts import build_cxns_for_frame
from .utils           import curry, flatmap, memoize, unique, flatten, invert
from os.path          import join
from glob             import glob


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


def get_schemas(fn):
    """ Returns list of ECG schemas from FrameNet frames. """
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


# class FnFrame(object):
#     def __init__(self, fn, frame):
#         self.fn, self.frame = fn, frame
#
#     def parents(self):
#         fn = self.fn
#         return [FnFrame(fn, fn.get_frame(f)) for f in self.frame.parents]
#
#     def ancestors(self):
#
#     # def elements(self):
#     #     for
#     #     return
#
#
# class Schema(object):
#     def __init__(self, fn_frame): pass
#
#
# class Construction(object):
#     def __init__(self, fn_frame): pass


@curry
def et_loader(base, path):
    #     *start, last = path
    fname = '%s.xml' % path if not path.endswith('.xml') else path
    return et.parse(join(base, fname)).getroot()


# TODO: This stuff needs to be better organized
URI         = 'http://framenet.icsi.berkeley.edu'
base_dir    = os.getenv('FN_HOME')
frame_dir   = join(base_dir, 'frame')
frame_root  = et_loader(frame_dir)
rel_root    = et_loader(base_dir)
et_roots    = lambda path: (frame_root(fname) for fname in glob('%s/*.xml' % path))


def maybe_int(kv_pair):
    """Make the value an int if the key contains ID.
    """
    k, v = kv_pair
    return k, (int(v) if 'ID' in k else v)


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


@memoize
def _frame_element_relations(root):
    """Builds the entire table off of the root XML element."""
    rtypes  = it(root, 'frameRelationType')
    cap     = lambda s: s[0].upper() + s[1:]
    items   = lambda fer, fr, rt: chain(fer.items(),
                                        [('relationType', rt.get('name'))],
                                        [('relation%s' % cap(k), v) for k, v in fr.items()])
    return [dict(map(maybe_int, items(fer, fr, rt)))
            for rt in rtypes for fr in rt for fer in fr]


@memoize
def frame_element_relations(xml_fname='frRelation', as_dataframe=False):
    if as_dataframe:
        return pd.DataFrame(_frame_element_relations(rel_root(xml_fname)))
    else:
        return _frame_element_relations(rel_root(xml_fname))


def frames(path=frame_dir):
    return flatmap(frame_element, et_roots(path))


@curry
def frame_element(root):
    """Builds a single frame/fe record, off of the root XML element."""
    attrs  = 'ID coreType name'.split()
    pairs  = lambda f, fe: chain(
                [(k, v) for k, v in fe.items() if k in attrs],
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
                 'relationSuperFrameName', 'subFEName')
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

    def as_ecg_contraint(relation):
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

    return final + '\n'.join(as_ecg_contraint(rel) for rel in frame.fe_relations)


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
    for k, v in types.items():
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


