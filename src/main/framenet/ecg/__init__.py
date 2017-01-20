"""Classes for constructing and formatting ECG constructions from valence patterns.

Question: What to do with subcase of Transitive_action, etc.?
"""

from typing import Iterable, List

from framenet.builder import build
from framenet.util import singleton


class Construction(object):
    def __init__(self, frame, parent='ArgumentStructure', n=1):
        self.frame = frame
        self.parent = parent
        self.constituents = []
        self.n = n
        self.annotations = []

    def add_constituent(self, constituent):
        self.constituents.append(constituent)

    def add_annotations(self, annotations):
        self.annotations += annotations

    def format_to_cxn(self):
        annotations = "/* {} */\n".format(self.annotations)
        final = annotations
        final += "construction {}_pattern_{}\n".format(self.frame, str(self.n))
        final += "     subcase of {}\n".format(self.parent)
        final += "     constructional\n"
        final += "      constituents\n"
        final += "        v: Verb\n"  # HACK
        for constituent in self.constituents:
            if constituent.gf != "Ext":
                final += "        {}\n".format(constituent.format_constituent())
        final += "      meaning: {}\n".format(self.frame)
        final += "       constraints\n"
        final += "         self.m \u27f7 v.m\n"  # HACK
        for constituent in self.constituents:
            final += "         {}\n".format(constituent.format_constraint())
        return final


class Constituent(object):
    """Represnts an ECG construction constituent. Contains: name (POS), frame element (role binding),
    and probabilities (omission, extraposition). P should be of the form [p1, p2].
    """

    def __init__(self, pt, fe, gf, probabilities):
        self.pt = pt
        self.fe = fe
        self.gf = gf
        self.probabilities = probabilities

    def format_constituent(self):
        return "{}: {} [{}, {}]".format(self.pt.lower(), self.pt, self.probabilities[0], self.probabilities[1])

    def format_constraint(self):
        if self.gf == "Ext":
            return "ed.profiledParticipant \u27f7 self.m.{}".format(self.fe)
        return "self.m.{} <--> {}.m".format(self.fe, self.pt.lower())


class Constraint:
    pass


class Role:
    def __init__(self, name, type_):
        self.name, self.type = name, type_


class Binding(Constraint):
    def __init__(self, left: List[Role], right: List[Role]):
        self.left, self.right = left, right


class Type:
    def __init__(self, name: str, parents: Iterable['Type']):
        self.name, self.parents = name, tuple(parents)

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def subtypeof(self, other) -> bool:
        if self == other:
            return True
        else:
            return any(p.subtypeof(other) for p in self.parents)

    __str__ = __repr__ = lambda self: '%s ^%s' % (self.name, self.parents)


@singleton
class Types(set):
    pass


def new_type(name, parents=()):
    ts = Types()
    if name not in ts and all(p in ts for p in parents):
        t = Type(name, parents)
        ts.add(t)
        return t
    else:
        raise TypeError('Type %s already exists' % name)


def ancestors(type_: Type) -> Iterable[Type]:
    yield type_
    for t in type_.parents:
        yield from ancestors(t)


class Schema(Type):
    def __init__(self, name: str, parents: List[Type], roles: List[str], constraints):
        super().__init__(name, parents)
        self.roles      = roles
        self.contraints = constraints

