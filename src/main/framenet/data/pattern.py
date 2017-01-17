"""Pattern matching-related stuff."""

from pprint         import pprint, pformat
from typing         import NamedTuple, List, Tuple, Dict, Callable, Union, Any
from framenet.util  import flatten, flatmap, singleton


Graph = NamedTuple('Graph', [('vertices', List[str]),
                             ('edges',    List[Tuple[str, str]])])

class Pattern:
    def __init__(self, graph):
        self.graph = graph
        self.next_state = dict(graph.edges)

    def match(self, xs, debug=False):
        ss = [self.graph.vertices[0]]
        next_state = lambda s: self.next_state[s]
        for x in xs:
            if debug:
                print('x:', x, 'states:', ss)
            if x in ss or '_' in ss:
                ss = flatmap(next_state, ss)
            else:
                return False
        else:
            return True


# A `Lexer` is a map from strings to lists of `Lexer`s.
# Lexer = Dict[str, List['Lexer']]

class Lexer:
    """A simple, immutable dictionary. """

    __str__ = __repr__ = lambda self: f'Lexer({self.d!r})'

    def __init__(self, d=None):
        self.d       = d if d is not None else {}
        self._keys   = tuple(self.d.keys())
        self._values = tuple(self.d.values())
        self._items  = tuple(self.d.items())
        self._hash   = hash(self._items)

    def get(self, s, default=None, wildcard='_'):
        v = self.d.get(s)
        if v is not None:
            return v
        else:
            return self.d.get(wildcard, default)

    def keys(self):                 return self._keys
    def values(self):               return self._values
    def items(self):                return self._items
    def __bool__(self):             return bool(self._items)
    def __getitem__(self, s):       return self.d[s]
    def __hash__(self):             return self._hash
    def __eq__(self, other):
        # print('>> eq called!')
        return type(other) == Lexer and other._items == self._items


# TODO: Hide this? No.
def mergemany(ls1: Tuple[Lexer, ...], ls2: Tuple[Lexer, ...], debug=False) -> Tuple[Lexer, ...]:
    ls1, ls2 = map(maybe_force, (ls1, ls2))

    if debug: print(f'mergemany({ls1}, {ls2})')

    if not ls1:
        return ls2
    elif not ls2:
        return ls1
    else:
        (l1, *r1), (l2, *r2) = ls1, ls2
        return (merge(l1, l2, debug),) + mergemany(tuple(r1), tuple(r2), debug)

# @dispatch(Lexer, Lexer)
def merge(l1: Lexer, l2: Lexer, debug=False) -> Lexer:
    l1, l2 = map(maybe_force, (l1, l2))

    if debug: print(f'merge({l1}, {l2})')

    if not l1:
        return l2
    elif not l2:
        return l1
    else:
        ss1, ss2 = set(l1.keys()), set(l2.keys())
        css      = ss1 & ss2
        merged   = {s: mergemany(l1[s], l2[s], debug) for s in css}
        merged.update({s: l1[s] for s in ss1 - ss2})
        merged.update({s: l2[s] for s in ss2 - ss1})
        return Lexer(merged)

def mapvalues(f, d: Dict) -> Dict:
    return {k: [f(x) for x in v] for k, v in d.items()}

def maybe_force(expr) -> Lexer:
    return expr() if isinstance(expr, Callable) else expr

# TODO: This cannot work in Python.
def force(lexer) -> Dict:
    seen = set()
    def force_many(ls: Tuple[Lexer, ...]):
        print(f'force_many({ls})')
        if not ls or ls in seen:
            return ls
        else:
            seen.add(ls)
            l, *rest = ls
            return (force_one(l),) + force_many(tuple(rest))

    def force_one(l):
        l = maybe_force(l)
        print(f'force_one({l})')
        if not l or l in seen:
            return l
        else:
            seen.add(l)
            return {s: force_many(ls) for s, ls in l.items()}

    return force_one(lexer)


class RegExp:
    """The State should contain the continuations. When a RegExp object gets constructed,
    it should return a new State with the possible continuations.
    """
    # State   = namedtuple('State', 'xs rx')
    __str__ = __repr__ = lambda self: '%s' % type(self).__name__

    def __iter__(self):
        yield from self

    def __call__(self, lexer: Lexer) -> Lexer:
        raise NotImplementedError()

    def __and__(self, other): return Seq(self, other)
    def __or__(self, other): return Alt(self, other)

    def match(self, xs, initial=None, debug=False) -> bool:
        # A `Lexer` is actually a state
        # lexers = [force(self(Lexer() if not initial else initial))]
        lexers = [self(Lexer() if not initial else initial)]

        if debug: pprint(lexers)

        for x in xs:
            if debug: print(f'trying to match {x} against {pformat(lexers)}')
            lexers = list(flatten(maybe_force(l).get(x, ()) for l in lexers))
            if debug: print(f'match returned {pformat(lexers)}')

            if not lexers:
                return False
        else:
            return True

class UnExp(RegExp):
    __str__ = __repr__ = lambda self: f'{type(self).__name__}({self.a!r})'
    def __init__(self, a): self.a = a
    def __iter__(self): yield from self.a

class BinExp(RegExp):
    __str__ = __repr__ = lambda self: f'{type(self).__name__}({self.a!r}, {self.b})'
    def __init__(self, a: RegExp, b: RegExp):
        # print(isinstance(a, RegExp), isinstance(b, RegExp), f"a: {a}, b: {b}")
        self.a, self.b = a, b
    def __iter__(self): yield from self.a; yield from self.b


@singleton
class Success(RegExp):
    """Always succeeds"""
    def __call__(self, lexer: Lexer) -> Lexer:
        return lexer

@singleton
class Failure(RegExp):
    """Always fails"""
    def __iter__(self): yield from set()
    def __call__(self, lexer): return dict()

class Lit(UnExp):
    def __call__(self, lexer) -> Lexer:
        return Lexer({str(self.a): (lexer,)})

class Seq(BinExp):
    def __call__(self, lexer: Lexer) -> Lexer:
        return self.a(lambda: self.b(lexer))

class Alt(BinExp):
    def __call__(self, lexer: Lexer) -> Lexer:
        return merge(self.a(lexer), self.b(lexer))

class Many(UnExp):
    def __call__(self, lexer):
        # return ((self.a & (lambda l: self(l))) | Epsilon())(lexer)
        return ((self.a & self) | Success())(lexer)

class Many1(UnExp):
    def __call__(self, lexer):
        return (self.a & Many(self.a))(lexer)

class Opt(UnExp):
    # def __init__(self, a): self.a = Alt(Success(), a)
    def __call__(self, lexer):
        return (self.a | Success())(lexer)

# class OneOrMore(UnExp):
#     def __call__(self, lexer):
#         return (ZeroOrMore(self.a) & self.a)(lexer)
#
# class ZeroOrMore(UnExp):
#     def __call__(self, lexer):
#         al = self.a(lexer)
#         this = dict()
#         this.update({s: mergemany(als, [this]) for s, als in al.items()})
#         pprint(this)
#         return merge(lexer, this)
