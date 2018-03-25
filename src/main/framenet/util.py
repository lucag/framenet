"""Utilities, some ECG-specific, some very general.
"""
import string, sys, wrapt, itertools, math

# @formatter:off
from functools          import partial
from pprint             import pformat, pprint
from typing             import TypeVar, NamedTuple, Any, Tuple, Union, Generic, Dict, Mapping, Sequence, List as TList
from collections        import namedtuple, Callable, Iterable, defaultdict, Hashable, MutableMapping, Iterator, Set
from functools          import reduce, wraps
from itertools          import chain, islice
from operator           import concat, itemgetter
from decorator          import decorate
from multipledispatch   import dispatch
from abc                import ABC, abstractmethod
# @formatter:on


# TODO: This had to be made "transparent".
def curry(func):
    """Decorator to curry a function. Typical usage:
    >>> @curry
    ... def foo(a, b, c):
    ...    return a + b + c

    The function still work normally:
    >>> foo(1, 2, 3)
    6

    And in various _curried forms:
    >>> foo(1)(2, 3)
    6
    >>> foo(1)(2)(3)
    6

    This also work with named arguments:
    >>> foo(a=1)(b=2)(c=3)
    6
    >>> foo(b=1)(c=2)(a=3)
    6
    >>> foo(a=1, b=2)(c=3)
    6
    >>> foo(a=1)(b=2, c=3)
    6

    And you may also change your mind on named arguments,
    But I don't know why you may want to do that:
    >>> foo(a=1, b=0)(b=2, c=3)
    6

    Finally, if you give more parameters than expected, the exception
    is the expected one, not some garbage produced by the currying
    mechanism:
    >>> foo(1, 2)(3, 4)
    Traceback (most recent call last):
    ...
    TypeError: foo() takes 3 positional arguments but 4 were given
    """

    def _curried(*args, **kw):
        if len(args) + len(kw) >= func.__code__.co_argcount:
            # print('args:', args, 'kwargs:', kwargs)
            return func(*args, **kw)
        else:
            # return _partial(func, *args, **kwargs)
            def _partial(*_args, **_kwargs):
                # print('args:', args, 'args2:', args2)
                # print('kwargs:', kwargs, 'kwargs2:', kwargs2)
                # print('_curried', inspect.signature(_curried))
                return _curried(*(args + _args), **dict(kw, **_kwargs))

            return _partial

    return _curried


# def curry2(func):
#     """Decorator to curry a function. Typical usage:
#     >>> @curry2
#     ... def foo(a, b, c):
#     ...    return a + b + c
#
#     The function still work normally:
#     >>> foo(1, 2, 3)
#     6
#
#     And in various _curried forms:
#     >>> foo(1)(2, 3)
#     6
#     >>> foo(1)(2)(3)
#     6
#
#     This also work with named arguments:
#     >>> foo(a=1)(b=2)(c=3)
#     6
#     >>> foo(b=1)(c=2)(a=3)
#     6
#     >>> foo(a=1, b=2)(c=3)
#     6
#     >>> foo(a=1)(b=2, c=3)
#     6
#
#     And you may also change your mind on named arguments,
#     But I don't know why you may want to do that:
#     >>> foo(a=1, b=0)(b=2, c=3)
#     6
#
#     Finally, if you give more parameters than expected, the exception
#     is the expected one, not some garbage produced by the currying
#     mechanism:
#     >>> foo(1, 2)(3, 4)
#     Traceback (most recent call last):
#     ...
#     TypeError: foo() takes 3 positional arguments but 4 were given
#     """
#
#     @wrapt.decorator
#     def _curried(wrapped, instance, args, kwargs):
#         if len(args) + len(kwargs) >= func.__code__.co_argcount:
#             print('args:', args, 'kwargs:', kwargs, 'co_argcount:', func.__code__.co_argcount, file=sys.stderr)
#             return wrapped(*args, **kwargs)
#         else:
#             # return _partial(func, *args, **kwargs)
#             def _partial(*_args, **_kwargs):
#                 print('args:',   args,   '_args:',   _args,   file=sys.stderr)
#                 print('kwargs:', kwargs, '_kwargs:', _kwargs, file=sys.stderr)
#                 # print('_curried', inspect.signature(_curried))
#                 return _curried(func, instance, *(args + _args), **dict(kwargs, **_kwargs))
#
#     return _curried(func)


def singleton(class_):
    """Singleton. Usage:
    >>> @singleton
    ... class foo:
    ...     pass
    >>> x, y, z = foo(), foo(), foo()
    >>> x.val = 'sausage'
    >>> y.val = 'eggs'
    >>> z.val = 'spam'
    >>> print(x.val)
    spam
    >>> print(y.val)
    spam
    >>> print(z.val)
    spam
    >>> print(x is y is z)
    True
    >>> print(foo.__name__)
    foo
    """

    class_.__only_instance__ = None

    @wrapt.decorator
    def _singleton(wrapped, instance, args, kwargs):
        if not class_.__only_instance__:
            class_.__only_instance__ = wrapped(*args, **kwargs)

        return class_.__only_instance__

    return _singleton(class_)

def memoize(f, cache=None):
    """A simple memoize implementation. It works by adding a ._cache dictionary
    to the decorated function. Usage:
    >>> @memoize
    ... def fib(n):
    ...    return 1 if n in (0, 1) else fib(n - 1) + fib(n - 2)
    >>> fib(0)
    1
    >>> fib(1)
    1
    >>> fib(10)
    89
    """
    @wrapt.decorator
    def _memoize(wrapped, instance, args, kwargs):
        # frozenset is used to ensure hashability
        key   = (args, frozenset(kwargs.value())) if kwargs else args
        _cache = f.__cache__
        # print('memoize: key:', key, 'args:', args, file=sys.stderr)
        # TODO: optimize lookup with sentinel, measure gains!
        if key not in _cache:
            # print('_memoize: _cache MISS for', key, id(_cache), file=sys.stderr)
            value = _cache[key] = wrapped(*args, **kwargs)
            return value
        # print('_memoize: _cache HIT for', key, id(_cache), file=sys.stderr)
        return _cache[key]

    # print('_memoize:', 'setting _cache', file=sys.stderr)
    f.__cache__ = cache or dict()
    return _memoize(f)

class Struct(object):
    """Create an instance with argument=value slots.
    This is for making a lightweight object whose class doesn't matter. (Author: P. Norvig)
    """

    def __init__(self, **entries): self.__dict__.update(entries)

    def __getitem__(self, item): return self.__dict__[item]

    def __eq__(self, other): return type(other) is Struct and self.__dict__ == other.__dict__

    def __hash__(self): return self.__dict__.__hash__()

    def as_dict(self): return self.__dict__

    def keys(self): return self.__dict__.keys()

    def values(self): return self.__dict__.values()

    def items(self): return self.__dict__.items()

    def __repr__(self):
        args = ['%s=%s' % (k, repr(v)) for (k, v) in vars(self).items()]
        return 'Struct(%s)' % ', '.join(sorted(args))

def update(x, **entries):
    """Destructively update a dict, or an object with slots, according to entries.
    >>> sorted(update({'a': 1}, a=10, b=20).items())
    [('a', 10), ('b', 20)]
    >>> update(Struct(a=1), a=10, b=20)
    Struct(a=10, b=20)
    """
    if isinstance(x, dict):
        x.update(entries)
    else:
        x.__dict__.update(entries)
    return x

@dispatch(Sequence)
def flatten(xss):
    # print(f'flatten(Sequence): {xss}')
    return reduce(concat, xss, [])

@dispatch(Iterable)
def flatten(it):
    # print(f'flatten(Iterable): {it}')
    ret = chain.from_iterable(it)
    if ret is None:
        print('flatten: returning None!!!', file=sys.stderr)
    return ret

@dispatch(Set)
def flatten(s):
    # print('flatten(%s)' % s)
    return reduce(lambda b, a: b | a, s, frozenset())

@dispatch(Callable, Iterator)
def flatmap(f, it):
    return flatten(map(f, it))

@dispatch(Callable, Sequence)
def flatmap(f, xs):
    # print('flatmap(_, Sequence)')
    return reduce(concat, map(f, xs), [])

# A value to signal an unset default
__unset__ = '__unset__'

@curry
def getattrs(names, default, obj):
    if default is __unset__:
        return tuple(getattr(obj, n) for n in names)
    else:
        return tuple(getattr(obj, n, default) for n in names)

@curry
def getitem(index, default, seq):
    try:
        if isinstance(index, (list, tuple)):
            # interpret index as a path
            p = seq
            for i in index:
                print('p:', p)
                p = p[i]
            return p
        else:
            return seq[index]
    except (IndexError, KeyError) as e:
        if default is __unset__:
            raise
        elif callable(default):
            return default(index, seq)
        else:
            return default

@curry
def getitems(indices, default, seq):
    return tuple(getitem(i, default, seq) for i in indices)

# TODO: these should not differentiate between singleton/non-singleton indices (and return different types...)
def iget(*indices, default=__unset__):
    # print('indices', indices)
    if not indices:
        raise ValueError('At least one index is needed')
    elif len(indices) == 1:
        return getitem(indices[0], default)
    else:
        return getitems(indices, default)

def aget(*names, default=__unset__):
    if not names:
        raise ValueError('At least one attribute name is needed')
    elif len(names) == 1:
        if default is __unset__:
            def try_getattr(obj):
                try:
                    return getattr(obj, names[0])
                except AttributeError as x:
                    print(f'problem {x} trying to get {names[0]} out of {obj}')
                    raise
            return try_getattr
        else:
            return lambda obj: getattr(obj, names[0], default)
    else:
        return getattrs(names, default)

class Stack(object):
    def __init__(self, xs=None):
        self.xs = xs if xs is None else []

    def get(self):
        return self.xs.pop()

    def put(self, x):
        return self.xs.append(x)

    def __bool__(self):
        return bool(self.xs)

class List(tuple):
    __slots__ = ()

    def empty(self): return True

Nil = List()

class Cons(List):
    __slots__  = ()
    head, tail = (property(itemgetter(i)) for i in range(2))
    __repr__   = __str__ = lambda self: 'List(%s)' % ', '.join(repr(x) for x in self)

    def empty(self): return False

    def __new__(cls, head, tail): return tuple.__new__(cls, (head, tail))

    def __iter__(self):
        x = self
        while x:
            if isinstance(x, Cons):
                yield x.head
                x = x.tail
            else:
                yield x
                return

@dispatch(List, List)
def append(xs, ys):
    if not xs:
        return ys
    else:
        return Cons(xs.head, append(xs.tail, ys))

def clist(*xs) -> List:
    if not xs:
        return Nil
    else:
        x, *rest = xs
        return Cons(x, clist(*rest))

def flip(f):
    return lambda tail, head: f(head, tail)

def creverse(cons: List) -> List:
    return reduce(flip(Cons), iter(cons), Nil)

class Queue(object):
    """A simple queue class. Usage:
    >>> q = Queue()
    >>> q.cons(1)
    >>> q
    Queue(1)
    >>> q.cons(2)
    >>> q
    Queue(1, 2)
    >>> q.snoc()
    1
    >>> q.cons(3)
    >>> q
    Queue(2, 3)
    >>> q.snoc()
    2
    >>> q.snoc()
    3
    >>> q.snoc()
    Traceback (most recent call last):
    ...
    ValueError: Empty queue

    Also, you can start from a list:
    >>> q2 = Queue(1, 2, 3)
    >>> bool(q2)
    True
    >>> q2.snoc()
    1
    >>> q2.snoc()
    2
    >>> q2.snoc()
    3
    >>> bool(q2)
    False
    >>> q2.cons('a')
    >>> q2
    Queue('a')
    >>> bool(q2)
    True
    >>> q2.snoc()
    'a'
    >>> bool(q2)
    False
    """

    def __init__(self, *xs):
        self.front, self.rear = clist(*xs), Nil

    def cons(self, x):
        self.rear = Cons(x, self.rear)

    put = cons

    def snoc(self):
        if not self._check():
            raise ValueError('Empty queue')
        else:
            car, self.front = self.front.car, self.front.cdr
            return car

    get = snoc

    def _check(self):
        # print('_check...')
        if not self.front:
            self.front, self.rear = creverse(self.rear), Nil
        return self.front

    __repr__ = __str__ = lambda self: 'Queue(%s)' % ', '.join(repr(x) for x in iter(self))

    def __bool__(self):
        return self._check() is not Nil

    def __iter__(self):
        return chain(iter(self.front), iter(creverse(self.rear)))

def graph_iterator(expand, state, frontier):
    def traverse():
        while frontier:
            node = frontier.get()
            for n in expand(node):
                if state(n) not in explored:
                    frontier.put(n)
            yield node

    explored = set()
    return traverse

@curry
def dfs_iterator(visited, expand, state, initial):
    def traverse(node):
        visited.add(state(node))
        for n in expand(node):
            if state(n) not in visited:
                yield from traverse(n)
        yield node

    return traverse(initial)

@curry
def take(n, it):
    """Take n elements from the iterable it. Usage:
    >>> list(take(3, range(10)))
    [0, 1, 2]
    >>> list(take(0, range(10)))
    []
    >>> list(take(11, range(10)))
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> list(take(-1, range(10)))
    Traceback (most recent call last):
    ...
    ValueError: Stop argument for islice() must be None or an integer: 0 <= x <= sys.maxsize.
    """
    return islice(it, 0, n)

@curry
def drop(n, it):
    """Drop n elements from iterable it. Usage:
    >>> list(drop(2, range(5)))
    [2, 3, 4]
    >>> list(drop(1, range(5)))
    [1, 2, 3, 4]
    >>> list(drop(0, range(5)))
    [0, 1, 2, 3, 4]
    >>> list(drop(10, range(5)))
    []
    >>> list(drop(-1, range(5)))
    Traceback (most recent call last):
    ...
    ValueError: Indices for islice() must be None or an integer: 0 <= x <= sys.maxsize.
    """
    return islice(it, n, None)

def delay(f, *args):
    cache = [None]

    def get():
        if cache[0] is None: cache[0] = f(*args)
        return cache[0]
    return get

def force(delayed_expr):
    return delayed_expr()

def invert(pairs):
    inv = defaultdict(list)
    for k, v in pairs: inv[v].append(k)
    return inv

@dispatch(Hashable)
def as_hashable(h):
    return h

@dispatch(dict)
def as_hashable(d):
    return tuple(d.items())

@dispatch(Sequence)
def unique(xs):
    marked = set()
    ys = []
    hs = map(as_hashable, xs)
    for x, h in zip(xs, hs):
        if h not in marked:
            ys.append(x)
            marked.add(h)
    return ys


@curry
def cata(f, tree):
    """A catamorphism:
    data Tree a = Node a [Tree a] deriving (Show)

    cata :: (a -> [b] -> b) -> Tree a -> b
    cata f (Node root children) = f root (map (cata f) children)
    """
    return f(tree.head(), map(cata(f), tree.tail()))


def juxt(*fs):
    """Juxtapose a variable number of function. Returns a function that applies
    the `fs` to a (possibly variable) list of arguments. Usage:
    >>> juxt(lambda x: x + 1, lambda x: x + 2)(1)
    [2, 3]
    """
    if not fs:
        return lambda *xs: []
    else:
        f, *rest = fs
        return lambda *xs: [f(*xs)] + juxt(*rest)(*xs)


def compose(*fs):
    """Function composition. Usage:
    >>> f = lambda x: x + 1
    >>> g = lambda x: x * 2
    >>> h = compose(f, g)
    >>> h(4)
    9
    """
    if not fs:
        return lambda x: x
    elif len(fs) == 1:
        return fs[0]
    else:
        f, *rest = fs
        return lambda x: f(compose(*rest)(x))


def identity(x):
    """The identity function that sends x in itself."""
    return x


@curry
def groupby(key, it):
    return itertools.groupby(sorted(it, key=key), key=key)


@curry
def groupwise(n, it):
    """Returns windows of n elements. Usage:
    >>> list(groupwise(2, range(5)))
    [(0, 1), (1, 2), (2, 3), (3, 4)]
    >>> list(groupwise(-1, range(5)))
    Traceback (most recent call last):
    ...
    ValueError: n must be nonnegative, not -1
    """
    if n < 0:
        raise ValueError('n must be nonnegative, not %d' % n)
    its = itertools.tee(it, n)
    return zip(*[drop(k, i) for k, i in enumerate(its)])


@curry
def reduceby(key, zero, step, it):
    keyed_groups = groupby(key, it)
    reducer      = lambda seq: reduce(step, seq, zero)
    return ((k, reducer(g)) for k, g in keyed_groups)


@curry
def remove(p, it):
    """Remove all items `i` such that `p(i) == True`
    or elements that evaluate to `True` if `p` is `None`.
    """
    _p = p or (lambda x: not x)
    return [i for i in it if not _p(i)]


def grouperby(key):
    return lambda it: groupby(key, it)


def reducerby(key, zero, step):
    return lambda it: reduceby(key, zero, step, it)


def pipe(*reducers):
    return lambda it: compose(reducers)(it)


# TODO: make this work. Do I need it though? Perhaps not.
class Nest:
    def __init__(self, key=__unset__, f=__unset__):
        self.key, self.f = key, f

    def key(self, f):
        return self

    def map(self, f):
        return self

    def fold(self, zero, step):
        return self

    def __call__(self, xs):
        return xs


A, B = map(TypeVar, ['A', 'B'])

def mergewith(binop, m1: Mapping[A, B], m2: Mapping[A, B]) -> Mapping[A, B]:
    # print(m1, m2)
    k1, k2 = set(m1.keys()), set(m2.keys())
    # print(k1, k2)
    common = k1 & k2
    return dict(chain(((k, binop(m1[k], m2[k])) for k in common),
                      ((k, m1[k]) for k in k1 - k2),
                      ((k, m2[k]) for k in k2 - k1)))


def merge_kv(m: Dict, k, v) -> Dict:
    """Destructively merge (k, v) item into dictionary m."""
    if k in m and m[k] != v:
        raise ValueError('Item (%s -> %s) already in dictionary %s (%s)' % (k, v, m, items))
    else:
        m[k] = v

    return m


def merge(m: Dict, items: Sequence[Tuple[str, Any]], merge_kv=merge_kv) -> Dict:
    """Safely merge `items` into mapping `m`, destructively.
    Raises exception if key is shared among `items`.
    """
    for k, v in items:
        merge_kv(m, k, v)

    return m


def isnan(x):
    try:              return math.isnan(x)
    except TypeError: return False


class frozendict(dict):
    def _blocked_attribute(obj):
        raise AttributeError('A frozendict cannot be modified.')

    _blocked_attribute = property(_blocked_attribute)

    __delitem__ = __setitem__ = clear = pop = popitem = setdefault = update = _blocked_attribute

    def __new__(cls, *args):
        new = dict.__new__(cls)
        super().__init__(new, *args)
        new._cached_hash = hash(tuple(new.items()))
        return new

    def __hash__(self):
        return self._cached_hash

    def __repr__(self):
        return f'frozendict({dict.__repr__(self)})'


if __name__ == "__main__":
    import doctest

    doctest.testmod()
