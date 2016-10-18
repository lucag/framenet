"""Utilities, some ECG-specific, some very general.
"""

import wrapt

from collections        import namedtuple, Callable, Sequence, Iterable, defaultdict
from functools          import reduce, wraps
from itertools          import chain, islice
from operator           import concat
from decorator          import decorate
from multipledispatch   import dispatch


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
            def _partial(*args2, **kwargs2):
                # print('args:', args, 'args2:', args2)
                # print('kwargs:', kwargs, 'kwargs2:', kwargs2)
                # print('_curried', inspect.signature(_curried))
                return _curried(*(args + args2), **dict(kw, **kwargs2))

            return _partial

    return _curried


# def _curried(f, *args, **kw):
#     if len(args) + len(kw) >= f.__code__.co_argcount:
#         # print('args:', args, 'kwargs:', kwargs)
#         return f(*args, **kw)
#     else:
#         # return partial(func, *args, **kwargs)
#         def _partial(*args2, **kwargs2):
#             # print('args:', args, 'args2:', args2)
#             # print('kwargs:', kwargs, 'kwargs2:', kwargs2)
#             # print('_curried', inspect.signature(_curried))
#             return _curried(f, *(args + args2), **dict(kw, **kwargs2))
#
#         return _partial


# @wrapt.decorator
# def curry2(wrapped, instance, args, kwargs):
#     """Decorator to curry a function. Typical usage:
#     >>> @curry2
#     ... def foo(a, b, c):
#     ...    "This is foo docstring!"
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
#     if len(args) + len(kwargs) >= wrapped.__code__.co_argcount:
#         return wrapped(*args, **kwargs)
#     else:
#         def _partial(*args2, **kwargs2):
#             return wrapped(*(args + args2), **dict(kwargs, **kwargs2))
#
#         return _partial


# noinspection PyPep8Naming
# @decorate
class singleton(object):
    """Singleton. Usage:
    >>> @singleton()
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

    def __init__(self, *args, **kwds):
        update(self, instance=None, args=args, kwds=kwds)

    def __call__(self, cls):
        @wraps(cls)
        def init(*args, **kwds):
            if self.instance is None:
                self.instance = cls(*self.args, **self.kwds)
            return self.instance

        return init


def memoize(f):
    """A simple memoize implementation. It works by adding a .cache dictionary
    to the decorated function.
    """
    def _memoize(func, *args, **kw):
        if kw:  # frozenset is used to ensure hashability
            key = args, frozenset(kw.items())
        else:
            key = args
        cache = func.cache  # attribute added by memoize
        if key not in cache:
            # print('memoize: cache miss')
            val = cache[key] = func(*args, **kw)
            return val
        # print('memoize: cache HIT')
        return cache[key]

    f.cache = {}
    return decorate(f, _memoize)


class Struct(object):
    """Create an instance with argument=value slots.
    This is for making a lightweight object whose class doesn't matter.
    """

    def __init__(self, **entries): self.__dict__.update(entries)

    def __eq__(self, other):
        return isinstance(other, Struct) and self.__dict__, other.__dict__

    def __hash__(self): return self.__dict__.__hash__()

    def __repr__(self):
        args = ['%s=%s' % (k, repr(v)) for (k, v) in vars(self).items()]
        return 'Struct(%s)' % ', '.join(sorted(args))


def update(x, **entries):
    """Update a dict, or an object with slots, according to entries.
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
    return reduce(concat, xss, [])


@dispatch(Iterable)
def flatten(it):
    return chain.from_iterable(it)


@dispatch(Callable, Iterable)
def flatmap(f, it):
    return flatten(map(f, it))


@dispatch(Callable, Sequence)
def flatmap(f, xs):
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
        return seq[index]
    except (IndexError, TypeError) as e:
        if default is __unset__:
            raise
        else:
            return default


@curry
def getitems(indices, default, seq):
    return tuple(getitem(i, default, seq) for i in indices)


def iget(*indices, default=__unset__):
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
            return lambda obj: getattr(obj, names[0])
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


Cons, Nil = namedtuple('Cons', 'car cdr'), ()


def as_iter(cons):
    while cons is not Nil:
        yield cons.car
        cons = cons.cdr


def append(xs, ys): pass


def clist(*xs):
    if not xs:
        return Nil
    else:
        x, *rest = xs
        return Cons(x, clist(*rest))


def flip(f):
    return lambda cdr, car: f(car, cdr)


def creverse(cons):
    return reduce(flip(Cons), as_iter(cons), Nil)


class Queue(object):
    """A simple queue class. Usage:
    >>> q = Queue()
    >>> q.cons(1)
    >>> q
    Q(1)
    >>> q.cons(2)
    >>> q
    Q(1, 2)
    >>> q.snoc()
    1
    >>> q.cons(3)
    >>> q
    Q(2, 3)
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
    Q('a')
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

    def __repr__(self):
        return 'Q(%s)' % ', '.join(repr(x) for x in iter(self))

    __str__ = __repr__

    def __bool__(self):
        return self._check() is not Nil

    def __iter__(self):
        return chain(as_iter(self.front), as_iter(creverse(self.rear)))


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


def delay(expr):
    cache = [None]

    def get():
        if cache[0] is None: cache[0] = expr
        return cache[0]
    return get


def force(delayed_expr):
    return delayed_expr()


def invert(pairs):
    inv = defaultdict(list)
    for k, v in pairs: inv[v].append(k)
    return inv


def unique(xs):
    marked = set()
    ys = []
    for x in xs:
        if x not in marked:
            ys.append(x)
            marked.add(x)
    return ys


if __name__ == "__main__":
    import doctest

    doctest.testmod()
