"""Some very simple HTML stuff."""
import string
import uuid
from itertools import starmap
from numbers import Number
from random import randrange, randint, choices
from textwrap import dedent
from typing import Iterable

from IPython.core.display import HTML
from jinja2 import Environment, PackageLoader, select_autoescape
from multipledispatch import dispatch

from framenet.util import flatten


def tag(elem):
    @dispatch(Iterable)
    def mkstr(it, sep=''):
        return sep.join(mkstr(elt) for elt in it)

    @dispatch((str, Number))
    def mkstr(obj):
        return str(obj)

    def e(*ts, **kwargs):
        if kwargs:
            attribs = ' '.join('%s="%s"' % (k.lower(), v) for k, v in kwargs.items())
            return f'<{elem} {attribs}>{mkstr(ts)}</{elem}>'
        else:
            return f'<{elem}>{mkstr(ts)}</{elem}>'
    return e

table, tr, td, th, b, ul, li, div, a = map(tag, 'table tr td th b ul li div a'.split())


def make_table_from_counts(cs, style):
    """Make a table from counts."""

    rows = [(td(i + 1, style=style),
             td(c,     style=style),
             td(' \u2192 '.join(ss)))
            for i, (ss, c) in enumerate(cs)
            if c > 4]
    #     pprint (rows)
    header = tr(map(th, ('', 'freq.', 'Pattern')))
    #     pprint(header)
    return table(header + '\n'.join(map(tr, rows)))


env = Environment(
    loader       = PackageLoader('framenet', 'ui/templates')
    , autoescape = select_autoescape(['html', 'xml'])
)

align_r = 'text-align: right'
r_arrow = ' \u2192 '
link_style = dedent("""
    <style type="text/css" media="screen">
    a.link-hover:link { color: #000; text-decoration: none; }
    a.link-hover:visited { color: #000; text-decoration: none; }
    a.link-hover:hover { text-decoration: underline; } 
    a.link-hover:active { color: #000; text-decoration: none; } 
    </style>""")

def mk_id(k=9):
    """Make a random id of length `k`."""
    return ''.join(choices(string.ascii_lowercase) + choices(string.digits + string.ascii_lowercase, k = k - 1))

def uuids():
    while True:
        yield str(uuid.uuid4())

def make_table_with_sentences(pattern_and_ss, total_count, style=align_r, include_sentences=True):
    """Table from (`pattern`, `sentence list`) pairs."""

    def slist(ss):
        return ul(li(s) for s in ss)

    def collapsible_pattern(_uuid, pattern, ss):
        pattern_a = a(r_arrow.join(pattern), **{
            'class': 'link-hover',
            'data-toggle': 'collapse',
            'href': f'#{_uuid}'})
        sentences = div(slist(ss), **{
            'id': _uuid,
            'class': 'collapse'})

        return pattern_a, sentences

    def styled_th(text, alignment):
        a = 'right' if alignment.lower() == 'r' else 'left'
        return th(text, style=f'text-align: {a};')

    header = tr(starmap(styled_th, zip(('', '$N$', 'Patterns'), ('r', 'r', 'l'))))
    rows = (tr(td(i + 1,   style=style),
               td(len(ss), style=style),
               td(collapsible_pattern(_uuid, pattern, ss), style='text-align: left; width: 100%;'))
            for i, (_uuid, (pattern, ss)) in enumerate(zip(uuids(), pattern_and_ss)))
    # return (#activate_popover.render()
    # # +
    # div(f'Total count: {total_count}')
    # # + toggle
    # + table(header + '\n'.join(rows)))
    return (link_style
            + div(f'Total count: {total_count}')
            + table(header + '\n'.join(rows), Class='table table-bordered', style='table-layout: auto !important;'))

