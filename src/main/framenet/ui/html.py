"""Some very simple HTML stuff."""
import string
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

table, tr, td, th, b, ul, li, div = map(tag, 'table tr td th b ul li div'.split())


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

activate_popover = env.get_template('activate_popover.html')
sent_button      = env.get_template('sent_button.html')

align_r = 'text-align: right'

data_template = dedent("""
    <div class="popover" role="tooltip">
        <div class="arrow"></div>
        <h3 class="popover-title"></h3>
        <div class="popover-content">
    </div>""")

r_arrow = ' \u2192 '

toggle = dedent("""
    <script>
    var drawer = $( ".drawer__toggle" );

    var toggleState = function (selector, one, two) {
      var elem = $( selector );
      elem.setAttribute('data-state', elem.getAttribute('data-state') === one ? two : one);
    };

    drawer.onclick = function (e) {
      toggleState('.drawer td', 'closed', 'open');
      e.preventDefault();
    };
    </script>""")

css = dedent("""
    .drawer tr[data-state=closed] {
        display: none;
    }
    .drawer tr[data-state=open] {
        display: inherit;
    }""")

def mk_id(k=9):
    """Make a random id of length `k`."""
    return ''.join(choices(string.ascii_lowercase) + choices(string.digits + string.ascii_lowercase, k = k - 1))

def make_table_with_sentences(pattern_and_ss, total_count, style=align_r, include_sentences=True):
    """Table from (`pattern`, `sentence list`) pairs."""

    def slist(ss):
        return ul(li(s) for s in ss)

    header = tr(map(th, ('', 'freq.', 'Patterns')))
    rows = flatten((tr(td(i + 1,   style=style),
                       td(len(ss), style=style),
                       td(r_arrow.join(pattern), style='vertical-align: top ; white-space: nowrap')),
                    tr(td(slist(ss), colspan='3'), id='', Class='drawer__toggle', style='display: none'))
                   for i, (pattern, ss) in enumerate(pattern_and_ss))
    return (activate_popover.render()
            + div(f'Total count: {total_count}')
            + toggle
            + table(header + '\n'.join(rows)))

