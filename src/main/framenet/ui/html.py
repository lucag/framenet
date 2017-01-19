"""Some very simple HTML stuff."""
from textwrap import dedent

from IPython.core.display import HTML
from jinja2 import Environment, PackageLoader, select_autoescape


def tag(elem):
    def e(t, **kwargs):
        if kwargs:
            avs = ' '.join('%s="%s"' % (k.lower(), v) for k, v in kwargs.items())
            return '<{0} {2}>{1}</{0}>'.format(elem, str(t), avs)
        else:
            return '<{0}>{1}</{0}>'.format(elem, str(t))
    return e

table, tr, td, th, b, ul, li = map(tag, 'table tr td th b ul li'.split())


def make_table_from_counts(cs, style):
    """Make a table from counts."""

    rows = [''.join((td(i + 1, style=style),
                     td(c, style=style),
                     td(' \u2192 '.join(ss))))
            for i, (ss, c) in enumerate(cs)
            if c > 4]
    #     pprint (rows)
    header = tr(''.join(map(th, ('', 'freq.', 'Pattern'))))
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

def make_table_with_sentences(pattern_and_ss, style=align_r):
    """Table from (`pattern`, `sentence list`) pairs."""

    def button(ss):
        return sent_button.render(data_content=ul(''.join(li(s) for s in ss)),
                                  data_placement='auto',
                                  data_template=data_template)

    header = tr(''.join(map(th, ('', 'freq.', 'Patterns', ''))))
    rows   = ['\n'.join((td(i + 1,   style=style),
                         td(len(ss), style=style),
                         td(' \u2192 '.join(pattern), style='vertical-align: top ; white-space: nowrap'),
                         td(button(ss))))
              for i, (pattern, ss) in enumerate(pattern_and_ss)]
    return activate_popover.render() + table(header + '\n'.join(tr(r) for r in rows))

