"""Some very simple HTML stuff."""

def tag(elem):
    def e(t, **kwargs):
        if kwargs:
            avs = ' '.join('%s="%s"' % (k, v) for k, v in kwargs.items())
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


align_r = 'text-align: right;'

def make_table_with_sentences(pattern_and_ss, style=align_r, collapse_sentences=False):
    """Table from (`pattern`, `sentence list`) pairs."""
    if not collapse_sentences:
        rows = ['\n'.join((td(i + 1,   style=style),
                       td(len(ss), style=style),
                       td(' \u2192 '.join(pattern), style='vertical-align: top; white-space: nowrap;'),
                       td(ul(''.join(li(s) for s in ss)), style='text-align: left;')))
            for i, (pattern, ss) in enumerate(pattern_and_ss)]
    else:
        rows = ['\n'.join((td(i + 1,   style=style),
                       td(len(ss), style=style),
                       td(' \u2192 '.join(pattern), style='vertical-align: top; white-space: nowrap;'),
                       # td(ul(''.join(li(s) for s in ss)), style='text-align: left;')
                           ))
            for i, (pattern, ss) in enumerate(pattern_and_ss)]
    if not collapse_sentences:
        header = tr(''.join(map(th, ('', 'freq.', 'Pattern', 'Sentences'))))
        return table(header + '\n'.join(map(tr, rows)))
    else:
        header = tr(''.join(map(th, ('', 'freq.', 'Pattern'))))
        return table(header + '\n'.join(map(tr, rows)))
