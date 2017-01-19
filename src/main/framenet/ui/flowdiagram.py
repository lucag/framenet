from jinja2 import Environment, PackageLoader, select_autoescape
from IPython.display import Javascript

env = Environment (
    loader                  = PackageLoader('framenet', 'ui/templates')
    # , autoescape          = select_autoescape(['html', 'js'])
    , variable_start_string = '/*{{'
    , variable_end_string   = '}}*/'
    , block_start_string    = '//{%'
    , block_end_string      = '%}//'
)

flow2_js    = env.get_template('flow2.js')
css_templ   = env.get_template('main.css')

def flowdiagram(data):
    rendered = flow2_js.render(arguments='"%s", element[0], "%s"' % (
        data.replace('\n', '\\n').replace('\t', '\\t'),
        css_templ.render().replace('\n', '\\n').replace('\t', '\\t')))
    # print(rendered)
    # return rendered
    return Javascript(data=rendered)
    # return flow_html.render(flow_call='flow("%s", element.get(0));' % data.replace('\n', '\\n'))
    # return Javascript(flow_templ.render(csv=csv))