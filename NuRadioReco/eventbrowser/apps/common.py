import dash_html_components as html
import numbers

def get_point_index(event_ids, selection):
    index = {}
    for i, event_id in enumerate(event_ids):
        if(not isinstance(event_id, basestring)):
            event_id = str(event_id)
        index[event_id] = i
    event_index = []
    for event_id in selection:
        if event_id in index.keys():
            event_index.append(index[event_id])
    return event_index

def get_properties_divs(obj, props_dic):
    props = []
    for display_prop in props_dic:
        if obj.has_parameter(display_prop['param']):
            if type(obj.get_parameter(display_prop['param'])) is dict:
                dict_entries = []
                dic = obj.get_parameter(display_prop['param'])
                for key in dic:
                    if isinstance(dic[key], numbers.Number):
                        dict_entries.append(
                            html.Div([
                                html.Div(key, className='custom-table-td'),
                                html.Div('{:.2f}'.format(dic[key]), className='custom-table-td custom-table-td-last')
                            ], className='custom-table-row')
                        )
                prop = html.Div(dict_entries, className='custom-table-td')
            else:
                if display_prop['unit'] is not None:
                    v = obj.get_parameter(display_prop['param'])/display_prop['unit']
                else:
                    v = obj.get_parameter(display_prop['param'])
                if isinstance(v,float) or isinstance(v, int):
                    prop = html.Div('{:.2f}'.format(v), className='custom-table-td custom-table-td-last')
                else:
                    prop = html.Div('{}'.format(v), className='custom-table-td custom-table-td-last')
            props.append(html.Div([
                html.Div(display_prop['label'], className='custom-table-td'),
                prop
            ], className='custom-table-row'))
    return props
