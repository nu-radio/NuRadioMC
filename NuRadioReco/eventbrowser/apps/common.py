def get_point_index(event_ids, selection):
    index = {}
    for i, event_id in enumerate(event_ids):
        if(not isinstance(event_id, basestring)):
            event_id = str(event_id)
        index[event_id] = i
    event_index = []
    for event_id in selection:
        if event_id in index.keys():
            print('event id {} is in index'.format(event_id))
            event_index.append(index[event_id])
    print('returning event index ', event_index)
    return event_index
