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
