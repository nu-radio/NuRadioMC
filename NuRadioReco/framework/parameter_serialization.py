from __future__ import absolute_import, division, print_function, unicode_literals

def serialize(object):
    reply = {}
    for entry in object:
        reply[str(entry)] = object[entry]
    return reply

def deserialize(object, parameter_enum):
    reply = {}
    for entry in parameter_enum:
        if str(entry) in object:
            reply[entry] = object[str(entry)]
    return reply
