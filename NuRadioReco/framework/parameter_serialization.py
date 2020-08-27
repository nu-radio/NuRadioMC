from __future__ import absolute_import, division, print_function, unicode_literals


def serialize(target_object):
    reply = {}
    for entry in target_object:
        reply[str(entry)] = target_object[entry]
    return reply


def deserialize(target_object, parameter_enum):
    reply = {}
    for entry in parameter_enum:
        if str(entry) in target_object:
            reply[entry] = target_object[str(entry)]
    return reply
