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


def serialize_covariances(target_object):
    reply = {}
    for entry in target_object:
        reply[(str(entry[0]), str(entry[1]))] = target_object[entry]
    return reply


def deserialize_covariances(target_object, parameter_enum):
    reply = {}
    for entry in target_object:
        first_key = None
        second_key = None
        for enum in parameter_enum:
            if str(enum) == entry[0]:
                first_key = enum
            if str(enum) == entry[1]:
                second_key = enum
        if first_key is not None and second_key is not None:
            reply[(first_key, second_key)] = target_object[entry]
    return reply
