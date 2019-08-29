from __future__ import absolute_import, division, print_function, unicode_literals


def serialize(object):
    reply = {}
    for entry in object:
        reply[str(entry)] = object[entry]
    return reply


def deserialize(object, parameter_enums):
    reply = {}

    if not isinstance(parameter_enums, list):
		parameter_enums = [parameter_enums]

	for parameter_enum in parameter_enums:
	    for entry in parameter_enum:
	        if str(entry) in object:
	            reply[entry] = object[str(entry)]
    
    return reply
    
