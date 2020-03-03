def set_key(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = value
    elif type(dictionary[key]) == list:
        dictionary[key].append(value)
    else:
        dictionary[key] = [dictionary[key], value]

    return dictionary


def dict_to_string(input_dict, separator=", "):
    combined_list = list()
    for key, value in input_dict.items():
        individual = "{} : {:.5f}".format(key, value)
        combined_list.append(individual)
    return separator.join(combined_list)


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)