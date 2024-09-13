
from argparse import Namespace

def get_class_info():
    class_info = Namespace()

    class_info.num_labels_city = 166
    class_info.num_labels_state = 157
    class_info.num_labels_country = 91
    class_info.num_labels_continent = 6

    return class_info
