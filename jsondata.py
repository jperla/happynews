import json

def save_data(filename, data):
    """Accepts filenamestring and a list of objects, probably dictionaries.
        Writes these to a file with each object pickled using json on each line.
    """
    with open(filename, 'w') as f:
        for i,d in enumerate(data):
            if i != 0:
                f.write('\n')
            f.write(json.dumps(d))

def read_data(filename):
    """Accepts filename string.
        Reads filename line by line and unpickles from json each line.
        Returns generator of objects.
    """
    with open(filename, 'r') as f:
        for r in f.readlines():
            yield json.loads(r)
