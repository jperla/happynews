"""
    jsondata helps you read data files encoded in json.
    Copyright (C) 2011 Joseph Perla

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
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

read = read_data # read_data function is deprecated
save = save_data
