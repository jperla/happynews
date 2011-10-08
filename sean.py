"""
    reads in sean gerrish's political discussants lexicons.
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
rows = [r.strip('\r\n ').split(',') for r in open('v4-lm.dat', 'r').readlines()]
rows = [(r[0], float(r[1]), float(r[2])) for r in rows]
v = list(sorted(rows, key=lambda k:k[2]))
