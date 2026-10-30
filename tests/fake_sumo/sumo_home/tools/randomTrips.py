"""A fake randomTrips.py: one trip every --period seconds from --begin to --end on the net's passenger edges, each
from a random edge (Python's random seeded with --seed), with the --trip-attributes; --validate also writes
routes.rou.xml in the working folder (as the real one does)."""
import argparse
import random
import re
import xml.etree.ElementTree as ET

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--net-file')
ap.add_argument('-o', '--output-trip-file')
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('-b', '--begin', type=float, default=0.0)
ap.add_argument('-e', '--end', type=float, default=3600.0)
ap.add_argument('-p', '--period', type=float, default=1.0)
ap.add_argument('--vclass', default='passenger')
ap.add_argument('--validate', action='store_true')
ap.add_argument('--remove-loops', action='store_true')
ap.add_argument('--prefix', default='')
ap.add_argument('--min-distance', type=float, default=0.0)
ap.add_argument('--trip-attributes', default='')
a = ap.parse_args()
net = ET.parse(a.net_file).getroot()
edges = sorted(e.get('id') for e in net.iter('edge') if e.get('function') != 'internal' and
               (not e.find('lane').get('allow') or a.vclass in e.find('lane').get('allow').split()))
rng = random.Random(a.seed)
attrs = dict(re.findall(r'(\w+)="([^"]*)"', a.trip_attributes))
root = ET.Element('routes')
k, t = 0, a.begin
while t < a.end:
    e = rng.choice(edges)
    ET.SubElement(root, 'trip', dict({'id': f'{a.prefix}{k}', 'depart': f'{t:.2f}', 'from': e, 'to': e}, **attrs))
    k += 1
    t = a.begin + k * a.period
ET.ElementTree(root).write(a.output_trip_file, encoding='utf-8', xml_declaration=True)
if a.validate:
    ET.ElementTree(ET.Element('routes')).write('routes.rou.xml', encoding='utf-8', xml_declaration=True)
print('Success.')
