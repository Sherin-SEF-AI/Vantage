"""A fake netconvert_carla.py: the net of tests/fake_carla's straight road (four lanes as edges, SUMO's axes: x east,
y north = CARLA's −y, shifted by the net offset), with one traffic light linked to CARLA's landmark '1000' (a
generated net has CARLA's lights). It reads the OpenDRIVE only to check it is one, and counts its calls in
netconvert_calls.txt beside the output (tests read it to see the cache work)."""
import argparse
import os
import sys

OFFSET = (100.0, 50.0)
LANES = (('east', 0.0, 1), ('east2', 3.5, 1), ('west', -3.5, -1), ('west2', -7.0, -1))   # (edge, CARLA y, +x or −x)


def net_xml(tls=True, town=''):
    ox, oy = OFFSET
    edges = []
    for eid, cy, d in LANES:
        y = oy - cy
        x0, x1 = (ox - 140.0, ox + 140.0) if d > 0 else (ox + 140.0, ox - 140.0)
        edges.append(f'    <edge id="{eid}" from="a_{eid}" to="b_{eid}" priority="1">\n'
                     f'        <lane id="{eid}_0" index="0" speed="13.89" length="280.00" '
                     f'shape="{x0:.2f},{y:.2f} {x1:.2f},{y:.2f}"/>\n    </edge>')
    tl = ('    <tlLogic id="J1" type="static" programID="0" offset="0">\n'
          '        <param key="linkSignalID:0" value="1000"/>\n'
          '        <phase duration="10" state="G"/>\n        <phase duration="3" state="y"/>\n'
          '        <phase duration="10" state="r"/>\n    </tlLogic>\n') if tls else ''
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<!-- fake SUMO net of {town or "the fake road"} -->\n<net version="1.20">\n'
            f'    <location netOffset="{ox:.2f},{oy:.2f}" convBoundary="0,0,280,20" projParameter="!"/>\n'
            + '\n'.join(edges) + '\n' + tl + '</net>\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('xodr_file')
    ap.add_argument('--output', '-o', default='net.net.xml')
    ap.add_argument('--guess-tls', action='store_true')
    a = ap.parse_args()
    text = open(a.xodr_file, encoding='utf-8').read()
    if '<OpenDRIVE' not in text:
        print('not an OpenDRIVE file', file=sys.stderr)
        sys.exit(1)
    if 'netconvert fails' in text:
        print('Error: netconvert failed on this map', file=sys.stderr)
        sys.exit(1)
    with open(a.output, 'w', encoding='utf-8') as fh:
        fh.write(net_xml(True, text.split('name="', 1)[-1].split('"', 1)[0]))
    calls = os.path.join(os.path.dirname(os.path.abspath(a.output)), 'netconvert_calls.txt')
    with open(calls, 'a', encoding='utf-8') as fh:
        fh.write(a.xodr_file + '\n')
