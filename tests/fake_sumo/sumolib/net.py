"""sumolib.net.readNet for the fake nets: edges with their permissions, the net offset."""
import traci


class _Edge:
    def __init__(self, eid, allow):
        self._id, self._allow = eid, allow

    def getID(self):
        return self._id

    def allows(self, vclass):
        return not self._allow or vclass in self._allow


class _Net:
    def __init__(self, d):
        self._d = d

    def getLocationOffset(self):
        return self._d['offset']

    def getEdges(self, withInternal=True):
        return [_Edge(k, v['allow']) for k, v in sorted(self._d['edges'].items())]


def readNet(filename, **kw):
    return _Net(traci.read_net(filename))
