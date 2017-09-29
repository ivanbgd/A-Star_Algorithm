"""
        *** Bidirectional A-Star algorithm ***

Useful for real road network graph problems, on a planar map/grid.

n - number of nodes 
m - number of edges

If the graph isn't dense, ie. it's sparse, it's better to implement
priority queue as heap than as array.
A graph is sparse when n and m are of the same order of magnitude.

Here, priority queue is implemented by using module heapq.

We put (dist, name) into heap, where dist is f(v), and f(v) = g(v) + h(v),
that is, known part + heuristic part.
Known part is correct distance from starting vertex, s, to this one, v.
Heuristic part is estimation from this vertex, v, to the target vertex, t.

This version of the algorithm assumes that we are given
coordinates of vertices, so we can calculate Euclidean
distances and use it as our heuristics function.
An alternative heuristic would be Manhattan distance,
but Euclidean is used in a case like this (road network).

This is Bidirectional A-Star algorithm, without landmarks.
This is directed search, thanks to heuristic (potential) function.
It represents Dijkstra with Potentials.
This means that this is just plain Dijkstra with new edge weights.
Actually, if heuristic (potential) function is identical to 0,
this is identical to the UCS Dijkstra.

It's probably the best to compute the heuristic on the fly.
A-Star is directed search, which means not many vertices
will be processed. If, on the other hand, we want to precompute
the heuristic function, we have to do that for every vertex of the graph,
because we don't know in advance which vertices will be processed.
In each query, the heuristic will be different, because target vertex
will be different, and heuristic is target-dependent.

A-Star with heuristics doesn't need the preprocessing phase.
We can have it, but it will only make the algorithm slower.
In other words, it doesn't make sense. That's because heuristic values depend
on a single query - they are not common to many queries in a graph.

This version of the algorithm doesn't reconstruct the shortest path.
It only computes its length and returns it.

This bidirectional variant is the fastest one.
"""


import sys
import math
import heapq


class AStar:
    def __init__(self, n, adj, cost, x, y):
        self.n = n;
        self.adj = adj
        self.cost = cost
        self.inf = n*10**6
        self.f = [[self.inf]*n, [self.inf]*n]       # forward & backward; f(v) = g(v) + h(v); these are new distances, with potential h (h(v) is computed on the fly); Dijkstra UCS now works with them (this is self.distance in Dijkstra); self.f can be a map (dictionary)
        self.g = [[self.inf]*n, [self.inf]*n]       # forward & backward;  this is the known part of the distance; h is the heuristic part; this is the true distance from starting vertex to v; self.g can be a map (dictionary)
        self.closedF = set()                        # forward closed set (processed nodes)
        self.closedR = set()                        # backward closed set (processed nodes)
        self.valid = [[True] * n, [True] * n]       # is vertex (name) valid or not - it's valid while name (vertex) is in open set (in heap)
        # Coordinates of the nodes
        self.x = x
        self.y = y

    # See the explanation of this method in the starter for friend_suggestion
    def clear(self):
        self.f = [[self.inf]*n, [self.inf]*n]
        self.g = [[self.inf]*n, [self.inf]*n]
        self.closedF.clear()
        self.closedR.clear()
        self.valid = [[True] * n, [True] * n]

    def heur(self, s, t, v):
        """ The so called "average function" by Goldberg. """
        pi_f = math.sqrt((self.x[v] - self.x[t])**2 + (self.y[v] - self.y[t])**2)
        pi_r = math.sqrt((self.x[s] - self.x[v])**2 + (self.y[s] - self.y[v])**2)
        pf = (pi_f - pi_r)/2.0
        pr = -pf
        return pf, pr

    def heurMax(self, s, t, v):
        """ Computing the Shortest Path - A-Star Search Meets Graph Theory (Microsoft, Goldberg)
            Goldberg calls it "max function".
            We can see it's slower than the above average function, which is consistent with his
            findings (BLA vs BLM, though that's ALT algorithm, but I guess that doesn't make a difference).
        """
        pi_t_v = math.sqrt((self.x[v] - self.x[t])**2 + (self.y[v] - self.y[t])**2)     # forward
        pi_s_v = math.sqrt((self.x[s] - self.x[v])**2 + (self.y[s] - self.y[v])**2)     # backward
        pi_s_t = self.pi_s_t
        pi_t_s = pi_s_t
        beta = int(pi_t_s) >> 3     # a constant that depends on pi_t(s) and/or pi_s(t) (their implementation uses a constant fraction of pi_t(s))
        pf = max(pi_t_v, pi_s_t - pi_s_v + beta)
        pr = -pf
        return pf, pr

    # See the explanation of this method in the starter for friend_suggestion
    def visit(self, open, side, name, s, t):
        closed = self.closedF if side == 0 else self.closedR
        for i in range(len(self.adj[side][name])):
            neighbor = self.adj[side][name][i]
            if neighbor in closed:     # also ok: if (self.f[neighbor], neighbor) in closed:
                continue
            temp_g = self.g[side][name] + self.cost[side][name][i]
            if (self.f[side][neighbor], neighbor) not in open[side]:
                hf, hr = self.heur(s, t, neighbor)
                self.g[side][neighbor] = temp_g
                self.f[side][neighbor] = temp_g + (hf if side == 0 else hr)     # h depends on the side!
                heapq.heappush(open[side], (self.f[side][neighbor], neighbor))
                continue
            if self.g[side][neighbor] > temp_g:
                hf, hr = self.heur(s, t, neighbor)
                self.g[side][neighbor] = temp_g
                self.f[side][neighbor] = temp_g + (hf if side == 0 else hr)     # h depends on the side!
                heapq.heappush(open[side], (self.f[side][neighbor], neighbor))

    # Returns the distance from s to t in the graph
    def query(self, s, t):
        self.clear()
        open = [[], []]     # list of two priority queues (that are implemented as min-heaps); q[0] is forward, q[1] is reverse - those are the two "open" sets; elements are tuples (f(v), v)

        #self.pi_s_t = math.sqrt((self.x[s] - self.x[t])**2 + (self.y[s] - self.y[t])**2)     # need only for heurMax(); comment it out for regular heur (heurAvg); backward

        hf, hr = self.heur(s, t, s)
        self.g[0][s] = 0
        self.f[0][s] = self.g[0][s] + hf
        heapq.heappush(open[0], (self.f[0][s], s))
        
        hf, hr = self.heur(s, t, t)
        self.g[1][t] = 0
        self.f[1][t] = self.g[1][t] + hr
        heapq.heappush(open[1], (self.f[1][t], t))

        while open[0] or open[1]:
            # the inner while loop removes and returns the best vertex
            best = None
            name = None
            while open[0]:
                best = heapq.heappop(open[0])
                name = best[1]
                if self.valid[0][name]:
                    self.valid[0][name] = False;
                    break
            self.visit(open, 0, name, s, t)   # forward
            if name in self.closedR:
                break
            self.closedF.add(name)      # also ok: self.closedF.add(best)

            # the inner while loop removes and returns the best vertex
            best = None
            name = None
            while open[1]:
                best = heapq.heappop(open[1])
                name = best[1]
                if self.valid[1][name]:
                    self.valid[1][name] = False;
                    break
            self.visit(open, 1, name, s, t)   # forward
            if name in self.closedF:
                break
            self.closedR.add(name)      # also ok: self.closedR.add(best)

        distance = self.inf

        # merge closedF & closedR
        self.closedF = self.closedF | self.closedR  # sets - O(1) lookup
        
        for u in self.closedF:
            if (self.g[0][u] + self.g[1][u] < distance):
                distance = self.g[0][u] + self.g[1][u]

        return distance if distance < self.inf else -1


if __name__ == '__main__':
    input = sys.stdin.read()    # Python 2
    data = list(map(int, input.split()))
    n, m = data[0:2]    # number of nodes, number of edges; nodes are numbered from 1 to n
    data = data[2:]
    x = [0] * n
    y = [0] * n
    adj = [[[] for _ in range(n)], [[] for _ in range(n)]]    # holds adjacency lists for every vertex in the graph; contains both forward and reverse arrays
    cost = [[[] for _ in range(n)], [[] for _ in range(n)]]   # holds weights of the edges; contains both forward and reverse arrays
    # coordinates x and y of the corresponding node
    for i in range(n):
        x[i] = data[i << 1]
        y[i] = data[(i << 1) + 1]
    data = data[2*n:]
    # directed edge (u, v) of length c from the node number u to the node number v
    for e in range(m):
        u = data[3*e]
        v = data[3*e+1]
        c = data[3*e+2]
        adj[0][u-1].append(v-1)
        cost[0][u-1].append(c)
        adj[1][v-1].append(u-1)
        cost[1][v-1].append(c)
    astar = AStar(n, adj, cost, x, y)
    data = data[3*m:]
    # the number of queries for computing the distance
    q = data[0]
    data = data[1:]
    # s and t are numbers ("names") of two nodes to compute the distance from s to t
    for i in range(q):
        s = data[i << 1]
        t = data[(i << 1) + 1]
        print(astar.query(s-1, t-1))
        
