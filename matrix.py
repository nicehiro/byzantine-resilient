import random
from typing import List
from attack import attacks

# decentralization matrix
# poor connection topo
# decentra_matrix = [
#     [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
#     [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
#     [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
#     [1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
# ]
# topo satisfy n >= 2f + 1
# decentra_matrix = [
#     [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
#     [1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
#     [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
#     [0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
#     [1, 1, 0, 1, 0, 1, 1, 0, 0, 0],
#     [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
#     [0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
#     [0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
#     [1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
# ]
# topo satisfy n >= 2f + 3
decentra_matrix = [
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 1, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 1, 1, 1],
    [0, 1, 0, 1, 1, 1, 1, 0, 1, 1],
    [0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
    [1, 1, 0, 0, 1, 0, 1, 1, 0, 0],
]
# topo satisfy n >= 4f + 3
# decentra_matrix = [
#     [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# ]


def make_matrix(nodes_n, connect_probs, byzantine_probs, attack):
    """Generate matrix by number of nodes, connection probability and byzantine probability.

    Args:
        nodes_n (int): number of nodes
        connect_probs (float): connection probability
        byzantine_probs (float): byzantine probability

    Returns:
        matrix (List): adj matrix
        attacks (List): attacks
    """
    matrix = []
    attack_matrix = []
    non_byzantines = []
    for i in range(nodes_n):
        if attack and random.random() < byzantine_probs:
            attack_matrix.append(attacks[attack]())
        else:
            attack_matrix.append(None)
            non_byzantines.append(i)
    for i in range(nodes_n):
        adj_i = []
        has_benign = False
        for j in range(nodes_n):
            if i == j or random.random() > connect_probs:
                adj_i.append(0)
            else:
                adj_i.append(1)
                if j in non_byzantines:
                    has_benign = True
        if not has_benign:
            # make sure every node has a benign neighbor except self
            if len(non_byzantines) <= 1:
                continue
            benigh_n = random.choice(non_byzantines)
            while benigh_n == i:
                benigh_n = random.choice(non_byzantines)
            adj_i[benigh_n] = 1
        matrix.append(adj_i)
    matrix = ensure_rooted(matrix, attack_matrix)
    return matrix, attack_matrix


def ensure_rooted(matrix: List[List], attacks: List):
    # ensure matrix is rooted
    # strong rooted
    sub_gs = []
    n = len(matrix)
    visited = {j: False for j in range(n)}
    while True:
        start = None
        for k, v in visited.items():
            if not v and attacks[k] is None:
                start = k
                break
        if start is None:
            break
        sub_g = [start]
        q = [start]
        while len(q) > 0:
            k = q.pop(0)
            visited[k] = True
            for v in matrix[k]:
                if v == 1 and not visited[v] and attacks[v] is None:
                    visited[v] = True
                    q.append(v)
                    sub_g.append(v)
        sub_gs.append(sub_g)
    for i in range(len(sub_gs) - 1):
        src = random.choice(sub_gs[i])
        dst = random.choice(sub_gs[i + 1])
        matrix[src][dst] = 1
    return matrix


if __name__ == "__main__":
    m, a = make_matrix(nodes_n=5, connect_probs=0.1, byzantine_probs=0.5, attack="max")
    # m = [[0, 0, 0, 1, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0]]
    # a = [1, None, 1, None, None]
    print(m)
    print(a)
    e_m = ensure_rooted(m, a)
    print(e_m)
