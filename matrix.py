import random
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
    for i in range(nodes_n):
        if attack and random.random() < byzantine_probs:
            attack_matrix.append(attacks[attack]())
        else:
            attack_matrix.append(None)
    for i in range(nodes_n):
        adj_i = []
        for j in range(nodes_n):
            if i == j or random.random() > connect_probs:
                adj_i.append(0)
            else:
                adj_i.append(1)
        matrix.append(adj_i)
    return matrix, attack_matrix
