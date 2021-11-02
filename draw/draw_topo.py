import sys
print(sys.path)

from train import decentra_matrix, attacks

import networkx as nx
import matplotlib.pyplot as plt


nodes_n = len(attacks)
byzantines = [i for i, a in enumerate(attacks) if a is not None]
non_byzantines = [i for i, a in enumerate(attacks) if a is None]

G = nx.DiGraph()
G.add_nodes_from(
    [(i, {"color": "green" if attacks[i] is None else "red"}) for i in range(nodes_n)]
)
options = {"node_size": 1000}
pos = nx.circular_layout(G)
# draw nodes
nx.draw_networkx_nodes(
    G, pos=pos, nodelist=non_byzantines, node_color="green", **options
)
nx.draw_networkx_nodes(G, pos=pos, nodelist=byzantines, node_color="red", **options)
# draw node label
nx.draw_networkx_labels(G, pos)

# draw edges
for i in range(nodes_n):
    for j in range(nodes_n):
        if decentra_matrix[i][j] != 0:
            color = "green" if attacks[j] is None else "red"
            style = "solid" if attacks[j] is None else "dashed"
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[[j, i]],
                width=1,
                style=style,
                arrows=True,
                arrowsize=10,
                edge_color=color,
                **options
            )

plt.savefig("topo.png")
