import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, deque

def visualize_family_tree(family):
    """
    Visualize the family as a layered (tree-like) directed graph:
      - Each person is a node labeled with ID, sex, genotype.
      - Directed edges father->child, mother->child.
      - Node color: genotype=0 => blue, genotype=1 => red.
      - Generations arranged top-to-bottom via BFS layering (founders at top).

    Args:
      family (list[dict]): 
        Each dict has keys like:
         - 'id' (int)
         - 'sex' ('M' or 'F')
         - 'father' (int or None)
         - 'mother' (int or None)
         - 'geno' (0 or 1)
         - ...

    Usage:
      >>> fam = [...]  # already built by your code
      >>> visualize_family_tree(fam)
    """

    # 1) Create a directed graph
    G = nx.DiGraph()

    # Add each person as a node, storing sex & genotype as attributes
    for person in family:
        pid = person['id']
        sex = person.get('sex', '?')
        geno = person.get('geno', 0)
        # We'll store these as node attributes
        G.add_node(pid, sex=sex, geno=geno)

    # Add edges father->child, mother->child
    for person in family:
        pid = person['id']
        fa = person['father']
        mo = person['mother']
        if fa is not None:
            G.add_edge(fa, pid)
        if mo is not None:
            G.add_edge(mo, pid)

    # 2) Assign BFS-based "generation" levels
    gen_map = {}
    queue = deque()

    # Founders: father=None, mother=None => generation=0
    founders = [p['id'] for p in family 
                if p['father'] is None and p['mother'] is None]
    for f_id in founders:
        gen_map[f_id] = 0
        queue.append(f_id)

    # BFS: child's gen = parent's gen + 1
    while queue:
        parent = queue.popleft()
        parent_gen = gen_map[parent]
        for child in G.successors(parent):
            if child not in gen_map:
                gen_map[child] = parent_gen + 1
                queue.append(child)

    # 3) Position nodes in rows by generation
    level_dict = defaultdict(list)
    for node_id, g in gen_map.items():
        level_dict[g].append(node_id)

    pos = {}
    for g in sorted(level_dict.keys()):
        level_dict[g].sort()
        x_offset = 0.0
        for n_id in level_dict[g]:
            pos[n_id] = (x_offset, -g)  # y = -g, x = x_offset
            x_offset += 1.5

    # 4) Prepare colors & labels
    colors = []
    labels = {}
    for node, data in G.nodes(data=True):
        gval = data.get('geno', 0)
        sex = data.get('sex', '?')
        # color by genotype
        c = 'red' if gval == 1 else 'blue'
        colors.append(c)
        # label shows ID, sex, genotype
        labels[node] = f"ID={node}\n(sex={sex}, g={gval})"

    # 5) Draw
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=800, alpha=0.9)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrows=True, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels=labels, font_color='white', font_weight='bold')

    plt.title("Family Tree (Layered by Generation)")
    plt.axis("off")
    plt.show()
