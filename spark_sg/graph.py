# simple_object_graph.py
import networkx as nx
import matplotlib.pyplot as plt


class ObjectGraph:
    """
    Tiny convenience wrapper over a directed NetworkX graph.

    • CONTAINS / CONSTRAINED are just string labels on edges
    • “root” nodes (parents) and “background” nodes get simple role flags
    """

    def __init__(self) -> None:
        self.G = nx.DiGraph()

    # ------------------------------------------------------------------ #
    # building helpers
    # ------------------------------------------------------------------ #
    def add_contains(self, parent: str, *children: str) -> None:
        self._add_edges(parent, children, etype="contains", role="root")

    def add_constrained(self, parent: str, *children: str) -> None:
        self._add_edges(parent, children, etype="constrained", role="root")

    def add_background(self, *nodes: str) -> None:
        for n in nodes:
            self.G.add_node(n, role="background")

    @classmethod
    def from_input(
        cls,
        contains: dict[str, list[str]],
        constrained: dict[str, list[str]],
        background: list[str],
    ) -> "ObjectGraph":
        g = cls()
        for p, ch in contains.items():
            g.add_contains(p, *ch)
        for p, ch in constrained.items():
            g.add_constrained(p, *ch)
        g.add_background(*background)
        return g

    # ------------------------------------------------------------------ #
    # visualisation (quick-and-dirty matplotlib)
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # pretty visualisation
    # ------------------------------------------------------------------ #
    def draw(
        self,
        *,
        figsize: tuple[int, int] = (10, 8),
        prog_priority: tuple[str, ...] = ("dot", "neato", "fdp"),
        save_path: str | None = None,
    ) -> None:
        """
        Draw the graph with edge labels and nicer styling.

        Parameters
        ----------
        figsize : (w, h)
            Size of the matplotlib figure.
        prog_priority : tuple
            Ordered list of Graphviz layout engines to try first.
        save_path : str | None
            If given, saves the figure to this path instead of calling plt.show().
        """
        import networkx as nx

        # ------ choose a layout -------------------------------------------------
        pos = None
        for prog in prog_priority:
            try:
                pos = nx.nx_pydot.graphviz_layout(self.G, prog=prog)
                break
            except Exception:
                continue
        if pos is None:  # Graphviz not available or failed
            try:
                pos = nx.kamada_kawai_layout(self.G)
            except Exception:
                pos = nx.spring_layout(self.G, seed=0)

        # ------ colour maps -----------------------------------------------------
        role_colour = {
            "root": "#e74c3c",  # red
            "background": "#95a5a6",  # gray
            None: "#2ecc71",  # regular objects (green)
        }
        edge_colour = {
            "contains": "#3498db",  # blue
            "constrained": "#f39c12",  # orange
        }

        # ------ draw ------------------------------------------------------------
        plt.figure(figsize=figsize)

        # nodes
        nx.draw_networkx_nodes(
            self.G,
            pos,
            node_size=900,
            linewidths=1.2,
            edgecolors="#333333",
            node_color=[role_colour[self.G.nodes[n].get("role")] for n in self.G.nodes],
        )

        # edges
        nx.draw_networkx_edges(
            self.G,
            pos,
            arrows=True,
            arrowsize=20,
            width=1.8,
            connectionstyle="arc3,rad=0.08",
            edge_color=[edge_colour[self.G.edges[e]["type"]] for e in self.G.edges],
        )

        # edge labels
        nx.draw_networkx_edge_labels(
            self.G,
            pos,
            edge_labels={e: self.G.edges[e]["type"] for e in self.G.edges},
            font_size=9,
            font_color="#555555",
            label_pos=0.55,
            rotate=False,
            bbox={"alpha": 0.0},
        )

        # node labels
        nx.draw_networkx_labels(
            self.G,
            pos,
            font_size=10,
            font_weight="bold",
            font_color="#ffffff",
        )

        plt.axis("off")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            plt.close()
        else:
            plt.show()

    # ------------------------------------------------------------------ #
    # island-grid v3  –  explicit rows, works with any Graphviz
    # ------------------------------------------------------------------ #
    def pretty_draw_grid(
        self,
        out_file: str,
        *,
        show: bool = False,  # headless by default
        fmt: str | None = None,  # infer from suffix if None
        wrap: int = 14,  # line-wrap long labels
        max_cols: int | None = None,  # islands per row (default = √N)
    ) -> None:
        """
        One cluster-island per root; islands placed in an explicit row×col
        grid so width *and* height are always limited – even if your
        Graphviz is older than 2.50 and ignores packmode=array.

        Parameters
        ----------
        out_file : str   – destination (.pdf, .svg, .png …)
        show     : bool  – open afterwards only if True
        fmt      : str   – output format; guessed from out_file if None
        wrap     : int   – wrap node labels every `wrap` chars  (0 = off)
        max_cols : int   – max islands per row; default = ceil(sqrt(#roots))
        """
        import graphviz, math, os, pathlib, textwrap, webbrowser, networkx as nx

        # ------------ palettes -------------------------------------------------
        ROLE_FILL = {"root": "#e74c3c", "background": "#95a5a6", None: "#2ecc71"}
        EDGE_CLR = {"contains": "#3498db", "constrained": "#f39c12"}

        # ------------ helpers --------------------------------------------------
        def _wrap(label: str) -> str:
            label = label.replace("_instance", "")
            return "\n".join(textwrap.wrap(label, wrap)) if wrap > 0 else label

        # ------------ classify nodes ------------------------------------------
        roots = [n for n, d in self.G.nodes(data=True) if d.get("role") == "root"]
        bg = [n for n, d in self.G.nodes(data=True) if d.get("role") == "background"]

        islands = {r: set(nx.dfs_preorder_nodes(self.G, r)) for r in roots}
        assigned = set().union(*islands.values())
        orphans = set(self.G.nodes) - assigned - set(bg)

        # ------------ grid parameters -----------------------------------------
        N = len(roots) or 1
        cols = max_cols or math.ceil(math.sqrt(N))
        # split roots into rows
        rows = [roots[i : i + cols] for i in range(0, N, cols)]

        dot = graphviz.Digraph(
            graph_attr=dict(rankdir="TB", splines="true", overlap="false", margin="0"),
            node_attr=dict(
                shape="ellipse",
                style="filled",
                fontname="Helvetica",
                fontsize="11",
                penwidth="1.2",
            ),
            edge_attr=dict(
                fontname="Helvetica", fontsize="10", penwidth="1.4", arrowsize="0.9"
            ),
        )

        # ------------ build each island cluster -------------------------------
        for root, nodes in islands.items():
            with dot.subgraph(name=f"cluster_{root}") as sg:
                sg.attr(label="", margin="8")
                for n in nodes:
                    role = self.G.nodes[n].get("role")
                    sg.node(
                        n,
                        label=_wrap(n),
                        fillcolor=ROLE_FILL[role],
                        fontcolor="#ffffff",
                    )

        # ------------ organise rows (rank=same + invisible edges) -------------
        for r_idx, row_roots in enumerate(rows):
            with dot.subgraph(name=f"row_{r_idx}") as sg:
                sg.attr(rank="same")  # keep this row horizontal
                # invisible chain to preserve order & spacing
                for i, n in enumerate(row_roots):
                    sg.node(n)  # already declared, just reference
                    if i > 0:
                        sg.edge(
                            row_roots[i - 1],
                            n,
                            style="invis",
                            weight="1",
                            arrowhead="none",
                        )

        # ------------ orphans & background ------------------------------------
        if orphans:
            with dot.subgraph(name="cluster_orphans") as sg:
                sg.attr(label="", margin="8")
                for n in orphans:
                    sg.node(
                        n,
                        label=_wrap(n),
                        fillcolor=ROLE_FILL[None],
                        fontcolor="#ffffff",
                    )

        if bg:
            with dot.subgraph(name="cluster_bg") as sg:
                sg.attr(label="", margin="8", rank="sink")
                for n in bg:
                    sg.node(
                        n,
                        label=_wrap(n),
                        fillcolor=ROLE_FILL["background"],
                        fontcolor="#ffffff",
                    )

        # ------------ coloured, labelled edges --------------------------------
        for u, v, d in self.G.edges(data=True):
            etype = d["type"]
            dot.edge(
                u, v, label=etype, color=EDGE_CLR[etype], fontcolor=EDGE_CLR[etype]
            )

        # ------------ write file (no auto-open unless show=True) --------------
        fmt = fmt or os.path.splitext(out_file)[1][1:] or "pdf"
        pathlib.Path(out_file).write_bytes(dot.pipe(format=fmt))

        if show:
            webbrowser.open(pathlib.Path(out_file).as_uri())
    # ------------------------------------------------------------------ #
    # vertical tree-islands  +  bounded-grid remainder
    # ------------------------------------------------------------------ #
    def pretty_draw_trees_grid(
        self,
        out_file: str,
        *,
        show: bool = False,       # perfect for headless servers
        fmt: str | None = None,   # auto-infer from suffix
        wrap: int = 16,           # soft-wrap long labels
        grid_cols: int = 6,       # max objects per grid row
    ) -> None:
        """
        • Every root (has no parent) → its own vertical cluster island.
        • Islands are stacked vertically (one per row) so width is bounded.
        • Any node with *zero* edges (and not a root) goes into a compact
          grid with ≤ `grid_cols` items per row.

        Needs `pip install graphviz`  +  the Graphviz binaries.
        """
        import graphviz, os, pathlib, textwrap, math, itertools, webbrowser
        import networkx as nx

        # ---------- palettes --------------------------------------------------
        ROLE_FILL = {"root": "#e74c3c", "background": "#95a5a6", None: "#2ecc71"}
        EDGE_CLR  = {"contains": "#3498db", "constrained": "#f39c12"}

        # ---------- helpers ---------------------------------------------------
        def _wrap(label: str) -> str:
            label = label.replace("_instance", "")
            return "\n".join(textwrap.wrap(label, wrap)) if wrap > 0 else label

        # ---------- classify nodes -------------------------------------------
        roots = [n for n, d in self.G.nodes(data=True) if d.get("role") == "root"]

        degree_zero = [n for n in self.G.nodes if self.G.degree(n) == 0]
        dangling    = [n for n in degree_zero if n not in roots]

        # connected components reachable from each root
        islands = {r: set(nx.dfs_preorder_nodes(self.G, r)) for r in roots}

        # ---------- DOT skeleton ---------------------------------------------
        dot = graphviz.Digraph(
            graph_attr=dict(
                rankdir="TB",         # trees grow downward
                splines="true",
                overlap="false",
                pack="true",
                packmode="clust",     # pack clusters tightly
                margin="0",
            ),
            node_attr=dict(
                shape="ellipse",
                style="filled",
                fontname="Helvetica",
                fontsize="11",
                penwidth="1.2",
            ),
            edge_attr=dict(
                fontname="Helvetica",
                fontsize="10",
                penwidth="1.4",
                arrowsize="0.9",
            ),
        )

        # ---------- build one vertical cluster per root -----------------------
        for idx, (root, nodes) in enumerate(islands.items()):
            with dot.subgraph(name=f"cluster_tree_{idx}") as sg:
                sg.attr(label="", margin="8")          # no border label
                # ensure root is drawn at the top of its cluster
                sg.node(root, label=_wrap(root),
                        fillcolor=ROLE_FILL["root"], fontcolor="#ffffff", rank="min")
                for n in nodes:
                    if n == root:
                        continue
                    role = self.G.nodes[n].get("role")
                    sg.node(n, label=_wrap(n),
                            fillcolor=ROLE_FILL.get(role, ROLE_FILL[None]),
                            fontcolor="#ffffff")

        # invisible chain to keep clusters vertical (one under another)
        for i in range(len(roots) - 1):
            dot.edge(roots[i], roots[i + 1],
                     style="invis", weight="10", arrowhead="none")

        # ---------- dangling objects grid ------------------------------------
        if dangling:
            rows = [
                list(itertools.islice(it, grid_cols))
                for it in (iter(dangling),)
                for _ in range(math.ceil(len(dangling) / grid_cols))
            ]
            for r, row_nodes in enumerate(rows):
                with dot.subgraph(name=f"grid_row_{r}") as sg:
                    sg.attr(rank="sink", label="", margin="6")
                    sg.attr(rank="same")               # one horizontal line
                    for n in row_nodes:
                        sg.node(
                            n,
                            label=_wrap(n),
                            fillcolor=ROLE_FILL[
                                "background"
                            ],  # ← gray instead of green
                            fontcolor="#ffffff",
                        )
                    # link invisibly to keep ordering & spacing
                    for a, b in zip(row_nodes, row_nodes[1:]):
                        sg.edge(a, b, style="invis", arrowhead="none", weight="1")

        # ---------- coloured edges with labels -------------------------------
        for u, v, d in self.G.edges(data=True):
            etype = d["type"]
            dot.edge(u, v, label=etype,
                     color=EDGE_CLR[etype], fontcolor=EDGE_CLR[etype])

        # ---------- write file ------------------------------------------------
        fmt = fmt or os.path.splitext(out_file)[1][1:] or "pdf"
        pathlib.Path(out_file).write_bytes(dot.pipe(format=fmt))

        if show:
            webbrowser.open(pathlib.Path(out_file).as_uri())

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _add_edges(
        self,
        parent: str,
        children: list[str],
        *,
        etype: str,
        role: str,
    ) -> None:
        self.G.add_node(parent, role=role)
        for c in children:
            self.G.add_node(c)  # default role = None
            self.G.add_edge(parent, c, type=etype)

if __name__ == "__main__":
    import pickle
    with open("/home/exx/Downloads/tmp_viz_info.pkl", "rb") as f:
        viz_info = pickle.load(f)
        
    contains = viz_info["contains"]
    constrained = viz_info["constrained"]
    background = viz_info["parentless_instances"]

    g = ObjectGraph.from_input(contains, constrained, background)
    
    g.pretty_draw_trees_grid("/home/exx/Downloads/out.pdf",show=False)
