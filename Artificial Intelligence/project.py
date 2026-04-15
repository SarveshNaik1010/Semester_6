"""
=============================================================================
Volumetric Fusion A* (VF-A*) Algorithm for Logistics Optimization
=============================================================================
Project: A* Algorithm for Resource Optimization in Logistics
Author : Sarvesh Naik | PRN: 3124045020142 | Roll No: 33
Dept   : B.Tech CSE, Third Year | Vishwakarma University, Pune
A.Y.   : 2025-2026
=============================================================================

Overview
--------
This implementation models a logistics road network as a directed weighted
graph and deploys a modified A* search algorithm — the Volumetric Fusion A*
(VF-A*) — that enables multiple delivery vehicles to:
  1. Find optimal routes independently using A*.
  2. Detect "Fusion Nodes" where two vehicles can meet en-route.
  3. Transfer cargo from one vehicle to another if volumetric constraints
     allow, retiring the now-empty "giving" vehicle.
  4. Maximize fleet utilization and minimize empty-mile travel.

Visualization Modules
---------------------
  • Fig 1 : Road network graph (all nodes color-coded by type)
  • Fig 2 : Individual A* routes for all vehicles before fusion
  • Fig 3 : Fusion event diagram — cargo transfer at fusion node
  • Fig 4 : Post-fusion routes with retired vehicles marked
  • Fig 5 : Vehicle utilization bar chart (before vs after fusion)
  • Fig 6 : Summary dashboard (metrics, legend, route table)
=============================================================================
"""

# ── Standard library ────────────────────────────────────────────────────────
import heapq
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── Visualisation ────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ── Reproducibility ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# =============================================================================
# SECTION 1 — DATA MODELS
# =============================================================================

# Node-type constants
NODE_PICKUP   = "pickup"
NODE_DELIVERY = "delivery"
NODE_FUSION   = "fusion"
NODE_DEPOT    = "depot"

# Colour palette used across all figures
PALETTE = {
    NODE_PICKUP   : "#3B82F6",   # blue
    NODE_DELIVERY : "#22C55E",   # green
    NODE_FUSION   : "#EAB308",   # yellow
    NODE_DEPOT    : "#EF4444",   # red
    "edge"        : "#94A3B8",
    "route_a"     : "#6366F1",   # indigo  – vehicle A
    "route_b"     : "#F97316",   # orange  – vehicle B
    "route_c"     : "#EC4899",   # pink    – vehicle C
    "route_d"     : "#14B8A6",   # teal    – vehicle D
    "fused_route" : "#10B981",   # emerald – post-fusion
    "retired"     : "#6B7280",   # grey
    "bg"          : "#0F172A",   # dark background
    "panel"       : "#1E293B",
    "text"        : "#F8FAFC",
    "subtext"     : "#94A3B8",
}

VEHICLE_COLORS = [
    PALETTE["route_a"], PALETTE["route_b"],
    PALETTE["route_c"], PALETTE["route_d"],
]


@dataclass
class Node:
    """Represents a location in the logistics graph."""
    node_id  : str
    node_type: str          # pickup | delivery | fusion | depot
    x        : float        # geographic x-coordinate (used for heuristic)
    y        : float        # geographic y-coordinate

    def __hash__(self):  return hash(self.node_id)
    def __eq__(self, o): return self.node_id == o.node_id


@dataclass
class Edge:
    """Directed road segment between two nodes."""
    source  : str
    target  : str
    distance: float         # km
    traffic : float         # multiplier ≥ 1.0

    @property
    def cost(self) -> float:
        """Effective travel cost = distance × traffic weight."""
        return self.distance * self.traffic


@dataclass
class Package:
    """A cargo item assigned to a vehicle."""
    pkg_id     : str
    weight     : float      # kg
    volume     : float      # m³
    destination: str        # node_id


@dataclass
class Vehicle:
    """A delivery vehicle operating in the logistics network."""
    vehicle_id  : str
    start_node  : str
    destination : str
    max_capacity: float     # m³
    packages    : List[Package] = field(default_factory=list)
    active      : bool = True
    route       : List[str] = field(default_factory=list)
    color       : str = "#6366F1"

    @property
    def current_load(self) -> float:
        return sum(p.volume for p in self.packages)

    @property
    def utilization(self) -> float:
        return self.current_load / self.max_capacity if self.max_capacity else 0.0

    def can_absorb(self, other: "Vehicle") -> bool:
        """True if this vehicle can take on all of other's cargo."""
        return (self.current_load + other.current_load) <= self.max_capacity


# =============================================================================
# SECTION 2 — GRAPH
# =============================================================================

class LogisticsGraph:
    """
    Directed weighted graph modelling the road network.
    G = (V, E) where V = logistics hubs, E = road connections.
    """

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, List[Edge]] = defaultdict(list)

    def add_node(self, node: Node):
        self.nodes[node.node_id] = node

    def add_edge(self, source: str, target: str,
                 distance: float, traffic: float = 1.0):
        self.edges[source].append(Edge(source, target, distance, traffic))

    def add_undirected_edge(self, a: str, b: str,
                            distance: float, traffic: float = 1.0):
        self.add_edge(a, b, distance, traffic)
        self.add_edge(b, a, distance, traffic)

    def get_neighbors(self, node_id: str) -> List[Edge]:
        return self.edges.get(node_id, [])

    def euclidean(self, a_id: str, b_id: str) -> float:
        """Straight-line distance — admissible heuristic h(n) for A*."""
        a, b = self.nodes[a_id], self.nodes[b_id]
        return math.hypot(a.x - b.x, a.y - b.y)


# =============================================================================
# SECTION 3 — A* SEARCH ENGINE
# =============================================================================

class AStarOptimizer:
    """
    Core A* pathfinder extended with Fusion Potential (Φ) awareness.

    Standard evaluation:   f(n) = g(n) + h(n)
    VF-A* evaluation:      f(n) = g(n) + h(n) − Φ(n)

    Φ(n) is non-zero only when a fusion opportunity is detected at node n,
    acting as a "negative cost" that pulls the search toward consolidation.
    """

    def __init__(self, graph: LogisticsGraph):
        self.graph = graph

    # ------------------------------------------------------------------
    # Plain A* (used for initial routes and return-to-depot paths)
    # ------------------------------------------------------------------
    def find_route(self, start: str, goal: str,
                   fusion_bonus: Dict[str, float] = None) -> Tuple[List[str], float]:
        """
        Returns (path, cost) from start to goal.
        fusion_bonus: optional dict {node_id: Φ value} for VF-A* mode.
        """
        if fusion_bonus is None:
            fusion_bonus = {}

        # Priority queue entries: (f_score, g_score, current_node, path)
        open_heap: List[Tuple[float, float, str, List[str]]] = []
        heapq.heappush(open_heap, (0.0, 0.0, start, [start]))

        visited: Dict[str, float] = {}   # node_id → best g_score seen

        while open_heap:
            f, g, current, path = heapq.heappop(open_heap)

            if current == goal:
                return path, g

            if current in visited and visited[current] <= g:
                continue
            visited[current] = g

            for edge in self.graph.get_neighbors(current):
                new_g = g + edge.cost
                phi   = fusion_bonus.get(edge.target, 0.0)
                h     = self.graph.euclidean(edge.target, goal)
                new_f = new_g + h - phi            # VF-A* formula

                if edge.target not in visited or visited.get(edge.target, math.inf) > new_g:
                    heapq.heappush(open_heap,
                                   (new_f, new_g, edge.target, path + [edge.target]))

        return [], math.inf   # No path found

    # ------------------------------------------------------------------
    # Fusion-aware route (slightly prefers nodes where fusion is possible)
    # ------------------------------------------------------------------
    def find_route_with_fusion(self, vehicle: Vehicle,
                               all_vehicles: List[Vehicle],
                               time_window: float = 5.0) -> Tuple[List[str], float]:
        """
        Computes A* route for `vehicle` biased toward nodes where another
        vehicle's projected path intersects (fusion opportunity).
        """
        fusion_bonus: Dict[str, float] = {}

        # Build a set of all nodes covered by other active vehicles' routes
        other_nodes: Dict[str, List[str]] = {}   # node → [vehicle_ids]
        for v in all_vehicles:
            if v.vehicle_id != vehicle.vehicle_id and v.active and v.route:
                for n in v.route:
                    other_nodes.setdefault(n, []).append(v.vehicle_id)

        # Assign Φ only to fusion-type nodes where another vehicle passes
        for node_id, vids in other_nodes.items():
            node = self.graph.nodes.get(node_id)
            if node and node.node_type == NODE_FUSION:
                # Φ increases if a feasible load transfer is possible
                for vid in vids:
                    other = next((v for v in all_vehicles
                                  if v.vehicle_id == vid), None)
                    if other and vehicle.can_absorb(other):
                        fusion_bonus[node_id] = fusion_bonus.get(node_id, 0) + 2.0

        return self.find_route(vehicle.start_node,
                               vehicle.destination,
                               fusion_bonus)

    # ------------------------------------------------------------------
    # Detect shared fusion nodes between two vehicle routes
    # ------------------------------------------------------------------
    @staticmethod
    def find_intersection(route_a: List[str], route_b: List[str],
                          graph: LogisticsGraph) -> Optional[str]:
        """
        Returns the first shared fusion-type node in both routes, or None.
        Earlier nodes in route_a take priority.
        """
        set_b = set(route_b)
        for node_id in route_a:
            if node_id in set_b:
                node = graph.nodes.get(node_id)
                if node and node.node_type == NODE_FUSION:
                    return node_id
        return None


# =============================================================================
# SECTION 4 — FUSION ENGINE
# =============================================================================

class FusionEngine:
    """
    Manages the multi-vehicle fusion lifecycle:
      • Detect fusion opportunities between active vehicle pairs.
      • Validate volumetric constraint before triggering a merge.
      • Transfer packages, retire the giving vehicle, reroute if needed.
    """

    def __init__(self, graph: LogisticsGraph, optimizer: AStarOptimizer):
        self.graph     = graph
        self.optimizer = optimizer
        self.fusion_log: List[dict] = []   # audit trail for visualisation

    def run(self, vehicles: List[Vehicle]) -> List[Vehicle]:
        """
        Main loop: iteratively detect and execute fusions until no more
        are possible.  Returns the updated vehicle list.
        """
        changed = True
        while changed:
            changed = False
            active  = [v for v in vehicles if v.active]

            for i, va in enumerate(active):
                for vb in active[i + 1:]:
                    fusion_node = AStarOptimizer.find_intersection(
                        va.route, vb.route, self.graph
                    )
                    if fusion_node is None:
                        continue

                    # Determine receiver (higher remaining capacity gets cargo)
                    receiver = va if va.max_capacity >= vb.max_capacity else vb
                    giver    = vb if receiver is va else va

                    if receiver.can_absorb(giver):
                        self._execute_fusion(receiver, giver,
                                             fusion_node, vehicles)
                        changed = True
                        break  # restart scan after a fusion
                if changed:
                    break

        return vehicles

    def _execute_fusion(self, receiver: Vehicle, giver: Vehicle,
                        fusion_node: str, all_vehicles: List[Vehicle]):
        """
        Transfers all packages from giver → receiver at the fusion node.
        Giver is retired and assigned a direct depot return route.
        Receiver's route is updated from the fusion node onward.
        """
        load_before_receiver = receiver.utilization
        load_before_giver    = giver.utilization

        # Transfer cargo
        receiver.packages.extend(giver.packages)
        giver.packages.clear()

        # Retire giver — find nearest depot for return path
        depot_id = self._nearest_depot(fusion_node)
        ret_route, _ = self.optimizer.find_route(fusion_node, depot_id)
        giver.route  = ret_route
        giver.active = False   # removed from active delivery queue

        # Receiver continues from fusion node to its original destination
        new_route, _ = self.optimizer.find_route(fusion_node,
                                                 receiver.destination)
        # Prepend the path up to (and including) the fusion node
        prefix = []
        for n in receiver.route:
            prefix.append(n)
            if n == fusion_node:
                break
        receiver.route = prefix + new_route[1:]  # avoid duplicating fusion node

        self.fusion_log.append({
            "receiver"          : receiver.vehicle_id,
            "giver"             : giver.vehicle_id,
            "fusion_node"       : fusion_node,
            "packages_transferred": len(receiver.packages) - (
                len(receiver.packages) - len(giver.packages) - len(receiver.packages)
            ),
            "util_receiver_before": load_before_receiver,
            "util_receiver_after" : receiver.utilization,
            "util_giver_before"   : load_before_giver,
        })

    def _nearest_depot(self, from_node: str) -> str:
        depots = [n.node_id for n in self.graph.nodes.values()
                  if n.node_type == NODE_DEPOT]
        if not depots:
            return from_node
        return min(depots, key=lambda d: self.graph.euclidean(from_node, d))


# =============================================================================
# SECTION 5 — SYNTHETIC DATASET (Pune Industrial Corridor)
# =============================================================================

def build_pune_network() -> LogisticsGraph:
    """
    Constructs a synthetic logistics network inspired by the Hinjewadi–
    Chakan–Talegaon industrial corridor around Pune, Maharashtra.

    Node layout (approximate coordinates in a normalised plane):
      Depots    : 2  (Hinjewadi, Chakan)
      Pickup    : 5  (warehouses / distribution centres)
      Fusion    : 6  (highway junctions / toll plazas)
      Delivery  : 12 (customer sites)
    """
    G = LogisticsGraph()

    # ── Depots ──────────────────────────────────────────────────────────────
    G.add_node(Node("D1", NODE_DEPOT,    1.0,  5.0))   # Hinjewadi Depot
    G.add_node(Node("D2", NODE_DEPOT,    9.0,  5.0))   # Chakan Depot

    # ── Pickup / Distribution centres ────────────────────────────────────────
    G.add_node(Node("P1", NODE_PICKUP,   2.0,  8.0))   # Hinjewadi Phase-2
    G.add_node(Node("P2", NODE_PICKUP,   2.0,  2.0))   # Hinjewadi Phase-3
    G.add_node(Node("P3", NODE_PICKUP,   8.0,  8.5))   # Chakan MIDC North
    G.add_node(Node("P4", NODE_PICKUP,   8.0,  1.5))   # Talegaon MIDC
    G.add_node(Node("P5", NODE_PICKUP,   5.0,  9.0))   # Bhosari

    # ── Fusion nodes (highway junctions) ────────────────────────────────────
    G.add_node(Node("F1", NODE_FUSION,   4.0,  7.0))   # Wakad Junction
    G.add_node(Node("F2", NODE_FUSION,   6.0,  7.0))   # Dehu Road Junction
    G.add_node(Node("F3", NODE_FUSION,   5.0,  5.0))   # Pimpri Junction
    G.add_node(Node("F4", NODE_FUSION,   4.0,  3.0))   # Ravet Junction
    G.add_node(Node("F5", NODE_FUSION,   6.0,  3.0))   # Chinchwad Junction
    G.add_node(Node("F6", NODE_FUSION,   5.0,  7.5))   # Moshi Junction

    # ── Delivery nodes (customer sites) ─────────────────────────────────────
    G.add_node(Node("DL1",  NODE_DELIVERY, 3.0,  9.0))
    G.add_node(Node("DL2",  NODE_DELIVERY, 7.0,  9.0))
    G.add_node(Node("DL3",  NODE_DELIVERY, 1.5,  6.5))
    G.add_node(Node("DL4",  NODE_DELIVERY, 2.5,  5.0))
    G.add_node(Node("DL5",  NODE_DELIVERY, 7.5,  6.5))
    G.add_node(Node("DL6",  NODE_DELIVERY, 8.5,  4.0))
    G.add_node(Node("DL7",  NODE_DELIVERY, 3.5,  1.5))
    G.add_node(Node("DL8",  NODE_DELIVERY, 6.5,  1.5))
    G.add_node(Node("DL9",  NODE_DELIVERY, 5.0,  2.0))
    G.add_node(Node("DL10", NODE_DELIVERY, 5.0,  6.0))
    G.add_node(Node("DL11", NODE_DELIVERY, 3.0,  4.5))
    G.add_node(Node("DL12", NODE_DELIVERY, 7.0,  4.5))

    # ── Road connections (undirected for simplicity) ─────────────────────────
    edges = [
        # Depots ↔ nearby hubs
        ("D1", "P1", 2.2, 1.1), ("D1", "P2", 2.2, 1.0),
        ("D1", "F1", 3.0, 1.2), ("D1", "F4", 3.2, 1.0),
        ("D2", "P3", 1.8, 1.1), ("D2", "P4", 2.5, 1.0),
        ("D2", "F2", 3.0, 1.2), ("D2", "F5", 3.2, 1.0),

        # Pickup ↔ Fusion
        ("P1", "F1", 2.1, 1.3), ("P1", "F6", 2.5, 1.1),
        ("P2", "F4", 2.0, 1.1), ("P2", "F3", 2.8, 1.2),
        ("P3", "F2", 1.9, 1.2), ("P3", "F6", 2.3, 1.1),
        ("P4", "F5", 1.8, 1.0), ("P4", "F4", 3.0, 1.3),
        ("P5", "F6", 1.5, 1.1), ("P5", "F2", 2.0, 1.0),

        # Fusion ↔ Fusion (inter-junction roads)
        ("F1", "F2", 2.0, 1.4), ("F1", "F3", 2.2, 1.2),
        ("F2", "F3", 2.0, 1.3), ("F2", "F6", 1.5, 1.1),
        ("F3", "F4", 2.2, 1.1), ("F3", "F5", 2.2, 1.0),
        ("F4", "F5", 2.0, 1.2), ("F6", "F3", 2.5, 1.2),

        # Fusion / Pickup ↔ Delivery
        ("F1", "DL1", 2.0, 1.1), ("F1", "DL3", 1.5, 1.0),
        ("F2", "DL2", 2.0, 1.1), ("F2", "DL5", 1.5, 1.2),
        ("F3", "DL4", 1.8, 1.0), ("F3", "DL10", 1.0, 1.0),
        ("F3", "DL11", 1.5, 1.1),("F3", "DL12", 1.5, 1.1),
        ("F4", "DL7", 2.0, 1.0), ("F4", "DL11", 1.2, 1.1),
        ("F5", "DL6", 1.5, 1.0), ("F5", "DL8", 2.0, 1.1),
        ("F5", "DL12", 1.2, 1.0),("F5", "DL9", 1.5, 1.2),
        ("P5", "DL2", 2.5, 1.2),
    ]
    for a, b, dist, traffic in edges:
        G.add_undirected_edge(a, b, dist, traffic)

    return G


def create_vehicles() -> List[Vehicle]:
    """
    Defines 4 delivery vehicles with deliberately low initial utilization
    so fusion events become meaningful.
    """
    vehicles = [
        Vehicle(
            vehicle_id   = "V1",
            start_node   = "P1",
            destination  = "DL5",
            max_capacity = 10.0,
            packages     = [
                Package("PKG-V1-A", 50, 1.8, "DL5"),
                Package("PKG-V1-B", 30, 1.5, "DL5"),
            ],
            color = VEHICLE_COLORS[0],
        ),
        Vehicle(
            vehicle_id   = "V2",
            start_node   = "P2",
            destination  = "DL12",
            max_capacity = 10.0,
            packages     = [
                Package("PKG-V2-A", 40, 1.6, "DL12"),
                Package("PKG-V2-B", 25, 1.2, "DL12"),
            ],
            color = VEHICLE_COLORS[1],
        ),
        Vehicle(
            vehicle_id   = "V3",
            start_node   = "P3",
            destination  = "DL10",
            max_capacity = 10.0,
            packages     = [
                Package("PKG-V3-A", 60, 2.0, "DL10"),
                Package("PKG-V3-B", 20, 1.0, "DL10"),
            ],
            color = VEHICLE_COLORS[2],
        ),
        Vehicle(
            vehicle_id   = "V4",
            start_node   = "P4",
            destination  = "DL9",
            max_capacity = 10.0,
            packages     = [
                Package("PKG-V4-A", 35, 1.4, "DL9"),
            ],
            color = VEHICLE_COLORS[3],
        ),
    ]
    return vehicles


# =============================================================================
# SECTION 6 — VISUALISATION HELPERS
# =============================================================================

def _node_positions(graph: LogisticsGraph) -> Dict[str, Tuple[float, float]]:
    return {nid: (n.x, n.y) for nid, n in graph.nodes.items()}


def _draw_graph_base(ax, graph: LogisticsGraph, pos: Dict,
                     title: str = "", alpha_edges: float = 0.4):
    """Draws the base road network (edges + nodes) on `ax`."""
    ax.set_facecolor(PALETTE["bg"])

    # Draw edges
    drawn = set()
    for src, edges in graph.edges.items():
        for e in edges:
            key = tuple(sorted([e.source, e.target]))
            if key in drawn:
                continue
            drawn.add(key)
            x0, y0 = pos[e.source]
            x1, y1 = pos[e.target]
            ax.plot([x0, x1], [y0, y1], color=PALETTE["edge"],
                    lw=1.0, alpha=alpha_edges, zorder=1)

    # Draw nodes
    node_type_order = [NODE_DEPOT, NODE_PICKUP, NODE_FUSION, NODE_DELIVERY]
    sizes = {NODE_DEPOT: 280, NODE_PICKUP: 220,
             NODE_FUSION: 180, NODE_DELIVERY: 140}

    for ntype in node_type_order:
        nodes = [n for n in graph.nodes.values() if n.node_type == ntype]
        xs = [pos[n.node_id][0] for n in nodes]
        ys = [pos[n.node_id][1] for n in nodes]
        ax.scatter(xs, ys, s=sizes[ntype], c=PALETTE[ntype],
                   zorder=3, edgecolors="white", linewidths=0.6)
        for n in nodes:
            ax.annotate(n.node_id,
                        xy=(pos[n.node_id][0], pos[n.node_id][1]),
                        xytext=(0, 6), textcoords="offset points",
                        fontsize=5.5, color=PALETTE["text"],
                        ha="center", zorder=4)

    if title:
        ax.set_title(title, color=PALETTE["text"], fontsize=9,
                     fontweight="bold", pad=6)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 10.5)
    ax.axis("off")


def _draw_route(ax, route: List[str], pos: Dict,
                color: str, lw: float = 2.0, label: str = ""):
    """Draws a route as coloured arrows on `ax`."""
    for i in range(len(route) - 1):
        x0, y0 = pos[route[i]]
        x1, y1 = pos[route[i + 1]]
        ax.annotate("",
                    xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=color, lw=lw,
                        mutation_scale=10,
                        connectionstyle="arc3,rad=0.06",
                    ), zorder=5)
    if route and label:
        sx, sy = pos[route[0]]
        ax.text(sx, sy - 0.3, label, fontsize=6, color=color,
                ha="center", fontweight="bold", zorder=6)


# =============================================================================
# SECTION 7 — FIGURE GENERATION
# =============================================================================

def fig1_network_overview(graph: LogisticsGraph, pos: Dict):
    """Figure 1: Road network with all nodes colour-coded."""
    fig, ax = plt.subplots(figsize=(8, 7),
                           facecolor=PALETTE["bg"])
    _draw_graph_base(ax, graph, pos, title="Fig 1 – Pune Industrial Corridor: Logistics Road Network")

    # Legend
    legend_items = [
        mpatches.Patch(color=PALETTE[NODE_DEPOT],    label="Depot"),
        mpatches.Patch(color=PALETTE[NODE_PICKUP],   label="Pickup / Warehouse"),
        mpatches.Patch(color=PALETTE[NODE_FUSION],   label="Fusion Node (Junction)"),
        mpatches.Patch(color=PALETTE[NODE_DELIVERY], label="Delivery Site"),
    ]
    ax.legend(handles=legend_items, loc="lower left",
              facecolor=PALETTE["panel"], edgecolor="none",
              labelcolor=PALETTE["text"], fontsize=7)

    # Node count annotation
    counts = defaultdict(int)
    for n in graph.nodes.values():
        counts[n.node_type] += 1
    info = (f"Nodes: {len(graph.nodes)}  |  "
            f"Edges: {sum(len(v) for v in graph.edges.values()) // 2}  |  "
            f"Pickups: {counts[NODE_PICKUP]}  |  "
            f"Fusions: {counts[NODE_FUSION]}  |  "
            f"Deliveries: {counts[NODE_DELIVERY]}")
    ax.text(5.25, 0.3, info, color=PALETTE["subtext"],
            fontsize=6.5, ha="center", style="italic")

    plt.tight_layout()
    return fig


def fig2_initial_routes(graph: LogisticsGraph, pos: Dict,
                        vehicles: List[Vehicle]):
    """Figure 2: Individual A* routes before any fusion."""
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=PALETTE["bg"])
    _draw_graph_base(ax, graph, pos,
                     title="Fig 2 – Initial A* Routes (Pre-Fusion)",
                     alpha_edges=0.25)

    for v in vehicles:
        _draw_route(ax, v.route, pos, v.color,
                    lw=2.5, label=f"{v.vehicle_id} ({v.utilization*100:.0f}%)")

    legend_items = [
        mpatches.Patch(color=v.color, label=f"{v.vehicle_id}: {v.start_node}→{v.destination} "
                                             f"({v.utilization*100:.0f}% loaded)")
        for v in vehicles
    ]
    ax.legend(handles=legend_items, loc="lower left",
              facecolor=PALETTE["panel"], edgecolor="none",
              labelcolor=PALETTE["text"], fontsize=7)

    plt.tight_layout()
    return fig


def fig3_fusion_event(graph: LogisticsGraph, pos: Dict,
                      vehicles: List[Vehicle],
                      fusion_log: List[dict]):
    """Figure 3: Highlights the fusion node and the two vehicles involved."""
    if not fusion_log:
        return None

    fig, axes = plt.subplots(1, min(len(fusion_log), 2),
                              figsize=(14, 6), facecolor=PALETTE["bg"])
    if len(fusion_log) == 1:
        axes = [axes]

    for idx, (ax, event) in enumerate(zip(axes, fusion_log[:2])):
        _draw_graph_base(ax, graph, pos,
                         title=f"Fig 3.{idx+1} – Fusion Event: "
                               f"{event['giver']} → {event['receiver']} "
                               f"@ {event['fusion_node']}",
                         alpha_edges=0.2)

        receiver_v = next(v for v in vehicles
                          if v.vehicle_id == event["receiver"])
        giver_v    = next(v for v in vehicles
                          if v.vehicle_id == event["giver"])

        _draw_route(ax, receiver_v.route, pos, receiver_v.color, lw=2.8,
                    label=f"{receiver_v.vehicle_id} (recv)")
        _draw_route(ax, giver_v.route, pos,
                    PALETTE["retired"], lw=2.0,
                    label=f"{giver_v.vehicle_id} (ret)")

        # Highlight fusion node
        fx, fy = pos[event["fusion_node"]]
        ax.scatter([fx], [fy], s=500, c=PALETTE[NODE_FUSION],
                   zorder=6, edgecolors="white", linewidths=1.5)
        ax.annotate(f"⚡ FUSION\n{event['fusion_node']}",
                    xy=(fx, fy), xytext=(fx + 0.7, fy + 0.5),
                    fontsize=7, color="#FDE047", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="#FDE047",
                                   lw=0.8), zorder=7)

        # Utilisation text box
        info = (f"Before  → Receiver: {event['util_receiver_before']*100:.0f}%  "
                f"| Giver: {event['util_giver_before']*100:.0f}%\n"
                f"After   → Receiver: {event['util_receiver_after']*100:.0f}%  "
                f"| Giver: retired (0%)")
        ax.text(0.5, 0.02, info, transform=ax.transAxes,
                fontsize=6.5, color=PALETTE["subtext"], ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc=PALETTE["panel"],
                          ec="none", alpha=0.8))

    plt.tight_layout()
    return fig


def fig4_post_fusion_routes(graph: LogisticsGraph, pos: Dict,
                             vehicles: List[Vehicle]):
    """Figure 4: Final routes after all fusions; retired vehicles shown grey."""
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=PALETTE["bg"])
    _draw_graph_base(ax, graph, pos,
                     title="Fig 4 – Post-Fusion Routes (Optimised Fleet)",
                     alpha_edges=0.2)

    for v in vehicles:
        if v.active:
            _draw_route(ax, v.route, pos, v.color, lw=3.0,
                        label=f"{v.vehicle_id}✓ ({v.utilization*100:.0f}%)")
        else:
            _draw_route(ax, v.route, pos, PALETTE["retired"], lw=1.5,
                        label=f"{v.vehicle_id}↩ (retired)")

    legend_items = (
        [mpatches.Patch(color=v.color,
                        label=f"{v.vehicle_id} – ACTIVE {v.utilization*100:.0f}%")
         for v in vehicles if v.active]
        +
        [mpatches.Patch(color=PALETTE["retired"],
                        label=f"{v.vehicle_id} – RETIRED (return to depot)")
         for v in vehicles if not v.active]
    )
    ax.legend(handles=legend_items, loc="lower left",
              facecolor=PALETTE["panel"], edgecolor="none",
              labelcolor=PALETTE["text"], fontsize=7)

    plt.tight_layout()
    return fig


def fig5_utilization_chart(vehicles: List[Vehicle],
                            util_before: Dict[str, float]):
    """Figure 5: Grouped bar chart – utilisation before vs after fusion."""
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    vid = [v.vehicle_id for v in vehicles]
    before = [util_before[v.vehicle_id] * 100 for v in vehicles]
    after  = [v.utilization * 100 for v in vehicles]

    x  = np.arange(len(vid))
    bw = 0.35

    bars_b = ax.bar(x - bw/2, before, bw, label="Before Fusion",
                    color="#4F46E5", alpha=0.85, zorder=3)
    bars_a = ax.bar(x + bw/2, after,  bw, label="After Fusion",
                    color="#10B981", alpha=0.85, zorder=3)

    # Value labels on bars
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{bar.get_height():.0f}%", ha="center",
                color=PALETTE["text"], fontsize=8, fontweight="bold")
    for bar, v in zip(bars_a, vehicles):
        label = f"{bar.get_height():.0f}%" + (" ↩" if not v.active else "")
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                label, ha="center",
                color=PALETTE["text"], fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(vid, color=PALETTE["text"], fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Volumetric Utilization (%)",
                  color=PALETTE["subtext"], fontsize=9)
    ax.set_title("Fig 5 – Vehicle Utilization: Before vs After Fusion",
                 color=PALETTE["text"], fontsize=10, fontweight="bold", pad=8)
    ax.tick_params(colors=PALETTE["subtext"])
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color=PALETTE["panel"], zorder=0)
    ax.legend(facecolor=PALETTE["panel"], edgecolor="none",
              labelcolor=PALETTE["text"], fontsize=8)

    # Fleet-level average improvement line
    avg_b = np.mean(before)
    avg_a = np.mean([v.utilization * 100 for v in vehicles if v.active])
    ax.axhline(avg_b, color="#6366F1", ls="--", lw=1.0, alpha=0.6)
    ax.axhline(avg_a, color="#10B981", ls="--", lw=1.0, alpha=0.6)
    ax.text(len(vid) - 0.1, avg_b + 1, f"Avg before: {avg_b:.1f}%",
            color="#6366F1", fontsize=7, ha="right")
    ax.text(len(vid) - 0.1, avg_a + 1, f"Avg after:  {avg_a:.1f}%",
            color="#10B981", fontsize=7, ha="right")

    plt.tight_layout()
    return fig


def fig6_summary_dashboard(vehicles: List[Vehicle],
                            util_before: Dict[str, float],
                            fusion_log: List[dict],
                            graph: LogisticsGraph):
    """Figure 6: High-level metrics summary dashboard."""
    fig = plt.figure(figsize=(14, 6), facecolor=PALETTE["bg"])
    gs  = gridspec.GridSpec(2, 4, figure=fig,
                            hspace=0.55, wspace=0.4)

    # ── KPI tiles ──────────────────────────────────────────────────────────
    active_vehicles = [v for v in vehicles if v.active]
    retired         = [v for v in vehicles if not v.active]
    avg_util_before = np.mean([util_before[v.vehicle_id] for v in vehicles]) * 100
    avg_util_after  = np.mean([v.utilization for v in active_vehicles]) * 100 if active_vehicles else 0
    improvement     = avg_util_after - avg_util_before
    fleet_reduction = len(retired) / len(vehicles) * 100

    kpis = [
        ("Total Vehicles", str(len(vehicles)), VEHICLE_COLORS[0]),
        ("Active After Fusion", str(len(active_vehicles)), "#10B981"),
        ("Avg Util Before", f"{avg_util_before:.1f}%", "#4F46E5"),
        ("Avg Util After",  f"{avg_util_after:.1f}%",  "#10B981"),
        ("Utilization Gain", f"+{improvement:.1f}%",   "#F59E0B"),
        ("Fleet Reduction",  f"{fleet_reduction:.0f}%", "#EF4444"),
        ("Fusion Events",    str(len(fusion_log)),       "#A855F7"),
        ("Nodes in Network", str(len(graph.nodes)),      PALETTE["subtext"]),
    ]

    for idx, (title, value, color) in enumerate(kpis):
        row, col = divmod(idx, 4)
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor(PALETTE["panel"])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

        ax.text(0.5, 0.65, value, transform=ax.transAxes,
                fontsize=20, fontweight="bold", color=color,
                ha="center", va="center")
        ax.text(0.5, 0.22, title, transform=ax.transAxes,
                fontsize=7.5, color=PALETTE["subtext"],
                ha="center", va="center")

        # Decorative top-border strip
        ax.add_patch(FancyBboxPatch((0, 0.92), 1, 0.08,
                                    boxstyle="square,pad=0",
                                    facecolor=color, transform=ax.transAxes,
                                    clip_on=False, zorder=5))

    fig.suptitle("Fig 6 – VF-A* Logistics Optimisation: Summary Dashboard",
                 color=PALETTE["text"], fontsize=12,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# SECTION 8 — MAIN PIPELINE
# =============================================================================

def run_vf_astar_pipeline():
    """
    Executes the complete VF-A* logistics optimisation pipeline and
    generates all visualisation figures.
    """
    print("=" * 65)
    print("  Volumetric Fusion A* (VF-A*) Logistics Optimisation")
    print("  Sarvesh Naik | Vishwakarma University, Pune")
    print("=" * 65)

    # ── Step 1: Build network ────────────────────────────────────────────
    print("\n[1/6] Building Pune Industrial Corridor road network...")
    graph = build_pune_network()
    pos   = _node_positions(graph)
    print(f"      Nodes: {len(graph.nodes)} | "
          f"Edges: {sum(len(v) for v in graph.edges.values()) // 2}")

    # ── Step 2: Initialise vehicles ──────────────────────────────────────
    print("[2/6] Initialising delivery vehicles and packages...")
    vehicles = create_vehicles()
    optimizer = AStarOptimizer(graph)

    # Record utilisation snapshot before fusion
    util_before: Dict[str, float] = {}
    for v in vehicles:
        util_before[v.vehicle_id] = v.utilization
        print(f"      {v.vehicle_id}: {v.start_node} → {v.destination} | "
              f"Load: {v.current_load:.1f}/{v.max_capacity:.1f} m³ "
              f"({v.utilization*100:.0f}%)")

    # ── Step 3: Compute initial A* routes ────────────────────────────────
    print("[3/6] Computing initial A* routes for all vehicles...")
    for v in vehicles:
        v.route, cost = optimizer.find_route(v.start_node, v.destination)
        print(f"      {v.vehicle_id}: {' → '.join(v.route)}  "
              f"(cost: {cost:.2f})")

    # ── Step 4: Run fusion engine ─────────────────────────────────────────
    print("[4/6] Running Volumetric Fusion Engine...")
    engine  = FusionEngine(graph, optimizer)
    vehicles = engine.run(vehicles)

    if engine.fusion_log:
        for event in engine.fusion_log:
            print(f"      ⚡ FUSION: {event['giver']} → {event['receiver']} "
                  f"@ node {event['fusion_node']} | "
                  f"Receiver util: "
                  f"{event['util_receiver_before']*100:.0f}% → "
                  f"{event['util_receiver_after']*100:.0f}%")
    else:
        print("      No fusion events occurred with this configuration.")

    # ── Step 5: Print post-fusion status ─────────────────────────────────
    print("[5/6] Post-fusion vehicle status:")
    for v in vehicles:
        status = "ACTIVE  " if v.active else "RETIRED "
        print(f"      {status} {v.vehicle_id}: "
              f"Load: {v.current_load:.1f}/{v.max_capacity:.1f} m³ "
              f"({v.utilization*100:.0f}%) | "
              f"Route: {' → '.join(v.route)}")

    # ── Step 6: Generate all figures ─────────────────────────────────────
    print("[6/6] Generating visualisations...")
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "savefig.facecolor": PALETTE["bg"],
        "font.family": "DejaVu Sans",
    })

    figures = {}
    figures["fig1"] = fig1_network_overview(graph, pos)
    figures["fig2"] = fig2_initial_routes(graph, pos, vehicles)
    figures["fig3"] = fig3_fusion_event(graph, pos, vehicles, engine.fusion_log)
    figures["fig4"] = fig4_post_fusion_routes(graph, pos, vehicles)
    figures["fig5"] = fig5_utilization_chart(vehicles, util_before)
    figures["fig6"] = fig6_summary_dashboard(vehicles, util_before,
                                             engine.fusion_log, graph)

    # Save figures
    import os
    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    saved = []
    for name, fig in figures.items():
        if fig is not None:
            path = f"{out_dir}/vfastar_{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            saved.append(path)
            print(f"      Saved: {path}")
            plt.close(fig)

    # Also save combined figure
    print("\n  Generating combined report figure...")
    fig_combined, axes = plt.subplots(2, 3, figsize=(20, 12),
                                       facecolor=PALETTE["bg"])
    fig_combined.suptitle(
        "VF-A* Logistics Optimisation — Complete Visualisation Report\n"
        "Sarvesh Naik | B.Tech CSE | Vishwakarma University, Pune | 2025-26",
        color=PALETTE["text"], fontsize=13, fontweight="bold", y=0.98
    )

    # Re-draw each panel into the combined figure
    titles_fns = [
        ("Network Overview",   lambda ax: _draw_mini_network(ax, graph, pos, vehicles, "network")),
        ("Initial Routes",     lambda ax: _draw_mini_network(ax, graph, pos, vehicles, "initial")),
        ("Fusion Events",      lambda ax: _draw_mini_fusion(ax, graph, pos, vehicles, engine.fusion_log)),
        ("Post-Fusion Routes", lambda ax: _draw_mini_network(ax, graph, pos, vehicles, "post")),
        ("Utilization Chart",  lambda ax: _draw_mini_util(ax, vehicles, util_before)),
        ("Summary Dashboard",  lambda ax: _draw_mini_summary(ax, vehicles, util_before,
                                                              engine.fusion_log, graph)),
    ]
    for ax, (title, fn) in zip(axes.flat, titles_fns):
        fn(ax)

    fig_combined.tight_layout(rect=[0, 0, 1, 0.96])
    combined_path = f"{out_dir}/vfastar_combined_report.png"
    fig_combined.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig_combined)
    saved.insert(0, combined_path)
    print(f"      Saved: {combined_path}")

    print("\n" + "=" * 65)
    print("  Pipeline complete.")
    print(f"  Fusion events  : {len(engine.fusion_log)}")
    print(f"  Active vehicles: {sum(1 for v in vehicles if v.active)}/{len(vehicles)}")
    avg_after = np.mean([v.utilization for v in vehicles if v.active]) * 100
    print(f"  Avg utilisation: {np.mean(list(util_before.values()))*100:.1f}% → {avg_after:.1f}%")
    print("=" * 65)

    return vehicles, engine.fusion_log, saved


# =============================================================================
# MINI-PANEL HELPERS (for combined figure)
# =============================================================================

def _draw_mini_network(ax, graph, pos, vehicles, mode):
    _draw_graph_base(ax, graph, pos,
                     title={"network": "Road Network",
                             "initial": "Initial Routes (A*)",
                             "post": "Post-Fusion Routes"}[mode],
                     alpha_edges=0.35)
    if mode == "initial":
        for v in vehicles:
            _draw_route(ax, v.route, pos, v.color, lw=2.0)
    elif mode == "post":
        for v in vehicles:
            clr = v.color if v.active else PALETTE["retired"]
            _draw_route(ax, v.route, pos, clr, lw=2.0)
    # legend
    if mode != "network":
        items = [mpatches.Patch(color=v.color if v.active else PALETTE["retired"],
                                label=f"{v.vehicle_id}{'✓' if v.active else '↩'}")
                 for v in vehicles]
        ax.legend(handles=items, loc="lower left", fontsize=5.5,
                  facecolor=PALETTE["panel"], edgecolor="none",
                  labelcolor=PALETTE["text"])


def _draw_mini_fusion(ax, graph, pos, vehicles, fusion_log):
    _draw_graph_base(ax, graph, pos, title="Fusion Events", alpha_edges=0.2)
    for event in fusion_log:
        receiver_v = next(v for v in vehicles if v.vehicle_id == event["receiver"])
        giver_v    = next(v for v in vehicles if v.vehicle_id == event["giver"])
        _draw_route(ax, receiver_v.route, pos, receiver_v.color, lw=2.5)
        _draw_route(ax, giver_v.route,    pos, PALETTE["retired"], lw=1.5)
        fx, fy = pos[event["fusion_node"]]
        ax.scatter([fx], [fy], s=350, c=PALETTE[NODE_FUSION],
                   zorder=6, edgecolors="white", linewidths=1.0)
        ax.annotate("⚡", xy=(fx, fy), fontsize=9, ha="center",
                    va="center", zorder=7, color="#FDE047")
    if not fusion_log:
        ax.text(0.5, 0.5, "No fusion events", transform=ax.transAxes,
                color=PALETTE["subtext"], ha="center", va="center", fontsize=9)


def _draw_mini_util(ax, vehicles, util_before):
    ax.set_facecolor(PALETTE["bg"])
    vid    = [v.vehicle_id for v in vehicles]
    before = [util_before[v.vehicle_id] * 100 for v in vehicles]
    after  = [v.utilization * 100 for v in vehicles]
    x      = np.arange(len(vid))
    bw     = 0.35
    ax.bar(x - bw/2, before, bw, color="#4F46E5", alpha=0.85, label="Before")
    ax.bar(x + bw/2, after,  bw, color="#10B981", alpha=0.85, label="After")
    ax.set_xticks(x); ax.set_xticklabels(vid, color=PALETTE["text"], fontsize=8)
    ax.set_ylim(0, 120)
    ax.set_title("Utilization Before vs After", color=PALETTE["text"],
                 fontsize=8, fontweight="bold")
    ax.tick_params(colors=PALETTE["subtext"])
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color=PALETTE["panel"])
    ax.legend(facecolor=PALETTE["panel"], edgecolor="none",
              labelcolor=PALETTE["text"], fontsize=7)


def _draw_mini_summary(ax, vehicles, util_before, fusion_log, graph):
    ax.set_facecolor(PALETTE["bg"]); ax.axis("off")
    active   = [v for v in vehicles if v.active]
    retired  = [v for v in vehicles if not v.active]
    avg_b    = np.mean([util_before[v.vehicle_id] for v in vehicles]) * 100
    avg_a    = np.mean([v.utilization for v in active]) * 100 if active else 0

    lines = [
        ("Total vehicles",          str(len(vehicles)),        VEHICLE_COLORS[0]),
        ("Active after fusion",      str(len(active)),          "#10B981"),
        ("Retired vehicles",         str(len(retired)),         "#EF4444"),
        ("Avg util before",          f"{avg_b:.1f}%",           "#4F46E5"),
        ("Avg util after",           f"{avg_a:.1f}%",           "#10B981"),
        ("Utilization improvement",  f"+{avg_a - avg_b:.1f}%",  "#F59E0B"),
        ("Fusion events",            str(len(fusion_log)),       "#A855F7"),
        ("Road network nodes",       str(len(graph.nodes)),      PALETTE["subtext"]),
    ]
    ax.set_title("Summary Metrics", color=PALETTE["text"],
                 fontsize=8, fontweight="bold")
    for i, (label, value, color) in enumerate(lines):
        y = 0.92 - i * 0.12
        ax.text(0.05, y, label + ":", transform=ax.transAxes,
                color=PALETTE["subtext"], fontsize=7.5, va="center")
        ax.text(0.75, y, value, transform=ax.transAxes,
                color=color, fontsize=8.5, fontweight="bold", va="center")


# =============================================================================
# INTERACTIVE SIMULATOR UI LAYER
# =============================================================================

class SimulationSnapshot:
    """Immutable snapshot of simulator state used for Back/Forward history."""
    def __init__(self, vehicles, position_indices, selected_vehicle_id,
                 fusion_log, step_counter, last_action):
        self.vehicles = copy.deepcopy(vehicles)
        self.position_indices = dict(position_indices)
        self.selected_vehicle_id = selected_vehicle_id
        self.fusion_log = copy.deepcopy(fusion_log)
        self.step_counter = step_counter
        self.last_action = last_action


class LogisticsSimulator:
    """Manages the step-by-step logistics simulation state for the UI."""

    def __init__(self):
        self.graph = build_pune_network()
        self.vehicles = create_vehicles()
        self.optimizer = AStarOptimizer(self.graph)
        self.fusion_engine = FusionEngine(self.graph, self.optimizer)
        self.position_indices = {v.vehicle_id: 0 for v in self.vehicles}
        self.selected_vehicle_id = self.vehicles[0].vehicle_id if self.vehicles else None
        self.last_action = "Initialized simulator"
        self.step_counter = 0
        self.history = []
        self.history_index = -1

        self._prepare_initial_routes()
        self._save_snapshot(self.last_action)

    def _prepare_initial_routes(self):
        for vehicle in self.vehicles:
            vehicle.route, _ = self.optimizer.find_route(vehicle.start_node,
                                                         vehicle.destination)

    def _save_snapshot(self, action: str):
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]

        snapshot = SimulationSnapshot(
            vehicles=self.vehicles,
            position_indices=self.position_indices,
            selected_vehicle_id=self.selected_vehicle_id,
            fusion_log=self.fusion_engine.fusion_log,
            step_counter=self.step_counter,
            last_action=action,
        )
        self.history.append(snapshot)
        self.history_index += 1

    def _restore_snapshot(self, snapshot: SimulationSnapshot):
        self.vehicles = copy.deepcopy(snapshot.vehicles)
        self.position_indices = dict(snapshot.position_indices)
        self.selected_vehicle_id = snapshot.selected_vehicle_id
        self.fusion_engine.fusion_log = copy.deepcopy(snapshot.fusion_log)
        self.step_counter = snapshot.step_counter
        self.last_action = snapshot.last_action

    def get_vehicle(self, vehicle_id: str) -> Optional[Vehicle]:
        return next((v for v in self.vehicles if v.vehicle_id == vehicle_id), None)

    def get_current_node(self, vehicle_id: str) -> str:
        vehicle = self.get_vehicle(vehicle_id)
        if not vehicle:
            return ""
        index = self.position_indices.get(vehicle_id, 0)
        return vehicle.route[min(index, len(vehicle.route) - 1)] if vehicle.route else vehicle.start_node

    def select_vehicle(self, vehicle_id: str):
        if vehicle_id not in self.position_indices:
            return {"status": "error", "message": "Unknown vehicle."}
        self.selected_vehicle_id = vehicle_id
        self.last_action = f"Selected vehicle {vehicle_id}."
        self._save_snapshot(self.last_action)
        return {"status": "ok", "message": self.last_action}

    def _find_route_index(self, vehicle: Vehicle, node_id: str) -> int:
        for idx, node in enumerate(vehicle.route):
            if node == node_id:
                return idx
        return min(self.position_indices.get(vehicle.vehicle_id, 0), len(vehicle.route) - 1)

    def _execute_fusion_if_possible(self, moved_vehicle: Vehicle, fusion_node: str) -> bool:
        if self.graph.nodes[fusion_node].node_type != NODE_FUSION:
            return False

        for other in self.vehicles:
            if other.vehicle_id == moved_vehicle.vehicle_id or not other.active:
                continue
            if self.get_current_node(other.vehicle_id) != fusion_node:
                continue

            receiver = moved_vehicle if moved_vehicle.max_capacity >= other.max_capacity else other
            giver = other if receiver is moved_vehicle else moved_vehicle

            if receiver.can_absorb(giver):
                self.fusion_engine._execute_fusion(receiver, giver, fusion_node, self.vehicles)
                self.position_indices[receiver.vehicle_id] = self._find_route_index(receiver, fusion_node)
                self.position_indices[giver.vehicle_id] = self._find_route_index(giver, fusion_node)
                return True

        return False

    def _can_move(self, vehicle: Vehicle) -> bool:
        current_index = self.position_indices.get(vehicle.vehicle_id, 0)
        return current_index < len(vehicle.route) - 1

    def next_step(self):
        if not self.selected_vehicle_id:
            return {"status": "error", "message": "No vehicle selected."}

        vehicle = self.get_vehicle(self.selected_vehicle_id)
        if not vehicle:
            return {"status": "error", "message": "Selected vehicle not found."}

        if not self._can_move(vehicle):
            self.last_action = f"{vehicle.vehicle_id} cannot move further; route complete."
            self._save_snapshot(self.last_action)
            return {"status": "ok", "message": self.last_action}

        self.position_indices[vehicle.vehicle_id] += 1
        next_node = self.get_current_node(vehicle.vehicle_id)
        self.last_action = f"{vehicle.vehicle_id} moved to {next_node}."

        fused = self._execute_fusion_if_possible(vehicle, next_node)
        if fused:
            self.last_action += " Fusion occurred here."

        self.step_counter += 1
        self._save_snapshot(self.last_action)
        return {"status": "ok", "message": self.last_action}

    def back_step(self):
        if self.history_index <= 0:
            return {"status": "error", "message": "Already at the first step."}
        self.history_index -= 1
        self._restore_snapshot(self.history[self.history_index])
        self.last_action = "Stepped back to a previous simulation state."
        return {"status": "ok", "message": self.last_action}

    def reset(self):
        self.__init__()
        return {"status": "ok", "message": "Simulator reset to initial state."}

    def get_state(self) -> dict:
        nodes = [
            {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "x": node.x,
                "y": node.y,
            }
            for node in self.graph.nodes.values()
        ]

        seen_edges = set()
        edges = []
        for source, edge_list in self.graph.edges.items():
            for edge in edge_list:
                key = tuple(sorted([edge.source, edge.target]))
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                edges.append({
                    "source": edge.source,
                    "target": edge.target,
                    "distance": edge.distance,
                    "traffic": edge.traffic,
                })

        vehicles = []
        for vehicle in self.vehicles:
            current_node = self.get_current_node(vehicle.vehicle_id)
            vehicles.append({
                "vehicle_id": vehicle.vehicle_id,
                "start_node": vehicle.start_node,
                "destination": vehicle.destination,
                "max_capacity": vehicle.max_capacity,
                "current_load": vehicle.current_load,
                "utilization": vehicle.utilization,
                "active": vehicle.active,
                "route": vehicle.route,
                "current_node": current_node,
                "color": vehicle.color,
                "packages": [
                    {
                        "pkg_id": pkg.pkg_id,
                        "weight": pkg.weight,
                        "volume": pkg.volume,
                        "destination": pkg.destination,
                    }
                    for pkg in vehicle.packages
                ],
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "vehicles": vehicles,
            "selected_vehicle_id": self.selected_vehicle_id,
            "step_counter": self.step_counter,
            "last_action": self.last_action,
            "fusion_log": self.fusion_engine.fusion_log,
        }


HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>VF-A* Logistics Interactive Simulator</title>
    <style>
        body { margin: 0; font-family: Arial, sans-serif; display: flex; min-height: 100vh; background: #020617; color: #F8FAFC; }
        #canvas-container { flex: 1; display: flex; align-items: center; justify-content: center; padding: 12px; }
        #networkCanvas { background: #0F172A; border: 1px solid #334155; border-radius: 10px; }
        .sidebar { width: 360px; padding: 18px; box-sizing: border-box; background: #111827; overflow-y: auto; }
        .panel { background: #1E293B; border: 1px solid #334155; border-radius: 12px; padding: 14px; margin-bottom: 16px; }
        .panel h2 { margin-top: 0; font-size: 1rem; color: #E2E8F0; }
        button { cursor: pointer; border: none; border-radius: 8px; padding: 10px 14px; margin-right: 8px; margin-top: 8px; background: #2563EB; color: #F8FAFC; font-weight: bold; }
        button.secondary { background: #0EA5E9; }
        button.disabled { opacity: 0.45; cursor: default; }
        .vehicle-item { display: block; padding: 10px 12px; border-radius: 10px; margin-bottom: 8px; background: #0F172A; border: 1px solid #334155; cursor: pointer; }
        .vehicle-item.selected { border-color: #22C55E; background: #132F3C; }
        .vehicle-item span.small { display: block; color: #94A3B8; font-size: 0.85rem; }
        #fusionLog { max-height: 220px; overflow-y: auto; font-size: 0.9rem; line-height: 1.4; }
        .legend-row { display: flex; align-items: center; margin-bottom: 6px; }
        .legend-color { width: 14px; height: 14px; display: inline-block; margin-right: 8px; border-radius: 3px; }
    </style>
</head>
<body>
    <div id="canvas-container">
        <canvas id="networkCanvas" width="740" height="740"></canvas>
    </div>
    <div class="sidebar">
        <div class="panel">
            <h2>VF-A* Logistics Simulator</h2>
            <div><strong>Step:</strong> <span id="stepCounter">0</span></div>
            <div><strong>Last action:</strong> <span id="lastAction">Initializing...</span></div>
        </div>
        <div class="panel">
            <h2>Truck Selection</h2>
            <div id="truckButtons"></div>
        </div>
        <div class="panel">
            <h2>Controls</h2>
            <button id="nextButton">Next</button>
            <button id="backButton" class="secondary">Back</button>
            <button id="resetButton" class="secondary">Reset</button>
        </div>
        <div class="panel">
            <h2>Selected Truck</h2>
            <div id="selectedInfo">Choose a truck or click its marker.</div>
        </div>
        <div class="panel">
            <h2>Fusion Log</h2>
            <div id="fusionLog">No fusion events yet.</div>
        </div>
        <div class="panel">
            <h2>Legend</h2>
            <div class="legend-row"><span class="legend-color" style="background:#EF4444"></span>Depot</div>
            <div class="legend-row"><span class="legend-color" style="background:#3B82F6"></span>Pickup</div>
            <div class="legend-row"><span class="legend-color" style="background:#EAB308"></span>Fusion Node</div>
            <div class="legend-row"><span class="legend-color" style="background:#22C55E"></span>Delivery</div>
            <div class="legend-row"><span class="legend-color" style="width: 20px; background: #6366F1"></span>Vehicle route</div>
        </div>
    </div>
    <script>
        let state = null;
        let markerMap = [];
        const canvas = document.getElementById('networkCanvas');
        const ctx = canvas.getContext('2d');
        const margin = 40;
        const scale = 64;
        const canvasSize = canvas.width;

        function mapPosition(node) {
            return {
                x: margin + node.x * scale,
                y: canvasSize - margin - node.y * scale,
            };
        }

        async function fetchState() {
            const response = await fetch('/state');
            state = await response.json();
            renderState();
        }

        async function sendAction(action, vehicleId = '') {
            const data = new URLSearchParams();
            data.append('action', action);
            if (vehicleId) data.append('vehicle_id', vehicleId);
            const response = await fetch('/action', {
                method: 'POST',
                body: data,
            });
            const result = await response.json();
            if (result.status === 'ok') {
                fetchState();
            } else {
                alert(result.message || 'Action failed');
            }
        }

        function renderState() {
            document.getElementById('stepCounter').textContent = state.step_counter;
            document.getElementById('lastAction').textContent = state.last_action;
            drawCanvas();
            populateTruckButtons();
            populateSelectedInfo();
            populateFusionLog();
        }

        function drawCanvas() {
            ctx.clearRect(0, 0, canvasSize, canvasSize);
            ctx.fillStyle = '#0F172A';
            ctx.fillRect(0, 0, canvasSize, canvasSize);

            const nodePositions = {};
            for (const node of state.nodes) {
                nodePositions[node.node_id] = mapPosition(node);
            }

            const drawnEdges = new Set();
            ctx.lineWidth = 1.1;
            ctx.strokeStyle = '#94A3B8';
            for (const edge of state.edges) {
                const key = [edge.source, edge.target].sort().join('-');
                if (drawnEdges.has(key)) continue;
                drawnEdges.add(key);
                const a = nodePositions[edge.source];
                const b = nodePositions[edge.target];
                ctx.beginPath();
                ctx.moveTo(a.x, a.y);
                ctx.lineTo(b.x, b.y);
                ctx.stroke();
            }

            for (const vehicle of state.vehicles) {
                if (!vehicle.route || vehicle.route.length < 2) continue;
                ctx.strokeStyle = vehicle.active ? vehicle.color : '#6B7280';
                ctx.lineWidth = vehicle.active ? 2.2 : 1.2;
                ctx.setLineDash(vehicle.active ? [] : [6, 4]);
                ctx.beginPath();
                for (let idx = 0; idx < vehicle.route.length; idx++) {
                    const nodeId = vehicle.route[idx];
                    const pos = nodePositions[nodeId];
                    if (idx === 0) ctx.moveTo(pos.x, pos.y);
                    else ctx.lineTo(pos.x, pos.y);
                }
                ctx.stroke();
            }
            ctx.setLineDash([]);

            markerMap = [];
            const collisions = {};
            state.vehicles.forEach(vehicle => {
                const nodeId = vehicle.current_node;
                collisions[nodeId] = collisions[nodeId] || [];
                collisions[nodeId].push(vehicle.vehicle_id);
            });

            for (const vehicle of state.vehicles) {
                const nodeId = vehicle.current_node;
                const pos = nodePositions[nodeId];
                const group = collisions[nodeId];
                const offset = group.indexOf(vehicle.vehicle_id) * 22 - ((group.length - 1) * 11);
                const markerX = pos.x + offset;
                const markerY = pos.y - 24;
                const radius = 10;

                ctx.beginPath();
                ctx.arc(markerX, markerY, radius, 0, Math.PI * 2);
                ctx.fillStyle = vehicle.active ? vehicle.color : '#6B7280';
                ctx.fill();
                ctx.lineWidth = state.selected_vehicle_id === vehicle.vehicle_id ? 3 : 1.5;
                ctx.strokeStyle = state.selected_vehicle_id === vehicle.vehicle_id ? '#FACC15' : '#FFFFFF';
                ctx.stroke();

                ctx.fillStyle = '#F8FAFC';
                ctx.font = '10px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(vehicle.vehicle_id, markerX, markerY + 4);

                markerMap.push({ vehicleId: vehicle.vehicle_id, x: markerX, y: markerY, radius });
            }

            for (const node of state.nodes) {
                const pos = nodePositions[node.node_id];
                let fill = '#3B82F6';
                if (node.node_type === 'depot') fill = '#EF4444';
                if (node.node_type === 'fusion') fill = '#EAB308';
                if (node.node_type === 'delivery') fill = '#22C55E';

                ctx.beginPath();
                ctx.arc(pos.x, pos.y, 16, 0, Math.PI * 2);
                ctx.fillStyle = fill;
                ctx.fill();
                ctx.lineWidth = 2;
                const selectedVehicle = state.vehicles.find(v => v.vehicle_id === state.selected_vehicle_id);
                const selectedHere = selectedVehicle && selectedVehicle.current_node === node.node_id;
                ctx.strokeStyle = selectedHere ? '#FFFFFF' : '#0F172A';
                ctx.stroke();

                ctx.fillStyle = '#F8FAFC';
                ctx.font = '11px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(node.node_id, pos.x, pos.y + 4);
            }
        }

        function populateTruckButtons() {
            const container = document.getElementById('truckButtons');
            container.innerHTML = '';
            for (const vehicle of state.vehicles) {
                const button = document.createElement('div');
                button.className = 'vehicle-item' + (vehicle.vehicle_id === state.selected_vehicle_id ? ' selected' : '');
                button.onclick = () => sendAction('select', vehicle.vehicle_id);
                button.innerHTML = `<strong>${vehicle.vehicle_id}</strong> <span class="small">${vehicle.active ? 'ACTIVE' : 'RETIRED'} • ${vehicle.current_node}</span>`;
                container.appendChild(button);
            }
        }

        function populateSelectedInfo() {
            const info = document.getElementById('selectedInfo');
            const vehicle = state.vehicles.find(v => v.vehicle_id === state.selected_vehicle_id);
            if (!vehicle) {
                info.textContent = 'No vehicle selected.';
                return;
            }
            info.innerHTML = `<strong>${vehicle.vehicle_id}</strong><br/>
                Status: ${vehicle.active ? 'Active' : 'Retired'}<br/>
                Current node: ${vehicle.current_node}<br/>
                Destination: ${vehicle.destination}<br/>
                Load: ${vehicle.current_load.toFixed(1)} / ${vehicle.max_capacity.toFixed(1)} m³<br/>
                Utilization: ${(vehicle.utilization * 100).toFixed(0)}%<br/>
                Route progress: ${vehicle.route.indexOf(vehicle.current_node) + 1}/${vehicle.route.length}`;
        }

        function populateFusionLog() {
            const log = document.getElementById('fusionLog');
            if (!state.fusion_log.length) {
                log.innerHTML = '<div>No fusion events yet.</div>';
                return;
            }
            log.innerHTML = state.fusion_log.map(event =>
                `<div><strong>${event.giver} → ${event.receiver}</strong> @ ${event.fusion_node}<br/>
                 Receiver before: ${(event.util_receiver_before * 100).toFixed(0)}% → ${(event.util_receiver_after * 100).toFixed(0)}%</div>`
            ).join('<hr style="border-color:#334155; margin:8px 0;">');
        }

        canvas.addEventListener('click', event => {
            const rect = canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;
            for (const marker of markerMap) {
                const dx = marker.x - x;
                const dy = marker.y - y;
                if (Math.sqrt(dx * dx + dy * dy) <= marker.radius + 4) {
                    sendAction('select', marker.vehicleId);
                    return;
                }
            }
        });

        document.getElementById('nextButton').addEventListener('click', () => sendAction('next'));
        document.getElementById('backButton').addEventListener('click', () => sendAction('back'));
        document.getElementById('resetButton').addEventListener('click', () => sendAction('reset'));

        fetchState();
    </script>
</body>
</html>
"""


class SimulatorHandler(BaseHTTPRequestHandler):
    def _send_response(self, code=200, body=b'', content_type='text/html'):
        self.send_response(code)
        self.send_header('Content-type', content_type)
        self.send_header('Content-length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self._send_response(200, HTML_PAGE.encode('utf-8'), 'text/html')
            return
        if self.path.startswith('/state'):
            body = json.dumps(self.server.simulator.get_state()).encode('utf-8')
            self._send_response(200, body, 'application/json')
            return
        self._send_response(404, b'Not found', 'text/plain')

    def do_POST(self):
        if self.path != '/action':
            self._send_response(404, b'Not found', 'text/plain')
            return
        length = int(self.headers.get('Content-Length', 0))
        payload = self.rfile.read(length).decode('utf-8')
        params = parse_qs(payload)
        action = params.get('action', [''])[0]
        vehicle_id = params.get('vehicle_id', [''])[0]
        result = {'status': 'error', 'message': 'Unknown action.'}

        if action == 'next':
            result = self.server.simulator.next_step()
        elif action == 'back':
            result = self.server.simulator.back_step()
        elif action == 'select':
            result = self.server.simulator.select_vehicle(vehicle_id)
        elif action == 'reset':
            result = self.server.simulator.reset()

        body = json.dumps(result).encode('utf-8')
        self._send_response(200, body, 'application/json')

    def log_message(self, format, *args):
        return


def main_interactive_app(host='localhost', port=8765):
    simulator = LogisticsSimulator()
    server = HTTPServer((host, port), SimulatorHandler)
    server.simulator = simulator

    def start_server():
        print(f"Starting interactive VF-A* simulator at http://{host}:{port}")
        server.serve_forever()

    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    webbrowser.open(f'http://{host}:{port}')
    try:
        server_thread.join()
    except KeyboardInterrupt:
        print("Shutting down simulator server.")
        server.shutdown()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    vehicles, fusion_log, saved_files = run_vf_astar_pipeline()