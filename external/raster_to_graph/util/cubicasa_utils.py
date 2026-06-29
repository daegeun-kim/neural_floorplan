"""Convert wall_graph.json to the training tensor format expected by the
original Raster-to-Graph training loop (engine.py / data_utils.py).

Design decisions for v1:
- Only wall_nodes are included (door/window centers added in v2).
- Pass-through nodes (connected only up+down or only left+right) are merged
  out so that all edge codes stay within 0-13 (matching the pretrained
  checkpoint's num_classes_edges=14 output head).
- All node semantic quadrant labels are set to 0 (no_type) because CubiCasa
  does not provide room-type labels in wall_graph.json.
- Coordinates are scaled to 512-canvas space matching dataset_demo.py.
- Graph adjacency slots are ordered [up, left, down, right] to match the
  edge-code bit positions used throughout the original codebase.
"""

import json
import torch
from util.graph_utils import graph_to_tensor
from util.edges_utils import get_edges_alldirections_rev

CANVAS_SIZE = 512


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def scale_coords(x, y, img_w, img_h):
    """Scale native raster coords to 512-canvas space (identical to dataset_demo.py)."""
    sf = CANVAS_SIZE / max(img_w, img_h)
    off_x = (CANVAS_SIZE - int(img_w * sf)) // 2
    off_y = (CANVAS_SIZE - int(img_h * sf)) // 2
    return int(round(x * sf + off_x)), int(round(y * sf + off_y))


# ---------------------------------------------------------------------------
# Direction helpers
# ---------------------------------------------------------------------------

_DIR_SLOT = {"up": 0, "left": 1, "down": 2, "right": 3}
_SLOT_DIR = {0: "up", 1: "left", 2: "down", 3: "right"}
_OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


def _classify_direction(from_x, from_y, to_x, to_y):
    """Return the cardinal direction from (from_x,from_y) toward (to_x,to_y)."""
    dx, dy = to_x - from_x, to_y - from_y
    if abs(dy) >= abs(dx):
        return "up" if dy < 0 else "down"
    return "left" if dx < 0 else "right"


# ---------------------------------------------------------------------------
# Stale-reference cleanup
# ---------------------------------------------------------------------------

def _purge_stale_refs(adj_by_dir, valid_nids):
    """Remove any adj_by_dir slot that points to a node not in valid_nids."""
    for dirs in adj_by_dir.values():
        for d in list(dirs.keys()):
            if dirs[d] not in valid_nids:
                del dirs[d]


def _largest_connected_component(nodes_512, adj_by_dir):
    """Keep only nodes reachable from the top-left-most node (largest BFS component).

    Disconnected nodes cause get_given_layers_random_region to loop forever because
    sampled_amount can exceed the reachable frontier size.
    """
    if len(nodes_512) <= 1:
        return nodes_512, adj_by_dir

    start = min(nodes_512, key=lambda nid: (nodes_512[nid][1], nodes_512[nid][0]))
    visited, queue = {start}, [start]
    while queue:
        nid = queue.pop()
        for neighbor in adj_by_dir.get(nid, {}).values():
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    if len(visited) == len(nodes_512):
        return nodes_512, adj_by_dir  # already connected

    nodes_512   = {nid: pos for nid, pos in nodes_512.items()   if nid in visited}
    adj_by_dir  = {nid: dirs for nid, dirs in adj_by_dir.items() if nid in visited}
    return nodes_512, adj_by_dir


# ---------------------------------------------------------------------------
# Pass-through node removal
# ---------------------------------------------------------------------------

def _remove_passthrough_nodes(nodes_512, adj_by_dir):
    """Merge out nodes that connect in exactly one axis (up+down or left+right).

    A pass-through node would produce edge code 14 ('0101', left+right) or
    15 ('1010', up+down), which the pretrained model's 14-class edge head
    cannot represent.  Merging them simplifies the graph without changing
    reachability.

    Returns updated (nodes_512, adj_by_dir) dicts (originals are not mutated).
    """
    nodes_512    = dict(nodes_512)
    adj_by_dir   = {nid: dict(dirs) for nid, dirs in adj_by_dir.items()}

    changed = True
    while changed:
        changed = False
        for nid in list(nodes_512.keys()):
            dirs = adj_by_dir.get(nid, {})
            if set(dirs.keys()) not in ({"up", "down"}, {"left", "right"}):
                continue
            # pass-through: connect the two neighbours directly, then remove nid
            dir_list = list(dirs.keys())
            a_dir, b_dir = dir_list[0], dir_list[1]
            a_nid = dirs[a_dir]
            b_nid = dirs[b_dir]

            # rewire: a's slot that pointed to nid -> b; b's slot that pointed to nid -> a
            a_dirs = adj_by_dir.get(a_nid, {})
            b_dirs = adj_by_dir.get(b_nid, {})
            a_slot_to_nid = _OPPOSITE[a_dir]  # direction from a toward nid
            b_slot_to_nid = _OPPOSITE[b_dir]

            if a_slot_to_nid in a_dirs and a_dirs[a_slot_to_nid] == nid:
                a_dirs[a_slot_to_nid] = b_nid
            if b_slot_to_nid in b_dirs and b_dirs[b_slot_to_nid] == nid:
                b_dirs[b_slot_to_nid] = a_nid

            del nodes_512[nid]
            del adj_by_dir[nid]

            # Full stale-reference sweep: some slots may not have been rewired above
            # (can happen when direction-slot collisions occurred during adjacency
            # construction, leaving unreachable back-pointers to nid).
            for other_dirs in adj_by_dir.values():
                for d in list(other_dirs.keys()):
                    if other_dirs[d] == nid:
                        del other_dirs[d]

            changed = True
            break  # restart scan after mutation

    return nodes_512, adj_by_dir


# ---------------------------------------------------------------------------
# Graph dict construction
# ---------------------------------------------------------------------------

def _build_graph_dict(nodes_512, adj_by_dir):
    """Build the 4-slot adjacency dict keyed by (x, y) integer tuples.

    Slot order matches edge-code bit positions: [up=0, left=1, down=2, right=3].
    Unused slots are (-1, -1).
    """
    graph = {}
    for nid, (nx, ny) in nodes_512.items():
        slots = [(-1, -1)] * 4
        for direction, adj_nid in adj_by_dir.get(nid, {}).items():
            ax, ay = nodes_512[adj_nid]
            slots[_DIR_SLOT[direction]] = (ax, ay)
        graph[(nx, ny)] = slots
    return graph


# ---------------------------------------------------------------------------
# BFS (quadtree) ordering
# ---------------------------------------------------------------------------

def _bfs_order(nodes_512, graph):
    """BFS from the top-left corner node, returning ordered nid list and layer_indices.

    Top-left = smallest y, then smallest x (matching make_quadtree_annot.py).
    Disconnected nodes are appended each in their own level at the end.
    """
    if not nodes_512:
        return [], []

    pos_to_nid = {pos: nid for nid, pos in nodes_512.items()}

    start_nid = min(nodes_512, key=lambda nid: (nodes_512[nid][1], nodes_512[nid][0]))
    visited   = {start_nid}
    levels    = [[start_nid]]

    while True:
        frontier = []
        for nid in levels[-1]:
            nx, ny = nodes_512[nid]
            for slot in graph[(nx, ny)]:
                if slot == (-1, -1):
                    continue
                slot_nid = pos_to_nid.get(slot)
                if slot_nid is not None and slot_nid not in visited:
                    visited.add(slot_nid)
                    frontier.append(slot_nid)
        if not frontier:
            break
        levels.append(frontier)

    # append disconnected nodes
    for nid in nodes_512:
        if nid not in visited:
            levels.append([nid])
            visited.add(nid)

    ordered_nids  = []
    layer_indices = []
    count = 0
    for level in levels:
        layer_indices.append(count)
        ordered_nids.extend(level)
        count += len(level)

    return ordered_nids, layer_indices


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def wall_graph_to_training_tensors(wall_graph_path):
    """Load wall_graph.json and return a training target dict.

    Returns None when the graph is degenerate after preprocessing (< 2 nodes).

    The returned dict is compatible with engine.train_one_epoch and the
    evaluate_iter / get_given_layers_random_region pipeline.
    """
    with open(wall_graph_path) as f:
        wg = json.load(f)

    img_w = wg["image_width"]
    img_h = wg["image_height"]

    # -- scale wall node coords to 512-canvas space --------------------------
    wall_nodes = {n["id"]: n for n in wg["nodes"] if n["type"] == "wall_node"}
    if len(wall_nodes) < 2:
        return None

    nodes_512 = {nid: scale_coords(n["x"], n["y"], img_w, img_h)
                 for nid, n in wall_nodes.items()}

    # -- build directed adjacency (wall edges only) --------------------------
    adj_by_dir = {nid: {} for nid in wall_nodes}
    for edge in wg["edges"]:
        s, e = edge["start"], edge["end"]
        if s not in wall_nodes or e not in wall_nodes:
            continue
        sx, sy = nodes_512[s]
        ex, ey = nodes_512[e]
        d_se = _classify_direction(sx, sy, ex, ey)
        d_es = _classify_direction(ex, ey, sx, sy)
        adj_by_dir[s][d_se] = e
        adj_by_dir[e][d_es] = s

    # -- remove pass-through nodes (edge codes 14/15 not in model head) ------
    nodes_512, adj_by_dir = _remove_passthrough_nodes(nodes_512, adj_by_dir)
    if len(nodes_512) < 2:
        return None

    # -- deduplicate coincident nodes (rare rounding collision) --------------
    seen_pos = {}
    dedup_nodes = {}
    for nid, pos in nodes_512.items():
        if pos not in seen_pos:
            seen_pos[pos] = nid
            dedup_nodes[nid] = pos
    nodes_512 = dedup_nodes
    _purge_stale_refs(adj_by_dir, set(nodes_512.keys()))

    # Re-run pass-through removal in case deduplication created new pass-throughs
    nodes_512, adj_by_dir = _remove_passthrough_nodes(nodes_512, adj_by_dir)
    if len(nodes_512) < 2:
        return None

    # -- keep only largest connected component (disconnected nodes cause ------
    # -- an infinite loop in get_given_layers_random_region) ------------------
    nodes_512, adj_by_dir = _largest_connected_component(nodes_512, adj_by_dir)
    if len(nodes_512) < 2:
        return None

    # -- build graph dict and BFS ordering -----------------------------------
    graph = _build_graph_dict(nodes_512, adj_by_dir)
    ordered_nids, layer_indices = _bfs_order(nodes_512, graph)
    if not ordered_nids:
        return None

    # -- assemble per-node tensors -------------------------------------------
    unnorm_pts, norm_pts, edge_codes, semantics = [], [], [], []

    for nid in ordered_nids:
        x, y = nodes_512[nid]
        unnorm_pts.append([x, y])
        norm_pts.append([x / CANVAS_SIZE, y / CANVAS_SIZE])

        # edge code: for each adjacency slot, '1' if occupied else '0'
        slots = graph[(x, y)]
        edge_str = "".join("0" if s == (-1, -1) else "1" for s in slots)
        edge_codes.append(get_edges_alldirections_rev(edge_str))
        semantics.append(0)  # no_type — no room labels available

    return {
        "unnormalized_points": torch.tensor(unnorm_pts, dtype=torch.int64),
        "points":              torch.tensor(norm_pts,   dtype=torch.float32),
        "edges":               torch.tensor(edge_codes, dtype=torch.int64),
        "semantic_left_up":    torch.zeros(len(semantics), dtype=torch.int64),
        "semantic_right_up":   torch.zeros(len(semantics), dtype=torch.int64),
        "semantic_right_down": torch.zeros(len(semantics), dtype=torch.int64),
        "semantic_left_down":  torch.zeros(len(semantics), dtype=torch.int64),
        "layer_indices":       torch.tensor(layer_indices, dtype=torch.int64),
        "graph":               graph_to_tensor(graph),
        "size":                torch.tensor([CANVAS_SIZE, CANVAS_SIZE], dtype=torch.int64),
    }
