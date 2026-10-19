"""WAD map parsing and grid navigation for a single Doom map.

The planner builds a 16-unit occupancy grid from the map's linedefs and sectors,
assumes doors can be opened, and runs A* over directed edges (step-ups above 24
units are impassable, drops are fine). Damaging floors are allowed but expensive.
"""
from __future__ import annotations

import heapq
import math
import struct
from dataclasses import dataclass, field

import numpy as np

CELL = 16
PLAYER_RADIUS = 17          # Doom player radius is 16; one unit of margin
# Solid decorations by WAD thing type -> radius. The planner used to see only walls, so routes ran
# through pillars and barrels (E1M9: a pillar beside a closed door; E1M1: the start barrel).
SOLID_THINGS = {2035: 10,                                              # barrel
                48: 16, 2028: 16, 85: 16, 86: 16,                      # tech pillar, lamps
                30: 16, 31: 16, 32: 16, 33: 16, 36: 16, 37: 16,        # green / red columns
                41: 16, 42: 16, 43: 16, 47: 16, 54: 32, 35: 16,        # eye, skull rock, trees, stalagmite, candelabra
                44: 16, 45: 16, 46: 16, 55: 16, 56: 16, 57: 16,        # torches
                25: 16, 26: 16, 27: 16, 28: 16}                        # impaled bodies
SKILL_FLAG = {1: 1, 2: 1, 3: 2, 4: 4, 5: 4}                             # WAD thing flags per skill
MAX_STEP = 24
PLAYER_HEIGHT = 56
DOOR_SPECIALS = {1, 26, 27, 28, 31, 32, 33, 34, 117, 118}
LOCKED_DOORS = {26: "blue", 32: "blue", 27: "yellow", 34: "yellow", 28: "red", 33: "red"}
EXIT_SPECIALS = {11, 51, 52, 124, 197, 198}
NORMAL_EXITS = {11, 52, 197}          # 51, 124, 198 are secret exits
DAMAGE_SECTORS = {4, 5, 7, 11, 16}
SQRT2 = math.sqrt(2)


@dataclass
class Line:
    v1: tuple[float, float]
    v2: tuple[float, float]
    flags: int
    special: int
    tag: int
    front: int          # sector index on the right side
    back: int           # sector index on the left side, -1 when one-sided


@dataclass
class MapData:
    name: str
    lines: list[Line]
    floor: np.ndarray            # per-sector floor height (static, from WAD)
    ceil: np.ndarray
    special: np.ndarray
    things: list[tuple[int, int, int, int, int]]
    door_sectors: set[int] = field(default_factory=set)
    sector_tag: np.ndarray | None = None
    lift_sectors: set[int] = field(default_factory=set)      # moving floors: step limits ignored
    floor_flat: list[str] | None = None                       # floor texture per sector (stair building follows it)
    moved_sectors: set[int] = field(default_factory=set)     # floors moved by a fired trigger, at their end height
    keys: frozenset = frozenset()                             # keys held in this world state (locked door faces)
    opened: frozenset = frozenset()                           # trigger tags fired in this world state

    @property
    def exit_lines(self) -> list[Line]:
        return [l for l in self.lines if l.special in EXIT_SPECIALS]


def load_map(wad_path: str, map_name: str) -> MapData:
    data = open(wad_path, "rb").read()
    n, off = struct.unpack("<ii", data[4:12])
    lumps = []
    for i in range(n):
        pos, size = struct.unpack("<ii", data[off + 16 * i: off + 16 * i + 8])
        lumps.append((data[off + 16 * i + 8: off + 16 * i + 16].rstrip(b"\0").decode("ascii"), pos, size))
    idx = next(i for i, l in enumerate(lumps) if l[0] == map_name)
    lump = {name: data[pos:pos + size] for name, pos, size in lumps[idx + 1: idx + 11]}

    def rows(name, fmt):
        sz = struct.calcsize(fmt)
        raw = lump[name]
        return [struct.unpack(fmt, raw[k:k + sz]) for k in range(0, len(raw), sz)]

    verts = rows("VERTEXES", "<hh")
    sides = rows("SIDEDEFS", "<hh8s8s8sh")
    secs = rows("SECTORS", "<hh8s8shhh")
    things = rows("THINGS", "<5h")
    lines = []
    for v1, v2, flags, special, tag, s1, s2 in rows("LINEDEFS", "<7h"):
        lines.append(Line(verts[v1], verts[v2], flags, special, tag,
                          sides[s1][5], sides[s2][5] if s2 != -1 else -1))
    m = MapData(map_name, lines,
                np.array([s[0] for s in secs], float), np.array([s[1] for s in secs], float),
                np.array([s[5] for s in secs], int), things, sector_tag=np.array([s[6] for s in secs], int),
                floor_flat=[s[2].rstrip(b"\0").decode("ascii", "replace") for s in secs])
    for l in lines:
        if l.special in DOOR_SPECIALS and l.back != -1:
            m.door_sectors.add(l.back)
    return m


def _seg_dist(px, py, ax, ay, bx, by):
    """Vectorised distance from points (px, py) to segment a-b."""
    dx, dy = bx - ax, by - ay
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / max(dx * dx + dy * dy, 1e-9), 0, 1)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


# Height-independent parts of a NavGrid, per map geometry (the lines list, shared by every world
# variant of a map): rebuilding a grid for live heights took 3.2 s on E1M8 and froze the player.
_GEOMETRY: dict[int, dict] = {}


class NavGrid:
    def __init__(self, m: MapData):
        self.m = m
        xs = [c for l in m.lines for c in (l.v1[0], l.v2[0])]
        ys = [c for l in m.lines for c in (l.v1[1], l.v2[1])]
        self.x0, self.y0 = min(xs) - CELL, min(ys) - CELL
        self.w = int((max(xs) - self.x0) // CELL) + 2
        self.h = int((max(ys) - self.y0) // CELL) + 2
        gx = self.x0 + (np.arange(self.w) + 0.5) * CELL
        gy = self.y0 + (np.arange(self.h) + 0.5) * CELL
        self.cx, self.cy = np.meshgrid(gx, gy)            # shape (h, w)
        geo = _GEOMETRY.get(id(m.lines))
        if geo is None or geo["lines"] is not m.lines:
            geo = _GEOMETRY[id(m.lines)] = self._geometry()
        self.sector = geo["sector"]
        floor, ceil = m.floor.copy(), m.ceil.copy()
        for s in m.door_sectors:                            # doors open to lowest neighbour ceiling - 4
            neigh = [l.front if l.back == s else l.back for l in m.lines if s in (l.front, l.back)]
            neigh = [n for n in neigh if n not in (-1, s)]
            if neigh:
                ceil[s] = min(m.ceil[n] for n in neigh) - 4
        self.floor_s, self.ceil_s = floor, ceil
        inside = self.sector >= 0
        sec = np.where(inside, self.sector, 0)
        self.floor = np.where(inside, floor[sec], np.nan)
        headroom = np.where(inside, ceil[sec] - floor[sec], 0)
        self.blocked = ~inside | (headroom < PLAYER_HEIGHT)
        for t in m.things:                                  # solid decorations present at skill 3, single player
            r = SOLID_THINGS.get(t[3])
            if r and (len(t) < 5 or (t[4] & SKILL_FLAG[3] and not t[4] & 16)):
                self.blocked |= np.hypot(self.cx - t[0], self.cy - t[1]) < r + PLAYER_RADIUS
        self.cost = np.ones((self.h, self.w))
        self.cost[inside & np.isin(m.special[sec], list(DAMAGE_SECTORS))] = 25.0
        self.blocked |= geo["solid_block"]
        self.cost += geo["solid_cost"]                      # prefer the middle of corridors
        flat = self.cost.reshape(-1)
        self.near_drop = np.zeros(self.cost.shape, bool)    # within 32 units of an edge one can fall off
        near_drop = self.near_drop.reshape(-1)
        for l, (idx, val) in zip(m.lines, geo["near"]):
            if idx is None:
                continue
            ff, bf = floor[l.front], floor[l.back]
            gap = min(ceil[l.front], ceil[l.back]) - max(ff, bf)
            if abs(ff - bf) > MAX_STEP or gap < PLAYER_HEIGHT:
                np.add.at(flat, idx, val)                   # keep off ledges
            if abs(ff - bf) > MAX_STEP and gap >= PLAYER_HEIGHT and not {l.front, l.back} & m.lift_sectors:
                near_drop[idx] = True
        self.blocked_edges = self._ledge_crossings()

    def _geometry(self) -> dict:
        solid_block = np.zeros(self.cx.shape, bool)
        solid_cost = np.zeros(self.cx.shape)
        near = []
        for l in self.m.lines:
            (ax, ay), (bx, by) = l.v1, l.v2
            d = _seg_dist(self.cx, self.cy, ax, ay, bx, by)
            if l.back == -1 or (l.flags & 1):
                solid_block |= d < PLAYER_RADIUS
                solid_cost += np.where(d < 40, (40 - d) / 8, 0)
                near.append((None, None))
            else:
                idx = np.flatnonzero(d.reshape(-1) < 32)
                near.append((idx, (32 - d.reshape(-1)[idx]) / 4))
        return {"lines": self.m.lines, "sector": self._locate_sectors(),
                "solid_block": solid_block, "solid_cost": solid_cost, "near": near}

    # -- geometry ---------------------------------------------------------
    def _locate_sectors(self) -> np.ndarray:
        """Point-in-sector by casting a ray to +x and taking the nearest crossed linedef. The ray is nudged off
        integer y: a cell centre level with a vertex tied two lines there and could pick the void side of a
        wall (E1M7: a whole row of cells read as outside the map, cutting a room in two)."""
        px, py = self.cx.ravel(), self.cy.ravel() + 1e-3
        best = np.full(px.shape, np.inf)
        sector = np.full(px.shape, -1)
        for l in self.m.lines:
            (x1, y1), (x2, y2) = l.v1, l.v2
            if y1 == y2:
                continue
            crosses = (np.minimum(y1, y2) <= py) & (py < np.maximum(y1, y2))
            xi = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            hit = crosses & (xi > px) & (xi - px < best)
            if not hit.any():
                continue
            cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
            side = np.where(cross < 0, l.front, l.back)       # right side = front
            best = np.where(hit, xi - px, best)
            sector = np.where(hit, side, sector)
        return sector.reshape(self.cx.shape)

    def _ledge_crossings(self) -> list[tuple[float, float, float, float, set]]:
        """Two-sided lines that cannot be crossed in at least one direction."""
        out = []
        # A manual door opens only from a face that carries the door special: E1M5's door at (-576, 1200)
        # has a bare face on one side, and the expert pressed use on it until it died.
        # A tagged door with a manual face is one-way too until its tag fires: E1M7's door at (1216, -2128)
        # opens by hand from the south only; from the north only switch 103 (tag 14) opens it, and the
        # planner routed through its bare face without visiting the switch.
        manual = {l.back for l in self.m.lines if l.special in DOOR_SPECIALS and l.back != -1
                  and (self.m.sector_tag[l.back] == 0 or self.m.sector_tag[l.back] not in self.m.opened)}
        # Crossing a teleport line from its front teleports the player (E1M5: the route skirted a teleport pad
        # and the expert was sent back 1,700 units each time it stepped in). End-level teleporters are goals.
        end_tags = {int(t) for t, sp in zip(self.m.sector_tag, self.m.special) if sp == END_LEVEL_SECTOR and t}
        for l in self.m.lines:
            if l.special in (39, 97) and l.back != -1 and l.tag not in end_tags:
                out.append((*l.v1, *l.v2, {l.front}))
                continue
            # A door with a manual face and a locked face opens only from the manual side without the key
            # (E1M6: the route to the yellow key ran into a blue-locked face of a door opened from behind).
            if l.special in LOCKED_DOORS and LOCKED_DOORS[l.special] not in self.m.keys and l.back != -1:
                out.append((*l.v1, *l.v2, {l.front}))
                continue
            if l.back != -1 and not l.flags & 1 and l.special not in DOOR_SPECIALS                     and (l.front in manual) != (l.back in manual):
                out.append((*l.v1, *l.v2, {l.back if l.front in manual else l.front}))
                continue
            if l.back == -1 or (l.flags & 1):
                continue
            if l.front in self.m.lift_sectors or l.back in self.m.lift_sectors:
                continue
            ff, bf = self.floor_s[l.front], self.floor_s[l.back]
            gap = min(self.ceil_s[l.front], self.ceil_s[l.back]) - max(ff, bf)
            bad = set()
            if gap < PLAYER_HEIGHT:
                bad = {l.front, l.back}
            elif bf - ff > MAX_STEP:
                bad = {l.front}           # cannot climb from front onto back
            elif ff - bf > MAX_STEP:
                bad = {l.back}
            if bad:
                out.append((*l.v1, *l.v2, bad))
        return out

    def to_cell(self, x: float, y: float) -> tuple[int, int]:
        return int((y - self.y0) // CELL), int((x - self.x0) // CELL)

    def to_world(self, r: int, c: int) -> tuple[float, float]:
        return float(self.cx[r, c]), float(self.cy[r, c])

    def sector_at(self, x: float, y: float) -> int:
        r, c = self.to_cell(x, y)
        if 0 <= r < self.h and 0 <= c < self.w:
            return int(self.sector[r, c])
        return -1

    # -- planning ---------------------------------------------------------
    def _edge_ok(self, r0, c0, r1, c1) -> bool:
        f0, f1 = self.floor[r0, c0], self.floor[r1, c1]
        if f1 - f0 > MAX_STEP and not (self.sector[r0, c0] in self.m.lift_sectors
                                        or self.sector[r1, c1] in self.m.lift_sectors):
            return False
        s0 = self.sector[r0, c0]
        if s0 == self.sector[r1, c1]:
            return True
        for ln in self._ledges_near(r0, c0):
            x1, y1, x2, y2, bad = ln
            if s0 in bad and _segments_cross(self.cx[r0, c0], self.cy[r0, c0], self.cx[r1, c1], self.cy[r1, c1], x1, y1, x2, y2):
                return False
        return True

    def _ledges_near(self, r, c):
        if not hasattr(self, "_ledge_index"):
            self._ledge_index = {}
            for ln in self.blocked_edges:
                x1, y1, x2, y2, _ = ln
                ra, ca = self.to_cell(min(x1, x2), min(y1, y2))
                rb, cb = self.to_cell(max(x1, x2), max(y1, y2))
                for rr in range(ra - 1, rb + 2):
                    for cc in range(ca - 1, cb + 2):
                        self._ledge_index.setdefault((rr, cc), []).append(ln)
        return self._ledge_index.get((r, c), ())

    def reachable(self, start: tuple[float, float]) -> np.ndarray:
        """Boolean mask of cells reachable from start under this grid's world state."""
        mask = np.zeros((self.h, self.w), bool)
        s = self.nearest_free(*self.to_cell(*start))
        if s is None:
            return mask
        stack = [s]
        mask[s] = True
        while stack:
            r, c = stack.pop()
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.h and 0 <= cc < self.w and not mask[rr, cc] and not self.blocked[rr, cc]                         and self._edge_ok(r, c, rr, cc):
                    mask[rr, cc] = True
                    stack.append((rr, cc))
        return mask

    def can_reach(self, goal: tuple[float, float]) -> np.ndarray:
        """Boolean mask of cells from which goal can be reached (reverse flood over directed edges).
        Free cells outside it are dead ends: drops the player cannot climb back out of."""
        mask = np.zeros((self.h, self.w), bool)
        s = self.nearest_free(*self.to_cell(*goal))
        if s is None:
            return ~mask
        stack = [s]
        mask[s] = True
        while stack:
            r, c = stack.pop()
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.h and 0 <= cc < self.w and not mask[rr, cc] and not self.blocked[rr, cc] \
                        and self._edge_ok(rr, cc, r, c):
                    mask[rr, cc] = True
                    stack.append((rr, cc))
        return mask

    def nearest_free(self, r: int, c: int, radius: int = 6) -> tuple[int, int] | None:
        best = None
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.h and 0 <= cc < self.w and not self.blocked[rr, cc]:
                    d = dr * dr + dc * dc
                    if best is None or d < best[0]:
                        best = (d, rr, cc)
        return None if best is None else (best[1], best[2])

    def plan(self, start: tuple[float, float], goal: tuple[float, float], max_expand: int = 200_000):
        """A* from world point to world point. Returns list of world waypoints or None."""
        s = self.nearest_free(*self.to_cell(*start))
        g = self.nearest_free(*self.to_cell(*goal))
        if s is None or g is None:
            return None
        gr, gc = g
        openq = [(0.0, 0.0, s)]
        came = {s: None}
        gcost = {s: 0.0}
        steps = ((0, 1, 1), (1, 0, 1), (0, -1, 1), (-1, 0, 1), (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2))
        expanded = 0
        while openq:
            _, cost, cur = heapq.heappop(openq)
            if cur == g:
                break
            if cost > gcost.get(cur, math.inf):
                continue
            expanded += 1
            if expanded > max_expand:
                return None
            r, c = cur
            for dr, dc, w in steps:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < self.h and 0 <= cc < self.w) or self.blocked[rr, cc]:
                    continue
                if dr and dc and (self.blocked[r, cc] or self.blocked[rr, c]):
                    continue
                if not self._edge_ok(r, c, rr, cc):
                    continue
                nc = cost + w * self.cost[rr, cc]
                if nc < gcost.get((rr, cc), math.inf):
                    gcost[(rr, cc)] = nc
                    came[(rr, cc)] = cur
                    heapq.heappush(openq, (nc + math.hypot(rr - gr, cc - gc), nc, (rr, cc)))
        if g not in came:
            return None
        path, cur = [], g
        while cur is not None:
            path.append(self.to_world(*cur))
            cur = came[cur]
        path.reverse()
        return path

    def walkable_line(self, a: tuple[float, float], b: tuple[float, float], keep_off_drops: bool = False) -> bool:
        """True when the straight segment a->b stays on free cells with legal transitions.
        keep_off_drops: also stay 32 units clear of edges one can fall off (beyond the start's own cell)."""
        dist = math.dist(a, b)
        n = max(2, int(dist / (CELL / 2)))
        prev = start = self.to_cell(*a)
        for i in range(1, n + 1):
            t = i / n
            cur = self.to_cell(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            if not (0 <= cur[0] < self.h and 0 <= cur[1] < self.w) or self.blocked[cur]:
                return False
            if keep_off_drops and cur != start and self.near_drop[cur]:
                return False
            if cur != prev:
                if not self._edge_ok(*prev, *cur):
                    return False
                if self.cost[cur] >= 25:      # do not shortcut across damaging floors
                    return False
                prev = cur
        return True


def _segments_cross(ax, ay, bx, by, cx, cy, dx, dy) -> bool:
    def orient(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)
    o1 = orient(ax, ay, bx, by, cx, cy)
    o2 = orient(ax, ay, bx, by, dx, dy)
    o3 = orient(cx, cy, dx, dy, ax, ay)
    o4 = orient(cx, cy, dx, dy, bx, by)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def path_length(path) -> float:
    return sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1)) if path else 0.0


TELEPORT_SPECIALS = {39, 97, 125, 126}
END_LEVEL_SECTOR = 11          # damage floor that ends the level when health drops below 11% (E1M8)


def teleport_exit(m: MapData) -> tuple[float, float] | None:
    """For maps without an exit line (E1M8): the middle of the teleport lines whose destination sector
    ends the level. Walking into it from outside crosses a teleport line."""
    end_tags = {int(m.sector_tag[i]) for i, sp in enumerate(m.special) if sp == END_LEVEL_SECTOR and m.sector_tag[i]}
    lines = [l for l in m.lines if l.special in TELEPORT_SPECIALS and l.tag in end_tags]
    if not lines:
        return None
    pts = [p for l in lines for p in (l.v1, l.v2)]
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def exit_target(m: MapData) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Stand point in front of the first exit switch, the switch midpoint, and the facing angle.
    Without an exit line, the end-level teleporter's middle (stand and target are the same point)."""
    normal = [l for l in m.exit_lines if l.special in NORMAL_EXITS]
    if not (normal or m.exit_lines):
        tp = teleport_exit(m)
        if tp is None:
            raise ValueError(f"{m.name}: no exit line and no end-level teleporter")
        return tp, tp, 0.0
    l = (normal or m.exit_lines)[0]
    (x1, y1), (x2, y2) = l.v1, l.v2
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    # Front side (where the player stands) is to the right of v1->v2.
    nx, ny = (y2 - y1), -(x2 - x1)
    norm = math.hypot(nx, ny)
    nx, ny = nx / norm, ny / norm
    stand = (mx + nx * 40, my + ny * 40)
    face = math.degrees(math.atan2(-ny, -nx)) % 360
    return stand, (mx, my), face
