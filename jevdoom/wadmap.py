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
MAX_STEP = 24
PLAYER_HEIGHT = 56
DOOR_SPECIALS = {1, 26, 27, 28, 31, 32, 33, 34, 117, 118}
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
                np.array([s[5] for s in secs], int), things, sector_tag=np.array([s[6] for s in secs], int))
    for l in lines:
        if l.special in DOOR_SPECIALS and l.back != -1:
            m.door_sectors.add(l.back)
    return m


def _seg_dist(px, py, ax, ay, bx, by):
    """Vectorised distance from points (px, py) to segment a-b."""
    dx, dy = bx - ax, by - ay
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / max(dx * dx + dy * dy, 1e-9), 0, 1)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy))


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
        self.sector = self._locate_sectors()
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
        self.cost = np.ones((self.h, self.w))
        self.cost[inside & np.isin(m.special[sec], list(DAMAGE_SECTORS))] = 25.0
        for l in m.lines:
            (ax, ay), (bx, by) = l.v1, l.v2
            d = _seg_dist(self.cx, self.cy, ax, ay, bx, by)
            solid = l.back == -1 or (l.flags & 1)
            if solid:
                self.blocked |= d < PLAYER_RADIUS
                self.cost += np.where(d < 40, (40 - d) / 8, 0)   # prefer the middle of corridors
            else:
                ff, bf = floor[l.front], floor[l.back]
                gap = min(ceil[l.front], ceil[l.back]) - max(ff, bf)
                if abs(ff - bf) > MAX_STEP or gap < PLAYER_HEIGHT:
                    self.cost += np.where(d < 32, (32 - d) / 4, 0)  # keep off ledges
        self.blocked_edges = self._ledge_crossings()

    # -- geometry ---------------------------------------------------------
    def _locate_sectors(self) -> np.ndarray:
        """Point-in-sector by casting a ray to +x and taking the nearest crossed linedef."""
        px, py = self.cx.ravel(), self.cy.ravel()
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
        for l in self.m.lines:
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

    def walkable_line(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        """True when the straight segment a->b stays on free cells with legal transitions."""
        dist = math.dist(a, b)
        n = max(2, int(dist / (CELL / 2)))
        prev = self.to_cell(*a)
        for i in range(1, n + 1):
            t = i / n
            cur = self.to_cell(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            if not (0 <= cur[0] < self.h and 0 <= cur[1] < self.w) or self.blocked[cur]:
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


def exit_target(m: MapData) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Stand point in front of the first exit switch, the switch midpoint, and the facing angle."""
    normal = [l for l in m.exit_lines if l.special in NORMAL_EXITS]
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
