"""
Input:  disks D_i = (x_i, y_i, r_i)
Output: maximum depth k*, and a point p attaining that depth.

Method implemented:
  1. Dualize each disk to two x-monotone boundary branches
        low_i(m) = y_i - x_i m - r_i sqrt(1 + m^2)
        up_i(m)  = y_i - x_i m + r_i sqrt(1 + m^2)
  2. For a fixed k, compute L_k and U_k by a sweep over pairwise branch
     intersections. L_k is kth from above among lower branches; U_k is kth
     from below among upper branches.
  3. Build the 2-variable linear feasibility problem for a dual line
        gamma(m) = alpha m + beta
     satisfying U_k(m) <= gamma(m) <= L_k(m) for all m.
  4. Recover the primal point from p* : b = alpha m + beta, namely
        p = (-alpha, beta).

The LP is solved with scipy.optimize.linprog.

Note: Floating-point computation is used.
"""

from dataclasses import dataclass
from math import acos, atan2, cos, isfinite, pi, sin, sqrt, tau
from typing import Iterable, List, Optional, Sequence, Tuple
import numpy as np
import math
from itertools import combinations
import random
import matplotlib.pyplot as plt
from scipy.optimize import linprog

EPS = 1e-10
ROOT_EPS = 1e-8
MERGE_EPS = 1e-8
LP_TOL = 1e-8


@dataclass(frozen=True)
class Disk:
    x: float
    y: float
    r: float

    def __post_init__(self) -> None:
        if self.r < -EPS:
            raise ValueError("Disk radius must be nonnegative")
        object.__setattr__(self, "x", float(self.x))
        object.__setattr__(self, "y", float(self.y))
        object.__setattr__(self, "r", max(0.0, float(self.r)))


@dataclass(frozen=True)
class BranchEvent:
    m: float
    i: int
    j: int


@dataclass
class LevelData:
    branch: str
    k: int
    vertices: List[Tuple[float, float]]
    anchor: Tuple[float, float]
    left_support: int
    right_support: int
    num_events: int
    used_degenerate_fallback: bool = False


@dataclass
class LPResult:
    feasible: bool
    alpha: Optional[float] = None
    beta: Optional[float] = None
    reason: str = ""
    solver: str = ""


# -----------------------------------------------------------------------------
# Basic disk-dual formulas
# -----------------------------------------------------------------------------


def as_disks(disks_like: Iterable[Tuple[float, float, float] | Disk]) -> List[Disk]:
    return [d if isinstance(d, Disk) else Disk(*d) for d in disks_like]


def branch_value(d: Disk, m: float, branch: str) -> float:
    """Dual boundary value b=f_i(m)."""
    s = sqrt(1.0 + m * m)
    if branch == "low":
        return d.y - d.x * m - d.r * s
    if branch == "up":
        return d.y - d.x * m + d.r * s
    raise ValueError("branch must be 'low' or 'up'")


def branch_slope_limit(d: Disk, branch: str, side: str) -> float:
    """Limit of db/dm as m -> -infinity ('left') or +infinity ('right')."""
    if branch == "low":
        return -d.x + d.r if side == "left" else -d.x - d.r
    if branch == "up":
        return -d.x - d.r if side == "left" else -d.x + d.r
    raise ValueError("branch must be 'low' or 'up'")


def _asymptotic_order_key(d: Disk, branch: str, side: str):
    """Key for vertical order on an unbounded side.

    For 'low', order is descending by value, i.e. kth from above.
    For 'up', order is ascending by value, i.e. kth from below.
    """
    s = branch_slope_limit(d, branch, side)
    # As m -> -infinity, value ~ s*m + y. Since m is negative, smaller slope
    # gives larger value and larger slope gives smaller value.
    # As m -> +infinity, larger slope gives larger value.
    if branch == "low" and side == "left":
        return (s, -d.y, -d.r, d.x, d.y)
    if branch == "low" and side == "right":
        return (-s, -d.y, -d.r, d.x, d.y)
    if branch == "up" and side == "left":
        return (-s, d.y, d.r, d.x, d.y)
    if branch == "up" and side == "right":
        return (s, d.y, d.r, d.x, d.y)
    raise ValueError("bad branch/side")


def kth_level_value_by_sort(disks: Sequence[Disk], branch: str, k: int, m: float) -> float:
    vals = [branch_value(d, m, branch) for d in disks]
    if branch == "low":
        vals.sort(reverse=True)
    elif branch == "up":
        vals.sort()
    else:
        raise ValueError("branch must be 'low' or 'up'")
    return vals[k - 1]


# -----------------------------------------------------------------------------
# Pairwise branch intersections
# -----------------------------------------------------------------------------


def pair_intersections(di: Disk, dj: Disk, branch: str, eps: float = EPS) -> List[float]:
    """Return finite m-coordinates where two same-type branches intersect.

    For same-type branches,

        A + B m + C sqrt(1+m^2) = 0,

    where C = -(r_i-r_j) for lower branches and +(r_i-r_j) for upper branches.
    Squaring gives a quadratic; roots are filtered against the original equation.
    """
    A = di.y - dj.y
    B = -(di.x - dj.x)
    dr = di.r - dj.r
    C = -dr if branch == "low" else dr

    roots: List[float] = []

    if abs(C) <= eps:
        if abs(B) <= eps:
            # Coincident or never-intersecting branches. Coincident branches do not
            # generate a finite swap event under symbolic perturbation.
            return []
        roots = [-A / B]
    else:
        qa = B * B - C * C
        qb = 2.0 * A * B
        qc = A * A - C * C

        if abs(qa) <= eps:
            if abs(qb) <= eps:
                return []
            roots = [-qc / qb]
        else:
            disc = qb * qb - 4.0 * qa * qc
            if disc < -eps:
                return []
            disc = max(0.0, disc)
            sd = sqrt(disc)
            roots = [(-qb - sd) / (2.0 * qa), (-qb + sd) / (2.0 * qa)]

    out: List[float] = []
    for m in roots:
        if not isfinite(m):
            continue
        val = branch_value(di, m, branch) - branch_value(dj, m, branch)
        scale = max(1.0, abs(branch_value(di, m, branch)), abs(branch_value(dj, m, branch)))
        if abs(val) <= ROOT_EPS * scale:
            if not any(abs(m - u) <= MERGE_EPS * max(1.0, abs(m), abs(u)) for u in out):
                out.append(m)
    out.sort()
    return out


def all_events(disks: Sequence[Disk], branch: str) -> List[BranchEvent]:
    n = len(disks)
    events: List[BranchEvent] = []
    for i in range(n):
        for j in range(i + 1, n):
            for m in pair_intersections(disks[i], disks[j], branch):
                events.append(BranchEvent(m, i, j))
    events.sort(key=lambda e: e.m)
    return events


def _same_x(a: float, b: float) -> bool:
    return abs(a - b) <= MERGE_EPS * max(1.0, abs(a), abs(b))


def _append_unique_vertex(vertices: List[Tuple[float, float]], m: float, y: float) -> None:
    if vertices:
        m0, y0 = vertices[-1]
        if abs(m - m0) <= MERGE_EPS * max(1.0, abs(m), abs(m0)) and abs(y - y0) <= 1e-7 * max(1.0, abs(y), abs(y0)):
            return
    vertices.append((m, y))


# -----------------------------------------------------------------------------
# Sweep-line kth-level extraction
# -----------------------------------------------------------------------------


def compute_level_sweepline(disks: Sequence[Disk], branch: str, k: int) -> LevelData:
    """Compute the kth level using a sweep over branch-intersection events.

    branch='low': kth highest lower branch, i.e. L_k.
    branch='up' : kth lowest upper branch, i.e. U_k.

    The returned vertices are the finite points where the kth-level support
    changes. An anchor point at m=0 is also returned; it is a safe additional LP
    constraint and is important in no-vertex cases.
    """
    n = len(disks)
    if not (1 <= k <= n):
        raise ValueError("k must lie in {1,...,n}")
    if branch not in {"low", "up"}:
        raise ValueError("branch must be 'low' or 'up'")

    order = list(range(n))
    order.sort(key=lambda idx: _asymptotic_order_key(disks[idx], branch, "left"))
    pos = [0] * n
    for rank, idx in enumerate(order):
        pos[idx] = rank

    left_support = order[k - 1]
    vertices: List[Tuple[float, float]] = []
    events = all_events(disks, branch)
    used_degenerate_fallback = False

    t = 0
    while t < len(events):
        m0 = events[t].m
        group: List[BranchEvent] = []
        while t < len(events) and _same_x(events[t].m, m0):
            group.append(events[t])
            t += 1

        # In general position the group has size 1 and the two curves are adjacent.
        # If degeneracies make this false, sort only after the event group as a
        # symbolic-perturbation fallback.
        fallback_needed = False
        for ev in group:
            pi, pj = pos[ev.i], pos[ev.j]
            if abs(pi - pj) != 1:
                fallback_needed = True
                break

        if fallback_needed:
            used_degenerate_fallback = True
            before_k = order[k - 1]

            # Probe just to the right of the event group.
            if t < len(events):
                next_m = events[t].m
                if next_m > m0:
                    probe = 0.5 * (m0 + next_m)
                else:
                    probe = m0 + max(1e-6, 1e-6 * max(1.0, abs(m0)))
            else:
                probe = m0 + max(1.0, abs(m0) + 1.0)

            reverse = branch == "low"
            order.sort(key=lambda idx: branch_value(disks[idx], probe, branch), reverse=reverse)
            for rank, idx in enumerate(order):
                pos[idx] = rank

            after_k = order[k - 1]
            if before_k != after_k:
                y = kth_level_value_by_sort(disks, branch, k, m0)
                _append_unique_vertex(vertices, m0, y)
            continue

        # Nondegenerate adjacent-swap processing.
        # Process independent same-x swaps from lower position to higher position
        # for determinism. In true general position, this loop has one event.
        group.sort(key=lambda ev: min(pos[ev.i], pos[ev.j]))
        for ev in group:
            pi, pj = pos[ev.i], pos[ev.j]
            if abs(pi - pj) != 1:
                # A previous same-x swap may have changed adjacency. Fall back by
                # rebuilding the order just right of the event and recording a kth
                # vertex if needed.
                used_degenerate_fallback = True
                before_k = order[k - 1]
                probe = m0 + max(1e-6, 1e-6 * max(1.0, abs(m0)))
                reverse = branch == "low"
                order.sort(key=lambda idx: branch_value(disks[idx], probe, branch), reverse=reverse)
                for rank, idx in enumerate(order):
                    pos[idx] = rank
                after_k = order[k - 1]
                if before_k != after_k:
                    y = kth_level_value_by_sort(disks, branch, k, m0)
                    _append_unique_vertex(vertices, m0, y)
                break

            pmin = min(pi, pj)
            if pmin == k - 1:
                # The adjacent swap crosses the kth-rank boundary, so the kth
                # level has a vertex at the crossing point.
                y = 0.5 * (branch_value(disks[ev.i], m0, branch) + branch_value(disks[ev.j], m0, branch))
                _append_unique_vertex(vertices, m0, y)

            # Swap the adjacent curves in vertical order.
            if pi < pj:
                a, b = pi, pj
            else:
                a, b = pj, pi
            order[a], order[b] = order[b], order[a]
            pos[order[a]] = a
            pos[order[b]] = b

    right_support = order[k - 1]
    anchor = (0.0, kth_level_value_by_sort(disks, branch, k, 0.0))

    return LevelData(
        branch=branch,
        k=k,
        vertices=vertices,
        anchor=anchor,
        left_support=left_support,
        right_support=right_support,
        num_events=len(events),
        used_degenerate_fallback=used_degenerate_fallback,
    )


# -----------------------------------------------------------------------------
# 2D linear programming feasibility
# -----------------------------------------------------------------------------


def build_corridor_constraints(
    disks: Sequence[Disk],
    lower_level: LevelData,
    upper_level: LevelData,
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, str]]]:
    """Build inequalities A*alpha + B*beta <= C.

    Also returns readable metadata for debugging.
    """
    constraints: List[Tuple[float, float, float]] = []
    meta: List[Tuple[float, float, str]] = []

    # gamma(m)=alpha*m+beta must be <= L_k(m).
    lower_points = list(lower_level.vertices)
    lower_points.append(lower_level.anchor)
    for m, y in lower_points:
        constraints.append((m, 1.0, y))
        meta.append((m, y, "gamma <= L_k"))

    # gamma(m)=alpha*m+beta must be >= U_k(m), i.e. -alpha*m - beta <= -U_k(m).
    upper_points = list(upper_level.vertices)
    upper_points.append(upper_level.anchor)
    for m, y in upper_points:
        constraints.append((-m, -1.0, -y))
        meta.append((m, y, "gamma >= U_k"))

    # Slope constraints for unbounded rays.
    L_left = disks[lower_level.left_support]
    L_right = disks[lower_level.right_support]
    U_left = disks[upper_level.left_support]
    U_right = disks[upper_level.right_support]

    # L_k left ray:  alpha >= slope(L_left, left)
    s = branch_slope_limit(L_left, "low", "left")
    constraints.append((-1.0, 0.0, -s))
    meta.append((s, 0.0, "alpha >= left slope of L_k"))

    # L_k right ray: alpha <= slope(L_right, right)
    s = branch_slope_limit(L_right, "low", "right")
    constraints.append((1.0, 0.0, s))
    meta.append((s, 0.0, "alpha <= right slope of L_k"))

    # U_k left ray:  alpha <= slope(U_left, left)
    s = branch_slope_limit(U_left, "up", "left")
    constraints.append((1.0, 0.0, s))
    meta.append((s, 0.0, "alpha <= left slope of U_k"))

    # U_k right ray: alpha >= slope(U_right, right)
    s = branch_slope_limit(U_right, "up", "right")
    constraints.append((-1.0, 0.0, -s))
    meta.append((s, 0.0, "alpha >= right slope of U_k"))

    return constraints, meta


def _solve_lp_with_scipy(constraints: Sequence[Tuple[float, float, float]]) -> Optional[LPResult]:
    if not constraints:
        return LPResult(True, 0.0, 0.0, solver="scipy")

    A = [[a, b] for a, b, _ in constraints]
    ub = [c for _, _, c in constraints]
    res = linprog(
        c=[0.0, 0.0],
        A_ub=A,
        b_ub=ub,
        bounds=[(None, None), (None, None)],
        method="highs",
    )
    if res.success:
        alpha, beta = float(res.x[0]), float(res.x[1])
        return LPResult(True, alpha, beta, solver="scipy")
    return LPResult(False, reason=res.message, solver="scipy")


def solve_corridor_lp(constraints: Sequence[Tuple[float, float, float]]) -> LPResult:
    res = _solve_lp_with_scipy(constraints)
    return res


def check_constraints(constraints: Sequence[Tuple[float, float, float]], alpha: float, beta: float) -> bool:
    for a, b, c in constraints:
        lhs = a * alpha + b * beta
        if lhs > c + 1e-7 * max(1.0, abs(c), abs(lhs)):
            return False
    return True


# -----------------------------------------------------------------------------
# Fixed-k feasibility and maximum-depth recovery
# -----------------------------------------------------------------------------


def verify_k_feasible_sweepline_lp(
    disks_like: Iterable[Tuple[float, float, float] | Disk],
    k: int,
    *,
    return_debug: bool = False,
):
    """Decide whether a k-deep point exists.

    Returns either LPResult or (LPResult, debug_dict) when return_debug=True.
    """
    disks = as_disks(disks_like)
    n = len(disks)

    if k < 0 or k > n:
        res = LPResult(False, reason="k outside [0,n]")
        return (res, {}) if return_debug else res
    if k == 0:
        res = LPResult(True, 0.0, 0.0, reason="k=0 is always feasible", solver="trivial")
        return (res, {}) if return_debug else res
    if n == 0:
        res = LPResult(False, reason="no disks and k>0")
        return (res, {}) if return_debug else res

    lower_level = compute_level_sweepline(disks, "low", k)
    upper_level = compute_level_sweepline(disks, "up", k)
    constraints, meta = build_corridor_constraints(disks, lower_level, upper_level)
    res = solve_corridor_lp(constraints)

    if res.feasible and res.alpha is not None and res.beta is not None:
        if not check_constraints(constraints, res.alpha, res.beta):
            res = LPResult(False, reason="LP solver returned a numerically infeasible point", solver=res.solver)

    if return_debug:
        debug = {
            "lower_level": lower_level,
            "upper_level": upper_level,
            "constraints": constraints,
            "constraint_metadata": meta,
            "num_constraints": len(constraints),
        }
        return res, debug
    return res


def deepest_point_sweepline_lp(
    disks_like: Iterable[Tuple[float, float, float] | Disk],
    *,
    verify_output_depth: bool = True,
) -> Tuple[int, Tuple[float, float]]:
    """Return (maximum_depth, primal_point) using binary search over k."""
    disks = as_disks(disks_like)
    n = len(disks)
    if n == 0:
        return 0, (0.0, 0.0)

    lo, hi = 0, n // 2
    best_k = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        res = verify_k_feasible_sweepline_lp(disks, mid)
        if res.feasible:
            best_k = mid
            lo = mid + 1
        else:
            hi = mid - 1

    if best_k == 0:
        return 0, (0.0, 0.0)

    res = verify_k_feasible_sweepline_lp(disks, best_k)
    if not res.feasible or res.alpha is None or res.beta is None:
        raise RuntimeError(f"Internal error: k={best_k} was feasible during search but no line was recovered")

    # If p=(a,c), then p*: b=-a*m+c.  Since the LP line is b=alpha*m+beta,
    # a=-alpha and c=beta.
    p = (-res.alpha, res.beta)

    if verify_output_depth:
        verified = depth_of_point(disks, p)
        if verified + 1e-9 < best_k:
            raise RuntimeError(
                f"Numerical verification failed: LP returned depth {best_k}, "
                f"but angular verifier found depth {verified}."
            )

    return best_k, p


# Backward-compatible shorter name.
deepest_point = deepest_point_sweepline_lp


# -----------------------------------------------------------------------------
# Independent primal verifier and visualization
# -----------------------------------------------------------------------------


def depth_of_point(
    disks_like: Iterable[Tuple[float, float, float] | Disk],
    p: Tuple[float, float],
) -> int:
    """Compute the depth of a point by an angular sweep in the primal plane.

    A disk D_i is fully contained in the oriented closed halfplane through p
    with unit normal u iff u . (c_i-p) >= r_i.  The depth is the minimum count
    over all orientations u.
    """
    disks = as_disks(disks_like)
    px, py = p
    events: List[Tuple[float, int]] = []
    count0 = 0
    twopi = 2.0 * pi

    for d in disks:
        vx = d.x - px
        vy = d.y - py
        dist = sqrt(vx * vx + vy * vy)
        if dist <= d.r + 1e-10:
            # If p lies inside the disk, no open set of halfplanes through p
            # fully contains that disk. It does not help the minimum.
            continue

        phi = atan2(vy, vx) % twopi
        half = acos(min(1.0, max(-1.0, d.r / dist)))
        start = (phi - half) % twopi
        end = (phi + half) % twopi

        if start <= end:
            if start <= 0.0 <= end:
                count0 += 1
            events.append((start, +1))
            events.append((end, -1))
        else:
            count0 += 1
            events.append((end, -1))
            events.append((start, +1))

    # To capture the minimum for closed halfplanes, process removals before
    # additions at the same angle.
    events.sort(key=lambda e: (e[0], e[1]))
    ans = count0
    cnt = count0
    i = 0
    while i < len(events):
        theta = events[i][0]
        delta = 0
        while i < len(events) and abs(events[i][0] - theta) <= 1e-12:
            delta += events[i][1]
            i += 1
        cnt += delta
        ans = min(ans, cnt)
    return max(0, ans)


def plot_disks_solution(
    disks_like: Iterable[Tuple[float, float, float] | Disk],
    point: Optional[Tuple[float, float]] = None,
    k: Optional[int] = None,
    *,
    ax=None,
    annotate: bool = False,
    show_centers: bool = True,
    show: bool = True,
    save_path: Optional[str] = None,
    padding_ratio: float = 0.08,
    color_points: list[int] = []
):
    """Plot disks and the output point."""

    disks = as_disks(disks_like)
    if point is None:
        k, point = deepest_point_sweepline_lp(disks)
    elif k is None:
        k = depth_of_point(disks, point)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if disks:
        xmin = min(d.x - d.r for d in disks)
        xmax = max(d.x + d.r for d in disks)
        ymin = min(d.y - d.r for d in disks)
        ymax = max(d.y + d.r for d in disks)
    else:
        xmin = xmax = point[0]
        ymin = ymax = point[1]

    px, py = point
    xmin, xmax = min(xmin, px), max(xmax, px)
    ymin, ymax = min(ymin, py), max(ymax, py)
    width = max(xmax - xmin, 1.0)
    height = max(ymax - ymin, 1.0)
    pad = padding_ratio * max(width, height)

    for idx, d in enumerate(disks):
        circle = plt.Circle((d.x, d.y), d.r, fill=False, linewidth=1.5, alpha=0.75)
        ax.add_patch(circle)
        if show_centers:
            if idx in color_points:
                ax.scatter([d.x], [d.y], marker=".", s=54, color='m')
            else:
                ax.scatter([d.x], [d.y], marker=".", s=16, color='black')
        if annotate:
            ax.text(d.x, d.y, f"D{idx}", fontsize=8, ha="left", va="bottom")

    # ax.scatter([px], [py], marker="*", s=180, label=f"depth={k}")
    ax.scatter([px], [py], marker="*", color='red', s=20, label=f"depth={k}")
    if annotate:
        ax.annotate(
            f"p=({px:.4g}, {py:.4g})\ndepth={k}",
            xy=(px, py),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Input disks and deepest output point")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize = 11)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
    if show:
        plt.show()
    return fig, ax, k, point


def generate_random_disks(
    n,
    center_range=(-10.0, 10.0),
    radius_range=(0.2, 1.5),
    seed=None,
    integer_centers=False,
    min_center_distance=None,
    max_attempts=100_000,
):
    """
    Parameters
    ----------
    n : int
        Number of disks.

    center_range : tuple[float, float]
        Range (lo, hi) for both x- and y-coordinates of disk centers.

    radius_range : tuple[float, float]
        Range (lo, hi) for disk radii.

    seed : int | None
        Random seed for reproducibility.

    integer_centers : bool
        If True, centers are sampled as integers. Otherwise they are sampled
        uniformly as floats.

    min_center_distance : float | None
        If given, enforce that every pair of disk centers has Euclidean
        distance at least this value. This is useful for avoiding nearly
        coincident centers.

    max_attempts : int
        Maximum rejection-sampling attempts when min_center_distance is used.

    Returns
    -------
    disks : list[tuple[float, float, float]]
        List of disks in the format (x, y, r).
    """
    if n < 0:
        raise ValueError("n must be nonnegative")

    cx_lo, cx_hi = center_range
    r_lo, r_hi = radius_range

    if cx_lo > cx_hi:
        raise ValueError("center_range must satisfy lo <= hi")
    if r_lo <= 0 or r_lo > r_hi:
        raise ValueError("radius_range must satisfy 0 < lo <= hi")

    rng = random.Random(seed)
    disks = []
    attempts = 0

    def sample_center():
        if integer_centers:
            return (
                float(rng.randint(math.ceil(cx_lo), math.floor(cx_hi))),
                float(rng.randint(math.ceil(cx_lo), math.floor(cx_hi))),
            )
        return (
            rng.uniform(cx_lo, cx_hi),
            rng.uniform(cx_lo, cx_hi),
        )

    while len(disks) < n:
        if attempts >= max_attempts:
            raise RuntimeError(
                "Failed to generate enough disks. Try reducing "
                "min_center_distance or increasing center_range."
            )

        attempts += 1
        x, y = sample_center()

        if min_center_distance is not None:
            ok = True
            for xj, yj, _ in disks:
                if math.hypot(x - xj, y - yj) < min_center_distance:
                    ok = False
                    break
            if not ok:
                continue

        r = rng.uniform(r_lo, r_hi)
        disks.append((x, y, r))

    return disks


def plot_radius_increase_vs_depth(
    disks,
    epsilon=0.25,
    max_increase=None,
    show=True,
    check_monotonic=True,
):
    """
    Parameters
    ----------
    disks : list[tuple[float, float, float]]
        Input disks as (x, y, r).

    epsilon : float
        Step size for the uniform radius increase.

    max_increase : float | None
        Maximum value of radius incease. If None, uses
            max pairwise center distance + max initial radius

    show : bool
        If True, calls plt.show().

    check_monotonic : bool
        If True, prints a warning if the computed depths are not non-increasing.

    Returns
    -------
    deltas : list[float]
        Radius increases.

    depths : list[int]
        Maximum depth for each radius increase.

    witnesses : list[tuple[float, float]]
        Witness point returned for each radius increase.
    """

    if not disks:
        raise ValueError("disks must be nonempty")

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    # Compute max pairwise distance among centers.
    max_center_dist = 0.0
    for i in range(len(disks)):
        xi, yi, _ = disks[i]
        for j in range(i + 1, len(disks)):
            xj, yj, _ = disks[j]
            max_center_dist = max(max_center_dist, math.hypot(xi - xj, yi - yj))

    max_radius = max(r for _, _, r in disks)

    if max_increase is None:
        max_increase = max_center_dist + max_radius

    if max_increase < 0:
        raise ValueError("max_increase must be nonnegative")

    # Generate radius increase values, including max_increase exactly.
    num_full_steps = int(math.floor(max_increase / epsilon))
    deltas = [i * epsilon for i in range(num_full_steps + 1)]

    if deltas[-1] < max_increase:
        deltas.append(max_increase)

    depths = []
    witnesses = []

    for delta in deltas:
        inflated_disks = [
            (x, y, r + delta)
            for x, y, r in disks
        ]

        k, p = deepest_point_sweepline_lp(inflated_disks)

        depths.append(k)
        witnesses.append(p)

    if check_monotonic:
        for i in range(len(depths) - 1):
            if depths[i + 1] > depths[i]:
                print(
                    "Warning: depth increased from "
                    f"{depths[i]} to {depths[i + 1]} "
                    f"between delta={deltas[i]} and delta={deltas[i + 1]}. "
                    "This should not happen theoretically."
                )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(deltas, depths, marker="o")
    # ax.step(deltas, depths, where="post")

    ax.set_xlabel("Uniform radius increase")
    ax.set_ylabel("Maximum depth")
    ax.set_title("Maximum depth vs uniform radius increase")
    ax.grid(True, alpha=0.3)

    ax.set_ylim(bottom=-0.25, top=max(depths) + 0.75)

    plt.tight_layout()

    if show:
        plt.show()

    return deltas, depths, witnesses
def TukeyMed(X,plot,specK = 0):
    """
    X: nx2 matrix
    plot: boolean for plotting options
    """
    n = len(X)
    
    # Store tuples of (k, a, b, c) representing constraint ax + by + c <= 0
    candidate_constraints = []
    participating_points = {}
    indices = range(n)
    epsilon = 1e-9
    
    # 1. Loop through all pairs
    for idx1, idx2 in combinations(indices, 2):
        p1 = X[idx1]
        p2 = X[idx2]
        
        # Line formulation: A*x + B*y + C = 0
        dy = p2[1] - p1[1]
        dx = p2[0] - p1[0]
        a = -dy
        b = dx
        c = - (a * p1[0] + b * p1[1])
        
        # Evaluate points
        vals = a * X[:, 0] + b * X[:, 1] + c
        
        n_above = np.sum(vals > epsilon)
        n_below = np.sum(vals < -epsilon)
        

        k = min(n_above, n_below)

        if n_above < n_below:
            # Constraint: val <= 0 -> ax + by + c <= 0
            candidate_constraints.append((k, a, b, c, idx1, idx2))
            
        # Case B: Below is smaller -> Keep ABOVE
        elif n_below < n_above:
            # Constraint: val >= 0 -> -val <= 0 -> -ax -by -c <= 0
            candidate_constraints.append((k, -a, -b, -c,idx1, idx2))
            
        # Case C: Equal parts
        else:

            candidate_constraints.append((k, a, b, c, idx1, idx2))   # Keep Below
            candidate_constraints.append((k, -a, -b, -c, idx1, idx2)) # Keep Above

    if not candidate_constraints:
        print("No valid constraints found.")
        return 0

    # 4. Find the highest k
    max_k = max(item[0] for item in candidate_constraints)
    if specK>0:
        max_k = specK-1
    median_found = False
    while (not median_found):
    # Filter constraints for this max k
        active_constraints = [item for item in candidate_constraints if item[0] == max_k]
        
        # 5. Compute Intersection
        inf = 1e9
        poly = np.array([[-inf, -inf], [inf, -inf], [inf, inf], [-inf, inf]])
        
        def clip_polygon(poly, a, b, c):
            new_poly = []
            if len(poly) == 0: return new_poly
            
            for j in range(len(poly)):
                p_curr = poly[j]
                p_next = poly[(j + 1) % len(poly)]
                
                val_curr = a * p_curr[0] + b * p_curr[1] + c
                val_next = a * p_next[0] + b * p_next[1] + c
                
                if val_curr <= 1e-9:
                    new_poly.append(p_curr)
                
                if (val_curr <= 1e-9 and val_next > 1e-9) or (val_curr > 1e-9 and val_next <= 1e-9):
                    denom = val_curr - val_next
                    if abs(denom) > 1e-12:
                        t = val_curr / denom
                        inter_p = p_curr + t * (p_next - p_curr)
                        new_poly.append(inter_p)
            return np.array(new_poly)
    
        for _, a, b, c, idx1, idx2 in active_constraints:
            poly = clip_polygon(poly, a, b, c)
       
        if len(poly) > 0:
            median_found = True
            boundary_constraints = []
           

            for constraint in active_constraints:
                _, a, b, c, idx1, idx2 = constraint
                vals = a*poly[:,0] + b*poly[:,1] + c
                if np.any(vals < 1e-8):
                    boundary_constraints.append(constraint)
            for _, a, b, c, idx1, idx2 in boundary_constraints:
                min_dist = 1e9
                for point in poly:
                    dist = np.abs(a*point[0] + b*point[1] + c)/np.sqrt(a*a + b*b)
                    min_dist = min(min_dist, dist)
                    
                participating_points[idx1] = min(participating_points.get(idx1, float('inf')), min_dist)
                participating_points[idx2] = min(participating_points.get(idx2, float('inf')), min_dist)
              
        else:
            max_k = max_k-1
        # --- PLOTTING ---
        if plot:
            if len(poly) > 0:
                plt.figure(figsize=(10, 10))
                plt.scatter(X[:,0], X[:,1], c='black', alpha=1.0, label='Points')
                # for point in X:
                #     plt.annotate(f"({point[0]:.2f}, {point[1]:.2f})", xy=(point[0], point[1]), xytext=(5, 5), textcoords='offset points', fontsize=8)
               # plt.annotate(f"k={max_k}", xy=(X[0,0], X[0,1]), xytext=(5, 5), textcoords='offset points', fontsize=12, color='red')
                plot_poly = np.vstack([poly, poly[0]])
                plt.plot(plot_poly[:,0], plot_poly[:,1], 'm-', linewidth=2, label=f'Tukey Median (k={max_k+1})')
                plt.fill(plot_poly[:,0], plot_poly[:,1], 'm', alpha=0.2)
                plt.title(f"Tukey Median: Depth {max_k+1}", fontsize = 24)
                plt.xlabel("X",fontsize = 20)
                plt.ylabel("Y",fontsize =20)
                plt.tick_params(axis='both', which='major', labelsize=20)
                #plt.legend(loc='best')
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.axis('equal')
                
                x_min, x_max = np.min(X[:,0]), np.max(X[:,0])
                y_min, y_max = np.min(X[:,1]), np.max(X[:,1])
                buff = (x_max - x_min) * 0.2 + 0.5
                plt.xlim(x_min - buff, x_max + buff)
                plt.ylim(y_min - buff, y_max + buff)
                plt.savefig("StandardTukey.png")
                plt.show()
    
    return max_k+1,poly, participating_points

def max_inradius(X):
    n = len(X)
    max_r = 0
    crit_lines = []
    best_triple = (0,0,0)
    crit_lines = []

    for i in range(len(poly)):
        p1 = poly[i]
        j = i+1
        if i == len(poly)-1:
            j = 0
        p2 = poly[j]
        dy = p2[1] - p1[1]
        dx = p2[0] - p1[0]
        a = -dy
        b = dx
        c = -(a*p1[0] + b*p1[1])
        if np.abs(a) > 1e-9 or np.abs(b) > 1e-9 or np.abs(c) > 1e-9:
            crit_lines.append((a,b,c))
           
    for triple in combinations(range(n), 3):
        p1, p2, p3 = X[triple[0]], X[triple[1]], X[triple[2]]
        A = 0.5 * abs(np.cross(p2 - p1, p3 - p1))
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)
        s = (a + b + c) / 2
        if s > 0:
            r = A / s
            if r > max_r:
          
                best_triple = triple
                max_r = r
    return max_r,crit_lines,best_triple
def polygon_lines(poly):
    lines = []

    centroid = np.mean(poly, axis=0)

    n = len(poly)

    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i+1) % n]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        a = -dy
        b = dx
        c = -(a*p1[0] + b*p1[1])

        if a*centroid[0] + b*centroid[1] + c > 0:
            a,b,c = -a,-b,-c

        norm = np.hypot(a,b)

        lines.append((a/norm,b/norm,c/norm))

    return lines
def max_inradius_poly(poly):

    lines = polygon_lines(poly)

    best_r = -np.inf
    best_center = None
    best_triple = None

    tol = 1e-8
    crit_lines = []
    for i in range(len(poly)):
        p1 = poly[i]
        p2 = poly[(i+1) % len(poly)]
        dy = p2[1] - p1[1]
        dx = p2[0] - p1[0]
        a = -dy
        b = dx
        c = -(a*p1[0] + b*p1[1])
        if np.abs(a) !=0 or np.abs(b) != 0 or np.abs(c) !=0:
            crit_lines.append((a,b,c))
      
    for triple in combinations(range(len(lines)),3):

        i,j,k = triple

        ai,bi,ci = lines[i]
        aj,bj,cj = lines[j]
        ak,bk,ck = lines[k]

        A = np.array([
            [ai,bi,1],
            [aj,bj,1],
            [ak,bk,1]
        ])

        rhs = -np.array([ci,cj,ck])

        if abs(np.linalg.det(A)) < 1e-10:
            continue

        x,y,r = np.linalg.solve(A,rhs)

        if r <= 0:
            continue

        feasible = True

        for a,b,c in lines:

            if a*x + b*y + c > -r + tol:
                feasible = False
                break

        if feasible and r > best_r:
            best_r = r
            best_center = (x,y)
            best_triple = [lines[i], lines[j], lines[k]]
            

    return best_r,best_triple,crit_lines
if __name__ == "__main__":
    # disks = [
    #     (10.000000, 0.000000, 0.5),
    #     (7.071068, 7.071068, 0.5),
    #     (0.000000, 10.000000, 0.5),
    #     (-7.071068, 7.071068, 0.5),
    #     (-10.000000, 0.000000, 0.5),
    #     (-7.071068, -7.071068, 0.5),
    #     (0.000000, -10.000000, 0.5),
    #     (7.071068, -7.071068, 0.5),
    # ]
    disks = generate_random_disks(
        n=15,
        center_range=(-10, 10),
        radius_range=(0.3, 1.2),
        #seed=12,
        seed = 42,
        min_center_distance=0.5,
    )
    n = 15
    points = np.array((n,2))
    for disk in disks:
        print(f"Disk: center=({disk[0]:.4f}, {disk[1]:.4f}), radius={disk[2]:.4f}")
        points = np.vstack((points,np.array([disk[0], disk[1]])))
    
    points = points[1:,:]
    i = 0;
    for point in points:
        print(f"Point:{i}: ({point[0]:.4f}, {point[1]:.4f})")
        i += 1

    k,poly, min_dists = TukeyMed(points,plot=True)#specK = 6)
   
    i = 0
    for p in poly:
        print(f"Poly vertex: ({p[0]:.4f}, {p[1]:.4f})", " poly vertex index: ", i)
        i += 1
   
        
    participating_points = [1,2,3,4,6,7,8,9,12,14]
    #participating_points = [0,3,14,5,1,11,6]
    #participating_points = [0,1,2,4,5,6,7,8]
    participating_points  = []
    m_disks = []
    crit_points = [poly[i] for i in range(len(poly))]
    rmax,best_triple,crit_lines  = max_inradius_poly(crit_points)
    
    plt.figure(figsize=(10,10))
    for i in range(len(crit_lines)):
        a,b,c = crit_lines[i]
        print(f"Line {i}: {a:.4f}x + {b:.4f}y + {c:.4f} = 0")
    
    poly_np = np.array(crit_points)

# plot vertices
    plt.scatter(points[:,0], points[:,1], c='blue', zorder=3)
    print('points: ', points[:,0],", ", points[:,1])
# label vertices


# draw polygon boundary
    poly_closed = np.vstack([poly_np, poly_np[0]])
    plt.plot(poly_closed[:,0], poly_closed[:,1], 'k-', lw=2, label = f'Tukey Median (k = 6)')
    
    plt.fill(poly_closed[:,0], poly_closed[:,1], 'm', alpha=0.2)
    
# plotting range
    xmin = -10
    xmax = 10
    ymin = -10
    ymax = 10

    xs = np.linspace(xmin, xmax, 500)

    for idx,(a,b,c) in enumerate(crit_lines):

        if abs(b) > 1e-10:
            ys = -(a*xs + c)/b
            plt.plot(xs, ys)

        else:
            xline = -c/a
            plt.axvline(xline, linestyle='--', label=f"L{idx}")

    plt.axis('equal')
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.grid(True)
    plt.title("Standard Tukey Median\nMax Depth k=6")
   # plt.savefig("StandardTukey.png")
   # plt.legend()
    plt.show()
    print("max radius: ", rmax)
    for i in range(len(disks)):
        disk = disks[i]
        if i in participating_points:
            y = list(disk)
            if i in [3,4,6]:
                y[2] = 0
               
                m_disks.append(tuple(y))
            else:
                y[2] = rmax+0.5
                m_disks.append(tuple(y))
            
                
        else:
            y = list(disk)
            min_dist = 1e9
            #y[2] = min_dists.get(i, 0.0) +rmax
            #if min_dists.get(i, 0.0) < 1e-7:
             #   y[2] = rmax+0.1
            y[2] = 0.5*rmax
            # for line in crit_lines: 
             
            #     a, b, c = line
            #     dist = abs(a*disk[0] + b*disk[1] + c) / math.sqrt(a*a + b*b)
            
            #     if dist < min_dist :
            #         min_dist = dist-0.1
            #         if min_dist<0:
            #             min_dist = 0.0
            #         y[2] = max(min_dist+rmax-0.1, 0.0)
            #         if i == 9:
            #             print("Disk 9 is ", min_dist, " away from line ", line)

                
            m_disks.append(tuple(y))
            
    k, p = deepest_point_sweepline_lp(m_disks)
    print("maximum depth:", k)
    print("point:", p)
    print("verified depth:", depth_of_point(m_disks, p))

    plot_disks_solution(m_disks, p, k)
    #plot_radius_increase_vs_depth(disks, epsilon=0.25)
