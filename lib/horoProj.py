"""Horocycle projection utils (Poincare model)."""
## https://github.com/HazyResearch/HoroPCA

import torch

# stable numerical results(if do not need to take gradient, MIN_NORM could take 1e-15)
MIN_NORM = 1e-6


def busemann(x, p, keepdim=True):
    """
    x: (..., d)
    p: (..., d)

    Returns: (..., 1) if keepdim==True else (...)
    """

    xnorm = x.norm(dim=-1, p=2, keepdim=True)
    pnorm = p.norm(dim=-1, p=2, keepdim=True)
    p = p / pnorm.clamp_min(MIN_NORM)
    num = torch.norm(p - x, dim=-1, keepdim=True) ** 2
    den = (1 - xnorm ** 2).clamp_min(MIN_NORM)
    ans = torch.log((num / den).clamp_min(MIN_NORM))
    if not keepdim:
        ans = ans.squeeze(-1)
    return ans


def circle_intersection_(r, R):
    """ Computes the intersection of a circle of radius r and R with distance 1 between their centers.

    Returns:
    x - distance from center of first circle
    h - height off the line connecting the two centers of the two intersection pointers
    """

    x = (1.0 - R ** 2 + r ** 2) / 2.0
    s = (r + R + 1) / 2.0
    sq_h = (s * (s - r) * (s - R) * (s - 1)).clamp_min(MIN_NORM)
    h = torch.sqrt(sq_h) * 2.0
    return x, h


def circle_intersection(c1, c2, r1, r2):
    """ Computes the intersections of a circle centered at ci of radius ri.

    c1, c2: (..., d)
    r1, r2: (...)
    """

    d = torch.norm(c1 - c2)  # (...)
    x, h = circle_intersection_(r1 / d.clamp_min(MIN_NORM), r2 / d.clamp_min(MIN_NORM))  # (...)
    x = x.unsqueeze(-1)
    h = h.unsqueeze(-1)
    center = x * c2 + (1 - x) * c1  # (..., d)
    radius = h * d  # (...)

    # The intersection is a hypersphere of one lower dimension, intersected with the plane
    # orthogonal to the direction c1->c2
    # In general, you can compute this with a sort of higher dimensional cross product?
    # For now, only 2 dimensions

    ortho = c2 - c1  # (..., d)
    assert ortho.size(-1) == 2
    direction = torch.stack((-ortho[..., 1], ortho[..., 0]), dim=-1)
    direction = direction / torch.norm(direction, keepdim=True).clamp_min(MIN_NORM)
    return center + radius.unsqueeze(-1) * direction  # , center - radius*direction


def busemann_to_horocycle(p, t):
    """ Find the horocycle corresponding to the level set of the Busemann function to ideal point p with value t.

    p: (..., d)
    t: (...)

    Returns:
    c: (..., d)
    r: (...)
    """
    # Busemann_p(x) = d means dist(0, x) = -d
    q = -torch.tanh(t / 2).unsqueeze(-1) * p
    c = (p + q) / 2.0
    r = torch.norm(p - q, dim=-1) / 2.0
    return c, r


def sphere_intersection(c1, r1, c2, r2):
    """ Computes the intersections of a circle centered at ci of radius ri.

    c1, c2: (..., d)
    r1, r2: (...)

    Returns:
    center, radius such that the intersection of the two spheres is given by
    the intersection of the sphere (c, r) with the hyperplane orthogonal to the direction c1->c2
    """

    d = torch.norm(c1 - c2, dim=-1)  # (...)
    x, h = circle_intersection_(r1 / d.clamp_min(MIN_NORM), r2 / d.clamp_min(MIN_NORM))  # (...)
    x = x.unsqueeze(-1)
    center = x * c2 + (1 - x) * c1  # (..., d)
    radius = h * d  # (...)
    return center, radius


def sphere_intersections(c, r):
    """ Computes the intersection of k spheres in dimension d.

    c: list of centers (..., k, d)
    r: list of radii (..., k)

    Returns:
    center: (..., d)
    radius: (...)
    ortho_directions: (..., d, k-1)
    """

    k = c.size(-2)
    assert k == r.size(-1)

    ortho_directions = []
    center = c[..., 0, :]  # (..., d)
    radius = r[..., 0]  # (...)
    for i in range(1, k):
        center, radius = sphere_intersection(center, radius, c[..., i, :], r[..., i])
        ortho_directions.append(c[..., i, :] - center)
    ortho_directions.append(torch.zeros_like(center))  # trick to handle the case k=1
    ortho_directions = torch.stack(ortho_directions, dim=-1)  # (..., d, k-1) [last element is 0]
    return center, radius, ortho_directions


# 2D projections
def project_kd(p, x, keep_ambient=True):
    """ Project n points in dimension d onto 'direction' spanned by k ideal points
    p: (..., k, d) ideal points
    x: (..., n, d) points to project

    Returns:
    projection_1: (..., n, s) where s = d if keep_ambient==True otherwise s = k
    projection_2: same as projection_1. this is guaranteed to be the ideal point in the case k = 1
    p: the ideal points
    """

    if len(p.shape) < 2:
        p = p.unsqueeze(0)
    if len(x.shape) < 2:
        x = x.unsqueeze(0)
    k = p.size(-2)
    d = x.size(-1)
    assert d == p.size(-1)
    busemann_distances = busemann(x.unsqueeze(-2), p.unsqueeze(-3), keepdim=False)  # (..., n, k)
    c, r = busemann_to_horocycle(p.unsqueeze(-3), busemann_distances)  # (..., n, k, d) (..., n, k)
    c, r, ortho = sphere_intersections(c, r)  # (..., n, d) (..., n) (..., n, d, k-1)
    # we are looking for a vector spanned by the k ideal points, orthogonal to k-1 given vectors
    # i.e. x @ p @ ortho = 0
    if ortho is None:
        direction = torch.ones_like(busemann_distances)  # (..., n, k)
        # print("ortho is None")
    else:
        ortho = ortho.detach()  # for gradient based method
        a = torch.matmul(p.unsqueeze(-3), ortho)  # (..., n, k, k-1) = (..., n, k, d) @ (..., n, d, k-1)
        u, s, v = torch.svd(a, some=False)  # a = u s v^T
        direction = u[..., -1]  # (..., n, k)
    direction = direction @ p  # (..., n, d)
    direction = direction / torch.norm(direction, dim=-1, keepdim=True).clamp_min(MIN_NORM)

    projection_1 = c - r.unsqueeze(-1) * direction
    projection_2 = c + r.unsqueeze(-1) * direction
    if not keep_ambient:
        _, _, v = torch.svd(p, some=False)  # P = USV^T => PV = US so last d-k columns of PV are 0
        projection_1 = (projection_1 @ v)[..., :k]
        projection_2 = (projection_2 @ v)[..., :k]
        p = (p @ v)[..., :k]

    # if torch.sum(torch.norm(projection_1, dim=-1) ** 2)>torch.sum(torch.norm(projection_2, dim=-1) ** 2):
    #     return projection_2@p.T
    # else:
    #     return projection_1@p.T
    return projection_1@p.T







