from random import sample
from math import sqrt, pow


def _get_cluster_assignment(points, centroids):
    """
    This function accepts:
    points: A list of 2D points (tuples) [(1, 2), (2, 3), ...] etc.
    centroids: A dictionary containing the locations of
    centroids {0: (1, 1), 1: (2, 4)} etc.

    and returns a dictionary where the key corresponds to
    a single point, and the value corresponds to the cluster
    that point belongs to (that with the min euclidian distance).
    This is a private helper function used within the k_means
    function defined below.
    """
    cluster_assignment = {}
    for point in points:
        min_dist = None
        p_x = point[0]
        p_y = point[1]
        for cluster, c_loc in centroids.items():
            c_x = c_loc[0]
            c_y = c_loc[1]
            euc_dist = sqrt(pow(p_x - c_x, 2) + pow(p_y - c_y, 2))
            if min_dist is None or euc_dist < min_dist:
                cluster_assignment[point] = cluster
                min_dist = euc_dist
    return cluster_assignment


def _get_centroids(cluster_assignment):
    """
    This function accepts
    cluster_assignment: A dictionary that maps a point to
    a particular cluster_id [0, k-1]

    and returns a dictionary containing the centroid locations
    after the locations have been updated to take into account the
    cluster assignment. This is a private helper function used 
    within the k_means function defined below.
    """
    # Find the points that belong to each group in a dictionary.
    rev_index = {}
    for point, cluster_id in cluster_assignment.items():
        if cluster_id not in rev_index:
            rev_index[cluster_id] = [point]
        else:
            rev_index[cluster_id].append(point)

    # Compute the recomputed centroids given the cluster assignment.
    centroids = {}
    for cluster_id, points in rev_index.items():
        num_points = len(points)
        x_mean = sum([p[0] for p in points]) / num_points
        y_mean = sum([p[1] for p in points]) / num_points
        centroids[cluster_id] = (x_mean, y_mean)

    return centroids


def k_means(points, k):
    """
    This functions accepts:
    points: A list of 2D points (tuples) [(1, 2), (2, 3), ...] etc.
    k: The number of cluster groups to assign points to

    and returns a dictionary containing the groups each provided point
    belongs to (an integer in [0, k-1]), and the location of the 
    centroids at the end of iteration. It uses the k-means 
    clustering algorithm found at: 
    https://en.wikipedia.org/wiki/K-means_clustering

    Note: an exception is thrown if k exceeds the number
    of points in points.

    EXAMPLE:
    points = [(1, 1), (2, 2)]
    k_means(points, 1) returns
        cluster_locations = {(1, 1): 0, (2, 2): 0}
        centroids = {0: (1.5, 1.5)}

    Thus calling cluster_locations[point] gives you the integer
    cluster assignment, and centroids[cluster_id] gives you
    the location of the centroid for that cluster_id.
    """

    # Check to ensure that we pass a valid value for num_groups.
    # Also check to ensure k > 0
    assert k > 0 and k <= len(points)

    # Select k random points to initialize location of centroids.
    centroids = {}
    init_sample = sample(points, k)
    for id, point in enumerate(init_sample):
        centroids[id] = point

    # Iterate until points are no longer assigned to a new cluster.
    # Note that we use an infinite loop since K-means is guaranteed
    # to converge. Note further that we keep track of the last cluster
    # assignment - if this doesn't change between iterations, then we
    # have converged to a local optimum and we break out of the loop.
    last_cluster_assignment = None
    while True:
        cluster_assignment = _get_cluster_assignment(points, centroids)
        centroids = _get_centroids(cluster_assignment)
        if cluster_assignment == last_cluster_assignment:
            break
        else:
            last_cluster_assignment = cluster_assignment

    return cluster_assignment, centroids
