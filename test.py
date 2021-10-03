import unittest
from k_means import (
    _get_cluster_assignment,
    _get_centroids,
    k_means
)


class TestKMeans(unittest.TestCase):

    def test_single_group_assignment(self):
        centroids = {
            0: (1, 1)
        }
        points = [(1, 1)]
        res = _get_cluster_assignment(points, centroids)
        self.assertEqual(res, {(1, 1): 0})

    def test_multiple_groups_assignment(self):
        centroids = {
            0: (1, 1),
            1: (2, 2),
            2: (3, 3)
        }
        points = [(1, 1.01), (3, 3.03), (2, 2.01), (4, 4)]
        res = _get_cluster_assignment(points, centroids)
        corr_res = {
            (1, 1.01): 0,
            (2, 2.01): 1,
            (3, 3.03): 2,
            (4, 4): 2
        }
        self.assertEqual(res, corr_res)

    def test_single_point_centroid(self):
        cluster_assignment = {(1, 1): 0}
        corr_res = {0: (1, 1)}
        res = _get_centroids(cluster_assignment)
        self.assertEqual(res, corr_res)

    def test_two_points_centroid(self):
        cluster_assignment = {(1, 1): 0, (1, 1.1): 0, (2, 2): 1, (2, 2.1): 1}
        corr_res = {0: (1, 1.05), 1: (2, 2.05)}
        res = _get_centroids(cluster_assignment)
        self.assertEqual(res, corr_res)

    def test_single_cluster_k_means(self):
        points = [(1, 1), (2, 2)]
        corr_ca = {(1, 1): 0, (2, 2): 0}
        corr_centroids = {0: (1.5, 1.5)}
        res_ca, res_centroids = k_means(points, 1)
        self.assertEqual(res_ca, corr_ca)
        self.assertEqual(res_centroids, corr_centroids)


if __name__ == '__main__':
    unittest.main()
