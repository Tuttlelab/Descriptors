"""
tests/test_descriptors.py

Unit tests for shape-descriptors package modules and mathematical helper functions.
Runs with standard unittest (`python -m unittest discover tests`) or pytest.
"""

import unittest
import numpy as np
from descriptors.vfi import calculate_sphericity
from descriptors.centering import find_largest_cluster

class TestDescriptors(unittest.TestCase):

    def test_sphericity_sphere(self):
        """Verify that points distributed on a sphere yield high sphericity (~1.0)."""
        phi = np.linspace(0, np.pi, 20)
        theta = np.linspace(0, 2 * np.pi, 20)
        phi, theta = np.meshgrid(phi, theta)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T

        sphericity = calculate_sphericity(points)
        self.assertGreater(sphericity, 0.70)

    def test_sphericity_too_few_points(self):
        """Verify handling of < 4 points."""
        points = np.array([[0, 0, 0], [1, 1, 1]])
        sphericity = calculate_sphericity(points)
        self.assertEqual(sphericity, 0.0)

    def test_find_largest_cluster(self):
        """Verify DBSCAN cluster identification."""
        cluster1 = np.random.randn(50, 3)
        cluster2 = np.random.randn(10, 3) + 100.0
        points = np.vstack([cluster1, cluster2])

        mask = find_largest_cluster(points, eps=5.0, min_samples=3)
        self.assertIsNotNone(mask)
        self.assertEqual(np.sum(mask), 50)

if __name__ == '__main__':
    unittest.main()
