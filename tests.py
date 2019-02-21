"""
Tests for the PrimordialOoze library.
"""
import unittest
import primordialooze as po

class TestOoze(unittest.TestCase):
    """
    All the tests for the PrimordialOoze package.
    """
    def test_instantiate(self):
        """
        Merely test whether we can successfully instantiate a Simulation instance.
        """
        _ = po.Simulation()

    def test_solve_conic(self):
        """
        Test that we can solve a simple 2D upside-down parabola problem.
        """
        def fitness(agent):
            """
            This is an upside-down parabola in 2D with max value at (z = (x, y) = 5 = (0, 0)).
            """
            x = agent[0]
            y = agent[1]
            return (-1.2 * x**2) - (0.75 * y**2) + 5

        nagents = 100
        sim = po.Simulation(nagents, fitnessfunc=fitness)
        best, value = sim.run()
        self.assertAlmostEqual(best[0], 0.0, places=4)
        self.assertAlmostEqual(best[1], 0.0, places=4)
        self.assertAlmostEqual(value, 5.0, places=4)

if __name__ == '__main__':
    unittest.main()
