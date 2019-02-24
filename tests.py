"""
Tests for the PrimordialOoze library.
"""
import numpy as np
import primordialooze as po
import unittest

class TestOoze(unittest.TestCase):
    """
    All the tests for the PrimordialOoze package.
    """
    def fitness(self, agent):
        """
        This is an upside-down parabola in 2D with max value at (z = (x, y) = 5 = (0, 0)).
        """
        x = agent[0]
        y = agent[1]
        return (-1.2 * x**2) - (0.75 * y**2) + 5.0

    def test_instantiate(self):
        """
        Merely test whether we can successfully instantiate a Simulation instance.
        """
        def fitnessfunc(agent):
            return 0.0

        _ = po.Simulation(100, (2,), fitnessfunc)

    def test_solve_conic_with_defaults(self):
        """
        Test that we can solve a simple 2D upside-down parabola problem.
        """
        nagents = 100
        sim = po.Simulation(nagents, shape=(2,), fitnessfunc=self.fitness)
        best, value = sim.run(niterations=10000, fitness=4.99999)
        msg = "(best, value): ({}, {})".format(best, value)
        self.assertGreaterEqual(best[0], -0.1, msg=msg)
        self.assertLessEqual(best[0], 1.0, msg=msg)
        self.assertGreaterEqual(best[1], -0.1, msg=msg)
        self.assertLessEqual(best[1], 1.0, msg=msg)
        self.assertAlmostEqual(value, 5.0, places=4, msg=msg)

    def test_seedfunction(self):
        """
        Test passing in our own seed function.
        """
        class Seedfunc:
            def __init__(self, shape):
                self._idx = 0
                self._shape = shape

            def __call__(self):
                agent = np.random.normal(0.0, self._idx / 10.0, size=self._shape)
                self._idx += 1
                return agent

        shape = (2,)
        nagents = 100
        seedfuncinstance = Seedfunc(shape)
        sim = po.Simulation(nagents, shape=shape, fitnessfunc=self.fitness, seedfunc=seedfuncinstance)
        best, value = sim.run(niterations=10000, fitness=4.99999)
        msg = "(best, value): ({}, {})".format(best, value)
        self.assertGreaterEqual(best[0], -1.0, msg=msg)
        self.assertLessEqual(best[0], 1.0, msg=msg)
        self.assertGreaterEqual(best[1], -1.0, msg=msg)
        self.assertLessEqual(best[1], 1.0, msg=msg)
        self.assertAlmostEqual(value, 5.0, places=4, msg=msg)

    def test_selectionfunction_take_one(self):
        """
        Test passing in a custom selection function that just takes the single top performer.
        """
        def selectionfunction(agents, fitnesses):
            return np.expand_dims(agents[0, :], 0)

        shape = (2,)
        nagents = 100
        sim = po.Simulation(nagents, shape=shape, fitnessfunc=self.fitness, selectionfunc=selectionfunction)
        _, _ = sim.run(niterations=100, fitness=4.99999)
        # We are just testing to make sure this doesn't crash. Convergence with this selection function would
        # take forever

if __name__ == '__main__':
    unittest.main()
