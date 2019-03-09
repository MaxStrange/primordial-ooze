"""
Tests for the PrimordialOoze library.
"""
import numpy as np
import os
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
        self.assertGreaterEqual(best[0], -1.0, msg=msg)
        self.assertLessEqual(best[0], 1.0, msg=msg)
        self.assertGreaterEqual(best[1], -1.0, msg=msg)
        self.assertLessEqual(best[1], 1.0, msg=msg)
        self.assertAlmostEqual(value, 5.0, places=3, msg=msg)

    def test_dump_csv(self):
        """
        Test that we can solve a simple 2D upside-down parabola problem.
        """
        nagents = 10000
        sim = po.Simulation(nagents, shape=(2,), fitnessfunc=self.fitness)
        best, value = sim.run(niterations=10000, fitness=4.99999)
        msg = "(best, value): ({}, {})".format(best, value)
        self.assertGreaterEqual(best[0], -1.0, msg=msg)
        self.assertLessEqual(best[0], 1.0, msg=msg)
        self.assertGreaterEqual(best[1], -1.0, msg=msg)
        self.assertLessEqual(best[1], 1.0, msg=msg)
        self.assertAlmostEqual(value, 5.0, places=3, msg=msg)
        fname = "stats.csv"
        sim.dump_history_csv(fname)
        os.remove(fname)

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
        self.assertAlmostEqual(value, 5.0, places=3, msg=msg)

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

    def test_crossoverfunction(self):
        """
        Test passing in our own crossover function. Our custom function does nothing at all, so we rely
        on elitism and mutations to reach convergence.
        """
        def xover(agents):
            return agents

        shape = (2,)
        nagents = 1000
        sim = po.Simulation(nagents, shape=shape, fitnessfunc=self.fitness, crossoverfunc=xover)
        best, value = sim.run(niterations=1000, fitness=4.99999)
        msg = "(best, value): ({}, {})".format(best, value)
        self.assertGreaterEqual(best[0], -1.0, msg=msg)
        self.assertLessEqual(best[0], 1.0, msg=msg)
        self.assertGreaterEqual(best[1], -1.0, msg=msg)
        self.assertLessEqual(best[1], 1.0, msg=msg)
        self.assertAlmostEqual(value, 5.0, places=3, msg=msg)

    def test_mutationfunction(self):
        """
        Test passing in our own mutation function. Our custom function mutates randomly chosen
        agents by replacing them with uniform random from [-1.0, 1.0).
        """
        def mutate(agents):
            nmutants = int(0.1 * agents.shape[0])
            mutants = np.random.uniform(-1.0, 1.0, size=(nmutants, agents.shape[1]))
            np.random.shuffle(agents)
            agents[0:nmutants, :] = mutants[:, :]
            return agents

        shape = (2,)
        nagents = 1000
        sim = po.Simulation(nagents, shape=shape, fitnessfunc=self.fitness, mutationfunc=mutate)
        best, value = sim.run(niterations=1000, fitness=4.99999)
        msg = "(best, value): ({}, {})".format(best, value)
        self.assertGreaterEqual(best[0], -1.0, msg=msg)
        self.assertLessEqual(best[0], 1.0, msg=msg)
        self.assertGreaterEqual(best[1], -1.0, msg=msg)
        self.assertLessEqual(best[1], 1.0, msg=msg)
        self.assertAlmostEqual(value, 5.0, places=3, msg=msg)

    def test_no_elitism(self):
        """
        Test having no elitism.
        """
        def elitism(idx):
            return 0.0

        shape = (2,)
        nagents = 1000
        sim = po.Simulation(nagents, shape=shape, fitnessfunc=self.fitness, elitismfunc=elitism)
        best, value = sim.run(niterations=1000, fitness=4.99999)
        msg = "(best, value): ({}, {})".format(best, value)
        self.assertGreaterEqual(best[0], -1.0, msg=msg)
        self.assertLessEqual(best[0], 1.0, msg=msg)
        self.assertGreaterEqual(best[1], -1.0, msg=msg)
        self.assertLessEqual(best[1], 1.0, msg=msg)
        self.assertAlmostEqual(value, 5.0, places=3, msg=msg)

    def test_decaying_elitism(self):
        """
        Test having an elitism function that decays to nothing over time.
        """
        def elitism(idx):
            return -1.0 * np.tanh(idx) + 1.0

        shape = (2,)
        nagents = 1000
        sim = po.Simulation(nagents, shape=shape, fitnessfunc=self.fitness, elitismfunc=elitism)
        best, value = sim.run(niterations=1000, fitness=4.99999)
        msg = "(best, value): ({}, {})".format(best, value)
        self.assertGreaterEqual(best[0], -1.0, msg=msg)
        self.assertLessEqual(best[0], 1.0, msg=msg)
        self.assertGreaterEqual(best[1], -1.0, msg=msg)
        self.assertLessEqual(best[1], 1.0, msg=msg)
        self.assertAlmostEqual(value, 5.0, places=3, msg=msg)

    def test_ramping_elitism(self):
        """
        Test using an elitism function that ramps up from nothing to most of the population.
        """
        def elitism(idx):
            return np.tanh(idx)

        shape = (2,)
        nagents = 1000
        sim = po.Simulation(nagents, shape=shape, fitnessfunc=self.fitness, elitismfunc=elitism)
        best, value = sim.run(niterations=1000, fitness=4.99999)
        msg = "(best, value): ({}, {})".format(best, value)
        self.assertGreaterEqual(best[0], -1.0, msg=msg)
        self.assertLessEqual(best[0], 1.0, msg=msg)
        self.assertGreaterEqual(best[1], -1.0, msg=msg)
        self.assertLessEqual(best[1], 1.0, msg=msg)
        self.assertAlmostEqual(value, 5.0, places=3, msg=msg)

    def test_parallelism_works(self):
        """
        Simply tests that we can converge while using multiprocessing on the fitness function.
        """
        # Note that there is an awful lot of overhead in creating a process pool,
        # so it really only makes sense to do this for large numbers of agents
        # and small numbers of iterations
        nagents = 10000
        sim = po.Simulation(nagents, shape=(2,), fitnessfunc=parallelizable_function, nworkers=None)  # nworkers=None=ncpus
        best, value = sim.run(niterations=100, fitness=4.99999)
        msg = "(best, value): ({}, {})".format(best, value)
        self.assertGreaterEqual(best[0], -1.0, msg=msg)
        self.assertLessEqual(best[0], 1.0, msg=msg)
        self.assertGreaterEqual(best[1], -1.0, msg=msg)
        self.assertLessEqual(best[1], 1.0, msg=msg)
        self.assertAlmostEqual(value, 5.0, places=3, msg=msg)

    def test_parallelism_works_again(self):
        """
        Simply tests that we can converge while using multiprocessing on the fitness function.
        """
        # Note that there is an awful lot of overhead in creating a process pool,
        # so it really only makes sense to do this for large numbers of agents
        # and small numbers of iterations
        nagents = 10000
        fitfunc = ParallelizableCallableClass()
        sim = po.Simulation(nagents, shape=(2,), fitnessfunc=fitfunc, nworkers=None)  # nworkers=None=ncpus
        best, value = sim.run(niterations=100, fitness=4.99999)
        msg = "(best, value): ({}, {})".format(best, value)
        self.assertGreaterEqual(best[0], -1.0, msg=msg)
        self.assertLessEqual(best[0], 1.0, msg=msg)
        self.assertGreaterEqual(best[1], -1.0, msg=msg)
        self.assertLessEqual(best[1], 1.0, msg=msg)
        self.assertAlmostEqual(value, 5.0, places=3, msg=msg)

    def test_min_num_agents(self):
        """
        Test that the final number of agents is not less than min_num_agents, even if mutations remove
        a bunch from the population each time.
        """
        def mutate(agents):
            """We just return the first 3 agents"""
            return agents[0:3, :]

        shape = (2,)
        nagents = 1000
        sim = po.Simulation(nagents, shape=shape, fitnessfunc=self.fitness, mutationfunc=mutate, min_agents_per_generation=75)
        _, _ = sim.run(niterations=1000, fitness=4.99999)
        self.assertEqual(sim._agents.shape[0], 75)

    def test_max_num_agents(self):
        """
        Test that the final number of agents is not greater than max_num_agents, even if mutations add
        new agents each time.
        """
        def mutate(agents):
            """Add a new agent each time"""
            agents = np.append(agents, np.array([[1, 2]]), axis=0)
            return agents

        shape = (2,)
        nagents = 1000
        sim = po.Simulation(nagents, shape=shape, fitnessfunc=self.fitness, mutationfunc=mutate, max_agents_per_generation=75, min_agents_per_generation=25)
        _, _ = sim.run(niterations=1000, fitness=4.99999)
        self.assertEqual(sim._agents.shape[0], 75)

    def test_statistics(self):
        """
        Test that we keep stats.
        """
        nagents = 100
        sim = po.Simulation(nagents, shape=(2,), fitnessfunc=self.fitness)
        best, value = sim.run(niterations=531, fitness=None)
        bestagents = sim.best_agents
        self.assertEqual(len(bestagents), 531)
        self.assertEqual(best.shape, bestagents[0].shape)
        for agent in bestagents:
            self.assertEqual(agent.shape, best.shape)
        for v1, v2 in zip(best, bestagents[-1]):
            self.assertAlmostEqual(v1, v2, places=2)

        self.assertAlmostEqual(value, sim.statistics[-1].maxval, places=2)
        lastidx = 0
        for stats in sim.statistics:
            self.assertGreaterEqual(stats.maxval, stats.avgval)
            self.assertGreaterEqual(stats.maxval, stats.minval)
            self.assertGreaterEqual(stats.avgval, stats.minval)
            if stats.generationidx != 0:
                self.assertEqual(stats.generationidx, lastidx + 1)
            lastidx = stats.generationidx

def parallelizable_function(agent):
    """
    This is an upside-down parabola in 2D with max value at (z = (x, y) = 5 = (0, 0)).
    """
    x = agent[0]
    y = agent[1]
    return (-1.2 * x**2) - (0.75 * y**2) + 5.0

class ParallelizableCallableClass:
    def __init__(self):
        self.a = -1.2
        self.b = 0.75
        self.c = 5.0

    def __call__(self, agent):
        x = agent[0]
        y = agent[1]
        return (self.a * x**2) - (self.b * y**2) + self.c

if __name__ == '__main__':
    unittest.main()
