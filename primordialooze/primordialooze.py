"""
PrimordialOoze is a genetic algorithm (GA) library for those who want something
very simple and don't want to spend time figuring out the more complicated libraries
that are out there.

See the README or the docstrings in this file for the documentation.
"""
import math
import multiprocessing
import numpy as np

class Simulation:
    """
    A GA simulation. The general workflow for this is:

    ```python
    import primordialooze as po
    import pandas
    import matplotlib.pyplot

    sim = po.Simulation(nagents, shape, fitnessfunction)
    bestagent, fitness = sim.run()

    # Dump and plot
    fname = "stats.csv"
    sim.dump_history_csv(fname)

    df = pandas.read_csv(fname)
    df = df.drop(['GenerationIndex'], axis=1)
    df.plot()
    plt.show()
    ```
    """

    def __init__(self, population, shape, fitnessfunc, *, seedfunc=None, selectionfunc=None,
                    crossoverfunc=None, mutationfunc=None, elitismfunc=None, nworkers=0,
                    max_agents_per_generation=None, min_agents_per_generation=None):
        """
        ## Args

        The following list contains the arguments that are needed. These do not have default values
        since the values for these will change dramatically depending on the problem.

        - **population**: The number of agents in the first generation. We will generate this many agents
                        in the initial generation, each of which is a Numpy Array of shape=`shape`.
                        They will be mutated according to `mutationfunc`, and evaluated each generation
                        by `fitnessfunc`.
        - **shape**: The shape of each agent in the population. Must be a list-like. The shape of the agents
                     must be a 1D array of whatever length like `(7,)`.
        - **fitnessfunc**: The function to use to evaluate the fitness of each agent in the generation.
                        Must have signature: `fitnessfunc(agent) -> scalar float`. This function
                        will be evaluated on every single agent in the gene pool at each generation.
                        If this function is slow, it probably makes sense to use multiprocessing, unless the
                        gene pool is quite small. See `nworkers`.

        ## Keyword Args

        These arguments contain (mostly) sensible defaults, but you should definitely make sure these
        defaults work for you. You will almost certainly want to change some of these to fit your problem.

        - **seedfunc**: The function to use to create the first generation of agents. The function must have
                            the signature `seedfunc() -> agent of shape 'shape'`. We call this function
                            `population` times. When `None`, defaults to uniform random
                            over the range [-1.0, 1.0) in each dimension.
        - **selectionfunc**: The function to use to select the agents that are allowed to breed to create the
                            next generation. Signature must be `selectionfunc(population, evaluations) -> selected_agents`,
                            where `population` is an n-dimensional array of shape (nagents, agent_length),
                            `evaluations` is an array of shape (nagents,); `evaluations[i]` contains
                            the fitness value for `population[i, :]`; `selected_agents` is an n-dimensional array
                            of shape (nagents_selected, agent_length), which must contain the selected agents.
                            `population` and `evaluations` are pre-sorted so that `population[0, :]`, corresponds
                            to `evalutaion[0]` and has the highest evaluation value. Agents which are not selected
                            are simply discarded, i.e., they will not appear in the next generation (unless randomly
                            created again as part of crossover/mutation).
                            If `None`, defaults to selecting the top ten percent.
        - **crossoverfunc**: Crossover function to use. Must have signature `crossoverfunc(agents) -> new_agents`,
                            where `agents` is an n-dimensional array of shape (nselected_agents, agent_length),
                            and where `new_agents` must be an n-dimensional array of shape (nagents, agent_length).
                            This function is applied after the selection function is used to determine which
                            agents will enter the new generation and this function is used exclusively on those
                            selected agents. Typically, `new_agents` will constitute the entirety of the new generation,
                            with one exception being if elitism is used (see below) and another exception being
                            if the mutation function adds new individuals to the gene pool, rather than just mutating
                            existing ones.
                            If `None`, defaults to 2-point crossover used on randomly selected pairs from the
                            breeding agents until `population` agents (or, if `elitismfunc` is None, `0.9 * population`).
        - **mutationfunc**: The function to use to apply mutations to the gene pool. The signature must be
                            `mutationfunc(agents) -> new_agents`, where `agents` is the value returned from
                            `crossoverfunc` and `new_agents` must be an n-dimensional array of shape (nagents, agent_length).
                            This function is applied to the result of `crossoverfunc`.
                            When `None`, defaults to setting each value in 0.05 of the agents to a random value,
                            where the random value is drawn from a Gaussian distribution of mean = the value being replaced
                            and stdev = 0.25.
        - **elitismfunc**: A function of signature `elitismfunc(generation_index) -> float in range [0.0, 1.0]`.
                            This function takes the index of the generation (0 for the first generation, 1 for the second, etc.)
                            and returns the fraction of top-performers to hold over as-is to the next generation.
                            The elites are duplicated and then, after the new
                            generation is created via the selectionfunc -> crossoverfunc -> mutationfunc pipeline, they are
                            reintroduced into the gene pool. This means that if the above pipeline generates 100 agents
                            and the elitism is set to take 10, the new generation will be composed of 110 agents. If this
                            is confusing, see `max_agents_per_generation` and `min_agents_per_generation`.
                            When `None`, defaults to a function that simply returns 0.1 (or 10%) of the gene pool regardless of the
                            generation.
        - **nworkers**: The number of processes to use to parallelize the fitness function. This will default to 0, which will
                            mean no parallelism at all. `None` will use the number of cores. Otherwise, should be a positive integer.
        - **max_agents_per_generation**: The maximum agents to allow into a generation. If the selection, crossover, mutation,
                                        and elitism functions are not handled properly, it is possible for the number of
                                        agents to change per generation. While this may be desired in some circumstances, it
                                        is often not. If this value is negative, we will allow the generations to grow to arbitrary
                                        size. If it is nonzero, after selection, crossover, mutation, and elitism, we will
                                        take all of the candidates as long as they do not number more than this value. If they do,
                                        we take this many at random.
                                        This value defaults to `None`, which means we use `population` as the max.
        - **min_agents_per_generation**: The minimum agents to allow making a new generation. If the selection, crossover, mutation,
                                        and elitism functions are not handled properly, it is possible for the number of
                                        agents to change per generation. While this may be desired in some circumstances, it
                                        is often not. If this value is negative or zero, we will allow the generations
                                        to shrink to zero, after which the simulation will stop. If it is nonzero, after selection,
                                        crossover, mutation, and elitism, we will cycle through the candidate agents in random
                                        order, duplicating them until this value is met. Note that we attempt to spread out the
                                        duplication evenly amongst all candidates.
                                        This value defaults to `None`, which means we use `population` as the min.
        """
        # Validate population
        if population <= 0:
            raise ValueError("Population must be > 0 but is {}".format(population))
        population = int(population)

        # Validate shape
        for i, dim in enumerate(shape):
            if dim <= 0:
                raise ValueError("Shape must contain no negative values, but contains {} at index {}".format(dim, i))
        try:
            _testagent = np.ndarray(shape=shape)
        except Exception:
            raise ValueError("There is something wrong with your shape parameter. It must be a list-like of integers greater than zero but is: {}.".format(shape))

        # Do not validate functions; may take too long

        # Validate nworkers
        if nworkers is None:
            nworkers = multiprocessing.cpu_count()
            if nworkers <= 0:
                raise ValueError("Something is wrong with multiprocessing.cpu_count(). Try passing in a number for nworkers instead of None.")
        elif nworkers < 0:
            raise ValueError("Nworkers must be zero (for no multiprocessing), None, or a positive integer, but is: {}".format(nworkers))
        nworkers = int(nworkers)

        # If we have negative max_agents, we actually want infinity
        if max_agents_per_generation is not None and max_agents_per_generation < 0:
            max_agents_per_generation = math.inf

        # We allow negative min_agents for compatibility with max_agents, but we just
        # interpret it as zero
        if min_agents_per_generation is not None and min_agents_per_generation < 0:
            min_agents_per_generation = 0

        self._initial_population_size = population
        self._shape = shape
        self._fitnessfunc = fitnessfunc
        self._seedfunc = self._default_seedfunc if seedfunc is None else seedfunc
        self._selectionfunc = self._default_selectionfunc if selectionfunc is None else selectionfunc
        self._crossoverfunc = self._default_crossoverfunc if crossoverfunc is None else crossoverfunc
        self._mutationfunc = self._default_mutationfunc if mutationfunc is None else mutationfunc
        self._elitismfunc = self._default_elitismfunc if elitismfunc is None else elitismfunc
        self._nworkers = nworkers
        self._max_agents_per_generation = population if max_agents_per_generation is None else max_agents_per_generation
        self._min_agents_per_generation = population if min_agents_per_generation is None else min_agents_per_generation
        self.statistics = []

        if self._max_agents_per_generation < self._min_agents_per_generation:
            raise ValueError("max_agents_per_generation {} is less than min_agents_per_generation {}".format(self._max_agents_per_generation, self._min_agents_per_generation))

    def dump_history_csv(self, fpath):
        """
        Saves this simulation's statistics as a CSV file at `fpath` in the form:

        ```
        Generation Index, Maximum, Minimum, Average
        ```
        """
        with open(fpath, 'w') as f:
            f.write("GenerationIndex, Maximum, Minimum, Average\n")
            for s in self.statistics:
                f.write("{}, {}, {}, {}\n".format(s.generationidx, s.maxval, s.minval, s.avgval))

    def run(self, niterations=100, fitness=None, printprogress=True):
        """
        Runs the constructed simulation.

        Either runs until `niterations` have passed, or runs until the best fitness is `fitness` or greater.
        Returns the best agent along with its fitness.

        ## Keyword Args

        - **niterations**: The number of iterations to run the simulation to. Defaults to 100. If `None`,
                           `fitness` will be used (and must not be None). If both this and `fitness` is
                           specified, we will stop as soon as one or the other condition is met.
        - **fitness**: The fitness level to converge on. As soon as one or more agents have this fitness level
                           or higher, the simulation will stop. Defaults to `None`. If `None` (the default),
                           `niterations` will be used (and must not be None). If this and `niterations` is
                           specified, we will stop as soon as one or the other condition is met.
        - **printprogress**: If `True` (the default), we will print a progress indication after each generation.

        ## Returns

        - The agent with the highest fitness score after the simulation ends.
        - The fitness of this agent.
        """
        # Validate args
        if niterations is None and fitness is None:
            raise ValueError("`niterations` and `fitness` must not both be None.")

        # First seed the gene pool
        listagents = [self._seedfunc() for _ in range(self._initial_population_size)]
        self._agents = np.array(listagents)
        self._fitnesses = np.zeros((self._initial_population_size,))

        iteridx = 0
        while not self._check_if_done(niterations, fitness, iteridx, printprogress):
            # Evaluate the gene pool
            self._fitnesses = self._evaluate_fitnesses()

            # Sort the fitnesses along with the agents and reverse
            sorted_indexes = np.argsort(self._fitnesses)[::-1]
            self._fitnesses = self._fitnesses[sorted_indexes]
            self._agents = self._agents[sorted_indexes]

            # Calculate statistics
            maxval = np.max(self._fitnesses)
            minval = np.min(self._fitnesses)
            avgval = np.mean(self._fitnesses)
            stats = Statistics(maxval, minval, avgval, iteridx)
            self.statistics.append(stats)

            # Elitism to duplicate the elites
            eliteratio = self._elitismfunc(iteridx)
            assert eliteratio <= 1.0, "The elitism function must produce a value between 0.0 and 1.0"
            assert eliteratio >= 0.0, "The elitism function must produce a value between 0.0 and 1.0"
            nelites = int(eliteratio * self._agents.shape[0])
            elites = np.copy(self._agents[0:nelites])
            elites = np.reshape(elites, (-1, self._agents.shape[1]))

            # Select breeding agents with selection function
            self._agents = self._selectionfunc(self._agents, self._fitnesses)
            assert len(self._agents.shape) == 2, "Selection function must return an ndarray of shape (nagents, agent_length), but has shape: {}".format(self._agents.shape)

            # Breed them using crossover
            self._agents = self._crossoverfunc(self._agents)
            assert len(self._agents.shape) == 2, "Crossover function must return an ndarray of shape (nagents, agent_length), but has shape: {}".format(self._agents.shape)

            # Mutate the results
            self._agents = self._mutationfunc(self._agents)
            assert len(self._agents.shape) == 2, "Mutation function must return an ndarray of shape (nagents, agent_length), but has shape: {}".format(self._agents.shape)

            # Construct the new gene pool from the mutation results and the elites
            ## Append any elites that were held over
            np.append(self._agents, elites, axis=0)

            ## Take as many as max_agents (but don't take more than we actually have), but randomized
            np.random.shuffle(self._agents)
            mx = min(self._max_agents_per_generation, self._agents.shape[0])
            self._agents = self._agents[0:mx, :]

            ## Now cycle through the agents, duplicating one at a time until we have at least min_agents
            i = 0
            while self._agents.shape[0] < self._min_agents_per_generation:
                self._agents = np.append(self._agents, np.expand_dims(self._agents[i], 0), axis=0)
                i += 1
                if i >= self._agents.shape[0]:
                    i = 0

            # Increment the generation index
            iteridx += 1

        # Sort the fitnesses along with the agents and reverse
        sorted_indexes = np.argsort(self._fitnesses)[::-1]
        self._fitnesses = self._fitnesses[sorted_indexes]
        self._agents = self._agents[sorted_indexes]

        if printprogress:
            print()

        # Return the fittest agent and its fitness score
        return self._agents[0, :], self._fitnesses[0]

    def _check_if_done(self, niterations, fitness, iteridx, prnt):
        """
        Returns `True` if the simulation is complete, `False` if not.
        """
        assert not (niterations is None and fitness is None), "niterations and fitness cannot both be None"
        if niterations is None:
            niterations = math.inf
        if fitness is None:
            fitness = math.inf

        # Check if the max fitness value is >= fitness
        finished_by_fitness = np.max(self._fitnesses) >= fitness

        # Check if iteridx + 1 >= niterations
        finished_by_iterations = (iteridx + 1) >= niterations

        # Now print an update if the user wants
        if prnt:
            maxsigns = 20
            if niterations != math.inf:
                # We are interested in niterations
                fraction_complete = iteridx / niterations
            else:
                # We are trying to converge on a particular value
                fraction_complete = np.max(self._fitnesses) / fitness
            npounds = int(fraction_complete * maxsigns)
            ndots = maxsigns - npounds
            msg = "Progress: [{}{}] Best Fitness: {} Worst Fitness: {}".format(
                "#" * npounds, "." * ndots, np.max(self._fitnesses), np.min(self._fitnesses)
            )
            print(msg, end="\r")

        return finished_by_fitness or finished_by_iterations

    def _evaluate_fitnesses(self):
        """
        Applies the fitness function to every agent currently in the gene pool
        and fills in self._fitnesses with this information.

        Will use multiprocessing if this class was initialized with it.
        """
        # If self._nworkers != 0, we are using multiprocessing, otherwise we aren't
        if self._nworkers == 0:
            # Don't use multiprocessing
            fitnesses = np.apply_along_axis(self._fitnessfunc, axis=1, arr=self._agents)
        else:
            # Make a pool
            # Split up the agents
            with multiprocessing.Pool(self._nworkers) as p:
                fitnesses = np.array(p.map(self._fitnessfunc, self._agents))
        return fitnesses

    def _default_seedfunc(self):
        """
        Default seed function to create the first generation of agents. Each time this is called, it creates
        a new agent from uniform random of shape `self._shape` over the values[-1.0, 1.0).
        """
        return np.random.uniform(low=-1.0, high=1.0, size=self._shape)

    def _default_selectionfunc(self, population, fitnesses):
        """
        Default selection function for selecting agents allowed to breed to create the next generation.
        Simply takes the top 10% of the given population and return them.

        Population and evaluations are pre-sorted so that index 0 is the fittest.

        This is guaranteed to take at least one agent.
        """
        tenpercent = int(population.shape[0] * 0.1)
        if tenpercent < 1:
            tenpercent = 1

        return self._agents[0:tenpercent, :]

    def _default_crossoverfunc(self, agents):
        """
        Applies 2-point crossover to the agents to generate children. Mating pairs are chosen at random
        without replacement until the next generation has `self._initial_population_size` agents in it
        unless self._elitismfunc == self._default_elitismfunc, in which case we only go to
        0.9 * `self._initial_population_size` agents (since 0.1 are kept by the elitism function).
        Once all agents have been chosen, we do it again. We repeat this process until the next generation
        has the right number of agents in it.

        Always mates at least one pair, unless the population is currently 1, in which case we simply
        return that agent unchanged.
        """
        nagents = self._initial_population_size

        # Determine how many agents to mate/create (we create one agent per parent - two per pair)
        if self._elitismfunc == self._default_elitismfunc:
            nagents = int(0.9 * nagents)

        if nagents < 2:
            nagents = 2

        if agents.shape[0] < 2:
            # We can't do much with less than 2 agents
            return agents

        # Create agents by choosing two agents randomly and swapping two parts of them
        created_agents = []
        so_far_mated = set()
        remaining = [i for i in range(agents.shape[0])]
        while len(created_agents) < nagents:
            # Draw a random index from the remaining indexes
            idx1 = np.random.choice(remaining)
            remaining.remove(idx1)

            # If that was the last one, we need to dump so-far-mated and
            # start going through them again
            if not remaining:
                remaining = list(so_far_mated)
                so_far_mated.clear()

            # Draw another
            idx2 = np.random.choice(remaining)
            remaining.remove(idx2)

            # Mate the two
            newa, newb = self._mate_two_agents(agents[idx1, :], agents[idx2, :])

            # Add the result to the list of agents we are going to return
            created_agents.append(newa)
            created_agents.append(newb)

            # Add to the set of so-far-mated
            so_far_mated.add(idx1)
            so_far_mated.add(idx2)

            # If we have run out of items in remaining, dump so-far-mated
            # and start cycling back through them
            if not remaining:
                remaining = list(so_far_mated)
                so_far_mated.clear()

        return np.array(created_agents)

    def _default_mutationfunc(self, agents):
        """
        Applies Gaussian noise to each value in 5% of agents, where mean=value and stdev=0.25.
        Always mutates at least one individual.
        """
        # TODO: Test that we always mutate at least on agent.
        # TODO: Test that the underlying distribution for a bunch of mutated points is gaussian, mean=x, stdev=0.25.
        nagents = int(0.05 * agents.shape[0])
        if nagents < 1:
            nagents = 1

        idxs = np.random.choice(agents.shape[0], size=nagents, replace=False)
        agents[idxs, :] = np.random.normal(agents[idxs, :], 0.25)
        return agents

    def _default_elitismfunc(self, genindex):
        """
        Simply returns 0.1, regardless of `genindex`. This means that 10% of the gene pool (the top
        10% specifically) will be re-injected into the next generation unchanged.
        """
        return 0.1

    def _mate_two_agents(self, a1, a2):
        """
        Returns two new agents after mating a1 and a2 via 2-point crossover.
        """
        assert len(a1.shape) == 1, "a1 must be a row vector, but has shape {}".format(a1.shape)
        assert len(a2.shape) == 1, "a2 must be a row vector, but has shape {}".format(a2.shape)

        # Find a random index
        i = np.random.choice(a1.shape[0])

        # Find another random index
        j = np.random.choice(a1.shape[0])

        # Sort them
        low, high = sorted([i, j])

        # Take a1[0:low] and a2[0:low] and swap them
        a1_up_to_low = a1[0:low]
        a2_up_to_low = a2[0:low]
        a1[0:low] = a2_up_to_low
        a2[0:low] = a1_up_to_low

        # Take a1[high:] and a2 [high:] and swap them
        a1_from_high = a1[high:]
        a2_from_high = a2[high:]
        a1[high:] = a2_from_high
        a2[high:] = a1_from_high

        return a1, a2

class Statistics:
    def __init__(self, maxval, minval, avgval, generationidx):
        self.maxval = maxval
        self.minval = minval
        self.avgval = avgval
        self.generationidx = generationidx

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    import pandas

    nagents = 1000
    sim = Simulation(nagents, shape=(2,), fitnessfunc=lambda agent: (-1.2* agent[0]**2) - (0.75 * agent[1]**2) + 5.0)
    best, value = sim.run(niterations=10000, fitness=4.99999)
    msg = "(best, value): ({}, {})".format(best, value)
    fname = "stats.csv"
    sim.dump_history_csv(fname)

    df = pandas.read_csv(fname)
    df = df.drop(['GenerationIndex'], axis=1)
    df.plot()
    plt.show()
