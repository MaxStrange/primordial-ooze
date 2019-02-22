"""
PrimordialOoze is a genetic algorithm (GA) library for those who want something
very simple and don't want to spend time figuring out the more complicated libraries
that are out there.

See the README or the docstrings in this file for the documentation.
"""
import multiprocessing
import numpy as np

class Simulation:
    """
    A GA simulation. The general workflow for this is:

    ```python
    import primordialooze as po

    sim = po.Simulation(nagents, shape, fitnessfunction)
    bestagent, fitness = sim.run()
    print(sim.results)
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
        - **shape**: The shape of each agent in the population. Must be a list-like.
        - **fitnessfunc**: The function to use to evaluate the fitness of each agent in the generation.
                        Must have signature: `def fitnessfunc(agent) -> scalar float`. This function
                        will be evaluated on every single agent in the gene pool at each generation.
                        If this function is slow, it probably makes sense to use multiprocessing.
                        See `nwokers`.

        ## Keyword Args

        These arguments contain (mostly) sensible defaults, but you should definitely make sure these
        defaults work for you. You will almost certainly want to change some of these to fit your problem.

        - **seedfunc**: The function to use to create the first generation of agents. The function must have
                        the signature `def seedfunc() -> agent of shape 'shape'`. We call this function
                        `population` times. When `None`, defaults to uniform random
                        over the range [-1.0, 1.0) in each dimension.
        - **selectionfunc**: The function to use to select the agents that are allowed to breed to create the
                            next generation. Signature must be `def selectionfunc(population, evaluations) -> selected_agents`,
                            where `population` is an n-dimensional array of shape (nagents, *agent_shape),
                            `evaluations` is an array of shape (nagents,); `evaluations[i]` contains
                            the fitness value for `population[i]`; `selected_agents` is an n-dimensional array
                            of shape (nagents_selected, *agent_shape), which must contain the selected agents.
                            `population` and `evaluations` are pre-sorted so that `population[0]`, corresponds
                            to `evalutaion[0]` and has the highest evaluation value. Agents which are not selected
                            are simply discarded, i.e., they will not appear in the next generation (unless randomly
                            created as part of crossover/mutation).
                            When `None`, defaults to selecting the top ten percent.
        - **crossoverfunc**: Crossover function to use. Must have signature `crossoverfunc(agents) -> new_agents`,
                            where `agents` is an n-dimensional array of shape (nselected_agents, *agent_shape),
                            and where `new_agents` must be an n-dimensional array of shape (nagents, *agent_shape).
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
                            `crossoverfunc` and `new_agents` must be an n-dimensional array of shape (nagents, *agent_shape).
                            This function is applied to the result of `crossoverfunc`.
                            When `None`, defaults to applying a random value to each value in a 0.05 of the agents,
                            where the random value is drawn from a Gaussian distribution of mean = the value being mutated
                            and stdev = 0.25.
        - **elitismfunc**: A function of signature `elitismfunc(generation_index) -> float in range [0.0, 1.0]`.
                        This function takes the index of the generation (0 for the first generation, 1 for the second, etc.)
                        and returns the fraction of top-performers to hold over as-is to the next generation.
                        Note that these agents are not removed from arguments sent to selectionfunc. This means that
                        selectionfunc will also get copies of the elites. The elites are duplicated and then, after the new
                        generation is created via the selectionfunc -> crossoverfunc -> mutationfunc pipeline, they are
                        reintroduced into the gene pool. This means that if the above pipeline generates 100 agents
                        and the elitism is set to take 10, the new generation will be composed of 110 agents. If this
                        is confusing, see `max_agents_per_generation` and `min_agents_per_generation`.
                        When `None`, defaults to a function that simply returns 0.1 (or 10%) of the gene pool regardless of the
                        generation.
        - **nworkers**: The number of processes to use to parallelize the various functions. This will default to 0, which will
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

    def _default_seedfunc(self):
        """
        Default seed function to create the first generation of agents. Each time this is called, it creates
        a new agent from uniform random of shape `self._shape` over the values[-1.0, 1.0).
        """
        # TODO: Test default function only produces individuals in range [-1.0, 1.0)
        return np.random.uniform(low=-1.0, high=1.0, size=self._shape)

    def _default_selectionfunc(self, population, evaluations):
        """
        Default selection function for selecting agents allowed to breed to create the next generation.
        Simply takes the top 10% of the given population and return them.

        Population and evaluations are pre-sorted so that index 0 is the fittest.

        This is guaranteed to take at least one agent.
        """
        # TODO: Test default function takes the top ten percent (and not just some random ten percent)
        tenpercent = int(population.shape[0] * 0.1)
        if tenpercent < 1:
            tenpercent = 1

        # Flatten the population into 2D - shape (nagents, flattened_agent)
        reshaped = np.reshape(population, (-1, np.product(population.shape[1:])))

        # Take the top ten percent
        topperformers = reshaped[0:tenpercent, :]

        # Reshape and return
        return np.reshape(topperformers, (tenpercent, *population.shape[1:]))

    def _default_crossoverfunc(self, agents):
        """
        Applies 2-point crossover to the agents to generate children. Mating pairs are chosen at random
        without replacement until the next generation has `self._initial_population_size` agents in it
        unless self._elitismfunc == self._default_elitismfunc, in which case we only go to
        0.9 * `self._initial_population_size` agents (since 0.1 are kept by the elitism function).

        Always mates at least one pair.
        """
        # TODO: Test that the crossover function samples without replacement
        # TODO: Test that the crossover function creates the right number of agents
        nagents = self._initial_population_size
        if self._elitismfunc == self._default_elitismfunc:
            nagents = int(0.9 * nagents)
        if nagents < 2:
            nagents = 2

        if nagents > agents.shape[0]:
            # Only one agent, just return it
            assert agents.shape[0] == 1
            return agents

        so_far_mated = set()
        remaining = [i for i in range(agents.shape[0])]
        created_agents = []
        while len(created_agents) < nagents:
            # Draw a random index from the remaining indexes
            idx1 = np.random.choice(remaining)
            remaining.remove(idx1)

            # If that was the last one, we need to dump so-far-mated and
            # start going through the them again
            if not remaining:
                remaining = list(so_far_mated)
                so_far_mated.clear()

            # Draw another
            idx2 = np.random.choice(remaining)
            remaining.remove(idx2)

            # Mate the two
            # TODO

            # Add the result to the list of agents we are going to return
            created_agents.append(newagent)

            # Add to the set of so-far-mated
            so_far_mated.add(idx1)
            so_far_mated.add(idx2)

            # If we have run out of items in remaining, dump so-far-mated
            # and start cycling back through them
            if not remaining:
                remaining = list(so_far_mated)
                so_far_mated.clear()

        new_agents = np.array(created_agents)
        return np.reshape(new_agents, (nagents, *self._shape))

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
        agents[idxs] = agents[idxs] + np.random.normal(agents[idxs], 0.25)
        return agents

    def _default_elitismfunc(self, genindex):
        """
        Simply returns 0.1, regardless of `genindex`. This means that 10% of the gene pool (the top
        10% specifically) will be re-injected into the next generation unchanged.
        """
        return 0.1
