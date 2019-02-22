# primordial-ooze

Dirt simple genetic algorithm library for Python

Docs are below.

## Simulation

A GA simulation. The general workflow for this is:

```python
import primordialooze as po

sim = po.Simulation(nagents, shape, fitnessfunction)
bestagent, fitness = sim.run()
print(sim.results)
```


### __init__

```python
def __init__(self, population, shape, fitnessfunc, *, seedfunc=None, selectionfunc=None,
                crossoverfunc=None, mutationfunc=None, elitismfunc=None, nworkers=0,
                max_agents_per_generation=None, min_agents_per_generation=None)
```

#### Args

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

#### Keyword Args

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

### run

```python
    def run(self, niterations=100, fitness=None)
```

Runs the constructed simulation.

Either runs until `niterations` have passed, or runs until the best fitness is `fitness` or greater.
Returns the best agent along with its fitness.

#### Keyword Args

- **niterations**: The number of iterations to run the simulation to. Defaults to 100. If `None`,
                    `fitness` will be used (and must not be None). If both this and `fitness` is
                    specified, we will stop as soon as one or the other condition is met.
- **fitness**: The fitness level to converge on. As soon as one or more agents have this fitness level
                or higher, the simulation will stop. Defaults to `None`. If `None` (the default),
                `niterations` will be used (and must not be None). If this and `niterations` is
                specified, we will stop as soon as one or the other condition is met.

#### Returns

- The agent with the highest fitness score after the simulation ends.
- The fitness of this agent.
