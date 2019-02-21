"""
PrimordialOoze is a genetic algorithm (GA) library for those who want something
very simple and don't want to spend time figuring out the more complicated libraries
that are out there.

See the README or the docstrings in this file for the documentation.
"""

class Simulation:
    """
    A GA simulation. The general workflow for this is:

    ```python
    import primordialooze as po

    sim = po.Simulation()
    bestagent, fitness = sim.run()
    print(sim.results)
    ```
    """

    def __init__(self):
        pass
