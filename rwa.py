from typing import Callable, Union

from net import Lightpath, Network
from ga import GeneticAlgorithm

ga: Union[GeneticAlgorithm, None] = None

def genetic_algorithm_callback(net: Network, k: int) -> Union[Lightpath, None]:
    """Callback function to perform RWA via genetic algorithm
    Args:
        net: Network topology instance
        k: number of alternate paths (ignored)
    Returns:
        Lightpath: if successful, returns both route and wavelength index as a
            lightpath
    """
    route, wavelength = ga.run(net, k)
    #print("route: ", route)
    #print("wavelenght", wavelength)
    if wavelength is not None and wavelength < net.nchannels:
        return Lightpath(route, wavelength)
    return None


def genetic_algorithm(pop_size: int, num_gen: int,
                      cross_rate: float, mut_rate: float) -> Callable:
    """Genetic algorithm as both routing and wavelength assignment algorithm
    This function just sets the parameters to the GA, so it acts as if it were
    a class constructor, setting a global variable as instance to the
    `GeneticAlgorithm` object in order to be further used by a callback
    function, which in turn returns the lightpath itself upon RWA success. This
    split into two classes is due to the fact that the class instance needs to
    be executed only once, while the callback may be called multiple times
    during simulation, namely one time per number of arriving call times number
    of load in Erlags (calls * loads)
    Note:
        Maybe this entire script should be a class and `ga` instance could be
        an attribute. Not sure I'm a good programmer.
    Args:
        pop_size: number of chromosomes in the population
        num_gen: number of generations towards evolve
        cross_rate: percentage of individuals to perform crossover
        mut_rate: percentage of individuals to undergo mutation
    Returns:
        callable: a callback function that calls the `GeneticAlgorithm` runner
            class, which finally and properly performs the RWA procedure
    """
    global ga
    #ga = GeneticAlgorithm(pop_size, num_gen, cross_rate, mut_rate)
    ga = GeneticAlgorithm(pop_size, num_gen, cross_rate, mut_rate)
    return genetic_algorithm_callback