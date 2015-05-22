# 1. peas + Webots
![alt text](https://raw.githubusercontent.com/rememmber/hwu/master/peas/docs/webots_peas_integration.png "Logo Title Text 1")
Legend:
* blue arrow - communication between peas and Webots;
* green arrow - communication inside Webots processes
* black solid/dashed arrows - sequential execution
* boxes - states

Run by:
  path_to_webots --mode=fast --stdout --stderr path_to_world.wbt

Requirements:
* Python 2.7.
* Numpy
* Webots

Optional:
* Pygraphviz

Supported platforms:
* OSX
* Win

Prerequisite:
* specify python_path (supervisor)
* set number for population and generations (python experiment)
* specify running time (supervisor)

TODO:
* There is a small memory leak in supervisor class
* Not all functionality is used of peas evolutionary algorithm in specified task (line following)
  * Evolving activation function
  * Evolving number of hidden layers
  * Currently max amount of nodes/substrate layer is specified (is it possible to make it variable?)
  * Nodes with recurrent connections

Line following experiment in 2 different worlds fitness evolution:
![alt text](https://raw.githubusercontent.com/rememmber/hwu/master/peas/Webots/controllers/advanced_genetic_algorithm_supervisor/stats/fitness_evolution.png "Logo Title Text 1")

,where the best fitness got the ANN:
![alt text](https://raw.githubusercontent.com/rememmber/hwu/master/peas/Webots/controllers/advanced_genetic_algorithm_supervisor/stats/gen13.png "Logo Title Text 1")

Observations:
* peas initialises connections with the same weight 0.7(7)
* the last fitness of a population individual is seem to be lost (may it be that the whole synchronisation is not alligned?)

# 2. HyperSharpNEAT + Webots


-------
Resources:
* http://link.springer.com/chapter/10.1007/978-3-642-55337-0_5
* http://eplex.cs.ucf.edu/uncategorised/software
* https://github.com/noio/peas
