# 1. peas + Webots
![alt text](https://raw.githubusercontent.com/rememmber/hwu/master/peas/docs/webots_peas_integration.png "Logo Title Text 1")

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
* Win (in progress)

Prerequisite:
* specify python_path (supervisor)
* set number for population and generations (python experiment)
* specify running time (supervisor)

TODO:
* There is a small memory leak in supervisor class
* Not all functionality is used of peas evolutionary algorithm in specified task (line following)

Line following experiment in 2 different worlds fitness evolution:
![alt text](https://raw.githubusercontent.com/rememmber/hwu/master/peas/Webots/controllers/advanced_genetic_algorithm_supervisor/stats/fitness_evolution.png "Logo Title Text 1")

,where the best fitness got the ANN:
![alt text](https://raw.githubusercontent.com/rememmber/hwu/master/peas/Webots/controllers/advanced_genetic_algorithm_supervisor/stats/gen13.png "Logo Title Text 1")

# 2. HyperSharpNEAT + Webots


-------
Resources:
* http://link.springer.com/chapter/10.1007/978-3-642-55337-0_5
* http://eplex.cs.ucf.edu/uncategorised/software
* https://github.com/noio/peas
