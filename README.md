# 1. peas + Webots
![alt text](https://raw.githubusercontent.com/rememmber/hwu/master/peas/docs/webots_peas_integration.png "Logo Title Text 1")
Requirements:
* Python 2.7.
* Numpy
* Webots

Optional:
* Pygraphviz

Supported platforms:
* OSX
* Win (in progress)

Run by:
path_to_webots --mode=fast --stdout --stderr path_to_world.wbt

Prerequisite:
* specify python_path (supervisor)
* set number for population and generations (python experiment)
* specify running time (supervisor)

TODO:
* There is a small memory leak in supervisor class
* Not all functionality is used of peas evolutionary algorithm in specified task (line following)
* Windows support
* Freeing resources after a single evolution

# 2. HyperSharpNEAT + Webots
