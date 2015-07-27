#!/usr/local/bin/python

# Python experiment setup for peas HyperNEAT implementation
# This setup is used by Webots to perform evolution of a controller for e-puck robot that tries to follow a line created as a 'world' in Webots environment

# Program name - line_following_webots.py
# Written by - Boris Mocialov (bm4@hw.ac.uk)
# Date and version No:  20.05.2015

# Webots supervisor class spawns this script, which is used to evolve a controller that is evaluated in Webots environment.
# The interaction between Webots environment and peas implementation is done via '.dat' files.
# peas implementation is only provided with the fitness of each generated controller - peas does no evaluation itself

# Fitness function is defined in Webots environment

### IMPORTS ###
import sys, os
from functools import partial
from itertools import product

# Libs
import numpy as np
np.seterr(invalid='raise')

# Local
sys.path.append(os.path.join(os.path.split(__file__)[0],'..','..')) 
from peas.methods.neat import NEATPopulation, NEATGenotype
from peas.methods.evolution import SimplePopulation
from peas.methods.wavelets import WaveletGenotype, WaveletDeveloper
from peas.methods.hyperneat import HyperNEATDeveloper, Substrate
from peas.tasks.linefollowing import LineFollowingTask

import time

import signal

#import admin

developer = None

class nonlocal:
    counter = 0

class Communication:
    PROVIDE_ANN = 1
    EVALUATION_RESULTS = 2

comm_direction = Communication.PROVIDE_ANN;

stats = {'fitness':0.0, 'dist':0, 'speed':0, 'nodes':0};

def pass_ann(node_types, ann):
    #dir = os.path.dirname('C:/Users/ifxboris/Desktop/hwu/Webots/controllers/advanced_genetic_algorithm_supervisor/fifofile')
    #if not os.path.exists(dir):
    #    os.makedirs(dir)

    fifo = open(os.path.join(os.path.dirname(__file__), '../../genes_1'), 'wb')
    fifo.write(' '.join(map(str, node_types)) + ' '+' '.join(map(str, ann)))  #str(len(ann)) + ' ' + ' '.join(map(str, ann))
    fifo.close()

    #fifo = open(os.path.join(os.path.dirname(__file__), '../../genes_bak'), 'wb')
    #fifo.write(' '.join(map(str, node_types)) + '\n'+' '.join(map(str, ann)))  #str(len(ann)) + ' ' + ' '.join(map(str, ann))
    #fifo.close()

    #print 'connectivity matrix length: '
    #print len(ann)

    comm_direction = Communication.EVALUATION_RESULTS
	
def get_stats():
    while not os.path.exists(os.path.join(os.path.dirname(__file__), '../../genes_fitness_1')):
	    time.sleep(1)
	    continue
    fitness = open(os.path.join(os.path.dirname(__file__), '../../genes_fitness_1'))
    comm_direction = Communication.PROVIDE_ANN
    fitness_data = fitness.read()
    fitness.close()
    time.sleep(5)
    os.remove(os.path.join(os.path.dirname(__file__), '../../genes_fitness_1'))
    return fitness_data

def evaluate(individual, task, developer):
    #many evaluations, single solve
    #print 'evaluate'
    #print individual.get_weights() # <-- every individual weight matrix
    
    phenotype = developer.convert(individual)
    #print individual.get_network_data()
    #print '\n'
    #print phenotype.get_full_connectivity_matrix()
    #print 'result: '

    #print phenotype.get_node_types()

    nodes_types_indexes = list()
    #j=0
    #print 'length: '+str(len(phenotype.get_full_connectivity_matrix()))

    #print 'check this'
    #print len(phenotype.get_full_connectivity_matrix())
    #print 'and this'
    #nonlocal.counter += 1
    #phenotype.visualize('temp'+str(nonlocal.counter)+'.png', inputs=10, outputs=2)
    #print individual.get_network_data()[1]
    #print individual.get_network_data()[0]

    #for i in range(len(phenotype.get_full_connectivity_matrix())):
    #    if(np.count_nonzero(phenotype.get_full_connectivity_matrix()[:,i]) > 0): #if at least one connection from node (i)
    #    	print 'inferring node '+str(i)+ ' is connected'
        	#nodes_types_indexes.append(float(['sin', 'bound', 'linear', 'gauss', 'sigmoid', 'abs'].index((individual.get_network_data()[1])[j])))
        	#j += 1
        #else:
        	#nodes_types_indexes.append(float(['sin', 'bound', 'linear', 'gauss', 'sigmoid', 'abs'].index('sigmoid')))

    #print nodes_types_indexes

    rest = 442 - len((individual.get_network_data()[1])[1:])
    rest_array = np.linspace(4., 4., rest)

    #print rest_array

    for idx, node_type in enumerate((individual.get_network_data()[1])[1:]):
        if (idx == len((individual.get_network_data()[1])[1:]) - 2):
        	nodes_types_indexes.extend(rest_array)
        nodes_types_indexes.append(float(['sin', 'bound', 'linear', 'gauss', 'sigmoid', 'abs'].index(node_type)))

    pass_ann(nodes_types_indexes, phenotype.get_connectivity_matrix()) #phenotype.get_connectivity_matrix()
    fitness = get_stats()

    print fitness

    #sys.exit()
	
    #stats = task.evaluate(phenotype)
    stats = {'fitness':float(fitness), 'dist':0, 'speed':0}  #what to do with dist and speed?
    if isinstance(individual, NEATGenotype):
        stats['nodes'] = len(individual.node_genes)
    elif isinstance(individual, WaveletGenotype):
        stats['nodes'] = sum(len(w) for w in individual.wavelets)
    #print '~',
    sys.stdout.flush()
    return stats
    
def solve(individual, task, developer):
    #phenotype = developer.convert(individual)
    #many evaluations, single solve
    #print 'solve'
    #phenotype = developer.convert(individual)
    #return int(self.evaluate(network) > 0.7) #task.solve(phenotype)
    #return task.solve(phenotype)
    #print 'solve'
    #print stats['fitness']
    #return stats['fitness'] > 0.7
    print 'fitness solve:'
    print individual.stats['fitness']
    return individual.stats['fitness'] > 0.15

def a_callback(self):
	print 'callback'


### SETUPS ###    
def run(method, setup, generations=15, popsize=10):
    task_kwds = dict(field='eight',
                     observation='eight',
                     max_steps=1000,
                     friction_scale=0.3,
                     damping=0.3,
                     motor_torque=10,
                     check_coverage=False,
                     flush_each_step=False,
                     initial_pos=(282, 300, np.pi*0.35))
    
    task = LineFollowingTask(**task_kwds)
	
    # The line following experiment has quite a specific topology for its network:    
    substrate = Substrate()

    substrate.add_nodes([(r, theta) for r in np.linspace(-10,10,21) for theta in np.linspace(-10, 10, 21)], 'input', is_input=True)

    #substrate.add_nodes([(r, theta) for r in np.linspace(-2,2,4) for theta in np.linspace(-2, 2, 4)], 'hidden')

    substrate.add_nodes([(r, theta) for r in np.linspace(0,0,1) for theta in np.linspace(0, 0, 1)], 'output')

    #substrate.add_connections('input', 'hidden', -1)
    substrate.add_connections('input', 'output', -1)

    geno_kwds = dict(feedforward=True, 
                     inputs=441,
                     outputs=1,
                     max_depth=3,
                     max_nodes=442,
                     weight_range=(-3.0, 3.0), 
                     prob_add_conn=0.3, 
                     prob_add_node=0.1,
                     bias_as_node=False,
                     types=['sigmoid'])

    geno = lambda: NEATGenotype(**geno_kwds)
    #geno = NEATGenotype(**geno_kwds)
    #test.visualize('test.png')
    #print str(geno.get_weights())

    pop = NEATPopulation(geno, popsize=popsize, target_species=8)

    developer = HyperNEATDeveloper(substrate=substrate, 
                                   add_deltas=False,
                                   sandwich=False,
                                   node_type='sigmoid')


    results = pop.epoch(generations=generations,
                        evaluator=partial(evaluate, task=task, developer=developer),
                        solution=partial(solve, task=task, developer=developer), 
                        )

    fitnesses = list()

    fifo = open(os.path.join(os.path.dirname(__file__), '../../best_solution_1'), 'a+')
    for champion in results['champions']:
        fitnesses.append(champion.stats['fitness'])
        phenotype = developer.convert(champion)
        nonlocal.counter += 1
        #dir = os.path.dirname('../../visual'+str(nonlocal.counter)+'_1.png')
        #if not os.path.exists(dir):
        #   os.makedirs(dir)
        #phenotype.visualize('../../visual'+str(nonlocal.counter)+'_1.png', inputs=441, outputs=1)
        
        rest = 442 - len((champion.get_network_data()[1])[1:])
        rest_array = np.linspace(4., 4., rest)
        
        for idx, node_type in enumerate((champion.get_network_data()[1])[1:]):
        	if (idx == len((champion.get_network_data()[1])[1:]) - 2):
				nodes_types_indexes.extend(rest_array)
        	nodes_types_indexes.append(float(['sin', 'bound', 'linear', 'gauss', 'sigmoid', 'abs'].index(node_type)))
        
        fifo.write('fitness: '+str(champion.stats['fitness'])+' || ' + ' '.join(map(str, node_types)) + ' '+' '.join(map(str, phenotype.get_connectivity_matrix()))+'\n')
    fifo.close()



    """ Visualize evolution. """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    print "Saving as " + os.path.join(os.getcwd(), '../../fitness_evolution_1')
    plt.figure()
    x = range(len(results['champions']))
    y = np.asarray(fitnesses)
    xa = plt.gca().get_xaxis()
    xa.set_major_locator(MaxNLocator(integer=True))
    plt.plot(x, y)
    plt.axis('on')
    plt.savefig(os.path.join(os.getcwd(), '../../fitness_evolution_1'), bbox_inches='tight', pad_inches=0)
    plt.close()

    return results

if __name__ == '__main__':
    print 'running peas line following + webots'
    parent = sys.argv[1]
    resnhn = run('nhn', 'hard')
    print 'Done'
    os.kill(parent, signal.SIGKILL)
