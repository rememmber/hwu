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

#import admin

developer = None

class nonlocal:
    counter = 0

class Communication:
    PROVIDE_ANN = 1
    EVALUATION_RESULTS = 2

comm_direction = Communication.PROVIDE_ANN;

def pass_ann(ann):
    #dir = os.path.dirname('C:/Users/ifxboris/Desktop/hwu/Webots/controllers/advanced_genetic_algorithm_supervisor/fifofile')
    #if not os.path.exists(dir):
    #    print 'does not exist'
    #    os.makedirs(dir)
    #else:
    #    print ' exists'

    fifo = open(os.path.join(os.path.dirname(__file__), '../../Webots/controllers/advanced_genetic_algorithm_supervisor/genes.txt'), 'wb')
    fifo.write(' '.join(map(str, ann)))  #str(len(ann)) + ' ' + ' '.join(map(str, ann))
    fifo.close()
    comm_direction = Communication.EVALUATION_RESULTS
	
def get_stats():
    while not os.path.exists(os.path.join(os.path.dirname(__file__), '../../Webots/controllers/advanced_genetic_algorithm_supervisor/genes_fitness.txt')):
	    time.sleep(1)
	    continue
    fitness = open(os.path.join(os.path.dirname(__file__), '../../Webots/controllers/advanced_genetic_algorithm_supervisor/genes_fitness.txt'))
    comm_direction = Communication.PROVIDE_ANN
    fitness_data = fitness.read()
    fitness.close()
    #time.sleep(5)
    os.remove(os.path.join(os.path.dirname(__file__), '../../Webots/controllers/advanced_genetic_algorithm_supervisor/genes_fitness.txt'))
    return fitness_data

def evaluate(individual, task, developer):
    #many evaluations, single solve
    #print 'evaluate'
    #print individual.get_weights() # <-- every individual weight matrix
    
    phenotype = developer.convert(individual)
    
    pass_ann(phenotype.get_connectivity_matrix())
    fitness = get_stats()
    print 'got fitness: '
    print fitness

    #print fitness

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
    #many evaluations, single solve
    #print 'solve'
    #phenotype = developer.convert(individual)
    return True #task.solve(phenotype)

def a_callback(self):
	print 'callback'


### SETUPS ###    
def run(method, setup, generations=10, popsize=10):
    task_kwds = dict(field='eight',
                     observation='eight',
                     max_steps=3000,
                     friction_scale=0.3,
                     damping=0.3,
                     motor_torque=10,
                     check_coverage=False,
                     flush_each_step=False,
                     initial_pos=(282, 300, np.pi*0.35))
    
    task = LineFollowingTask(**task_kwds)
	
    # The line following experiment has quite a specific topology for its network:    
    substrate = Substrate()

    substrate.add_nodes([(r, theta) for r in np.linspace(0,1,2)
                              for theta in np.linspace(-1, 1, 5)], 'input')

    #substrate.add_nodes([(r, theta) for r in np.linspace(0,1,3)
    #                          for theta in np.linspace(-1, 1, 3)], 'bias')

    substrate.add_nodes([(r, theta) for r in np.linspace(0,1,3)
                              for theta in np.linspace(-1, 1, 3)], 'bias')
						
    substrate.add_nodes([(r, theta) for r in np.linspace(0,0,1)
                              for theta in np.linspace(-1, 1, 2)], 'layer')

    substrate.add_connections('input', 'bias',-1)
    substrate.add_connections('bias', 'layer', -2)
    #substrate.add_connections('layer',-3)
        
    geno_kwds = dict(feedforward=True, 
                     inputs=10,
                     outputs=2,
                     weight_range=(-3.0, 3.0), 
                     prob_add_conn=0.1, 
                     prob_add_node=0.03,
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

    fifo = open(os.path.join(os.path.dirname(__file__), '../../Webots/controllers/advanced_genetic_algorithm_supervisor/best_solution.txt'), 'a+')
    for champion in results['champions']:
        phenotype = developer.convert(champion)
        #nonlocal.counter += 1
        #phenotype.visualize('visual'+str(nonlocal.counter)+'.png', inputs=10, outputs=2)
        fifo.write(' '.join(map(str, phenotype.get_connectivity_matrix()))+'\n')
    fifo.close()

    return results

if __name__ == '__main__':
    print 'running peas line following + webots'
    resnhn = run('nhn', 'hard')