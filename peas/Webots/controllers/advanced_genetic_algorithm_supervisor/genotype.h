#ifndef GENOTYPE_H
#define GENOTYPE_H

//   File:          genotype.h
//   Description:   General-purpose genotype class with mutation and crossover operations
//   Project:       Advanced exercises in Cyberbotics' Robot Curriculum
//   Author:        Yvan Bourquin - www.cyberbotics.com
//   Date:          January 6, 2010

#include <stdio.h>

// abstract type definition
typedef struct _Genotype_ *Genotype;

// set/get global number of genes
void genotype_set_size(int size);
int genotype_get_size();

// create new genotypes
Genotype genotype_create();
Genotype genotype_clone(Genotype g);

// release memory associated with g
void genotype_destroy(Genotype g);

// set/get fitness
void genotype_set_fitness(Genotype g, double fitness);
double genotype_get_fitness(Genotype g);
  
// for Emitter/Receiver transmission
const double *genotype_get_genes(Genotype g);

// read/write from stream
void genotype_fwrite(Genotype g, FILE *fd);
void genotype_fread(Genotype g, FILE *fd);

void print_genotype(Genotype g);

double drand();

#endif
