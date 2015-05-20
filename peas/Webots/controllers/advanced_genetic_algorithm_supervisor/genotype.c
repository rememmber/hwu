#include "genotype.h"
#include <math.h>
#include <assert.h>
#include <stdlib.h>

static const double MUTATION_PROBABLITY = 0.1;
static const double MUTATION_DEVIATION  = 0.2;

struct _Genotype_ {
  double *genes;   // genome
  double fitness;  // fitness
};

static int genotype_size = -1;

int genotype_get_size() {
  return genotype_size;
}

void genotype_set_size(int size) {
  genotype_size = size;
}

Genotype genotype_create() {
  //assert(genotype_size > 0);
  Genotype gen = malloc(sizeof(struct _Genotype_));
  gen->fitness = 0.0;
  gen->genes = malloc(genotype_size * sizeof(double));

  // initialize with random uniform numbers in the range [0,1]
  int i;
  for (i = 0; i < genotype_size; i++)
    gen->genes[i] = 0.0;
    
  return gen;
}

Genotype create_genotype_from_file(FILE *fd) {
  int i;
  int ret;
  int numberArray[1];
  
  for (i = 0; i < 1; i++){
    fscanf(fd, "%d", &numberArray[i] );
  }
    
  //printf("number is: %d\n", numberArray[0]);
  
  //assert(genotype_size > 0);
  Genotype gen = malloc(sizeof(struct _Genotype_));
  gen->fitness = 0.0;
  gen->genes = malloc(numberArray[0] * sizeof(double));
  
    for (i = 0; i < numberArray[0]-1; i++) {
    ret = fscanf(fd, "%lf", &gen->genes[i]);
    if (ret == EOF)
      fprintf(stderr, "Cannot decode the genotype file\n");
  }
  
  genotype_set_size(numberArray[0]-1);

  return gen;
}

void genotype_destroy(Genotype g) {
  free(g->genes);
  free(g);
}

Genotype genotype_clone(Genotype g) {

  Genotype clone = genotype_create();

  int i;
  for (i = 0; i < genotype_size; i++)
    clone->genes[i] = g->genes[i];

  clone->fitness = g->fitness;
  return clone;
}

double drand(){
  double dr;
  int r;
  
  r = rand();
  dr = (double)(r)/(double)(RAND_MAX);
  return(dr);
}

void genotype_set_fitness(Genotype g, double fitness) {
  g->fitness = fitness;
}

double genotype_get_fitness(Genotype g) {
  return g->fitness;
}

const double *genotype_get_genes(Genotype g) {
  return g->genes;
}

void genotype_fread(Genotype g, FILE *fd) {
  int i;
  for (i = 0; i < genotype_size; i++) {
    int ret = fscanf(fd, "%lf", &g->genes[i]);
    if (ret == EOF)
      fprintf(stderr, "Cannot decode the genotype file\n");
  }
}

void genotype_fwrite(Genotype g, FILE *fd) {
  int i;
  for (i = 0; i < genotype_size; i++)
    fprintf(fd, " %lf", g->genes[i]);
    
  fprintf(fd, "\n");
}

void print_genotype(Genotype g) {
  int i;
  printf("size is: %d\n", genotype_size);
  for (i = 0; i < genotype_size; i++)
    printf(" %lf", g->genes[i]);
    
  printf("\n");
}