/*
 * Examples.hpp
 *
 *  Created on: Oct 22, 2018
 *      Author: brian
 */

#ifndef EXAMPLES_HPP_
#define EXAMPLES_HPP_

// Standard includes
#include <iostream>
#include <ostream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <complex>
#include <cstdio>
#include <ctime>
#include <vector>
#include <mpi.h>
#include <iomanip>

using namespace std;


// local includes
#include "SimplexSearch.hpp"
#include "PNOL_Algorithm.hpp"
#include "PNOL_Objective.hpp"
#include "ExampleObjectives.hpp"
#include "LevenbergMarquardt.hpp"
#include "LevenbergMarquardtMPI.hpp"
#include "GeneticAlgorithm.hpp"
#include "GeneticAlgorithmMPI.hpp"
#include "BFGS_with_linesearch.hpp"
#include "BFGS_with_linesearch_MPI.hpp"
#include "BFGS_with_bnd_linesearch_MPI.hpp"
#include "BFGS_bnd_linesearch.hpp"
#include "BFGS_bnd_linesearch_MPI_SW.hpp"


// example functions
void testBFGSBndMPISW();
void testBFGSBnd();
void testBFGSBnd_MPI();
void testLMExpMPI();
void testBFGS_MPI();
void testBFGS_booth();
void testBFGS();
void testHessian();
void testGAParallel();
void testGA();
void testLMExp();
void testLMCubicLinearCoef();
void testSimplexSearch();
void testCreateObject();
void testGradientEvaluation();
void testObjectEvaluation( Objective & obj );
void testGradientApproxMultMPI();
void testGradientApproxMultMPIRecur();

#endif /* EXAMPLES_HPP_ */
