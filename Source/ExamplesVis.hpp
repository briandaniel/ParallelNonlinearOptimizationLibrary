/*
 * ExamplesVis.hpp
 *
 *  Created on: Oct 24, 2018
 *      Author: brian
 */

#ifndef EXAMPLESVIS_HPP_
#define EXAMPLESVIS_HPP_




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
#include <string>

using namespace std;


// local includes
#include "SimplexSearch.hpp"
#include "PNOL_Algorithm.hpp"
#include "PNOL_Objective.hpp"
#include "ExampleObjectives.hpp"
#include "LevenbergMarquardt.hpp"
#include "GeneticAlgorithm.hpp"


// VTK function library includes
#include "triangleStripFunctions.hpp"
#include "surfacePlotFunctions.hpp"
#include "generalVTKFunctions.hpp"
#include "vtkColorMaps.hpp"
#include "VTKRenderContainer.hpp"
#include "vtkFunctionLibrary_examples.hpp"

// local functions
void testGAWithVis();

#endif /* EXAMPLESVIS_HPP_ */
