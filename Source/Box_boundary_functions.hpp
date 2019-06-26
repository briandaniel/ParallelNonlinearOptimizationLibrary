/*
 * Box_boundary_functions.hpp
 *
 *  Created on: Nov 12, 2018
 *      Author: brian
 */

#ifndef BOX_BOUNDARY_FUNCTIONS_HPP_
#define BOX_BOUNDARY_FUNCTIONS_HPP_


// Parallel commands
#ifndef ROOT_ID
#define ROOT_ID 0 // id of root process
#endif

// standard includes
#include <vector>
#include <ctime>    // For time()
#include <cstdlib>  // For srand() and rand()

// Utility includes
#include "ImportExport/importExport.hpp"
#include "UtilityFunctions/utilityFunctions.hpp"

using namespace std;


// Function definitions
void checkBoxBounds( vector <double> & X, vector <double> & Xlb, vector <double> & Xub );
void setHardRandValues( vector <double> & X, vector <double> & Xlb, vector <double> & Xub );



#endif /* BOX_BOUNDARY_FUNCTIONS_HPP_ */
