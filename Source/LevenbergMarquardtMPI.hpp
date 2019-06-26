/*
 * LevenbergMarquardtMPI.hpp
 *
 *  Created on: Nov 7, 2018
 *      Author: brian
 */

#ifndef LEVENBERGMARQUARDTMPI_HPP_
#define LEVENBERGMARQUARDTMPI_HPP_

// Parallel commands
#define ROOT_ID 0	 // id of root processor

// standard includes
#include <vector>
#include <ctime>    // For time()
#include <cstdlib>  // For srand() and rand()

// Utility includes
#include "UtilityFunctions/utilityFunctions.hpp"

// local includes
#include "PNOL_Algorithm.hpp"

using namespace std;

class LevMarqMPI : public MultiAlgorithm {

  private:

	// parameters
	double lambda0;
	double dXGrad;
	double xMinDiff;
	int maxIter;
	double lambdaFactor;  // Lambda factor: should be > 1
	int verbose; // verbose output from optimization?

  public:
	// main optimization function (instantiated algorithm function)
	void findMin( vector <double> & X, vector <double> & f0,vector <double> & fOpt  );

	// set the parameters
	void setParams( double lambda0In, double lambdaFactorIn, double dXGradIn, double maxIterIn, double xMinDiffIn, int verboseIn )
	{  maxIter = maxIterIn; xMinDiff = xMinDiffIn; verbose = verboseIn; dXGrad = dXGradIn; lambda0 = lambda0In; lambdaFactor = lambdaFactorIn; }

    // default constructor/destructor
	LevMarqMPI()
	{
		dXGrad = 1e-7;
		lambda0 = 0.001;
		maxIter = 10000;
		xMinDiff = 1e-7;
		verbose = 1;
		lambdaFactor = 10;
	}
	~LevMarqMPI(){}


};





#endif /* LEVENBERGMARQUARDTMPI_HPP_ */
