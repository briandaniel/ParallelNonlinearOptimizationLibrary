/*
 * LevenbergMarquardt.hpp
 *
 *  Created on: Oct 23, 2018
 *      Author: brian
 */

#ifndef LEVENBERGMARQUARDT_HPP_
#define LEVENBERGMARQUARDT_HPP_

// standard includes
#include <vector>
#include <ctime>    // For time()
#include <cstdlib>  // For srand() and rand()

// Utility includes
#include "UtilityFunctions/utilityFunctions.hpp"

// local includes
#include "PNOL_Algorithm.hpp"

using namespace std;

class LevMarq : public MultiAlgorithm {

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
	LevMarq()
	{
		dXGrad = 1e-7;
		lambda0 = 0.001;
		maxIter = 10000;
		xMinDiff = 1e-7;
		verbose = 1;
		lambdaFactor = 10;
	}
	~LevMarq(){}


};




#endif /* LEVENBERGMARQUARDT_HPP_ */
