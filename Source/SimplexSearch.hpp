/*
 * SimplexSearch.hpp
 *
 *  Created on: Oct 22, 2018
 *      Author: brian
 */

#ifndef SIMPLEXSEARCH_HPP_
#define SIMPLEXSEARCH_HPP_

// standard includes
#include <vector>
#include <ctime>    // For time()
#include <cstdlib>  // For srand() and rand()

// Utility includes
#include "UtilityFunctions/utilityFunctions.hpp"

// local includes
#include "PNOL_Algorithm.hpp"

using namespace std;

class SimplexSearch : public Algorithm {

  private:

	// simplex parameters
	double alpha, rho, gamma, sigma;
	double initRandMax, xMinDiff;

	int maxIter;
	bool verbose; // verbose output from optimization?

  public:
	// main optimization function (instantiated algorithm function)
	void findMin( vector <double> & X, double & f0, double &fOpt  );

	// Evaluate simplex varaibles
	double evaluateVariableArray(  double * x, vector <double> & X );
	void evaluateVariableSet(  double ** xvec, int Nsimplex, vector <double> & X, double * fvec );


	// set the parameters
	void setSimplexParams( double alphaIn,  double gammaIn, double rhoIn, double sigmaIn, int maxIterIn,
			double initRandMaxIn, double xMinDiffIn, bool verboseIn )
	{ alpha = alphaIn; gamma = gammaIn; rho = rhoIn; sigma = sigmaIn;
	  maxIter = maxIterIn; initRandMax = initRandMaxIn; xMinDiff = xMinDiffIn; verbose = verboseIn; }

    // default constructor/destructor
	SimplexSearch()
	{
		alpha = 1.0; gamma = 2.0; rho = 0.5; sigma = 0.5;
		maxIter = 10000;
		initRandMax = 1;
		xMinDiff = 1e-7;
		verbose = 0;
	}
	~SimplexSearch(){}


};

// other local includes
void simplexSort(double * fvec, double ** xvec, int Nd );
double simplexDiff( double ** xvec, int Nd, int Nsimplex );

#endif /* SIMPLEXSEARCH_HPP_ */
