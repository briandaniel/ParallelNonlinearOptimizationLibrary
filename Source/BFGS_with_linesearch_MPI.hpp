/*
 * BFGS_with_linesearch_MPI.hpp
 *
 *  Created on: Nov 6, 2018
 *      Author: brian
 */

#ifndef BFGS_WITH_LINESEARCH_MPI_HPP_
#define BFGS_WITH_LINESEARCH_MPI_HPP_



// standard includes
#include <vector>
#include <ctime>    // For time()
#include <cstdlib>  // For srand() and rand()

// Utility includes
#include "UtilityFunctions/utilityFunctions.hpp"

// local includes
#include "PNOL_Algorithm.hpp"
#include "BFGS_with_linesearch.hpp"

using namespace std;



class BFGS_MPI : public Algorithm {

  private:

	// line search parameters
	double c1, c2;
	double maxAlphaMult;
	double alphaGuess;
	int maxIterLineSearch;

	// BFGS parameters
	double dXGrad;
	double dXHess;
	double xMinDiff;
	double minGrad2Norm;

	int maxIter;
	bool initHessFD; // use initial hessian finite difference

	bool verbose; // verbose output from optimization?


  public:
	// main optimization function (instantiated algorithm function)
	void findMin( vector <double> & X, double & f0, double &fOpt  );
	double lineSearchObj( double alpha, vector <double> & X, vector <double> & p  );
	void evalAlphaPoolMPI( vector <double> & alphaPool, vector <double> & phiPool, vector <double> & X, vector <double> & p  );
	void lineSearchZoom( double alpha_lo, double alpha_hi, double phi_lo, double phi_hi, double dphi_lo_dalpha, double dphi_hi_dalpha,
			double phi0, double dphi0dalpha, vector <double> & X, vector <double> & p, double & alphaOpt, double & phiOpt, double & dphiOptdalpha  );
	void secantLineSearch( vector <double> & X, double FX,
			vector <double> & dFdX, vector <double> & p, double & alphaOpt, double & Fopt  );



	// set the parameters
	void setParams( double c1In, double c2In, double maxAlphaMultIn, double alphaGuessIn, int maxIterLineSearchIn, double dXGradIn, double dXHessIn,
			double maxIterIn, double xMinDiffIn, double minGrad2NormIn, bool initHessFDIn, bool verboseIn )
	{
		c1 = c1In;
		c2 = c2In;
		maxAlphaMult = maxAlphaMultIn;
		alphaGuess = alphaGuessIn;
		maxIterLineSearch = maxIterLineSearchIn;
		dXGrad = dXGradIn;
		dXHess = dXHessIn;
		maxIter = maxIterIn;
		xMinDiff = xMinDiffIn;
		minGrad2Norm = minGrad2NormIn;
		initHessFD = initHessFDIn;
		verbose = verboseIn;
	}

    // default constructor/destructor
	BFGS_MPI()
	{
		c1 = 1e-4;
		c2 = 0.1;
		maxAlphaMult = 4;
		alphaGuess = 1;
		maxIterLineSearch = 1000;

		dXGrad = 1e-6;
		dXHess = 1e-3;
		maxIter = 10000;
		xMinDiff = 1e-5;
		minGrad2Norm = 1e-5;
		verbose = 0;
		initHessFD = 0;
	}


	~BFGS_MPI(){}


};

// local functions
void findPoolBounds( vector<double> & alphaPool, vector<double> & phiPool, double alpha0, double phi0,
		double & alpha_lo, double & alpha_hi, double & phi_lo, double & phi_hi);


#endif /* BFGS_WITH_LINESEARCH_MPI_HPP_ */
