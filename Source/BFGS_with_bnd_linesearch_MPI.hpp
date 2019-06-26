/*
 * BFGS_with_bnd_linesearch_MPI.hpp
 *
 *  Created on: Nov 7, 2018
 *      Author: brian
 */

#ifndef BFGS_WITH_BND_LINESEARCH_MPI_HPP_
#define BFGS_WITH_BND_LINESEARCH_MPI_HPP_


// Parallel commands
#define ROOT_ID 0	 // id of root processor



// standard includes
#include <vector>
#include <ctime>    // For time()
#include <cstdlib>  // For srand() and rand()

// Utility includes
#include "ImportExport/importExport.hpp"

// local includes
#include "PNOL_Algorithm.hpp"
#include "BFGS_with_linesearch.hpp"
#include "BFGS_with_linesearch_MPI.hpp"
#include "Box_boundary_functions.hpp"

using namespace std;



class BFGSBnd_MPI : public AlgorithmBnd {

  private:

	// line search parameters
	double c1, c2;
	double maxAlphaMult;
	double alphaGuess;
	int maxIterLineSearch;
	double alphaMin;

	// BFGS parameters
	double dXGrad;
	double dXHess;
	double xMinDiff;
	double minGrad2Norm;
	double FStepTolerance;

	int maxIter;
	bool initHessFD; // use initial hessian finite difference

	bool verbose; // verbose output from optimization?


  public:
	// main optimization function (instantiated algorithm function)
	void findMinBnd( vector <double> & X, vector <double> & Xlb, vector <double> & Xub, double & f0, double & fOpt  );
	void mainBFGSLoop( double & F, vector <double> & X, vector<double> & dFdX, vector<vector<double> > & D,
			vector <double> & Xlb, vector <double> & Xub, vector<double> & dX, vector<double> & constantX, vector<bool> & constantIndicator,
			bool & optimFlag, bool & recurFlag );
	double lineSearchObj( double alpha, vector <double> & X, vector <double> & p, vector<double> & constantX, vector<bool> & constantIndicator  );
	void evalAlphaPoolMPI( vector <double> & alphaPool, vector <double> & phiPool, vector <double> & X, vector <double> & p,
			vector<double> & constantX, vector<bool> & constantIndicator  );
	void lineSearchZoom( double alpha_lo, double alpha_hi, double phi_lo, double phi_hi, double dphi_lo_dalpha, double dphi_hi_dalpha,
			double phi0, double dphi0dalpha, vector <double> & X, vector <double> & p, double & alphaOpt, double & phiOpt, double & dphiOptdalpha,
			vector<double> & constantX, vector<bool> & constantIndicator  );
	void secantLineSearchBnd( vector <double> & X, vector <double> & Xlb, vector <double> & Xub, double FX,
			vector <double> & dFdX, vector <double> & p, double & alphaOpt, double & Fopt, vector<double> & constantX, vector<bool> & constantIndicator  );
	void boundaryAssessment( double & F, vector <double> & X, vector <double> & p,	vector<double> & dFdX, vector<vector<double> > & D,
			vector <double> & Xlb, vector <double> & Xub, vector<double> & dX, vector<double> & constantX, vector<bool> & constantIndicator,
			bool & optimFlag, bool & recurFlag);


	// set the parameters
	void setParams( double c1In, double c2In, double alphaMinIn, double maxAlphaMultIn, double alphaGuessIn, int maxIterLineSearchIn, double dXGradIn,
			double dXHessIn, double maxIterIn, double xMinDiffIn, double minGrad2NormIn, double FStepToleranceIn, bool initHessFDIn, bool verboseIn )
	{
		c1 = c1In;
		c2 = c2In;
		alphaMin = alphaMinIn;
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
		FStepTolerance = FStepToleranceIn;
	}

    // default constructor/destructor
	BFGSBnd_MPI()
	{
		c1 = 1e-4;
		c2 = 0.1;
		alphaMin = 1e-16;
		maxAlphaMult = 4;
		alphaGuess = 1;
		maxIterLineSearch = 1000;
		FStepTolerance = 1e-5;

		dXGrad = 1e-6;
		dXHess = 1e-3;
		maxIter = 10000;
		xMinDiff = 1e-5;
		minGrad2Norm = 1e-5;
		verbose = 0;
		initHessFD = 0;
	}


	~BFGSBnd_MPI(){}


};


double computeAlphaBnd( vector <double> & X, vector <double> & Xlb, vector <double> & Xub, vector <double> & p );
void checkAlphaPoolBnd( bool & bndIndicator, vector <double> & alphaPool, vector <double> & X, vector <double> & Xlb, vector <double> & Xub,
		vector <double> & p, vector<double> & constantX, vector<bool> & constantIndicator );


#endif /* BFGS_WITH_BND_LINESEARCH_MPI_HPP_ */
