/*
 * BFGS_bnd_linesearch_MPI_SW.hpp
 *
 *  Created on: Nov 16, 2018
 *      Author: brian
 */

#ifndef BFGS_BND_LINESEARCH_MPI_SW_HPP_
#define BFGS_BND_LINESEARCH_MPI_SW_HPP_




// standard includes
#include <vector>
#include <ctime>    // For time()
#include <cstdlib>  // For srand() and rand()

// Utility includes
#include "UtilityFunctions/utilityFunctions.hpp"

// local includes
#include "PNOL_Algorithm.hpp"
#include "BFGS_bnd_linesearch.hpp"

using namespace std;



class BFGS_Bnd_MPI_SW : public AlgorithmBnd {

  private:

	// line search parameters
	double c1, c2;
	double dalpha;
	double alphaGuess;
	double alphaTol;
	double alphaMult;
	int maxIterLineSearch;

	// BFGS parameters
	double bndTol;
	double dXGrad;
	double dXHess;
	double xMinDiff;
	double minGrad2Norm;

	vector<double> dXGradVec;
	vector<double> initialScalingVec;

	int maxIter;
	int totalIter;
	bool initHessFD; // use initial hessian finite difference

	int verbose; // verbose output from optimization?


	// local variables
	int Nprocs;
	int procID;
	bool optimFlag;
	int recurFlag;


  public:
	// main optimization function (instantiated algorithm function)
	void findMinBnd( vector <double> & X, vector <double> & Xlb, vector <double> & Xub, double & f0, double &fOpt  );
	void mainBFGSLoop( double & F, vector <double> & X, vector<double> & dFdX, vector<vector<double> > & D,
			vector <double> & Xlb, vector <double> & Xub, vector<double> & dX, vector<double> & constantX,
			vector<bool> & constantIndicator );
	void evaluateAlphaPoolAndDerivativesIndicator( vector <double> & alphaPool, vector<int> evalIndicator,
			vector <double> & X, vector <double> & p, vector<double> & constantX, vector<bool> & constantIndicator,
			vector <double> & phiPool, vector <double> & dphidalphaPool );
	void evaluateAlphaPoolAndDerivatives( vector <double> & alphaPool,	vector <double> & X, vector <double> & p,
			vector<double> & constantX, vector<bool> & constantIndicator, vector <double> & phiPool, vector <double> & dphidalphaPool );
	double lineSearchObj( double alpha, vector <double> & X, vector <double> & p, vector<double> & constantX, vector<bool> & constantIndicator );
	double lineSearchFDDerivative( double alpha, double phialpha, vector <double> & X, vector <double> & p,
			vector<double> & constantX, vector<bool> & constantIndicator );
	void lineSearchZoomBnd( double alpha_a, double alpha_b, double phi_a, double phi_b, double dphi_a_dalpha, double dphi_b_dalpha,
			double phi0, double dphi0dalpha, vector <double> & X, vector <double> & p, vector<double> & constantX, vector<bool> & constantIndicator,
			int & iter_ls, double & alphaOpt, double & phiOpt, double & dphiOptdalpha  );
	void cubicInterpolationLineSearchBnd( vector <double> & X,vector <double> & Xlb, vector <double> & Xub,
			double FX, vector <double> & dFdX, vector <double> & p, vector<double> & constantX, vector<bool> & constantIndicator,
			double & alphaOpt, double & Fopt  );
	void boundaryAssessment( double & F, vector <double> & X, vector <double> & p, vector<double> & dFdX, vector<vector<double> > & D,
			vector <double> & Xlb, vector <double> & Xub, vector<double> & dX, vector<double> & constantX, vector<bool> & constantIndicator );


	// set the parameters
	void setParams( double c1In, double c2In, double dalphaIn, double alphaGuessIn, double alphaTolIn, double alphaMultIn,
			int maxIterLineSearchIn, double bndTolIn, double dXGradIn, double dXHessIn, double maxIterIn,
			double xMinDiffIn, double minGrad2NormIn, bool initHessFDIn, int verboseIn )
	{
		c1 = c1In;
		c2 = c2In;
		dalpha = dalphaIn;
		alphaGuess = alphaGuessIn;
		alphaTol = alphaTolIn;
		alphaMult = alphaMultIn;
		maxIterLineSearch = maxIterLineSearchIn;
		bndTol = bndTolIn;
		dXGrad = dXGradIn;
		dXHess = dXHessIn;
		maxIter = maxIterIn;
		xMinDiff = xMinDiffIn;
		minGrad2Norm = minGrad2NormIn;
		initHessFD = initHessFDIn;
		verbose = verboseIn;
	}

    // default constructor/destructor
	BFGS_Bnd_MPI_SW()
	{
		c1 = 1e-4;
		c2 = 0.9;
		dalpha = 1e-6;
		alphaGuess = 1;
		alphaTol = 1e-20;
		alphaMult = 2;
		maxIterLineSearch = 50;

		bndTol = 1e-5;
		dXGrad = 1e-6;
		dXHess = 1e-3;
		maxIter = 10000;
		xMinDiff = 1e-5;
		minGrad2Norm = 1e-5;
		initHessFD = 0;

		verbose = 0;

		// Get the number of processes and the current processor ID
		MPI_Comm_size(MPI_COMM_WORLD, &Nprocs);
		MPI_Comm_rank(MPI_COMM_WORLD, &procID);

		optimFlag = true;
		recurFlag = 0;

		totalIter = 0;

	}

	void setGradVec( vector <double> & dXGradVecIn )
	{
		dXGradVec.clear();
		for(int i = 0; i < dXGradVecIn.size(); i++ )
		{
			dXGradVec.push_back(dXGradVecIn[i]);
		}
	}

	void setinitialScalingVec( vector <double> & initialScalingVecIn )
	{
		initialScalingVec.clear();
		for(int i = 0; i < initialScalingVecIn.size(); i++ )
		{
			initialScalingVec.push_back(initialScalingVecIn[i]);
		}
	}

	~ BFGS_Bnd_MPI_SW(){}


};

// local functions
void computeZoomRegion(vector <double> & alphaPool, vector <double> & phiPool, vector <double> & dphidalphaPool,
		 double & alpha_a, double & alpha_b, double & phi_a, double & phi_b, double & dphi_a_dalpha, double & dphi_b_dalpha );

#endif /* BFGS_BND_LINESEARCH_MPI_SW_HPP_ */
