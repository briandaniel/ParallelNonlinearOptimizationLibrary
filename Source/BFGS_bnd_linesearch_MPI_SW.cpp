/*
 * BFGS_bnd_linesearch_MPI_SW.cpp
 *
 *  Created on: Nov 16, 2018
 *      Author: brian
 */


#include "BFGS_bnd_linesearch_MPI_SW.hpp"


void BFGS_Bnd_MPI_SW::findMinBnd( vector <double> & X, vector <double> & Xlb, vector <double> & Xub, double & f0, double & fOpt  ){

	totalIter = 0;
	MPI_Barrier(MPI_COMM_WORLD);

	if( procID == 0 )
	{
		cout << endl << "----------------------------------------------------------------------------------------------------------" << endl ;
		cout << "  Starting parallel bounded BFGS with line search. " << endl;
		cout << "  X0 = "; print1DVector(X);
		cout << "----------------------------------------------------------------------------------------------------------" << endl ;
	}
	int Nparam = X.size();

	// Check that the initial guess is within the boundary
	checkBoxBounds( X, Xlb, Xub );

	// vectors
	vector<double> constantX(Nparam,0);
	vector<bool> constantIndicator(Nparam,false);

 	vector<double> X0 (Nparam, dXGrad);
 	for(int k = 0; k < Nparam; k++){ X0[k] = X[k]; }

 	vector<double> dX (Nparam, dXGrad);
	vector<double> dFdX (Nparam, 0);
	if(dXGradVec.size() > 0)
 	{
 		for(int i =0; i < Nparam; i++ )
		{
 			dX[i] = dXGradVec[i];
		}
 	}


	vector<vector<double> > B(Nparam, vector<double>(Nparam));
	vector<vector<double> > D(Nparam, vector<double>(Nparam));

	// Compute initial hessian approximation
	if( initHessFD )
	{
		vector <double> dXHessian(X.size(),dXHess);
		objPtr->hessianApproximation(X,dXHessian,B);

		// compute approximate hessian inverse, D
		matrixInverse( B, D);

	}
	else
	{
		// otherwise use the identity
		for(int i = 0; i < Nparam; i++)
		{
			for(int j = 0; j < Nparam; j++)
			{
				D[i][j] = 0.0;
				if(i == j)
				{
					D[i][j] = 1.0;
					if(initialScalingVec.size() > 0)
					{
						D[i][j] = initialScalingVec[i];
					}
				}
			}
		}
	}



	// Initial gradient approximation
	objPtr->gradientApproximationMPIRecur(X,dX,dFdX,constantX,constantIndicator);


	double F = objPtr->objEvalRecur(X,constantX,constantIndicator);
	f0 = F;

	// Run optimization
	optimFlag = true;
	recurFlag = 0;
	mainBFGSLoop( F,  X, dFdX, D, Xlb, Xub, dX, constantX, constantIndicator );


	// Barrier after solution
	MPI_Barrier(MPI_COMM_WORLD);

    // Save function optimum result
    fOpt = F;

    if( procID == 0 )
    {
    	cout << endl << "----------------------------------------------------------------------------------------------------------" << endl ;
    	cout << "  Completed bounded BFGS." << endl;
    	cout << "  X0 = "; print1DVector(X0);
    	cout << "  Xopt = "; print1DVector(X);
    	cout << "  f0 = " << f0 << ", fOpt = " << fOpt << endl;
    	cout << "----------------------------------------------------------------------------------------------------------" << endl << endl ;
    }



}


void BFGS_Bnd_MPI_SW::mainBFGSLoop( double & F, vector <double> & X, vector<double> & dFdX, vector<vector<double> > & D,
		vector <double> & Xlb, vector <double> & Xub, vector<double> & dX, vector<double> & constantX,
		vector<bool> & constantIndicator)
{
	int Nparam = X.size();

	vector<double> dFdX_prev (Nparam, 0);
	vector<double> p (Nparam, 0);
	vector<double> s (Nparam, 0);
	vector<double> g (Nparam, 0);
	vector<double> Xprev(Nparam,0);

	for(int i = 0; i < Nparam; i++){ dFdX_prev[i] = dFdX[i]; }

    // other variables
    int iter = 0;
    double xdiff = xMinDiff*2;
    double grad2Norm  = 2*minGrad2Norm;
    // Optimization loop
    while (optimFlag && iter < maxIter && xdiff > xMinDiff  && grad2Norm > minGrad2Norm && totalIter < maxIter)
    {

		if( verbose > 0 && procID == 0 && mod(iter,1) == 0 )
		{
			cout << endl << "Iter = " << iter << " of bounded BFGS search starting with previous F = " << F << "." << endl;
		}
    	// ----------- 1. Compute search direction --------------//
    	matrixVectorMultiply( D, dFdX, p );
    	for(int i = 0; i < Nparam; i++){ p[i] = -p[i]; }


    	// ----------- 2. Compute search scaling (alpha) --------------//
    	double alpha, Fopt;
    	cubicInterpolationLineSearchBnd( X, Xlb, Xub, F, dFdX, p,constantX,constantIndicator, alpha, Fopt );


    	// ----------- 3. Update variables and hessian --------------//
    	// Update X
    	for(int i = 0; i < Nparam; i++)
    	{
    		Xprev[i] = X[i]; // store previous values
    		X[i] = X[i] + alpha*p[i];
    	}
    	F = Fopt;

    	// store previous gradient evaluation
    	for(int i = 0; i < Nparam; i++){ dFdX_prev[i] = dFdX[i]; }

    	// Compute updated gradient
    	objPtr->gradientApproximationMPIRecur(X,dX,dFdX,constantX,constantIndicator);


    	// Compute Hessian update
      	for(int i = 0; i < Nparam; i++)
		{
			s[i] = alpha*p[i];
			g[i] = dFdX[i] - dFdX_prev[i];
		}
    	updateHessianInv( D, g, s );


    	// ----------- 4. Check boundary --------------//
    	boundaryAssessment( F, X, p, dFdX, D, Xlb, Xub, dX, constantX, constantIndicator );


    	// Compute differences and norms
    	xdiff = 0;
    	for(int i = 0; i < Nparam; i++){ xdiff += fabs( X[i] - Xprev[i] ); }
    	grad2Norm = vector2Norm ( dFdX );

    	// Display current step
    	if( verbose > 0 && procID == 0 && mod(iter,1) == 0 )
		{
			cout << "  Step completed with F = " << F << " and mean abs xdiff is " << xdiff << " and the grad2norm = " << grad2Norm << endl;

			if( verbose > 1)
			{
				cout << "  X = "; print1DVector(X);
				cout << "  Xlb = "; print1DVector(Xlb);
				cout << "  Xub = "; print1DVector(Xub);
				cout << "  dFdX = "; print1DVector(dFdX);
				cout << "  p = "; print1DVector(p);
			}
		}
        iter = iter+1;
        totalIter = totalIter + 1;
    	MPI_Barrier(MPI_COMM_WORLD);


    }

}

void BFGS_Bnd_MPI_SW::cubicInterpolationLineSearchBnd( vector <double> & X,vector <double> & Xlb, vector <double> & Xub,
		double FX, vector <double> & dFdX, vector <double> & p, vector<double> & constantX, vector<bool> & constantIndicator,
		double & alphaOpt, double & Fopt  )
{
	bool success = false;
	bool bndIndicator = false;
	double alphaMax;
	double phiOpt, dphiOptdalpha;
	double phii;

	alphaOpt = 0;
	phiOpt = FX;
	Fopt = FX;

	// Initial values, with initial slope from gradient computation
	double alpha0 = 0;
	double phi0 = FX;
	double dphi0dalpha = dotProd( dFdX, p );

	// Initial alpha pool
	int Npool = Nprocs+1;
	vector <int> evalIndicator( Npool, 1);
	vector <double> alphaPool( Npool, 0);
	vector <double> phiPool( Npool, 0);
	vector <double> dphidalphaPool( Npool, 0);

	// set first value of pool to initial value
	evalIndicator[0] = 0; // dont evaluate first point
	alphaPool[0] = alpha0;
	phiPool[0] = phi0;
	dphidalphaPool[0] = dphi0dalpha;

	// Compute max value of alpha
	alphaMax = computeAlphaBnd( X, Xlb, Xub, p );

	// Initial step guess
	double alphai = alphaGuess;
	if(alphai > alphaMax)
	{
		alphai = alphaMax;
	}

	// Compute the starting values of alpha
	double delta_alpha = alphai/Nprocs;
	for(int i = 1; i < Npool; i++ )
	{
		alphaPool[i] = delta_alpha*i;
	}

	if( verbose > 1 && procID == 0 )
	{
		cout << endl << "Starting line search in direction " << endl;
		cout << "p = "; print1DVector(p);
		cout << " with alpha0 = " << alphai << endl;
	}

	int iter_ls = 0;
	bool extendFlag = true;
	bool zoomFlag = false;
	while (iter_ls < maxIterLineSearch && extendFlag)
	{

		evaluateAlphaPoolAndDerivativesIndicator( alphaPool, evalIndicator, X, p, constantX, constantIndicator,	phiPool,  dphidalphaPool );



		// 1. Check if interval is large enough (based on magnitude)
		for(int i = 1; i < Npool; i++ )
		{
			if( ( phiPool[i] > phi0 + c1*alphaPool[i]*dphi0dalpha ) || (  phiPool[i] >= phiPool[0] && iter_ls > 1 ) )
			{
				extendFlag = false;
				zoomFlag = true;
			}
		}



		// 2. Check if close enough to optimum
		for(int i = 1; i < Npool; i++ )
		{
			if ( fabs(dphidalphaPool[i]) <= fabs( c2*dphi0dalpha) )
			{
				extendFlag = false;
				zoomFlag = false;
			}
		}


		// 3. Check if interval is large enough (based on slope)
		for(int i = 1; i < Npool; i++ )
		{
			if( dphidalphaPool[i] >= 0)
			{
				extendFlag = false;
				zoomFlag = true;
				success = true;
			}
		}

		// 4. If interval still requires extension, check that the boundary was not already met
		if( extendFlag && alphaPool[Npool-1] == alphaMax )
		{
			extendFlag = false;
			zoomFlag = false;
			bndIndicator = true;
			success = false;
		}

		// 5. Compute new step since no good interval was found
		// Extend search interval as no reasonable interval was found
		if(extendFlag)
		{
			double alphai = (Nprocs+1)*alphaPool[Npool-1];
			if( alphai > alphaMax )
			{
				alphai = alphaMax;
			}

			// compute new pool
			linspace( alphaPool[Npool-1], alphai, Npool, alphaPool );
			for(int i = 0; i < Npool; i++){evalIndicator[i] = 1;}
			evalIndicator[0] = 0; // dont evaluate first point
			phiPool[0] = phiPool[Npool-1];
			dphidalphaPool[0] = dphidalphaPool[Npool-1];

		}
		iter_ls++;



		MPI_Barrier(MPI_COMM_WORLD);

	}

	// If zoom flag was called, compute zoom
	if( zoomFlag )
	{
		double alpha_a, alpha_b, phi_a, phi_b, dphi_a_dalpha, dphi_b_dalpha;
		computeZoomRegion( alphaPool, phiPool, dphidalphaPool,	alpha_a, alpha_b, phi_a, phi_b, dphi_a_dalpha, dphi_b_dalpha );
		// Only zoom if alpha steps are large enough
		if( ( alpha_b - alpha_a > alphaTol ) )
		{
			lineSearchZoomBnd( alpha_a, alpha_b, phi_a, phi_b, dphi_a_dalpha, dphi_b_dalpha, phi0, dphi0dalpha,
				X,  p, constantX, constantIndicator, iter_ls,  alphaOpt, phiOpt, dphiOptdalpha );
		}
		else
		{
			double phiMin;
			int indexMin;
			vectorMin ( phiPool, phiPool.size(), phiMin, indexMin);
			alphaOpt = alphaPool[indexMin];
			phiOpt = phiMin;
			dphiOptdalpha = dphidalphaPool[indexMin];
		}
		// save the result
		Fopt = phiOpt;
	}
	else // Otherwise optimum is simply the best member of the current pool
	{
		// Optimum is always best member of pool
		double phiMin;
		int indexMin;
		vectorMin ( phiPool, phiPool.size(), phiMin, indexMin);
		alphaOpt = alphaPool[indexMin];
		phiOpt = phiMin;
		dphiOptdalpha = dphidalphaPool[indexMin];

		// set Fopt
		Fopt = phiOpt;
	}

	// If the search exited because it took too long, set the optimum as the best current option
    if( verbose > 0 && procID == 0 )
    {
    	if(!bndIndicator)
    	{
    		cout << "  Line search completed with alpha = " << alphaOpt << " and F = " << Fopt << " after " << iter_ls << " iterations. "
    				<< "Note: alphaMax = " << alphaMax << endl ;
    	}
    	else
    	{
        	cout << "  ! Line search terminated at boundary with alpha = " << alphaOpt << " and F = " << Fopt << " after " << iter_ls << " iterations. "
    				<< "Note: alphaMax = " << alphaMax << endl ;
    	}
    }


}

void computeZoomRegion(vector <double> & alphaPool, vector <double> & phiPool, vector <double> & dphidalphaPool,
		 double & alpha_a, double & alpha_b, double & phi_a, double & phi_b, double & dphi_a_dalpha, double & dphi_b_dalpha )
{

	double phiMin;
	int indexMin;
	vectorMin ( phiPool, phiPool.size(), phiMin, indexMin);

	// If the slope is positive at the minimum, use the point to the left (the values of alpha are always monotonically increasing)
	if( dphidalphaPool[indexMin] > 0 )
	{
		alpha_a = alphaPool[indexMin-1];
		alpha_b = alphaPool[indexMin];

		phi_a = phiPool[indexMin-1];
		phi_b = phiPool[indexMin];

		dphi_a_dalpha = dphidalphaPool[indexMin-1];
		dphi_b_dalpha = dphidalphaPool[indexMin];
	}
	else // Otherwise, use the point to the right
	{
		alpha_a = alphaPool[indexMin];
		alpha_b = alphaPool[indexMin+1];

		phi_a = phiPool[indexMin];
		phi_b = phiPool[indexMin+1];

		dphi_a_dalpha = dphidalphaPool[indexMin];
		dphi_b_dalpha = dphidalphaPool[indexMin+1];
	}

}


void computeZoomPool(double alpha_a, double alpha_b, double phi_a, double phi_b, double dphi_a_dalpha, double dphi_b_dalpha,
		vector <double> & alphaPool, vector <double> & phiPool, vector <double> & dphidalphaPool, vector <int> & evalIndicator )
{
	int Npool = alphaPool.size();
	double alpha_c = cubicInterpMinSimple( alpha_a, alpha_b, phi_a, phi_b, dphi_a_dalpha, dphi_b_dalpha );

	if(alpha_c == ( alpha_a  + alpha_b )/2 )
	{
		linspace( alpha_a, alpha_b, Npool, alphaPool );
	}
	else
	{
		int Nlinear = Npool-1;
		vector<double> alphaLinear(Nlinear,0);
		linspace( alpha_a, alpha_b, Nlinear, alphaLinear );


		alphaPool[0] = alphaLinear[0];
		int iLinear = 1;
		for(int i = 1; i < alphaPool.size(); i++)
		{
			if( alpha_c >= alphaLinear[iLinear-1] && alpha_c <= alphaLinear[iLinear] )
			{
				alphaPool[i] = alpha_c;
				alpha_c = -1;
			}
			else
			{
				alphaPool[i] = alphaLinear[iLinear];
				iLinear++;
			}
		}

	}

	// Set eval indicator to on
	for(int i = 0; i < Npool; i++){evalIndicator[i] = 1;}

	// The first and last values were computed in the last step, so place those and do not recompute
	phiPool[0] = phi_a;
	dphidalphaPool[0] = dphi_a_dalpha;
	evalIndicator[0] = 0;

	phiPool[Npool-1] = phi_b;
	dphidalphaPool[Npool-1] = dphi_b_dalpha;
	evalIndicator[Npool-1] = 0;


}

void BFGS_Bnd_MPI_SW::lineSearchZoomBnd( double alpha_a, double alpha_b, double phi_a, double phi_b, double dphi_a_dalpha, double dphi_b_dalpha,
		double phi0, double dphi0dalpha, vector <double> & X, vector <double> & p, vector<double> & constantX, vector<bool> & constantIndicator,
		int & iter_ls, double & alphaOpt, double & phiOpt, double & dphiOptdalpha  )
{

	// Initial alpha pool
	int Npool = Nprocs+2;
	vector <int> evalIndicator( Npool, 1);
	vector <double> alphaPool( Npool, 0);
	vector <double> phiPool( Npool, 0);
	vector <double> dphidalphaPool( Npool, 0);


	bool zoomFlag = true;
	bool success = false;
	while (iter_ls < maxIterLineSearch && ( alpha_b - alpha_a > alphaTol ) && zoomFlag )
	{

		// 0. Compute new alpha pool, and evaluate it
		computeZoomPool(alpha_a, alpha_b, phi_a, phi_b, dphi_a_dalpha, dphi_b_dalpha, alphaPool, phiPool, dphidalphaPool, evalIndicator );
		evaluateAlphaPoolAndDerivativesIndicator( alphaPool, evalIndicator, X, p, constantX, constantIndicator,	phiPool,  dphidalphaPool );


		// 1. Check if close enough to optimum first
		for(int i = 1; i < Npool-1; i++)
		{
			if( (phiPool[i] <= phi0 + c1*alphaPool[i]*dphi0dalpha) && ( fabs(dphidalphaPool[i]) <= fabs(c2*dphi0dalpha)) )
			{
				zoomFlag = false;
				success = true;
			}
		}

		// 2. If the optimum is not found yet, decide new interval
		if(!success)
		{
			computeZoomRegion( alphaPool, phiPool, dphidalphaPool, alpha_a, alpha_b, phi_a, phi_b, dphi_a_dalpha, dphi_b_dalpha );
		}

		iter_ls ++;
		MPI_Barrier(MPI_COMM_WORLD);

	}

	// Optimum is always best member of pool
	double phiMin;
	int indexMin;
	vectorMin ( phiPool, phiPool.size(), phiMin, indexMin);


	alphaOpt = alphaPool[indexMin];
	phiOpt = phiMin;
	dphiOptdalpha = dphidalphaPool[indexMin];


	// Display current search alphas
	if(verbose > 2 && procID == ROOT_ID)
	{
		cout << "Completed zooom search with... " << endl;
		cout << "alphaPool = "; print1DVector(alphaPool);
		cout << "phiPool = "; print1DVector(phiPool);
		cout << "   so phiOpt = FOpt = " << phiOpt << endl;
		cout << "-----" << endl;
	}
}



void BFGS_Bnd_MPI_SW::evaluateAlphaPoolAndDerivativesIndicator( vector <double> & alphaPool, vector<int> evalIndicator,
		vector <double> & X, vector <double> & p, vector<double> & constantX, vector<bool> & constantIndicator,
		vector <double> & phiPool, vector <double> & dphidalphaPool )
{
	vector <double> alphaPoolTemp;
	vector <double> phiPoolTemp;
	vector <double> dphidalphaPoolTemp;

	for(int i = 0; i < evalIndicator.size(); i++)
	{
		if(evalIndicator[i] == 1)
		{
			alphaPoolTemp.push_back(alphaPool[i]);
			phiPoolTemp.push_back(phiPool[i]);
			dphidalphaPoolTemp.push_back(dphidalphaPool[i]);
		}
	}
	evaluateAlphaPoolAndDerivatives( alphaPoolTemp,	X, p, constantX, constantIndicator, phiPoolTemp, dphidalphaPoolTemp );

	int idxTemp = 0;
	for(int i = 0; i < evalIndicator.size(); i++)
	{
		if(evalIndicator[i] == 1)
		{
			alphaPool[i] = alphaPoolTemp[idxTemp];
			phiPool[i] = phiPoolTemp[idxTemp];
			dphidalphaPool[i] = dphidalphaPoolTemp[idxTemp];

			idxTemp ++;
		}
	}



	// Display current search alphas
	if(verbose > 2 && procID == ROOT_ID)
	{
		cout << "alphaPool = "; print1DVector(alphaPool);
		cout << "phiPool = "; print1DVector(phiPool);
		cout << "dphidalphaPool = "; print1DVector(dphidalphaPool);
		cout << "-----" << endl;
	}
}




void BFGS_Bnd_MPI_SW::evaluateAlphaPoolAndDerivatives( vector <double> & alphaPool,	vector <double> & X, vector <double> & p,
		vector<double> & constantX, vector<bool> & constantIndicator, vector <double> & phiPool, vector <double> & dphidalphaPool )
{
	// Ensure that all processes are using values on ROOT
	MPI_Bcast(alphaPool.data(), alphaPool.size(), MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);
	MPI_Bcast(X.data(), X.size(), MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);
	MPI_Bcast(p.data(), p.size(), MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);
	MPI_Bcast(constantX.data(), constantX.size(), MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);
	// Broadcast the constant indicator requires change in the type
	vector <int> constantIndicatorTemp (constantIndicator.size(),0);
	for(int i = 0; i < constantIndicator.size(); i++){ constantIndicatorTemp[i] = (int) constantIndicator[i]; }
	MPI_Bcast(constantIndicatorTemp.data(), constantIndicatorTemp.size(), MPI_INT, ROOT_ID, MPI_COMM_WORLD);
	for(int i = 0; i < constantIndicator.size(); i++){ constantIndicator[i] = (bool) constantIndicatorTemp[i]; }

	// Display current search alphas


	int N = alphaPool.size();

	double * phi_local = new double[N];
	double * phi = new double [N];
	double * dphidalpha_local = new double[N];
	double * dphidalpha = new double [N];
	for(int i = 0; i < N; i++ )
	{
		phi_local[i] = 0;
		phi[i] = 0;
		dphidalpha_local[i] = 0;
		dphidalpha[i] = 0;
	}


	// compute load balance
	vector<int>procEvalID(N,0);
	int procIdx = 0;
	for(int k = 0; k < N; k++)
	{
		if( procIdx > Nprocs - 1 )
		{
			procIdx = 0;
		}
		procEvalID[k] = procIdx;

		procIdx = procIdx+1;
	}


	// evaluate the pool
	for(int i = 0; i < N; i++ )
	{
		if ( procEvalID[i] == procID  )
		{
			phi_local[i] = lineSearchObj( alphaPool[i], X, p, constantX, constantIndicator  );
			dphidalpha_local[i] = lineSearchFDDerivative( alphaPool[i], phi_local[i], X, p, constantX, constantIndicator );
		}
	}


	for(int i = 0; i < N; i++ )
	{

		if( phi_local[i] != phi_local[i] || isinf(phi_local[i]) )
		{
			phi_local[i] = 1e10;
			cout << "Line search crashed with " << endl;
			cout << phi_local[i] << endl;
			cout << "Ending search..." << endl;
		}

	}


	// collect result
	MPI_Reduce(phi_local, phi, N, MPI_DOUBLE, MPI_SUM, ROOT_ID, MPI_COMM_WORLD);
	MPI_Reduce(dphidalpha_local, dphidalpha, N, MPI_DOUBLE, MPI_SUM, ROOT_ID, MPI_COMM_WORLD);
	for(int i = 0; i < N; i++ )
	{
		phiPool[i] = phi[i];
		dphidalphaPool[i] = dphidalpha[i];

	}
	// Make sure that everyone has the same values as root
	MPI_Bcast(phiPool.data(), phiPool.size(), MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);
	MPI_Bcast(dphidalphaPool.data(), phiPool.size(), MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);

	for(int i = 0; i < N; i++ )
	{

		if( phiPool[i] == 1e10 )
		{
			cout << "Line search crashed ... ending search..." << endl;
			optimFlag = false;
		}

	}

	delete [] phi_local;
	delete [] phi;
	delete [] dphidalpha_local;
	delete [] dphidalpha;
}



double BFGS_Bnd_MPI_SW::lineSearchObj( double alpha, vector <double> & X, vector <double> & p, vector<double> & constantX, vector<bool> & constantIndicator )
{
	vector <double> Xalphap(X.size(),0);

	for(int i = 0; i < X.size(); i++)
	{
		Xalphap[i] = X[i] + alpha*p[i];
	}

	double value = objPtr->objEvalRecur(Xalphap,constantX,constantIndicator);
	return value;

}



double BFGS_Bnd_MPI_SW::lineSearchFDDerivative( double alpha, double phialpha, vector <double> & X, vector <double> & p,
		vector<double> & constantX, vector<bool> & constantIndicator )
{
	vector <double> Xalphap_dalpha(X.size(),0);

	for(int i = 0; i < X.size(); i++)
	{
		Xalphap_dalpha[i] = X[i] + (alpha+dalpha)*p[i];
	}

	double Falpha_dalpha = objPtr->objEvalRecur(Xalphap_dalpha,constantX,constantIndicator);
	double dFdalpha = ( Falpha_dalpha - phialpha )/dalpha;

	return dFdalpha;

}






void BFGS_Bnd_MPI_SW::boundaryAssessment( double & F, vector <double> & X, vector <double> & p, vector<double> & dFdX, vector<vector<double> > & D,
		vector <double> & Xlb, vector <double> & Xub, vector<double> & dX, vector<double> & constantX, vector<bool> & constantIndicator )
{
	int j;
	int irecur;
	vector<int> idxCurrRecur;

	int Ndim_current = X.size();
	vector <double> constantX_current(Ndim_current,0);
	vector <bool> constantIndicator_current(Ndim_current,false);

	int Ndim = constantX.size();
	bool bndFlag = false;

	// 1. Check if the search direction or the steepest descent direction point out of the domain
	// 	  if so, eliminate that variable.
	int iCurrent = 0;
	for(int i = 0; i < Ndim; i++ )
	{
		if( !constantIndicator[i] )
		{
			if( (fabs(X[iCurrent] - Xlb[iCurrent]) < bndTol )  &&  ( (p[iCurrent] < 0) || (dFdX[iCurrent] > 0) ) )
			{
				bndFlag = true;

				constantIndicator[i] = true;
				constantX[i] = X[iCurrent];

				constantIndicator_current[iCurrent] = true;
				constantX_current[iCurrent] = X[iCurrent];
				idxCurrRecur.push_back(i);
			}
			else if ( (fabs(X[iCurrent] - Xub[iCurrent]) < bndTol )  && ( (p[iCurrent] > 0) || (dFdX[iCurrent] < 0) )  )
			{
				bndFlag = true;

				constantIndicator[i] = true;
				constantX[i] = X[iCurrent];

				constantIndicator_current[iCurrent] = true;
				constantX_current[iCurrent] = X[iCurrent];
				idxCurrRecur.push_back(i);
			}
			iCurrent++;
		}
	}


	// Number of constant variables
	int Nconst = 0;
	for(int i = 0; i < constantIndicator.size(); i++ )
	{
		Nconst = Nconst + constantIndicator[i];
	}

	if(bndFlag && verbose > 0 && procID == 0 )
	{
		cout << endl << "-------------------------------------------------------------------------------------------------" << endl;
		cout << "Optimizer reached box boundary and found that the steepest descent is directed outside of the boundary at" << endl;
		cout << "    constantIndicator = "; print1DVector(constantIndicator);
		cout << "    constantX = "; print1DVector(constantX);
		cout << endl << "Variable info:" << endl;
		cout << "    X = "; print1DVector(X);
		cout << "    F = "; cout << F << endl;
		cout << "    Xlb = "; print1DVector(Xlb);
		cout << "    Xub = "; print1DVector(Xub);
		cout << "    dFdX = "; print1DVector(dFdX);
		cout << "-------------------------------------------------------------------------------------------------" << endl;

	}

	// Recursive call to the minimizer with variables at the boundary eliminated
	int NdimRecur = Ndim - Nconst; // number of variables remaining
	if( bndFlag && NdimRecur > 0) // only continue if at least one variable remains
	{

		 double FRecur = F;
		 vector <double> XRecur(NdimRecur,0);
		 vector <double> dFdXRecur(NdimRecur,0);
		 vector <double> XlbRecur(NdimRecur,0);
		 vector <double> XubRecur(NdimRecur,0);
		 vector <double> dXRecur(NdimRecur,0);

		 vector<vector<double> > DRecur (NdimRecur, vector<double>(NdimRecur));
		 double f0Recur;
		 double fRecur;

		 irecur = 0;
		 for( iCurrent = 0; iCurrent < Ndim_current; iCurrent++)
		 {
			 if( !constantIndicator_current[iCurrent] )
			 {
				 FRecur = F;
				 XRecur[irecur] = X[iCurrent];
				 dFdXRecur[irecur] = dFdX[iCurrent];
				 XlbRecur[irecur] = Xlb[iCurrent];
				 XubRecur[irecur] = Xub[iCurrent];
				 dXRecur[irecur] = dX[iCurrent];

				 irecur++;
			 }
		 }
		 // Start with steepest descent direction
		 setIdentity( DRecur );
		 // use scaling if it was defined
		 j = 0;
		 for(int i = 0; i < Ndim; i++)
		 {
			if( !constantIndicator[i] && initialScalingVec.size() > 0 )
			{
				DRecur[j][j] = initialScalingVec[i];
				j++;
			}
		 }

		 // 2. Call the main solution with recursive variables
  		 recurFlag = true; // set recur flag to true
		 mainBFGSLoop( FRecur, XRecur, dFdXRecur, DRecur, XlbRecur, XubRecur, dXRecur, constantX, constantIndicator );


		 // 3. Set the values of the original variable
		 irecur = 0;
		 for( iCurrent = 0; iCurrent < Ndim_current; iCurrent++ )
		 {
			 if( !constantIndicator_current[iCurrent] )
			 {
				 F = FRecur;
				 X[iCurrent] = XRecur[irecur];
				 dFdX[iCurrent] = dFdXRecur[irecur];
				 Xlb[iCurrent] = XlbRecur[irecur];
				 Xub[iCurrent] = XubRecur[irecur];
				 dX[iCurrent] =  dXRecur[irecur];

				 irecur++;
			 }
		 }

		 // Turn off the constant indicator before gradient computation
		 for(int k = 0; k < idxCurrRecur.size(); k++ )
		 {
			 constantIndicator[idxCurrRecur[k]] = false;
		 }

		// Reset D as identity
		setIdentity( D );
		 // use scaling if it was defined
		 j = 0;
		 for(int i = 0; i < Ndim; i++)
		 {
			if( !constantIndicator[i] && initialScalingVec.size() > 0 )
			{
				D[j][j] = initialScalingVec[i];
				j++;
			}
		 }


		// 4. Compute the gradient of the original variables
		objPtr->gradientApproximationMPIRecur(X, dX, dFdX, constantX, constantIndicator);


		// 5. Check if solution can continue in the domain interior
		bool continueFlag = false;
    	for( iCurrent = 0; iCurrent < Ndim_current; iCurrent++ )
		{
    		if( constantIndicator_current[iCurrent] )
			{
    			if( (fabs(X[iCurrent] - Xlb[iCurrent]) < bndTol )  &&  (  dFdX[iCurrent] < 0 ) )
				{
					continueFlag = true;
				}
				else if ( (fabs(X[iCurrent] - Xub[iCurrent]) < bndTol )  && ( dFdX[iCurrent] > 0 )  )
				{
					continueFlag = true;
				}
			}
		}


		Nconst = 0;
		for(int i = 0; i < constantIndicator.size(); i++ )
		{
			Nconst = Nconst + constantIndicator[i];
		}
		if( Nconst == 0 )
			recurFlag = false;

		if( continueFlag )
		{
			optimFlag = true;
			if(verbose > 1 && procID == 0 )
			{
				cout << endl << "-------------------------------------------------------------------------------------------------" << endl;
				cout << "     Optimization continuing after recursive boundary optimization, "<< endl;
				cout << "     as the gradient through the boundary points to the domain interior." << endl;
				cout << "-------------------------------------------------------------------------------------------------" << endl;
			}

		}
		else
		{
			optimFlag = false;
			if( verbose > 1 && procID == 0 )
			{
				cout << endl << "-------------------------------------------------------------------------------------------------" << endl;
				cout << "     Optimization exiting after recursive boundary optimization, " << endl;
				cout << "     as the gradient through the boundary still points out of the domain." << endl;
				cout << "-------------------------------------------------------------------------------------------------" << endl;
			}
		}


	}
	else if (NdimRecur == 0)
	{
		optimFlag = false;
		if( verbose > 1 && procID == 0 )
		{
			cout << endl << "-------------------------------------------------------------------------------------------------" << endl;
			cout << "      NO VARIABLES LEFT TO OPTIMIZE!!!! " << endl;
			cout << "-------------------------------------------------------------------------------------------------" << endl;
		}
	}



}

