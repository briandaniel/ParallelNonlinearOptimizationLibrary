/*
 * BFGS_with_linesearch_MPI.cpp
 *
 *  Created on: Nov 6, 2018
 *      Author: brian
 */

#include "BFGS_with_linesearch_MPI.hpp"



void BFGS_MPI::findMin( vector <double> & X, double & f0, double & fOpt  ){

	int Nparam = X.size();

	// Get the number of processes and the current processor ID
	int Nprocs, procID;
	MPI_Comm_size(MPI_COMM_WORLD, &Nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &procID);

	// vectors
	vector<double> Xprev(Nparam,0);
 	vector<double> dX (Nparam, dXGrad);
	vector<double> dFdX (Nparam, 0);
	vector<double> dFdX_prev (Nparam, 0);


	vector<double> p (Nparam, 0);
	vector<double> s (Nparam, 0);
	vector<double> g (Nparam, 0);


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
				}
			}
		}
	}


	// Initial gradient approximation
	objPtr->gradientApproximationMPI(X,dX,dFdX);
	for(int i = 0; i < Nparam; i++){ dFdX_prev[i] = dFdX[i]; }
	double F = objPtr->objEval(X);
	f0 = F;

    // other variables
    int iter = 0;
    double xdiff = xMinDiff*2;
    double grad2Norm  = 2*minGrad2Norm;
    // Optimization loop
    while (iter < maxIter && xdiff > xMinDiff  && grad2Norm > minGrad2Norm )
    {

    	// store previous gradient evaluation
    	for(int i = 0; i < Nparam; i++){ dFdX_prev[i] = dFdX[i]; }

    	// Compute search direction
    	matrixVectorMultiply( D, dFdX, p );
    	for(int i = 0; i < Nparam; i++){ p[i] = -p[i]; }


    	// Compute scaling
    	double alpha, Fopt;
    	secantLineSearch( X, F, dFdX, p, alpha, Fopt  );



    	// Update X
    	for(int i = 0; i < Nparam; i++)
    	{
    		Xprev[i] = X[i]; // store previous values
    		X[i] = X[i] + alpha*p[i];
    	}
    	F = Fopt;


    	// Compute updated gradient
    	objPtr->gradientApproximationMPI(X,dX,dFdX);

    	// Compute Hessian update
      	for(int i = 0; i < Nparam; i++)
		{
			s[i] = alpha*p[i];
			g[i] = dFdX[i] - dFdX_prev[i];
		}
    	updateHessianInv( D, g, s );


    	// cout << "X = "; print1DVector(X);
    	// cout << "D = "; print2DVector(D);

    	xdiff = 0;
    	for(int i = 0; i < Nparam; i++){ xdiff += fabs( X[i] - Xprev[i] ); }
    	grad2Norm = vector2Norm ( dFdX );

        if( verbose == true && mod(iter,1) == 0 && procID == 0 )
		{
			cout << "---> At iter = " << iter << " the mean abs xdiff is " << xdiff << " and the grad2norm = " << 	grad2Norm << endl;
			cout << "                 with a minimum function evaluation of " <<  F << endl;
		}

        iter = iter+1;
    }

    fOpt = F;

    if( verbose == true && procID == 0)
    {
    	cout << endl << "-----------------------------------------------------------------------------------" << endl ;

    	cout << "Completed bfgs." << endl;
    	cout << "f0 = " << f0 << ", fOpt = " << fOpt << " with variable:" << endl;
    	cout << "X = "; print1DVector(X);
    	cout << "-----------------------------------------------------------------------------------" << endl << endl ;
    }



}




double BFGS_MPI::lineSearchObj( double alpha, vector <double> & X, vector <double> & p  )
{
	vector <double> Xalphap(X.size(),0);

	for(int i = 0; i < X.size(); i++)
	{
		Xalphap[i] = X[i] + alpha*p[i];
	}

	double value = objPtr->objEval(Xalphap);
	return value;

}



void BFGS_MPI::evalAlphaPoolMPI( vector <double> & alphaPool, vector <double> & phiPool, vector <double> & X, vector <double> & p  )
{

	// Get the number of processes and the current processor ID
	int Nprocs, procID;
	MPI_Comm_size(MPI_COMM_WORLD, &Nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &procID);

	int N = phiPool.size();

	double * phi_local = new double[N];
	double * phi = new double [N];
	for(int i = 0; i < N; i++ )
	{
		phi_local[i] = 0;
		phi[i] = 0;
	}

	// Compute load distribution
	// number of function evaluations


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
			phi_local[i] = lineSearchObj( alphaPool[i], X, p );
		}
	}

	// collect result
	MPI_Allreduce(phi_local, phi, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


	for(int i = 0; i < N; i++ )
	{
		phiPool[i] = phi[i];
	}


	// clean up
	delete [] phi_local;
	delete [] phi;

}


void BFGS_MPI::secantLineSearch( vector <double> & X, double FX,
		vector <double> & dFdX, vector <double> & p, double & alphaOpt, double & Fopt  )
{
	// Get the number of processes and the current processor ID
	int Nprocs, procID;
	MPI_Comm_size(MPI_COMM_WORLD, &Nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &procID);

	// As many evaluations as available processors
	int Npool = Nprocs;
	vector<double> alphaPool(Npool,0);
	vector<double> phiPool(Npool,0);
	vector<double> alphaPoolPrev(Npool,-1);
	vector<double> phiPoolPrev(Npool,0);
	vector<double> poolSecantSlope(Npool,0);

	alphaOpt = 0;
	Fopt = FX;

	// Initial values
	double alpha0 = 0;
	double phi0 = FX;
	// Compute initial slope from gradient computation
	double dphi0dalpha = dotProd( dFdX, p );


	// Initial pool
	int idxMin = -ceil((Npool-1.0)/2.0);
	int idxMax = floor((Npool-1.0)/2.0);
	double r = pow(maxAlphaMult, 1.0/(double) idxMax);
	int idx = idxMin;
	for(int k =0; k < Npool; k++)
	{
		alphaPool[k] = alphaGuess*pow(r,idx);
		idx++;
	}

	bool firstFlag = true;
	bool zoomFlag = false;
	int iter = 0;
	while (iter < maxIterLineSearch && firstFlag )
	{

		// evaluate the pool
		evalAlphaPoolMPI( alphaPool, phiPool, X,  p  );

		// 1. check suff. decrease
		for(int i = 0; i < Npool; i++)
		{
			if( phiPool[i] > phi0 + c1*alphaPool[i]*dphi0dalpha  )
			{
				zoomFlag = true;
				firstFlag = false;
			}
		}

		// 2. Check curvature against secant slopes
		poolSecantSlope[0] = (phiPool[0] - phi0)/(alphaPool[0] - alpha0);
		for(int i = 1; i < Npool; i++)
		{
			poolSecantSlope[i] = ( phiPool[i] - phiPool[i-1])/(alphaPool[i] - alphaPool[i-1]);
		}

		if(firstFlag)
		{
			for(int i = 0; i < Npool; i++)
			{
				if ( fabs(poolSecantSlope[i]) <= fabs( c2*dphi0dalpha) )
				{
					zoomFlag = false;
					firstFlag = false;
				}
			}
		}


		// 3. Check for positive (secant) slopes, if available, zoom in
		if(firstFlag)
		{
			for(int i = 0; i < Npool; i++)
			{
				if(poolSecantSlope[i] >= 0)
				{
					zoomFlag = true;
					firstFlag = false;
				}
			}
		}

		// 4. Otherwise extend the interval and loop
		if(firstFlag)
		{
			double alphaMax; int indexMax;
			vectorMax ( alphaPool, alphaPool.size(), alphaMax, indexMax);
			r = pow( maxAlphaMult, 1.0/(double) Npool );
			for(int i = 0; i < Npool; i++)
			{
				alphaPoolPrev[i] = alphaPool[i];
				phiPoolPrev[i] = phiPool[i];

				double power = i+1;
				alphaPool[i] = alphaMax*pow(r,power);

			}
		}

	  	if( verbose == true && mod(iter,1) == 0  && procID == 0)
		{
	  		/*
	  		if(firstFlag)
	  		{

				cout << "Line search at iter = " << iter << " has not found a reasonable region. Continuing line search with larger region  " << endl;
				cout << "alpha = "; print1DVector(alphaPool);
				cout << "Previous values are" << endl;
				cout << "alpha prev = "; print1DVector(alphaPoolPrev);
				cout << "phi prev = "; print1DVector(phiPoolPrev);
	  		}
			else if (zoomFlag)
			{
				cout << "Line search at iter = " << iter << " has found a reasonable region. Zooming. " << endl;
				cout << "values are" << endl;
				cout << "alpha = "; print1DVector(alphaPool);
				cout << "phi = "; print1DVector(phiPool);
			}
			else
			{
				cout << "Line search completed during first loop" << endl;
			}
			*/

		}

		iter++;
	}

	// find pool bounds
	double alpha_lo, alpha_hi, phi_lo, phi_hi;
	if(alphaPoolPrev[0] < 0)
	{
		findPoolBounds( alphaPool, phiPool, alpha0, phi0, alpha_lo, alpha_hi, phi_lo, phi_hi);
	}
	else
	{

		vector<double> alphaPoolEval(Npool*2,0);
		vector<double> phiPoolEval(Npool*2,0);
		for(int i = 0; i < Npool; i++)
		{
			alphaPoolEval[i] = alphaPoolPrev[i];
			alphaPoolEval[i+Npool] = alphaPool[i];

			phiPoolEval[i] = phiPoolPrev[i];
			phiPoolEval[i+Npool] = phiPool[i];
		}
		findPoolBounds( alphaPoolEval, phiPoolEval, alpha0, phi0, alpha_lo, alpha_hi, phi_lo, phi_hi);
	}
  	if( verbose == true && procID == 0)
	{
  		// cout << " alpha lo = " << alpha_lo << " alpha hi = " << alpha_hi << " phi lo = " << phi_lo << " phi hi = " << phi_hi << endl;
	}


  	// Second loop (zoom)
  	iter = 0;
	vector <double> alphaPool2(Npool+2,0);
	vector <double> phiPool2(Npool+2,0);
	vector <double> t(Npool+2,0);
	while (iter < maxIterLineSearch && zoomFlag )
	{
		// 1. Compute new pool locations
		/*
		linspace(0,1,Npool+2,t);
		double w_lo = 1;
		double w_hi = 1;
		if( phi_hi > phi0 )
		{
			w_lo = 1;
			w_hi = pow(abs(phi0)/abs(phi_hi-phi0), 0.25);
			w_hi = 1;
		}
		for(int i = 0; i < Npool+2; i++)
		{
			alphaPool2[i] = (w_lo*alpha_lo*t[i] + w_hi*alpha_hi*(1-t[i])) / (w_lo*t[i] + w_hi*t[i] );
		}
		alphaPool2[0] = alpha_hi;
		alphaPool2[Npool+1] = alpha_lo;
		phiPool2[0] = phi_hi;
		phiPool2[Npool+1] = phi_lo;
		*/
		linspace(alpha_lo,alpha_hi,Npool+2,alphaPool2);
		phiPool2[0] = phi_lo;
		phiPool2[Npool+1] = phi_hi;
		alphaPool2[0] = alpha_lo;
		alphaPool2[Npool+1] = alpha_hi;


		// 2. evaluate new pool
		for(int i = 0; i < Npool; i++)
		{
			alphaPool[i] = alphaPool2[i+1];
			phiPool[i] = phiPool2[i+1];
 		}
		evalAlphaPoolMPI( alphaPool, phiPool, X,  p  ); // actual evaluation
		for(int i = 0; i < Npool; i++)
		{
			alphaPool2[i+1] = alphaPool[i];
			phiPool2[i+1] = phiPool[i];
		}


		// 3. Check curvature against secant slopes
		for(int i = 0; i < Npool; i++)
		{
			poolSecantSlope[i] = ( phiPool2[i+1] - phiPool2[i])/(alphaPool2[i+1] - alphaPool2[i]);
		}


		for(int i = 0; i < Npool; i++)
		{

			if ( fabs(poolSecantSlope[i]) <= fabs( c2*dphi0dalpha) )
			{
				zoomFlag = false;
			}
		}

		if(zoomFlag)
			findPoolBounds( alphaPool2, phiPool2, alpha0, phi0, alpha_lo, alpha_hi, phi_lo, phi_hi);




	  	if( verbose == true && mod(iter,1) == 0  && procID == 0)
		{
	  		if(zoomFlag)
	  		{

	  			//cout << "Zoom search at iter = " << iter << " has not found a reasonable region." << endl;
	  			//cout << "     phi0 = " << phi0 << endl;
				//cout << "     alpha pool  = "; print1DVector(alphaPool2);
				//cout << "     at phi pool = "; print1DVector(phiPool2);
	  		}
	  		else
	  		{
	  			//cout << "Zoom search ending at iter = " << iter << " has found a reasonable region." << endl;

				//cout << "alpha pool  = "; print1DVector(alphaPool2);
				//cout << "at phi pool = "; print1DVector(phiPool2);
	  		}
		}

		iter++;
	}


	// Place minimum in export variable
	double phiMin;
	vectorMin ( phiPool2, phiPool2.size(), phiMin, idxMin);
	double alphaMin = alphaPool2[idxMin];

	alphaOpt = alphaMin;
	Fopt = phiMin;



}



void findPoolBounds( vector<double> & alphaPool, vector<double> & phiPool, double alpha0, double phi0,
		double & alpha1, double & alpha2, double & phi1, double & phi2)
{
	double phiMin;
	int idxMin;
	vectorMin ( phiPool, phiPool.size(), phiMin, idxMin);
	double alphaMin = alphaPool[idxMin];


	if(phi0 < phiMin) // If the starting position is still the best, search only the region closest to the starting location
	{
        alpha1 = alpha0;
        alpha2 = alphaPool[0];

        phi1 = phi0;
        phi2 = phiPool[0];
	}
    else if ( phi0 >= phiMin && idxMin == 0 ) // If the best point is the first, have to use the starting point
    {
		alpha1 = alpha0;
		alpha2 = alphaPool[idxMin+1];

		phi1 = phi0;
		phi2 = phiPool[idxMin+1];
    }
	else // Otherwise just use the two points that neighbor the best point
	{
		alpha1 = alphaPool[idxMin-1];
		alpha2 = alphaPool[idxMin+1];

		phi1 = phiPool[idxMin-1];
		phi2 = phiPool[idxMin+1];

	}
}

























