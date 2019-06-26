/*
 * BFGS_with_bnd_linsearch_MPI.cpp
 *
 *  Created on: Nov 7, 2018
 *      Author: brian
 */



#include "BFGS_with_bnd_linesearch_MPI.hpp"



void BFGSBnd_MPI::findMinBnd( vector <double> & X, vector <double> & Xlb, vector <double> & Xub, double & f0, double & fOpt  ){


	// Get the number of processes and the current processor ID
	int Nprocs, procID;
	MPI_Comm_size(MPI_COMM_WORLD, &Nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &procID);

	int Nparam = X.size();

	// vectors
	vector<double> constantX(Nparam,0);
	vector<bool> constantIndicator(Nparam,false);

 	vector<double> dX (Nparam, dXGrad);
	vector<double> dFdX (Nparam, 0);

	vector<vector<double> > B(Nparam, vector<double>(Nparam));
	vector<vector<double> > D(Nparam, vector<double>(Nparam));

	// Check that the initial guess is within the boundary
	checkBoxBounds( X, Xlb, Xub );


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
		setIdentity( D );
	}


	// Initial gradient approximation
	objPtr->gradientApproximationMPI(X,dX,dFdX);
	double F = objPtr->objEval(X);
	f0 = F;
	bool optimFlag = true;
	bool recurFlag = false;

	// Main solution loop call
	mainBFGSLoop( F, X, dFdX, D, Xlb, Xub, dX, constantX, constantIndicator, optimFlag, recurFlag );

	fOpt = F;


    if( verbose == true && procID == 0)
    {
    	cout << endl << "-----------------------------------------------------------------------------------" << endl ;

    	cout << "Completed bounded bfgs." << endl;
    	cout << "f0 = " << f0 << ", fOpt = " << fOpt << " with variable:" << endl;
    	cout << "X = "; print1DVector(X);
    	cout << "Xlb = "; print1DVector(Xlb);
    	cout << "Xub = "; print1DVector(Xub);
    	cout << "-----------------------------------------------------------------------------------" << endl << endl ;
    }



}


void BFGSBnd_MPI::mainBFGSLoop( double & F, vector <double> & X, vector<double> & dFdX, vector<vector<double> > & D,
		vector <double> & Xlb, vector <double> & Xub, vector<double> & dX, vector<double> & constantX,
		vector<bool> & constantIndicator, bool & optimFlag, bool & recurFlag)
{

	// Ensure that all processors have the same data
	MPI_Bcast(&F, 1, MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);
	MPI_Bcast(X.data(), X.size(), MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);
	MPI_Bcast(dFdX.data(), dFdX.size(), MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);



	double Fprev = 2*F;



	// Get the number of processes and the current processor ID
	int Nprocs, procID;
	MPI_Comm_size(MPI_COMM_WORLD, &Nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &procID);
	if( verbose == true && procID == 0)
	{
		cout << endl << "-----------------------------------------------------------------------------------" << endl ;

		cout << "Starting bounded BFGS loop with." << endl;
		cout << "    X = "; print1DVector(X);
		cout << "    F(X) = " << F << endl;
		cout << "    Xlb = "; print1DVector(Xlb);
		cout << "    Xub = "; print1DVector(Xub);
		cout << "    dFdX = "; print1DVector(dFdX);
		// cout << "    D = "; print2DVector(D);

		cout << "-----------------------------------------------------------------------------------" << endl << endl ;
	}


	int Nparam = X.size();

	vector<double> Xprev(Nparam,0);

	vector<double> dFdX_prev (Nparam, 0);

	vector<double> p (Nparam, 0);
	vector<double> s (Nparam, 0);
	vector<double> g (Nparam, 0);


	// store previous gradient evaluation
	for(int i = 0; i < Nparam; i++){ dFdX_prev[i] = dFdX[i]; }

	// Compute search direction for next iteration
	matrixVectorMultiply( D, dFdX, p );
	for(int i = 0; i < Nparam; i++){ p[i] = -p[i]; }


    // other variables
    int iter = 0;
    double xdiff = xMinDiff*2;
    double grad2Norm  = 2*minGrad2Norm;
    double alpha = alphaMin*2;
    // Optimization loop
    while (iter < maxIter && xdiff > xMinDiff  && grad2Norm > minGrad2Norm && alpha > alphaMin && optimFlag)
    {


    	// Compute scaling
    	double  Fopt;
    	secantLineSearchBnd( X, Xlb, Xub, F, dFdX, p, alpha, Fopt, constantX, constantIndicator   );

    	// If the line search failed to meet the tolerance, try using the steepest descent direction instead
    	if( F - Fopt < FStepTolerance )
    	{
    		if(procID == ROOT_ID){
    			cout << "Line search failed in the quasi-newton direction. Recomputing gradient and attempting steepest descent instead." << endl;
    		}
	    	objPtr->gradientApproximationMPIRecur(X, dX, dFdX, constantX, constantIndicator);

    		for(int i = 0; i < Nparam; i++)
    		{
    			p[i] = -dFdX[i];
    		}
        	secantLineSearchBnd( X, Xlb, Xub, F, dFdX, p, alpha, Fopt, constantX, constantIndicator   );
    	}


    	// Update X
    	for(int i = 0; i < Nparam; i++)
    	{
    		Xprev[i] = X[i]; // store previous values
    		X[i] = X[i] + alpha*p[i];
    	}
    	Fprev = F;
    	F = Fopt;


    	// Compute updated gradient
    	objPtr->gradientApproximationMPIRecur(X, dX, dFdX, constantX, constantIndicator);


    	// Compute Hessian update
      	for(int i = 0; i < Nparam; i++)
		{
			s[i] = alpha*p[i];
			g[i] = dFdX[i] - dFdX_prev[i];
		}
      	if( dotProd( g, s) != 0 )
      	{
      		updateHessianInv( D, g, s );
      	}


    	// store previous gradient evaluation
    	for(int i = 0; i < Nparam; i++){ dFdX_prev[i] = dFdX[i]; }

    	// Compute search direction for next iteration
    	matrixVectorMultiply( D, dFdX, p );
    	for(int i = 0; i < Nparam; i++){ p[i] = -p[i]; }

    	// If the solution failed, it is likely that the approximation error is too large, and the solution should cease
    	if(F > Fprev)
    	{
    		optimFlag = false;
    	}

    	xdiff = 0;
    	for(int i = 0; i < Nparam; i++){ xdiff += fabs( X[i] - Xprev[i] ); }
    	grad2Norm = vector2Norm ( dFdX );

        if( verbose == true && mod(iter,1) == 0 && procID == 0 )
		{
			cout << endl << endl << "---> At iter = " << iter << " the mean abs xdiff is " << xdiff << " and the grad2norm = " << 	grad2Norm << endl;
        	cout << "    X = "; print1DVector(X);
        	cout << "    dFdX = "; print1DVector(dFdX);
        	cout << "    p = "; print1DVector(p);
			cout << "    with a minimum function evaluation of " <<  F << endl;

		}


        /*
    	if(procID == 0 )
    	{
    	cout << "  alpha = " << alpha << endl;
    	cout << "  X = "; print1DVector(X);
    	cout << "  Xprev = "; print1DVector(Xprev);
    	cout << "  F " << F << endl;
    	cout << "  dFdX_prev = "; print1DVector(dFdX_prev);
    	cout << "  dFdX = "; print1DVector(dFdX);
    	cout << "  D = "; print2DVector(D);
    	}
		*/

    	if(!recurFlag)
    		boundaryAssessment(  F, X, p, dFdX, D, Xlb, Xub, dX, constantX, constantIndicator, optimFlag, recurFlag );

        iter = iter+1;
    }


}


double BFGSBnd_MPI::lineSearchObj( double alpha, vector <double> & X, vector <double> & p, vector<double> & constantX, vector<bool> & constantIndicator  )
{
	vector <double> Xalphap(X.size(),0);

	for(int i = 0; i < X.size(); i++)
	{
		Xalphap[i] = X[i] + alpha*p[i];
	}

	double value = objPtr->objEvalRecur(Xalphap,constantX,constantIndicator);
	return value;

}



void BFGSBnd_MPI::evalAlphaPoolMPI( vector <double> & alphaPool, vector <double> & phiPool, vector <double> & X,
		vector <double> & p, vector<double> & constantX, vector<bool> & constantIndicator  )
{



	// Get the number of processes and the current processor ID
	int Nprocs, procID;
	MPI_Comm_size(MPI_COMM_WORLD, &Nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &procID);


	// Ensure all processes are using the same alphaPool as computed on ROOT_ID
	MPI_Bcast(alphaPool.data(), alphaPool.size(), MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);
	// Display current search alphas
	if(procID == ROOT_ID)
	{
		cout << "Evaluating alphaPool =  [";
		for(int i = 0; i < alphaPool.size(); i++){ cout << alphaPool[i] << "  ";}
		cout << "]" << '\r' << flush;
	}

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
			phi_local[i] = lineSearchObj( alphaPool[i], X, p, constantX, constantIndicator  );

		}
	}


	for(int i = 0; i < N; i++ )
	{

		if( phi_local[i] != phi_local[i] || isinf(phi_local[i]) )
		{
			cout << "Line search crashed with " << endl;
			cout << phi_local[i] << endl;
			cout << "Quitting..." << endl;
			exit(0);
		}

	}


	// collect result
	MPI_Reduce(phi_local, phi, N, MPI_DOUBLE, MPI_SUM, ROOT_ID, MPI_COMM_WORLD);
	for(int i = 0; i < N; i++ )
	{
		phiPool[i] = phi[i];
	}
	MPI_Bcast(phiPool.data(), phiPool.size(), MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);





	// clean up
	delete [] phi_local;
	delete [] phi;

}



void BFGSBnd_MPI::secantLineSearchBnd( vector <double> & X, vector <double> & Xlb, vector <double> & Xub, double FX,
		vector <double> & dFdX, vector <double> & p, double & alphaOpt, double & Fopt, vector<double> & constantX, vector<bool> & constantIndicator  )
{
	MPI_Barrier(  MPI_COMM_WORLD );

	int idxMin, idxMax, idx;
	double r;

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
	bool bndIndicator = false;

	idxMin = -ceil((Npool-1.0)/2.0);
	idxMax = floor((Npool-1.0)/2.0);
	r = pow(maxAlphaMult, 1.0/(double) idxMax);
	idx = idxMin;
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

		// Check the bounds
		checkAlphaPoolBnd( bndIndicator, alphaPool, X, Xlb, Xub, p, constantX, constantIndicator  );

		// evaluate the pool
		evalAlphaPoolMPI( alphaPool, phiPool, X,  p, constantX, constantIndicator   );

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
		if(firstFlag && !bndIndicator)
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
		else if( bndIndicator ) // Discontinue extensions if at the boundary
		{
			firstFlag = false;
		 	if( verbose == true && procID == 0)
			{
		 		cout << endl << "Line search reached boundary. Attempting to find an acceptable point in the domain interior." << endl;
				cout << "alpha = "; print1DVector(alphaPool);
				cout << "phi = "; print1DVector(phiPool);
			}
		}



	  	if( verbose == true && mod(iter,1) == 0  && procID == 0)
		{
  			// cout << "Computing line search search with alpha pool terminating at = "<< alphaPool[Npool-1] << '\r' << flush;

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
		MPI_Barrier(  MPI_COMM_WORLD );

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


		evalAlphaPoolMPI( alphaPool, phiPool, X,  p, constantX, constantIndicator   ); // actual evaluation
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




	  	if( verbose == true && mod(iter,1) == 0  && procID == ROOT_ID)
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

		//5 . Check that alpha is not below the minimum set value
	  	if( vectorMax(alphaPool2) < alphaMin )
	  	{
	  		zoomFlag = 0;
	  	}

		iter++;
	}


	// Place minimum in export variable
	double phiMin;
	vectorMin ( phiPool, phiPool.size(), phiMin, idxMin);
	alphaOpt = alphaPool[idxMin];
	Fopt = phiMin;

	MPI_Bcast(&alphaOpt, 1, MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);
	MPI_Bcast(&Fopt, 1, MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);

}




double computeAlphaBnd( vector <double> & X, vector <double> & Xlb, vector <double> & Xub, vector <double> & p )
{
	double alphaBnd;


	int Nprm = X.size();

	for(int i = 0; i < Nprm; i++)
	{

		double alpha1i, alpha2i;
		alpha1i = ( Xub[i] - X[i] )/p[i];
		alpha2i = ( Xlb[i] - X[i] )/p[i];


		double alphaBndi;
		if( alpha1i > 0 )
		{
			alphaBndi = alpha1i;
		}
		else if( alpha2i > 0 )
		{
			alphaBndi = alpha2i;
		}
		else
		{
			alphaBndi = 0;
		}

		if(i == 0)
		{
			alphaBnd = alphaBndi;
		}

		if( alphaBnd > alphaBndi)
		{
			alphaBnd = alphaBndi;
		}
	}


	return alphaBnd;

}


void checkAlphaPoolBnd( bool & bndIndicator, vector <double> & alphaPool, vector <double> & X, vector <double> & Xlb, vector <double> & Xub,
		vector <double> & p, vector<double> & constantX, vector<bool> & constantIndicator )
{

	int Npool = alphaPool.size();
	bndIndicator = false;
	double alphaBnd = computeAlphaBnd(  X, Xlb, Xub, p );


	for(int i = 0; i < Npool; i++ )
	{
		if( alphaPool[i] > alphaBnd )
			bndIndicator = true;
	}


	if(bndIndicator)
	{
		double deltaAlpha = alphaBnd/(Npool);
		for(int i = 0; i < Npool; i++)
		{
			alphaPool[i] = deltaAlpha*(i+1);
		}
	}

	// If any alpha are negative, set to zero.
	for(int i = 0; i < Npool; i++ )
	{
		if(alphaPool[i] < 0)
			alphaPool[i] = 0;
	}

}




void BFGSBnd_MPI::boundaryAssessment( double & F, vector <double> & X, vector <double> & p,	vector<double> & dFdX, vector<vector<double> > & D,
		vector <double> & Xlb, vector <double> & Xub, vector<double> & dX, vector<double> & constantX, vector<bool> & constantIndicator,
		bool & optimFlag, bool & recurFlag)
{
	// Get the number of processes and the current processor ID
	int Nprocs, procID;
	MPI_Comm_size(MPI_COMM_WORLD, &Nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &procID);


	int Ndim = constantX.size();
	bool bndFlag = false;

	int iRecur = 0;
	for(int i = 0; i < Ndim; i++ )
	{
		if( !constantIndicator[i] )
		{
			if( (fabs(X[iRecur] - Xlb[iRecur]) < dXGrad)  &&  (p[iRecur] < 0) )
			{
				bndFlag = true;
				constantIndicator[i] = true;
				constantX[i] = X[iRecur];
			}
			else if ( fabs(X[iRecur] - Xub[iRecur]) < dXGrad  && (p[iRecur] > 0) )
			{
				bndFlag = true;
				constantIndicator[i] = true;
				constantX[i] = X[iRecur];
			}
			iRecur++;
		}
	}

	int Nconst = 0;
	for(int i = 0; i < constantIndicator.size(); i++ )
	{
		Nconst = Nconst + constantIndicator[i];
	}

	if(bndFlag && procID == ROOT_ID && verbose)
	{
		cout << endl << "-------------------------------------------------------------------------------------------------" << endl;
		cout << " Optimizer reached box boundary and found that the steepest descent is directed outside of the boundary at" << endl;
		cout << "    indicator "; print1DVector(constantIndicator);
		cout << "    the constant values are "; print1DVector(constantX);
		cout << endl << " Full variable info:" << endl;
		cout << "    X = "; print1DVector(X);
		cout << "    F = "; cout << F << endl;
		cout << "    Xlb = "; print1DVector(Xlb);
		cout << "    Xub = "; print1DVector(Xub);
		cout << "    dFdX = "; print1DVector(dFdX);
		cout << "    D = "; print2DVector(D);
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

		 int irecur = 0;
		 for(int i = 0; i < Ndim; i++)
		 {
			 if( !constantIndicator[i] )
			 {
				 XRecur[irecur] = X[i];
				 dFdXRecur[irecur] = dFdX[i];
				 XlbRecur[irecur] = Xlb[i];
				 XubRecur[irecur] = Xub[i];
				 dXRecur[irecur] = dX[i];

				 int jrecur = 0;
				 for(int j = 0; j < Ndim; j++)
				 {
					 if( !constantIndicator[j] )
					 {
						 DRecur[irecur][jrecur] = D[i][j];
						 jrecur++;
					 }
				 }
				 irecur++;
			 }
		 }

		 // Call the main solution with recursive variables
  		 recurFlag = true; // set recur flag to true
		 mainBFGSLoop( FRecur, XRecur, dFdXRecur, DRecur, XlbRecur, XubRecur, dXRecur, constantX, constantIndicator, optimFlag, recurFlag );


		 // Check to see if the gradient in the direction of the previously removed variables has improved
		 irecur = 0;
		 for(int i = 0; i < Ndim; i++)
		 {
			 if( !constantIndicator[i] )
			 {
				 X[i] = XRecur[irecur];
				 dFdX[i] = dFdXRecur[irecur];
				 Xlb[i] = XlbRecur[irecur];
				 Xub[i] = XubRecur[irecur];
				 dX[i] =  dXRecur[irecur];

				 irecur++;
			 }
		 }

		// Reset D as identity
		for(int i = 0; i < Ndim; i++)
		{
			for(int j = 0; j < Ndim; j++)
			{
				D[i][j] = 0.0;
				if(i == j)
				{
					D[i][j] = 1.0;
				}
			}
		}
		 objPtr->gradientApproximationMPI(X,dX,dFdX);

        iRecur = 0;
		for(int i = 0; i < Ndim; i++ )
		{
			constantIndicator[i] = false;
			if( !constantIndicator[i] )
			{
				if( (fabs(X[iRecur] - Xlb[iRecur]) < dXGrad)  &&  (dFdX[iRecur] > 0) )
				{
					bndFlag = true;
					constantIndicator[i] = true;
					constantX[i] = X[iRecur];
				}
				else if ( fabs(X[iRecur] - Xub[iRecur]) < dXGrad  && (dFdX[iRecur] < 0) )
				{
					bndFlag = true;
					constantIndicator[i] = true;
					constantX[i] = X[iRecur];
				}
				iRecur++;
			}
		}
		Nconst = 0;
		for(int i = 0; i < constantIndicator.size(); i++ )
		{
			Nconst = Nconst + constantIndicator[i];
		}

		if(Nconst > 0)
		{
			optimFlag = false;
			recurFlag = false;
			if(procID == ROOT_ID && verbose)
			{
				cout << "Optimization exiting after recursive boundary optimization, "<< endl
				 << " as the gradient through the boundary still points out of the domain." << endl;
			}
		}
		else
		{
			optimFlag = true;
			recurFlag = false;

			if(procID == ROOT_ID && verbose)
			{
				cout << "Optimization continuing after recursive boundary optimization, "<< endl
					 << " as the gradient through the boundary points to the domain interior." << endl;
			}
		}


	}



}


























