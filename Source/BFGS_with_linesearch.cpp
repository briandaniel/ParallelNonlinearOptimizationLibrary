/*
 * BFGS_with_linesearch.cpp
 *
 *  Created on: Nov 5, 2018
 *      Author: brian
 */

#include "BFGS_with_linesearch.hpp"



void BFGS::findMin( vector <double> & X, double & f0, double & fOpt  ){

	int Nparam = X.size();



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
	objPtr->gradientApproximation(X,dX,dFdX);
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
    	cubicInterpolationLineSearch( X, F, dFdX, p, alpha, Fopt  );


    	// Update X
    	for(int i = 0; i < Nparam; i++)
    	{
    		Xprev[i] = X[i]; // store previous values
    		X[i] = X[i] + alpha*p[i];
    	}
    	F = Fopt;


    	// Compute updated gradient
    	objPtr->gradientApproximation(X,dX,dFdX);

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

        if( verbose == true && mod(iter,1) == 0 )
		{
			cout << "At iter = " << iter << " the mean abs xdiff is " << xdiff << " and the grad2norm = " << 	grad2Norm << endl;
			cout << " with a minimum function evaluation of " <<  F << endl;
		}

        iter = iter+1;
    }


    fOpt = F;

    if( verbose == true )
    {
    	cout << endl << "-----------------------------------------------------------------------------------" << endl ;

    	cout << "Completed bfgs." << endl;
    	cout << "f0 = " << f0 << ", fOpt = " << fOpt << " with variable:" << endl;
    	cout << "X = "; print1DVector(X);
    	cout << "-----------------------------------------------------------------------------------" << endl << endl ;
    }



}




double BFGS::lineSearchObj( double alpha, vector <double> & X, vector <double> & p  )
{
	vector <double> Xalphap(X.size(),0);

	for(int i = 0; i < X.size(); i++)
	{
		Xalphap[i] = X[i] + alpha*p[i];
	}

	double value = objPtr->objEval(Xalphap);
	return value;

}



double BFGS::lineSearchFDDerivative( double alpha, double phialpha, vector <double> & X, vector <double> & p  )
{
	vector <double> Xalphap_dalpha(X.size(),0);

	for(int i = 0; i < X.size(); i++)
	{
		Xalphap_dalpha[i] = X[i] + (alpha+dalpha)*p[i];
	}

	double Falpha_dalpha = objPtr->objEval(Xalphap_dalpha);
	double dFdalpha = ( Falpha_dalpha - phialpha )/dalpha;

	return dFdalpha;

}


void BFGS::cubicInterpolationLineSearch( vector <double> & X, double FX,
		vector <double> & dFdX, vector <double> & p, double & alphaOpt, double & Fopt  )
{
	double alpha_lo, alpha_hi, phi_lo, phi_hi, dphi_lo_dalpha, dphi_hi_dalpha;
	double dphiOptdalpha;

	alphaOpt = 0;
	Fopt = FX;


	// Initial values
	double alpha0 = 0;
	double phi0 = FX;
	// Compute initial slope from gradient computation
	double dphi0dalpha = dotProd( dFdX, p );


	// set previous step to intial values
	double alphaim1 = alpha0;
	double phiim1 = phi0;
	double dphiim1dalpha = dphi0dalpha;


	// Initial step guess
	double alphai = alphaGuess;

	int iter = 0;
	while (iter < maxIterLineSearch )
	{
		double phii = lineSearchObj(alphai, X, p);
		double dphiidalpha = lineSearchFDDerivative( alphai, phii, X, p  );

		// 1. check suff. decrease
		if( (phii > phi0 + c1*alphai*dphi0dalpha ) || (  phii >= phiim1 && iter > 1 ) )
		{
			alpha_lo = alphaim1;
			phi_lo = phiim1;
			dphi_lo_dalpha = dphiim1dalpha;

			alpha_hi = alphai;
			phi_hi = phii;
			dphi_hi_dalpha = dphiidalpha;

			// Zooom function and break
			lineSearchZoom(  alpha_lo, alpha_hi, phi_lo, phi_hi, dphi_lo_dalpha, dphi_hi_dalpha,
					phi0, dphi0dalpha, X, p, alphaOpt, Fopt, dphiOptdalpha  );

		    if( verbose == true )
		    {
		    	cout << "Line search ending after " << iter << " iterations and failing sufficient decrease condition and zooming." << endl;
		    }
			break;
		}


		// 2. Check curvature
		if ( fabs(dphiidalpha) <= fabs( c2*dphi0dalpha) )
		{
			alphaOpt = alphai;
			Fopt = phii;

		    if( verbose == true )
			{
				  cout << "Line search ending after " << iter << " iterations and satisfying curvature condition." << endl;
			}
			break;
		}


		// 3. Check whether to shorten interval or extend
		if( dphiidalpha >= 0)
		{

			alpha_lo = alphai;
			phi_lo = phii;
			dphi_lo_dalpha = dphiidalpha;

			alpha_hi = alphaim1;
			phi_hi = phiim1;
			dphi_hi_dalpha = dphiim1dalpha;

			// Zooom function and break
			lineSearchZoom(  alpha_lo, alpha_hi, phi_lo, phi_hi, dphi_lo_dalpha, dphi_hi_dalpha,
						phi0, dphi0dalpha, X, p, alphaOpt, Fopt, dphiOptdalpha  );

			 if( verbose == true )
			{
				  cout << "Line search ending after " << iter << " iterations and zooming due to positive derivative at endpoint." << endl;
			}
			break;

		}

		// store previous values
		alphaim1 = alphai;
		phiim1 = phii;
		dphiim1dalpha = dphiidalpha;

		// Extend search interval as no reasonable interval was found
		alphai = 2*alphai;

	    if( verbose == true && mod(iter,1) == 0 )
		{
			cout << "Line search at iter = " << iter << " failed. Continuing line search with larger region alpha =  " << alphai << endl;
			cout << "Current values are: " << endl;
			cout << alphaim1 << "  " << phiim1 << "  " <<  dphiim1dalpha << endl;
			cout << alphai << "  " << phii << "  " <<  dphiidalpha << endl;

		}
		iter++;
	}



}




void BFGS::lineSearchZoom( double alpha_lo, double alpha_hi, double phi_lo, double phi_hi, double dphi_lo_dalpha, double dphi_hi_dalpha,
		double phi0, double dphi0dalpha, vector <double> & X, vector <double> & p, double & alphaOpt, double & phiOpt, double & dphiOptdalpha  )
{
	int iter = 0;
	while (iter < maxIterLineSearch)
	{

		double alphaj = cubicInterpMin( alpha_lo, alpha_hi, phi_lo, phi_hi, dphi_lo_dalpha, dphi_hi_dalpha, X, p );
		double phij = lineSearchObj(alphaj, X, p);
		double dphijdalpha = lineSearchFDDerivative( alphaj, phij, X, p  );


		// 1. Check suff. decrease
		if( phij > phi0 + c1*alphaj*dphi0dalpha || phij >= phi_lo )
		{
			alpha_hi = alphaj;
			phi_hi = phij;
			dphi_hi_dalpha = dphijdalpha;
		}
		else
		{
			// 2. check curvature
			if( fabs(dphijdalpha) <= fabs(c2*dphi0dalpha) )
			{

				alphaOpt = alphaj;
				phiOpt = phij;
				dphiOptdalpha = dphijdalpha;

			   if( verbose == true )
				{
					  cout << "Zoom ending after " << iter << " iterations and satisfying curvature condition." << endl;
				}
				break;

			}

			// 3.
			if (dphijdalpha*(alpha_hi - alpha_lo) >= 0)
			{
				alpha_hi = alpha_lo;
				phi_hi = phi_lo;
				dphi_hi_dalpha = dphi_lo_dalpha;
			}

			alpha_lo = alphaj;
			phi_lo = phij;
			dphi_lo_dalpha = dphijdalpha;


		}


		iter ++;
	}

	if( verbose == true && iter == maxIterLineSearch-1 )
	{
		  cout << "Zoom failed after " << iter << " iterations !!!!! "  << endl;
	}
}


double cubicInterpMin( double alpha_lo, double alpha_hi, double phi_lo, double phi_hi, double dphi_lo_dalpha, double dphi_hi_dalpha,
		vector <double> & X, vector <double> & p )
{

	double d1 = dphi_lo_dalpha + dphi_hi_dalpha - 3*( phi_lo -phi_hi)/(alpha_lo - alpha_hi);
	double d2 = sign( alpha_hi - alpha_lo )*sqrt( pow(d1,2) - dphi_lo_dalpha*dphi_hi_dalpha);
	double alphaNew = alpha_hi - ( alpha_hi - alpha_lo )*( dphi_hi_dalpha + d2 - d1 )/( dphi_hi_dalpha - dphi_lo_dalpha + 2*d2 );

	// if outside the region, use bisection instead
	if( alpha_lo < alpha_hi)
	{
		if(alphaNew < alpha_lo)
		{
			alphaNew = (alpha_hi + alpha_lo)/2;
		}
	}
	else
	{
		if(alphaNew < alpha_hi)
		{
			alphaNew = (alpha_hi + alpha_lo)/2;
		}
	}


	return alphaNew;
}



void updateHessianInv( vector<vector<double> > & D, vector<double> & g, vector<double> & s )
{
	int Nparam = g.size();
	vector<vector<double> > M1(Nparam, vector<double>(Nparam));
	vector<vector<double> > M2(Nparam, vector<double>(Nparam));
	vector<vector<double> > M3(Nparam, vector<double>(Nparam));
	vector<vector<double> > A(Nparam, vector<double>(Nparam));

	double rho = 1/dotProd( g, s);

	for(int i = 0; i < Nparam; i++)
	{
		for(int j =0; j < Nparam; j++)
		{
			M1[i][j] = 0;
			M2[i][j] = 0;

			if (i == j)
			{
				M1[i][j] = 1;
				M2[i][j] = 1;
			}

			M1[i][j] = M1[i][j] - rho*s[i]*g[j];
			M2[i][j] = M2[i][j] - rho*g[i]*s[j];
			M3[i][j] = rho*s[i]*s[j];

		}
	}



	matrixMultiply( M1, D, A );
	matrixMultiply( A, M2, D );

	for(int i = 0; i < Nparam; i++)
	{
		for(int j =0; j < Nparam; j++)
		{
			D[i][j] = D[i][j] + M3[i][j];
		}
	}

}



