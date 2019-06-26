/*
 * BFGS_bnd_linesearch.cpp
 *
 *  Created on: Nov 15, 2018
 *      Author: brian
 */




#include "BFGS_bnd_linesearch.hpp"



void BFGS_Bnd::findMinBnd( vector <double> & X, vector <double> & Xlb, vector <double> & Xub, double & f0, double & fOpt  ){


	totalIter = 0;

	if( verbose >= 0 && getProcID() == ROOT_ID)
	{
		cout << endl << "----------------------------------------------------------------------------------------------------------" << endl ;
		cout << "  Starting bounded BFGS with line search. " << endl;
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
 	if(dXGradVec.size() > 0)
 	{
 		for(int i =0; i < Nparam; i++ )
		{
 			dX[i] = dXGradVec[i];
		}
 	}


	vector<double> dFdX (Nparam, 0);


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
	objPtr->gradientApproximationRecur(X,dX,dFdX,constantX,constantIndicator);
	double F = objPtr->objEvalRecur(X,constantX,constantIndicator);
	f0 = F;

	// Run optimization
	bool optimFlag = true;
	int recurFlag = 0;
	mainBFGSLoop( F,  X, dFdX, D, Xlb, Xub, dX, constantX, constantIndicator, optimFlag, recurFlag);


    // Save function optimum result
    fOpt = F;

    if( verbose >= 0 && getProcID() == ROOT_ID)
    {
    	cout << endl << "----------------------------------------------------------------------------------------------------------" << endl ;
    	cout << "  Completed bounded BFGS." << endl;
    	cout << "  X0 = "; print1DVector(X0);
    	cout << "  Xopt = "; print1DVector(X);
    	cout << "  f0 = " << f0 << ", fOpt = " << fOpt << endl;
    	cout << "----------------------------------------------------------------------------------------------------------" << endl << endl ;
    }



}


void BFGS_Bnd::mainBFGSLoop( double & F, vector <double> & X, vector<double> & dFdX, vector<vector<double> > & D,
		vector <double> & Xlb, vector <double> & Xub, vector<double> & dX, vector<double> & constantX,
		vector<bool> & constantIndicator, bool & optimFlag, int & recurFlag )
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
    while (optimFlag && iter < maxIter && xdiff > xMinDiff  && grad2Norm > minGrad2Norm && totalIter < maxIter )
    {
		if( verbose > 0 && mod(iter,1) == 0 && getProcID() == ROOT_ID)
		{
			cout << endl << "Iter = " << iter << " of bounded BFGS search starting with previous F = " << F << "." << endl;
		}
    	// ----------- 1. Compute search direction --------------//
    	matrixVectorMultiply( D, dFdX, p );
    	for(int i = 0; i < Nparam; i++){ p[i] = -p[i]; }


    	// ----------- 2. Compute search scaling (alpha) --------------//
    	double alpha, Fopt;
    	cubicInterpolationLineSearchBnd( X, Xlb, Xub, F, dFdX, p,constantX,constantIndicator , alpha, Fopt );


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
    	objPtr->gradientApproximationRecur(X,dX,dFdX,constantX,constantIndicator);

    	// Compute Hessian update
      	for(int i = 0; i < Nparam; i++)
		{
			s[i] = alpha*p[i];
			g[i] = dFdX[i] - dFdX_prev[i];
		}
    	updateHessianInv( D, g, s );


    	// ----------- 4. Check boundary --------------//
    	boundaryAssessment( F, X, p, dFdX, D, Xlb, Xub, dX, constantX, constantIndicator, optimFlag, recurFlag);


    	// Compute differences and norms
    	xdiff = 0;
    	for(int i = 0; i < Nparam; i++){ xdiff += fabs( X[i] - Xprev[i] ); }
    	grad2Norm = vector2Norm ( dFdX );

    	// Display current step
    	if( verbose > 1 && mod(iter,1) == 0 && getProcID() == ROOT_ID )
		{
			cout << "  Step completed with F = " << F << " and mean abs xdiff is " << xdiff << " and the grad2norm = " << grad2Norm << endl;
			cout << "  X = "; print1DVector(X);
			cout << "  Xlb = "; print1DVector(Xlb);
			cout << "  Xub = "; print1DVector(Xub);
			cout << "  dFdX = "; print1DVector(dFdX);
			cout << "  p = "; print1DVector(p);
		}
        iter = iter+1;
        totalIter = totalIter +1;


    }

}

void BFGS_Bnd::cubicInterpolationLineSearchBnd( vector <double> & X,vector <double> & Xlb, vector <double> & Xub,
		double FX, vector <double> & dFdX, vector <double> & p, vector<double> & constantX, vector<bool> & constantIndicator,
		double & alphaOpt, double & Fopt  )
{
	bool success = false;
	bool bndIndicator = false;
	double alphaMax;
	double dphiOptdalpha;
	double phii;

	alphaOpt = 0;
	Fopt = FX;

	// Initial values, with initial slope from gradient computation
	double alpha0 = 0;
	double phi0 = FX;
	double dphi0dalpha = dotProd( dFdX, p );

	// set previous step to intial values
	double alphaim1 = alpha0;
	double phiim1 = phi0;
	double dphiim1dalpha = dphi0dalpha;

	// Compute max value of alpha
	alphaMax = computeAlphaBnd( X, Xlb, Xub, p );

	// Initial step guess
	double alphai = alphaGuess;
	if(alphai > alphaMax)
	{
		alphai = alphaMax;
	}


	if( verbose > 1 && getProcID() == 0 && getProcID() == ROOT_ID)
	{
		cout << endl << "Starting line search in direction " << endl;
		cout << "p = "; print1DVector(p);
		cout << " with alpha0 = " << alphai << " implying X = ";

		vector <double> Xalphap(X.size(),0);
		for(int i = 0; i < X.size(); i++)
		{
			Xalphap[i] = X[i] + alphai*p[i];
		}
		print1DVector(Xalphap);

	}

	int iter_ls = 0;
	while (iter_ls < maxIterLineSearch )
	{
		phii = lineSearchObj(alphai, X, p,constantX,constantIndicator);
		double dphiidalpha = lineSearchFDDerivative( alphai, phii, X, p, constantX, constantIndicator );
		// cout << " dphidalpha = " << dphiidalpha << endl;

		// 1. Check if interval is large enough (based on magnitude)
		if( (phii > phi0 + c1*alphai*dphi0dalpha ) || (  phii >= phiim1 && iter_ls > 1 ) )
		{
			// Zoom function and break
			lineSearchZoomBnd(  alphaim1, alphai, phiim1, phii, dphiim1dalpha, dphiidalpha,
					phi0, dphi0dalpha, X, p, constantX, constantIndicator, iter_ls, alphaOpt, Fopt, dphiOptdalpha  );
			success = true;
			break;
		}


		// 2. Check if close enough to optimum
		if ( fabs(dphiidalpha) <= fabs( c2*dphi0dalpha) )
		{
			alphaOpt = alphai;
			Fopt = phii;
			success = true;
			break;
		}


		// 3. Check if interval is large enough (based on slope)
		if( dphiidalpha >= 0)
		{
			// Zooom function and break
			lineSearchZoomBnd(  alphaim1, alphai, phiim1, phii, dphiim1dalpha, dphiidalpha,
						phi0, dphi0dalpha, X, p, constantX, constantIndicator, iter_ls, alphaOpt, Fopt, dphiOptdalpha  );
			success = true;
			break;
		}

		// 4. Check if boundary was reached
		if( alphai == alphaMax )
		{
			alphaOpt = alphai;
			Fopt = phii;
			bndIndicator = true;
			success = true;
			break;
		}

		// store previous values
		alphaim1 = alphai;
		phiim1 = phii;
		dphiim1dalpha = dphiidalpha;


		// 5. Compute new step since no interval was found
		// Extend search interval as no reasonable interval was found
		alphai = 2*alphai;
		if( alphai > alphaMax )
		{
			alphai = alphaMax;
		}

		iter_ls++;
	}


	// If the search exited because it took too long, set the optimum as the best current option
	if(!success)
	{
		if( phiim1 < phii )
		{
			alphaOpt = alphaim1;
			Fopt = phiim1;
		}
		else if ( phi0 < phii )
		{
			alphaOpt = 0;
			Fopt = phi0;
		}
		else
		{
			alphaOpt = alphai;
			Fopt = phii;
		}
	}

    if( verbose > 0 && getProcID() == ROOT_ID)
    {
    	if(!bndIndicator)
    	{
    		cout << "  Line search completed with alpha = " << alphaOpt << " and F = " << Fopt << " after " << iter_ls << " iterations. "
    				<< "Note: alphaMax = " << alphaMax << endl << endl;
    	}
    	else
    	{
        	cout << "  ! Line search terminated at boundary with alpha = " << alphaOpt << " and F = " << Fopt << " after " << iter_ls << " iterations. "
    				<< "Note: alphaMax = " << alphaMax << endl << endl;
    	}
    }


}




void BFGS_Bnd::lineSearchZoomBnd( double alpha_a, double alpha_b, double phi_a, double phi_b, double dphi_a_dalpha, double dphi_b_dalpha,
		double phi0, double dphi0dalpha, vector <double> & X, vector <double> & p, vector<double> & constantX, vector<bool> & constantIndicator,
		int & iter_ls, double & alphaOpt, double & phiOpt, double & dphiOptdalpha  )
{
	if( verbose > 1 && getProcID() == 0 )
	{
		cout << "Starting zoom line search ... " << endl;
	}
	bool success = false;

	while (iter_ls < maxIterLineSearch && ( alpha_b - alpha_a > alphaTol ) )
	{

		// 0. Compute new alpha guess, and evaluate it
		double alpha_c = cubicInterpMinSimple( alpha_a, alpha_b, phi_a, phi_b, dphi_a_dalpha, dphi_b_dalpha );
		double phi_c = lineSearchObj(alpha_c, X, p, constantX, constantIndicator);
		double dphi_c_dalpha = lineSearchFDDerivative( alpha_c, phi_c, X, p, constantX, constantIndicator  );

		if( verbose > 1 && getProcID() == 0  )
		{
			cout << endl << "In zoom ... " << endl;
			cout << "p = "; print1DVector(p);
			cout << " with alpha0 = " << alpha_c << " implying X = ";

			vector <double> Xalphap(X.size(),0);
			for(int i = 0; i < X.size(); i++)
			{
				Xalphap[i] = X[i] + alpha_c*p[i];
			}
			print1DVector(Xalphap);
		}

		// 1. Check if interval can be made smaller
		double phi_min = phi_a;
		if ( phi_b < phi_a ){ phi_min = phi_b; }
		if( phi_c > phi0 + c1*alpha_c*dphi0dalpha || phi_c >= phi_min )
		{
			if( phi_a < phi_b)
			{
				alpha_b = alpha_c;
				phi_b = phi_c;
				dphi_b_dalpha = dphi_c_dalpha;
			}
			else
			{
				alpha_a = alpha_c;
				phi_a = phi_c;
				dphi_a_dalpha = dphi_c_dalpha;
			}

		}
		else
		{
			// 2. Check if close enough to optimum
			if( fabs(dphi_c_dalpha) <= fabs(c2*dphi0dalpha) )
			{
				alphaOpt = alpha_c;
				phiOpt = phi_c;
				dphiOptdalpha = dphi_c_dalpha;
				success = true;
				break;
			}

			// 3. Select new interval so that the minimum is still inside
			if( dphi_c_dalpha < 0 )
			{
				alpha_a = alpha_c;
				phi_a = phi_c;
				dphi_a_dalpha = dphi_c_dalpha;
			}
			else
			{
				alpha_b = alpha_c;
				phi_b = phi_c;
				dphi_b_dalpha = dphi_c_dalpha;
			}

		}

		iter_ls ++;
	}

	// If the search exited because it took too long, set the optimum as the best current option
	if(!success)
	{
		if( phi_a < phi_b)
		{
			alphaOpt = alpha_a;
			phiOpt = phi_a;
			dphiOptdalpha = dphi_a_dalpha;
		}
		else
		{
			alphaOpt = alpha_b;
			phiOpt = phi_b;
			dphiOptdalpha = dphi_b_dalpha;
		}
	}

}







double BFGS_Bnd::lineSearchObj( double alpha, vector <double> & X, vector <double> & p, vector<double> & constantX, vector<bool> & constantIndicator )
{
	vector <double> Xalphap(X.size(),0);

	for(int i = 0; i < X.size(); i++)
	{
		Xalphap[i] = X[i] + alpha*p[i];
	}

	double value = objPtr->objEvalRecur(Xalphap,constantX,constantIndicator);
	return value;

}



double BFGS_Bnd::lineSearchFDDerivative( double alpha, double phialpha, vector <double> & X, vector <double> & p,
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






void BFGS_Bnd::boundaryAssessment( double & F, vector <double> & X, vector <double> & p, vector<double> & dFdX, vector<vector<double> > & D,
		vector <double> & Xlb, vector <double> & Xub, vector<double> & dX, vector<double> & constantX, vector<bool> & constantIndicator,
		bool & optimFlag, int & recurFlag)
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

	if(bndFlag && verbose && getProcID() == ROOT_ID)
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
		 mainBFGSLoop( FRecur, XRecur, dFdXRecur, DRecur, XlbRecur, XubRecur, dXRecur, constantX, constantIndicator, optimFlag, recurFlag );


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
		objPtr->gradientApproximationRecur(X, dX, dFdX, constantX, constantIndicator);


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
			if(verbose && getProcID() == ROOT_ID)
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
			if( verbose > 0 && getProcID() == ROOT_ID)
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
		if( verbose > 0 && getProcID() == ROOT_ID)
		{
			cout << endl << "-------------------------------------------------------------------------------------------------" << endl;
			cout << "      NO VARIABLES LEFT TO OPTIMIZE!!!! " << endl;
			cout << "-------------------------------------------------------------------------------------------------" << endl;
		}
	}



}






// Assumes alpha_a < alpha_b
double cubicInterpMinSimple( double alpha_a, double alpha_b, double phi_a, double phi_b, double dphi_a_dalpha, double dphi_b_dalpha )
{

	double d1 = dphi_a_dalpha + dphi_b_dalpha - 3*( phi_a -phi_b)/(alpha_a - alpha_b);
	double d2 = sign( alpha_b - alpha_a )*sqrt( pow(d1,2) - dphi_a_dalpha*dphi_b_dalpha);
	double alphaNew = alpha_b - ( alpha_b - alpha_a )*( dphi_b_dalpha + d2 - d1 )/( dphi_b_dalpha - dphi_a_dalpha + 2*d2 );

	// if outside the region, use bisection instead
	if( alphaNew < alpha_a || alphaNew > alpha_b || alphaNew != alphaNew)
	{
		alphaNew = ( alpha_a  + alpha_b )/2;
	}

	return alphaNew;
}
