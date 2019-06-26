/*
 * SimplexSearch.cpp
 *
 *  Created on: Oct 22, 2018
 *      Author: brian
 */


#include "SimplexSearch.hpp"



void SimplexSearch::findMin( vector <double> & X, double & f0, double &fOpt  )
{

	// Problem size0
	int Ndim = X.size();

    // There are n+1 points in the simplex where n is the number of dimensions
    int Nsimplex = Ndim+1;;

    if( verbose == true )
    {
    	cout << endl << "------------------------------------------------------------" << endl ;

    	cout << "Computing simplex search with settings " << endl;
    	cout << "alpha = " << alpha << ", gamma = " << gamma << ", rho = " << rho << ", sigma = " << sigma << endl;

    	cout << "------------------------------------------------------------" << endl << endl;
    }

    // Variable declarations
    double** xvec = new double *[Nsimplex];
    for( int k = 0; k < Nsimplex; k++ )
        xvec[k] = new double [Ndim];
    double* xbar = new double [Ndim];
    double* xref = new double [Ndim];
    double* xe = new double [Ndim];
    double* xc = new double [Ndim];
    double * fvec = new double [Nsimplex];

    double fbar, fref, fe, fc;
    double fmin_old;

    // First set
    for(int j = 0; j < Ndim; j++)
    {
    	xvec[0][j] = X[j];
    }


    if( verbose == true )
    {
    	cout << "Initial guess: " << endl;
    	print1DArrayLine(xvec[0],Ndim,5,"X0");
    	cout << endl;

    }
    // Add randomness to other guesses
	srand((unsigned)time(0)); // have to seed the random number generator
    for( int i = 1; i < Nsimplex; i++ )
    {
    	for(int j = 0; j < Ndim; j++)
    	{
    		xvec[i][j] = xvec[0][j] + initRandMax * (timeRand() - 0.5) * 2.0;
    	}
    }

    // Evaluate initial set of variables
    evaluateVariableSet(  xvec, Nsimplex, X, fvec );

    f0 = fvec[0];

    // Sort initial variables
    simplexSort( fvec, xvec, Ndim );


    // other variables
	double fmin = fvec[0]*100;
    int iter = 0;
    double xdiff = xMinDiff*2;
    while (iter < maxIter && xdiff > xMinDiff)
    {

		fmin_old = fmin;
		fmin = fvec[0];

        // Calculate centroid of first n elements (where n = Ndim) (leave out x_n+1)
		// Set values to zero first
		for (int i = 0; i < Ndim; i++)
		{
			xbar[i] = 0.0;
		}
		// Now calculate centroid
		for (int i = 0; i < Ndim; i++)
		{
			for (int k = 0; k < Ndim; k++)
			{
				xbar[i] = xbar[i] + xvec[k][i] / ((double)Ndim);
			}
		}

		// Calculate reflection
		// cout << endl << "Computing reflection" << endl;
		for (int i = 0; i < Ndim; i++)
		{
			xref[i] = xbar[i] + alpha*(xbar[i] - xvec[Nsimplex - 1][i]);
		}
        // Evaluate reflection
        fref = evaluateVariableArray(  xref, X );


		// Based on value of fref, decide what to do
		// 1. either simply add it as a new point (if it wasnt better than the best)
		if (fref >= fvec[0] && fref < fvec[Ndim - 1])
		{
			for (int i = 0; i < Ndim; i++)
			{
				xvec[Nsimplex - 1][i] = xref[i];
				fvec[Nsimplex - 1] = fref;
			}
		}
		// 2. or expand (if it was better than the best)
		else if (fref < fvec[0])
		{
			// cout << endl << "Computing expansion" << endl;
			// Calculate expansion if appropriate
			for (int i = 0; i < Ndim; i++)
			{
				xe[i] = xbar[i] + gamma*(xbar[i] - xvec[Nsimplex - 1][i]);
			}
            fe = evaluateVariableArray(  xe, X );

			// Decide what to do based on expansion value
			// (a) If expansion is better than the reflected use the expansion
			if (fe < fref)
			{
				for (int i = 0; i < Ndim; i++)
				{
					xvec[Nsimplex - 1][i] = xe[i];
					fvec[Nsimplex - 1] = fe;
				}
			}
			// (b) if not, just use the reflection
			else
			{
				for (int i = 0; i < Ndim; i++)
				{
					xvec[Nsimplex - 1][i] = xref[i];
					fvec[Nsimplex - 1] = fref;
				}
			}
		}
		// 3. if expansion is worse than the worst, contract instead
		else
		{
			// cout << endl << "Computing contraction" << endl;
			for (int i = 0; i < Ndim; i++)
			{
				xc[i] = xbar[i] + rho*( xvec[Nsimplex - 1][i] - xbar[i] );
			}
            fc = evaluateVariableArray( xc, X );

			// Decide what to do based on the value of the contraction
			// (a) If contracted is better than the worst, keep it
			if (fc < fvec[Nsimplex - 1])
			{
				for (int i = 0; i < Ndim; i++)
				{
					xvec[Nsimplex - 1][i] = xc[i];
					fvec[Nsimplex - 1] = fc;
				}
			}
			// (b) if not, reduce the whole system toward the best point, keep the best point fixed
			// (since everything else failed at this point)
			else
			{
				for (int k = 1; k < Nsimplex; k++)
				{
					for (int i = 0; i < Ndim; i++)
					{
						xvec[k][i] = xvec[0][i] + sigma*(xvec[k][i] - xvec[0][i]);
					}

    			}

                // Recompute solution at new locations
			    evaluateVariableSet(  xvec, Nsimplex, X, fvec );

			}
		}

        // Sort values
        simplexSort( fvec, xvec, Ndim );

        // Calculate inf norm of xdifference
        xdiff = simplexDiff( xvec, Ndim, Nsimplex );

        if( verbose == true && mod(iter,10) == 0 )
        {
        	cout << "At iter = " << iter << " the xdiff is " << xdiff << " with a minimum function evaluation of " << fvec[0] << endl;
        }

        iter = iter+1;
    }

    // Save optimal values
    fOpt = fvec[0];
    for(int j = 0; j < Ndim; j++)
    {
    	X[j] = xvec[0][j];
    }

    if( verbose == true )
    {
    	cout << endl << "-----------------------------------------------------------------------------------" << endl ;

    	cout << "Completed simplex search." << endl;
    	cout << "f0 = " << f0 << ", fOpt = " << fOpt << " with variable:" << endl;
    	print1DArrayLine(xvec[0],Ndim,5,"X");
    	cout << "-----------------------------------------------------------------------------------" << endl << endl ;
    }



    // Clean up
    delete [] fvec;
	delete [] xbar;
    delete [] xref;
    delete [] xe;
    delete [] xc;

    for( int k = 0; k < Nsimplex; k++ )
        delete [] xvec[k];

    delete [] xvec;



}


double SimplexSearch::evaluateVariableArray(  double * x, vector <double> & X )
{
	// set variable
	for(int j = 0; j < X.size(); j++)
	{
		X[j] = x[j];
	}

	// evaluate objective from objective pointer
	double f = objPtr->objEval(X);

	return f;

}

void SimplexSearch::evaluateVariableSet(  double ** xvec, int Nsimplex, vector <double> & X, double * fvec )
{

    for( int i = 0; i < Nsimplex; i++ )
    {
    	fvec[i] = evaluateVariableArray(  xvec[i],  X );
    }

}




// Sorts the xvector
void simplexSort(double * fvec, double ** xvec, int Nd ){

    int Nsimplex = Nd + 1;
	int min = 1;

    // Temporary variables
    double** xvecTemp = new double *[Nsimplex];
    for( int k = 0; k < Nsimplex; k++ )
            xvecTemp[k] = new double [Nd];
    double * fvecTemp = new double [Nsimplex];


    // find largest value to be used (sillily) later
    double fvecMax;
	int indexMax;
    vectorMax ( fvec, Nsimplex, fvecMax, indexMax);



    // Sort
	for (int k = 0; k < Nsimplex; k++)
	{
		for (int i = 0; i < Nsimplex; i++)
		{
			if (fvec[i] < fvec[min])
			{
				min = i;
			}

		}

		for (int j = 0; j < Nd; j++)
			xvecTemp[k][j] = xvec[min][j];

		fvecTemp[k] = fvec[min];

		fvec[min] = 2*fvecMax ;
	}


	for (int k = 0; k < Nsimplex; k++)
	{
		fvec[k] = fvecTemp[k];
		for (int j = 0; j < Nd; j++)
			xvec[k][j] = xvecTemp[k][j];

	}

    // Clean up temporary variables
    for( int k = 0; k < Nsimplex; k++ )
        delete [] xvecTemp[k];

    delete [] xvecTemp;
    delete [] fvecTemp;

}



double simplexDiff( double ** xvec, int Nd, int Nsimplex )
{
    double xDiffMax = 0;
    for (int k = 1; k < Nsimplex; k++)
    {
        for (int j = 0; j < Nd; j++)
        {
            double diff = fabs( xvec[0][j] - xvec[k][j]);

            if ( diff > xDiffMax )
            {
                xDiffMax = diff;
            }
        }

    }

    return xDiffMax;
}
