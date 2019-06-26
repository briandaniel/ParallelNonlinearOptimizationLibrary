/*
 * LevenbergMarquardtMPI.cpp
 *
 *  Created on: Nov 7, 2018
 *      Author: brian
 */



#include "LevenbergMarquardtMPI.hpp"

void LevMarqMPI::findMin( vector <double> & X, vector <double> & F0,vector <double> & FOpt  )
{
	// Get the number of processes and the current processor ID
	int Nprocs, procID;
	MPI_Comm_size(MPI_COMM_WORLD, &Nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &procID);

	double chiSq, chiSqPrev;
	double lambda = lambda0;

	// Problem size
	int Nparam = X.size();
	int Ndata = F0.size();

	// Storage
	vector<vector<double> > A(Nparam, vector<double>(Nparam));
	vector<vector<double> > J(Ndata, vector<double>(Nparam));
	vector<vector<double> > JT(Nparam, vector<double>(Ndata));
	vector<vector<double> > JTJ(Nparam, vector<double>(Nparam));

	vector <double> rhs( Nparam, 0.0);
	vector <double> dX( Nparam, dXGrad );
	vector <double> delta ( Nparam, 0 );
	vector <double> F( F0.size(), 0);
	vector <double> Fprev( F0.size(), 0);
	vector <double> sigma( Nparam, 0);
	vector <double> Xprev(Nparam,0);


	// Initial f values
	mObjPtr->objEval(X,F0);
	for(int k = 0; k < Ndata; k++ )
	{
		F[k] = F0[k];
		Fprev[k] = F[k];
	}
	for(int k = 0; k < Nparam; k++ )
		Xprev[k] = X[k];

	chiSq = pow(vector2Norm(F),2);

	int iter = 0;
	double xdiff2Norm = xMinDiff*2;
	while (iter < maxIter )
	{


		// Update the gradient
		mObjPtr->gradientApproximationMPI(X, dX, J);


		// compute LHS matrix
		matrixTranspose(J,JT);
		matrixMultiply(JT,J,JTJ);
		for(int i = 0; i < Nparam; i++)
		{
			for(int j = 0; j < Nparam; j++)
			{
				A[i][j] = JTJ[i][j];
				// In the Marquardt version the matrix is
				// J^TJ + lambda * diag (J^TJ)
				if( i == j)
				{
					A[i][j] = (1+lambda)*JTJ[i][j];
				}
			}
		}



		// compute RHS vector
		matrixVectorMultiply(JT,F,rhs);
		for(int i = 0; i < Nparam; i++)
			rhs[i] = -rhs[i];

		// Solve for sigma
		luSolve( A, rhs, sigma );

		// store previous
		for(int k = 0; k < Nparam; k++ )
			Xprev[k] = X[k];
		for(int k = 0; k < Ndata; k++)
			Fprev[k] = F[k];

		// update parameters
		for(int i = 0; i < Nparam; i++)
		{
			X[i] = X[i] + sigma[i];
		}

		// Update F
		mObjPtr->objEval(X,F);


		// Update chi
		double chiSqPrev = chiSq;
		chiSq = pow(vector2Norm(F),2);

		if( chiSq >= chiSqPrev || chiSq != chiSq )
		{

	    	if( verbose >= 1 && procID == ROOT_ID )
	    	{
	    		cout << "Step " << iter << " failed with chiSq = " << chiSq << ", chiSqPrev = " << chiSqPrev;
	    	    cout << ",  increasing lambda: " << lambda/lambdaFactor <<" --> " << lambda <<  endl;

	    	}
			// If new step is worse than previous step, reset X and increase lambda
			chiSq = chiSqPrev;
		   	for(int i = 0; i < Nparam; i++)
			{
				X[i] = Xprev[i];
			}
			for(int k = 0; k < Ndata; k++ )
			{
				F[k] = Fprev[k];
			}
			lambda = lambda*lambdaFactor;

		}
		else
		{
			// otherwise keep new step and reduce lambda
			lambda = lambda/lambdaFactor;

	    	// Check stopping criterion
	    	xdiff2Norm =  vector2Norm ( sigma );
	    	if( xdiff2Norm < xMinDiff)
	    		break;
		}



		if( verbose >= 1 && mod(iter,10) == 0 && procID == ROOT_ID )
		{
			cout << "At iter = " << iter << " the xdiff 2Norm = " << xdiff2Norm <<
					", chi^2 = " << chiSq <<  ", and params: ";
			print1DVector(X);
		}

		iter++;
	}


	for(int k = 0; k < Ndata; k++ )
	{
		FOpt[k] = F[k];
	}
	if( verbose >= 0  && procID == ROOT_ID )
	{
		cout << endl << "-----------------------------------------------------------------------------------" << endl ;

		cout << "Completed Levenberg Marquardt." << endl;
		cout << "At iter = " << iter << " the xdiff 2Norm = " << xdiff2Norm <<
							", chi^2 = " << chiSq << ", and  optimal params: ";
		cout << endl;
		print1DVector(X);
		cout << "-----------------------------------------------------------------------------------" << endl << endl ;
	}

}

