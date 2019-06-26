/*
 * PNOL_Objective.cpp
 *
 *  Created on: Oct 23, 2018
 *      Author: brian
 */


#include "PNOL_Objective.hpp"


void Objective::gradientApproximation( vector <double> & X, vector <double> & dX, vector <double> & dFdX )
{

	int N = X.size();

	vector <double> XdX (N,0);

	double F = objEval(X);
	for(int i = 0; i < N; i++ )
	{

		for(int j = 0; j < N; j++ )
		{
			XdX[j] = X[j];
		}
		XdX[i] = XdX[i] + dX[i];

		double FdX = objEval(XdX);

		dFdX[i] = (FdX - F)/dX[i];
	}

}


// computes finite difference approximation of the hessian B
void  Objective::hessianApproximation( vector <double> & X, vector <double> & dX, vector<vector<double>> & B )
{
	int N = X.size();

	vector <double> XdXi (N,0);
	vector <double> XdXj (N,0);
	vector <double> XdXij (N,0);

	double F = objEval(X);

	// Only compute upper right Hessian
	for(int i = 0; i < N; i++ )
	{
		for(int j = i; j < N; j++)
		{

			for(int k = 0; k < N; k++ )
			{
				XdXi[k] = X[k];
				XdXj[k] = X[k];
				XdXij[k] = X[k];
			}
			XdXi[i] = XdXi[i] + dX[i];
			XdXj[j] = XdXj[j] + dX[j];
			XdXij[i] = XdXij[i] + dX[i];
			XdXij[j] = XdXij[j] + dX[j];

			double FdXi = objEval(XdXi);
			double FdXj = objEval(XdXj);
			double FdXij = objEval(XdXij);

			B[i][j] = ( FdXij - FdXi - FdXj + F )/(dX[i]*dX[j]);

		}
	}


	// Simply assign lower left hessian based on symmetry
	for(int i = 0; i < N; i++ )
	{
		for(int j = 0; j < i; j++)
		{
			B[i][j] = B[j][i];
		}
	}


}


void Objective::gradientApproximationMPI( vector <double> & X, vector <double> & dX, vector <double> & dFdX )
{
	double F_local = 0;
	double F = 0;
	double * FdX_local = new double[ X.size() ];
	double * FdX = new double [X.size()];
	for(int i = 0; i < X.size(); i++ )
	{
		FdX_local[i] = 0;
		FdX[i] = 0;
	}

	// Get the number of processes and the current processor ID
	int Nprocs, procID;
	MPI_Comm_size(MPI_COMM_WORLD, &Nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &procID);

	// Compute load distribution
	// number of function evaluations
	int Neval = X.size() + 1;

	// compute load balance
	vector<int>procEvalID(Neval,0);
	int procIdx = 0;
	for(int k = 0; k < Neval; k++)
	{
		if( procIdx > Nprocs - 1 )
		{
			procIdx = 0;
		}
		procEvalID[k] = procIdx;

		procIdx = procIdx+1;
	}


	// compute the gradient
	int N = X.size();

	vector <double> XdX (N,0);
	for(int i = 0; i < Neval; i++ )
	{
		if ( i < N && procEvalID[i] == procID )
		{
			for(int j = 0; j < N; j++ )
			{
				XdX[j] = X[j];
			}
			XdX[i] = XdX[i] + dX[i];

		    FdX_local[i] = objEval(XdX);

		}
		else if ( procEvalID[i] == procID  )
		{
			F_local = objEval(X);
		}
	}

	MPI_Allreduce(&F_local, &F, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(FdX_local, FdX, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	for(int i = 0; i < N; i++ )
	{
		dFdX[i] = (FdX[i] - F)/dX[i];
	}


	// clean up
	delete [] FdX_local;
	delete [] FdX;
}





void MultiObjective::gradientApproximation( vector <double> & X, vector <double> & dX, vector< vector<double> > & J )
{

	int N = X.size();

	vector <double> XdX (N,0);

	// create F and FdX of correct size to match data
	vector <double> F(J.size(),0);
	vector <double> FdX(J.size(),0);

	// evaluate at the starting value
	objEval(X,F);

	for(int j = 0; j <  N; j++ )
	{

		for(int k = 0; k < N; k++ )
		{
			XdX[k] = X[k];
		}
		XdX[j] = XdX[j] + dX[j];

		objEval(XdX, FdX);

		for(int i = 0; i < J.size(); i++ )
		{
			J[i][j] = (FdX[i] - F[i])/dX[j];
		}
	}


}




void MultiObjective::gradientApproximationMPI( vector <double> & X, vector <double> & dX, vector< vector<double> > & J )
{

	int Nprm = X.size();
	int Ndata = J.size();

	// create F and FdX of correct size to match data
	vector <double> F(J.size(),0);
	vector <double> FdX(J.size(),0);


	double * F_local = new double [ Ndata ];
	double ** FdX_local = new double * [ Nprm ];
	for(int i = 0; i < Nprm; i++ )
	{
		FdX_local[i] = new double [ Ndata];
		for( int j = 0; j < Ndata; j++ )
		{
			FdX_local[i][j] = 0.0;
			F_local[j] = 0.0;
		}
	}

	// Get the number of processes and the current processor ID
	int Nprocs, procID;
	MPI_Comm_size(MPI_COMM_WORLD, &Nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &procID);

	// Compute load distribution
	// number of function evaluations
	int Neval = Nprm + 1;

	// compute load balance
	vector<int>procEvalID(Neval,0);
	int procIdx = 0;
	for(int k = 0; k < Neval; k++)
	{
		if( procIdx > Nprocs - 1 )
		{
			procIdx = 0;
		}
		procEvalID[k] = procIdx;

		procIdx = procIdx+1;
	}


	vector <double> XdX (Nprm,0);
	for(int j = 0; j < Neval; j++ )
	{

		if ( j < Nprm && procEvalID[j] == procID )
		{

			for(int k = 0; k < Nprm; k++ )
			{
				XdX[k] = X[k];
			}
			XdX[j] = XdX[j] + dX[j];

			objEval(XdX, FdX);

			for(int k = 0; k < Ndata; k++ )
			{
				FdX_local[j][k] = FdX[k];
			}
		}
		else if ( procEvalID[j] == procID  )
		{
			objEval(X,F);
			for(int k = 0; k < Ndata; k++ )
			{
				F_local[k] = F[k];
			}
		}
	}

	MPI_Allreduce(F_local, F.data(), Ndata, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	for(int j = 0; j < Nprm; j++ )
	{
		MPI_Allreduce(FdX_local[j], FdX.data(), Ndata, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		for(int i = 0; i < Ndata; i++ )
		{
			J[i][j] = (FdX[i] - F[i])/dX[j];
		}
	}


	// clean up
	for(int i = 0; i < Nprm; i++ )
	{
		delete [] FdX_local[i];
	}
	delete [] FdX_local;
	delete [] F_local;

}



double Objective::objEvalRecur( vector <double> & Xrecur, vector <double> & constantX, vector<bool> & constantIndicator )
{

	int Nrecur = Xrecur.size();
	int Nparam = constantX.size();

	vector <double> X(Nparam,0);

	int iRecur = 0;
	for(int i = 0; i <Nparam; i++)
	{
		if(constantIndicator[i])
		{
			X[i] = constantX[i];
		}
		else
		{
			X[i] = Xrecur[iRecur];
			iRecur++;
		}
	}

	// Ensure that all processes are using values on ROOT
	double F = objEval(X);


	return F;



}


// gradient approximation with constants
void Objective::gradientApproximationRecur( vector <double> & X, vector <double> & dX, vector <double> & dFdX,
		vector <double> & constantX, vector<bool> & constantIndicator )
{

	int N = X.size();

	vector <double> XdX (N,0);

	double F = objEvalRecur(X, constantX, constantIndicator);
	for(int i = 0; i < N; i++ )
	{

		for(int j = 0; j < N; j++ )
		{
			XdX[j] = X[j];
		}
		XdX[i] = XdX[i] + dX[i];

		double FdX = objEvalRecur(XdX, constantX, constantIndicator);

		dFdX[i] = (FdX - F)/dX[i];
	}

}




// parallel gradient approximation with constants
void Objective::gradientApproximationMPIRecur( vector <double> & X, vector <double> & dX, vector <double> & dFdX,
		vector <double> & constantX, vector<bool> & constantIndicator )
{


	double F_local = 0;
	double F = 0;
	double * FdX_local = new double[ X.size() ];
	double * FdX = new double [X.size()];
	for(int i = 0; i < X.size(); i++ )
	{
		FdX_local[i] = 0;
		FdX[i] = 0;
	}

	// Get the number of processes and the current processor ID
	int Nprocs, procID;
	MPI_Comm_size(MPI_COMM_WORLD, &Nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &procID);

	// Ensure that all processes are using values on ROOT
	MPI_Bcast(X.data(), X.size(), MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);
	MPI_Bcast(dX.data(), dX.size(), MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);
	MPI_Bcast(constantX.data(), constantX.size(), MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);
	// Broadcast the constant indicator requires change in the type
	vector <int> constantIndicatorTemp (constantIndicator.size(),0);
	for(int i = 0; i < constantIndicator.size(); i++){ constantIndicatorTemp[i] = (int) constantIndicator[i]; }
	MPI_Bcast(constantIndicatorTemp.data(), constantIndicatorTemp.size(), MPI_INT, ROOT_ID, MPI_COMM_WORLD);
	for(int i = 0; i < constantIndicator.size(); i++){ constantIndicator[i] = (bool) constantIndicatorTemp[i]; }

	// Compute load distribution
	// number of function evaluations
	int Neval = X.size() + 1;

	// compute load balance
	vector<int>procEvalID(Neval,0);
	int procIdx = 0;
	for(int k = 0; k < Neval; k++)
	{
		if( procIdx > Nprocs - 1 )
		{
			procIdx = 0;
		}
		procEvalID[k] = procIdx;

		procIdx = procIdx+1;
	}


	// compute the gradient
	int N = X.size();

	vector <double> XdX (N,0);
	for(int i = 0; i < Neval; i++ )
	{
		if ( i < N && procEvalID[i] == procID )
		{
			for(int j = 0; j < N; j++ )
			{
				XdX[j] = X[j];
			}
			XdX[i] = XdX[i] + dX[i];

			FdX_local[i] = objEvalRecur(XdX, constantX, constantIndicator);

		}
		else if ( procEvalID[i] == procID  )
		{
			F_local = objEvalRecur(X, constantX, constantIndicator);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Reduce(&F_local, &F, 1, MPI_DOUBLE, MPI_SUM, ROOT_ID, MPI_COMM_WORLD);
	MPI_Reduce(FdX_local, FdX, N, MPI_DOUBLE, MPI_SUM, ROOT_ID, MPI_COMM_WORLD);
	MPI_Bcast(&F, 1, MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);
	MPI_Bcast(FdX, N, MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);



	for(int i = 0; i < N; i++ )
	{
		dFdX[i] = (FdX[i] - F)/dX[i];
	}


	// clean up
	delete [] FdX_local;
	delete [] FdX;



}














