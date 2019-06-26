/*
 * Examples.cpp
 *
 *  Created on: Oct 22, 2018
 *      Author: brian
 */


#include "Examples.hpp"


void testBFGSBndMPISW()
{

	// Object
	RosenbrockObject obj;
	// PowerObject obj;
	// obj.setPower(2);

	// Vector of optization variables
	int Nparam = 3;
	vector <double> X(Nparam,2);
	vector <double> Xlb(Nparam,-1);
	vector <double> Xub(Nparam,5);
	X[0] = -1;

	// Algorithm
	BFGS_Bnd_MPI_SW bfgs;
	bfgs.setObjPtr(obj);

	// Run bnd bfgs search
	/*
	void setParams( double c1In, double c2In, double dalphaIn, double alphaGuessIn, double alphaTolIn, double alphaMultIn,
			int maxIterLineSearchIn, double dXGradIn, double dXHessIn, double maxIterIn,
			double xMinDiffIn, double minGrad2NormIn, bool initHessFDIn, bool verboseIn )
	*/
	bfgs.setParams(1e-4, 0.8, 1e-6, 1, 1e-10, 2, 50, 1e-5, 1e-6, 1e-3, 200, 1e-5, 1e-5, 0, 2);


	double f0, fOpt;
	bfgs.findMinBnd( X, Xlb, Xub, f0, fOpt );

	cout << "Optimization used " << obj.getEvals() << " function evaluations." << endl;

}



void testBFGSBnd()
{

	// Object
	RosenbrockObject obj;
	// PowerObject obj;
	// obj.setPower(2);

	// Vector of optization variables
	int Nparam = 5;
	vector <double> X(Nparam,2);
	vector <double> Xlb(Nparam,-5);
	vector <double> Xub(Nparam,5);



	// Algorithm
	BFGS_Bnd bfgs;
	bfgs.setObjPtr(obj);

	// Run bnd bfgs search
	/*
	void setParams( double c1In, double c2In, double dalphaIn, double alphaGuessIn, double alphaTolIn, double alphaMultIn,
			int maxIterLineSearchIn, double dXGradIn, double dXHessIn, double maxIterIn,
			double xMinDiffIn, double minGrad2NormIn, bool initHessFDIn, bool verboseIn )
	*/
	bfgs.setParams(1e-4, 0.8, 1e-6, 1, 1e-10, 2, 50, 1e-5, 1e-6, 1e-3, 200, 1e-5, 1e-5, 0, 1);


	double f0, fOpt;
	bfgs.findMinBnd( X, Xlb, Xub, f0, fOpt );

	cout << "Optimization used " << obj.getEvals() << " function evaluations." << endl;

}






void testBFGSBnd_MPI()
{

	// Object
	RosenbrockObject obj;
	// PowerObject obj;
	// obj.setPower(2);

	// Vector of optization variables
	int Nparam = 10;
	vector <double> X(Nparam,3);
	vector <double> Xlb(Nparam,-5);
	vector <double> Xub(Nparam,5);
	Xlb[0] = -1;
	X[0] = -0.5;
	X[1] = 3;

	// Algorithm
	BFGSBnd_MPI bfgs;
	bfgs.setObjPtr(obj);

	// Run bnd bfgs search
	bfgs.setParams(1e-4, 0.1, 1e-16, 4, 1, 1000,  1e-7,1e-3,200,1e-5,1e-5,1e-5,0,1);


	double f0, fOpt;
	bfgs.findMinBnd( X, Xlb, Xub, f0, fOpt );

	cout << "Optimization used " << obj.getEvals() << " function evaluations." << endl;

}







void testLMExpMPI(){

	// Initial parameter guess
	int Nparam = 3;

	vector <double> X(Nparam,0.1);
	vector <double> dX(X.size(),1e-6);

	// objective
	ExpCurveObjective mObj;

	// gradient 2D vector
	int Ndata = mObj.getDataSize();
	vector<vector<double> > J(Ndata, vector<double>(Nparam));
	vector <double> F0(Ndata,0.0);
	vector <double> FOpt(Ndata,0.0);

	// test gradient
	mObj.gradientApproximationMPI(X,dX,J);

	// Set algorithm
	LevMarqMPI lmOptimizer;
	lmOptimizer.setObjPtr(mObj);
	lmOptimizer.setParams(0.001, 10, 1e-6, 100, 1e-6, true );


	// find minimum
	lmOptimizer.findMin(X,F0,FOpt);




}


void testBFGS_MPI()
{

	// Object
	RosenbrockObject obj;
	// PowerObject obj;
	// obj.setPower(2);

	// Vector of optization variables
	int Nparam = 10;
	vector <double> X(Nparam,10);


	// Algorithm
	BFGS_MPI bfgs;
	bfgs.setObjPtr(obj);

	// Rim simplex search
	bfgs.setParams(1e-4, 0.1, 4, 1, 1000,  1e-7,1e-3,200,1e-5,1e-5,0,1);


	double f0, fOpt;
	bfgs.findMin( X, f0, fOpt );

	cout << "Optimization used " << obj.getEvals() << " function evaluations." << endl;

}




void testBFGS_booth()
{

	// Object
	// BoothFunction obj;
	GoldsteinFunction obj;

	// Vector of optization variables
	int Nparam = 2;
	vector <double> X(Nparam,1);


	// Algorithm
	BFGS bfgs;
	bfgs.setObjPtr(obj);

	// Rim simplex search
	// bfgs.setParams(1e-7,1e-3,100,1e-5,1,1);
	bfgs.setParams(1e-4, 0.9, 1e-6, 1, 1000,  1e-7,1e-3,100,1e-5,1e-5,0,1);


	double f0, fOpt;
	bfgs.findMin( X, f0, fOpt );


	cout << "Optimization used " << obj.getEvals() << " function evaluations." << endl;




}



void testBFGS()
{

	// Object
	RosenbrockObject obj;
	// PowerObject obj;
	// obj.setPower(2);

	// Vector of optization variables
	int Nparam = 5;
	vector <double> X(Nparam,3);


	// Algorithm
	BFGS bfgs;
	bfgs.setObjPtr(obj);

	// Rim simplex search
	// bfgs.setParams(1e-7,1e-3,100,1e-5,1,1);
	bfgs.setParams(1e-4, 0.9, 1e-6, 1, 1000,  1e-7,1e-3,100,1e-5,1e-5,0,1);


	double f0, fOpt;
	bfgs.findMin( X, f0, fOpt );

	cout << "Optimization used " << obj.getEvals() << " function evaluations." << endl;

}










void testHessian(){

	int Nparam = 4;

	vector <double> X(Nparam,3.0);
	vector <double> dX(X.size(),1e-3);
	vector<vector<double> > B(Nparam, vector<double>(Nparam));

	PowerObject obj;
	obj.setPower(2);

	obj.hessianApproximation(X,dX,B);


	for(int i = 0; i < Nparam; i++)
	{
		for(int j = 0; j < Nparam; j++)
		{

			B[i][j] = j + i;
			if(i == j)
				B[i][j] =B[i][j] +  (i+1)*10;


		}
	}
	vector<vector<double> > Binv(Nparam, vector<double>(Nparam));


	print2DVector(B);
	matrixInverse( B,  Binv );

	print2DVector(Binv);




}



void testGAParallel(){

	// Object
	PowerObjectSlow obj;
	obj.setPower(2);

	// Vector of optization variables
	int NOptim = 4;
	vector <double> X;
	for(int k = 0; k < NOptim; k++)
		X.push_back(3.0);

	// bounds
	vector <double> Xlb( NOptim, -10);
	vector <double> Xub( NOptim, 10);

	// Algorithm
	GeneticAlgorithmMPI ga;
	ga.setObjPtr(obj);

	// void setGAParams( int NpopIn, int maxGenerationsIn, double eliteFracIn, double crossFracIn, double eliteMutationFracIn,
	//		 double mutationSizeIn, double eliteMutationSizeIn, double initialPopScalingIn,
	//		 double NstaticGenerations, bool verboseIn)
	ga.setGAParams( 150, 10, 0.2, 0.4, 0.1, 0.5, 0.01, 0.5, 50, 1 );

	double f0, fOpt;
	ga.findMinBnd( X, Xlb, Xub, f0, fOpt );



}




void testGA(){

	// Object
	PowerObject obj;
	obj.setPower(2);

	// Vector of optization variables
	int NOptim = 4;
	vector <double> X;
	for(int k = 0; k < NOptim; k++)
		X.push_back(3.0);

	// bounds
	vector <double> Xlb( NOptim, -10);
	vector <double> Xub( NOptim, 10);

	// Algorithm
	GeneticAlgorithm ga;
	ga.setObjPtr(obj);

	// void setGAParams( int NpopIn, int maxGenerationsIn, double eliteFracIn, double crossFracIn, double eliteMutationFracIn,
	//		 double mutationSizeIn, double eliteMutationSizeIn, double initialPopScalingIn,
	//		 double NstaticGenerations, bool verboseIn)
	ga.setGAParams( 100, 1000, 0.1, 0.30, 0.2, 0.5, 0.01, 0.5, 20, 1, 0 );

	double f0, fOpt;
	ga.findMinBnd( X, Xlb, Xub, f0, fOpt );





}



void testLMExp(){

	cout << endl << "Testing LM on nonlinear exp least squares problem " << endl;

	// Initial parameter guess
	int Nparam = 3;

	vector <double> X(Nparam,0.1);
	vector <double> dX(X.size(),1e-6);

	// objective
	ExpCurveObjective mObj;

	// gradient 2D vector
	int Ndata = mObj.getDataSize();
	vector<vector<double> > J(Ndata, vector<double>(Nparam));
	vector <double> F0(Ndata,0.0);
	vector <double> FOpt(Ndata,0.0);

	// test gradient
	mObj.gradientApproximation(X,dX,J);

	// Set algorithm
	LevMarq lmOptimizer;
	lmOptimizer.setObjPtr(mObj);
	lmOptimizer.setParams(0.001, 10, 1e-6, 100, 1e-6, true );


	// find minimum
	lmOptimizer.findMin(X,F0,FOpt);




}


void testLMCubicLinearCoef(){

	cout << endl << "Testing LM on cubic linear coef least squares problem " << endl;
	// Initial parameter guess
	int Nparam = 4;

	vector <double> X(Nparam,0.1);
	vector <double> dX(X.size(),1e-6);

	// objective
	CubicObjective mObj;

	// gradient 2D vector
	int Ndata = mObj.getDataSize();
	vector<vector<double> > J(Ndata, vector<double>(Nparam));
	vector <double> F0(Ndata,0.0);
	vector <double> FOpt(Ndata,0.0);

	// test gradient
	mObj.gradientApproximation(X,dX,J);

	// Set algorithm
	LevMarq lmOptimizer;
	lmOptimizer.setObjPtr(mObj);

	// lambda = 0 for a linear problem
	lmOptimizer.setParams( 0.001, 10, 1e-6, 10, 1e-5, true );


	// find minimum
	lmOptimizer.findMin(X,F0,FOpt);




}





void testSimplexSearch(){



	// Object
	// PowerObject obj;
	// obj.setPower(2);
	// ExpCurveObjectiveSingle obj;
	RosenbrockObject obj;

	// Vector of optization variables
	int NOptim = 10;
	vector <double> X;
	for(int k = 0; k < NOptim; k++)
		X.push_back(3.0);


	// Algorithm
	SimplexSearch simpSearch;
	simpSearch.setObjPtr(obj);

	// Rim simplex search
	simpSearch.setSimplexParams( 1.0,  2.0, 0.5, 0.5, 10000, 1.0, 1e-7, 1 );

	double f0, fOpt;
	simpSearch.findMin( X, f0, fOpt );

	cout << "Optimization used " << obj.getEvals() << " function evaluations." << endl;

}







void testCreateObject(){

	vector <double> X;

	for(int k = 0; k < 3; k++)
		X.push_back(3.0);

	PowerObject obj;
	obj.setPower(3);

	cout << obj.getPower() << endl;
	cout << obj.objEval(X) << endl;

	testObjectEvaluation(obj);

	X.clear();
}


void testGradientEvaluation( ) {

	PowerObject obj;
	obj.setPower(3);

	int Ndim = 5;

	vector <double> X(Ndim,3);
	vector <double> dX(Ndim,1e-6);
	vector <double> dFdX(Ndim,0);
	vector <double> dFdX_MPI(Ndim,0);

	obj.gradientApproximation(X,dX,dFdX);

	obj.gradientApproximationMPI(X,dX,dFdX_MPI);

	// Get the number of processes and the current processor ID
	int Nprocs, procID;
	MPI_Comm_size(MPI_COMM_WORLD, &Nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &procID);

	if(procID == 0)
	{
		print1DVector(dFdX);
		print1DVector(dFdX_MPI);
	}


}


void testObjectEvaluation( Objective & obj ) {

	vector <double> X;

	for(int k = 0; k < 3; k++)
		X.push_back(3.0);


	cout << obj.objEval(X) << endl;

	X.clear();

}




void testGradientApproxMultMPI(){

	// Get the number of processes and the current processor ID
	int Nprocs, procID;
	MPI_Comm_size(MPI_COMM_WORLD, &Nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &procID);


	cout << endl << "Testing gradient approx mult mpi" << endl;
	// Initial parameter guess
	int Nparam = 4;

	vector <double> X(Nparam,0.1);
	vector <double> dX(X.size(),1e-6);

	// objective
	CubicObjective mObj;

	// gradient 2D vector
	int Ndata = mObj.getDataSize();
	vector<vector<double> > J(Ndata, vector<double>(Nparam));
	vector <double> F0(Ndata,0.0);
	vector <double> FOpt(Ndata,0.0);

	// test gradient
	mObj.gradientApproximation(X,dX,J);
	if(procID == 0) {print2DVector(J);}
	mObj.gradientApproximationMPI(X,dX,J);
	if(procID == 0) {print2DVector(J);}
}



void testGradientApproxMultMPIRecur(){

	// Get the number of processes and the current processor ID
	int Nprocs, procID;
	MPI_Comm_size(MPI_COMM_WORLD, &Nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &procID);


	if(procID == 0){ cout << endl << "Testing gradient approx mult mpi recur " << endl; }
	// Initial parameter guess
	int Nparam = 8;

	vector <double> X(Nparam,0.1);
	for(int i = 0; i < Nparam; i++)
	{
		X[i] = i*0.1;
	}
	vector <double> dX(X.size(),1e-6);
	vector <double> dFdX(X.size(),1e-6);

	// objective
	RosenbrockObject obj;

	// gradient 2D vector
	obj.gradientApproximation(X,dX,dFdX);

	if(procID == 0)
	{
		print1DVector(X);
		print1DVector(dFdX);
		cout << "----" << endl;
	}

	vector<double> constantX(Nparam,0);
	vector<bool> constantIndicator(Nparam,false);
	obj.gradientApproximationRecur(X,dX,dFdX,constantX,constantIndicator);

	if(procID == 0)
	{
		print1DVector(X);
		print1DVector(dFdX);
		cout << "----" << endl;
	}


	constantX[3] = X[3];
	constantIndicator[3] = true;

	vector<double> Xrecur(Nparam-1,0.1);
	int idxRecur = 0;
	for(int i = 0; i < Nparam; i++)
	{
		if(!constantIndicator[i])
		{
			Xrecur[idxRecur] = X[i];
			idxRecur++;
		}

	}
	vector<double> dXrecur(Nparam-1,1e-6);
	vector<double> dFdXrecur(Nparam-1,1e-6);

	obj.gradientApproximationRecur(Xrecur,dXrecur,dFdXrecur,constantX,constantIndicator);
	if(procID == 0)
	{
		print1DVector(Xrecur);
		print1DVector(dFdXrecur);
		cout << "----" << endl;
	}

}








