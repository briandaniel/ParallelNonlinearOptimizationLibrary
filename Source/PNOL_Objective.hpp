/*
 * PNOL_Objective.hpp
 *
 *  Created on: Oct 22, 2018
 *      Author: brian
 */

#ifndef PNOL_OBJECTIVE_HPP_
#define PNOL_OBJECTIVE_HPP_

using namespace std;

// Parallel commands
#ifndef ROOT_ID
#define ROOT_ID 0 // id of root process
#endif

// Standard includes
#include <vector>
#include <mpi.h>

// local includes

// Evaluates to a single double return
class Objective{

  public:
	// Pure virtual objective evaluation
	virtual double objEval( vector <double> & X ) = 0;

	// gradient evaluation
	void gradientApproximation( vector <double> & X, vector <double> & dX, vector <double> & dFdX );

	// Hessian approximate
	void hessianApproximation( vector <double> & X, vector <double> & dX, vector<vector<double>> & H );

	// parallel gradient evaluation
	void gradientApproximationMPI( vector <double> & X, vector <double> & dX, vector <double> & dFdX );

	// Object evaluation with constant members
	double objEvalRecur( vector <double> & Xrecur, vector <double> & constantX, vector<bool> & constantIndicator );

	// gradient approximation with constants
	void gradientApproximationRecur( vector <double> & X, vector <double> & dX, vector <double> & dFdX, vector <double> & constantX, vector<bool> & constantIndicator );

	// parallel gradient approximation with constants
	void gradientApproximationMPIRecur( vector <double> & X, vector <double> & dX, vector <double> & dFdX, vector <double> & constantX, vector<bool> & constantIndicator );

};


// Evaluates to multiple objective outputs F
class MultiObjective{

  public:
	// Pure virtual objective evaluation
	virtual void objEval( vector <double> & X, vector <double> & F ) = 0;

	void gradientApproximation( vector <double> & X, vector <double> & dX, vector< vector<double> > & J );

	void gradientApproximationMPI( vector <double> & X, vector <double> & dX, vector< vector<double> > & J );
};


#endif /* PNOL_OBJECTIVE_HPP_ */
