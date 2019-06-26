/*
 * Box_boundary_functions.cpp
 *
 *  Created on: Nov 12, 2018
 *      Author: brian
 */


#include "Box_boundary_functions.hpp"

void checkBoxBounds( vector <double> & X, vector <double> & Xlb, vector <double> & Xub )
{

	// Get current processor ID
	int procID;
	MPI_Comm_rank(MPI_COMM_WORLD, &procID);

	for ( int i = 0; i < X.size(); i++ )
	{
		if(X[i] - Xlb[i] < -abs(Xlb[i])/1000 || X[i] - Xub[i] > abs(Xub[i])/1000)
		{
			if(procID == 0)
			{
				cout << endl << "!!!!----------------- WARNING -----------------!!!!" << endl;
				cout << "X[" << i << "] = " << X[i] << " is outside of bounds Xlb[i] = " << Xlb[i] << ", Xub[i] = " << Xub[i] << endl;
			}

			X[i] = (Xlb[i] + Xub[i] )/2.0;

			if(procID == 0)
			{
				cout << " Replaced X[" << i << "] with " << X[i] << endl;
				cout << "!!!!----------------- END WARNING -----------------!!!!" << endl << endl;
			}

		}
	}


}


// Sets random values between Xlb and Xub using hardware random number generator
void setHardRandValues( vector <double> & X, vector <double> & Xlb, vector <double> & Xub )
{

	// Get current processor ID
	int procID;
	MPI_Comm_rank(MPI_COMM_WORLD, &procID);

	if(procID == ROOT_ID)
	{
		for ( int i = 0; i < X.size(); i++ )
		{
			X[i] = ( Xub[i] - Xlb[i] )*hardRand() + Xlb[i];
		}

		cout << " Set random values of X: "; print1DVector(X);
	}
	MPI_Bcast( X.data(), X.size(), MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);

}
