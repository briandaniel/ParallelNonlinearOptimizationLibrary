/*
 * ExamplesVis.cpp
 *
 *  Created on: Oct 24, 2018
 *      Author: brian
 */



#include "ExamplesVis.hpp"

void testGAWithVis()
{


	// Object
	PowerObject obj;
	obj.setPower(2);

	// Vector of optization variables
	int NOptim = 2;
	vector <double> X;
	for(int k = 0; k < NOptim; k++)
		X.push_back(7.0);

	// bounds
	vector <double> Xlb( NOptim, -10);
	vector <double> Xub( NOptim, 10);

	// Algorithm
	GeneticAlgorithm ga;
	ga.setObjPtr(obj);

	// void setGAParams( int NpopIn, int maxGenerationsIn, double eliteFracIn, double crossFracIn, double eliteMutationFracIn,
	//		 double mutationSizeIn, double eliteMutationSizeIn, double initialPopScalingIn,
	//		 double NstaticGenerations, bool verboseIn, bool graph)
	ga.setGAParams( 150, 1000, 0.3, 0.5, 0.2, 0.5, 0.01, 0.5, 50, 1, 1 );

	double f0, fOpt;
	ga.findMinBnd( X, Xlb, Xub, f0, fOpt );



}










