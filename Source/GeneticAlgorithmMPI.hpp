/*
 * GeneticAlgorithmMPI.hpp
 *
 *  Created on: Nov 6, 2018
 *      Author: brian
 */

#ifndef GENETICALGORITHMMPI_HPP_
#define GENETICALGORITHMMPI_HPP_



// standard includes
#include <vector>
#include <ctime>    // For time()
#include <cstdlib>  // For srand() and rand()
#include <mpi.h>

// Utility includes
#include "UtilityFunctions/utilityFunctions.hpp"
#include "VTKRenderContainer.hpp"



// local includes
#include "PNOL_Algorithm.hpp"

using namespace std;


// local functions
#include "GeneticAlgorithm.hpp"


class GeneticAlgorithmMPI : public AlgorithmBnd {

  private:

	// GA parameters
	int Npop;
	int maxGenerations;

	double eliteFrac, crossFrac, eliteMutationFrac;
	double mutationSize, eliteMutationSize;
	double initialPopScaling;
	double NstaticGenerations;

	bool verbose; // verbose output from optimization?

  public:
	// main optimization function (instantiated algorithm function)
	void findMinBnd( std::vector <double> & X, std::vector <double> & Xlb, std::vector <double> & Xub, double & f0 , double & fOpt );

	// set the parameters
	void setGAParams( int NpopIn, int maxGenerationsIn, double eliteFracIn, double crossFracIn, double eliteMutationFracIn,
			 double mutationSizeIn, double eliteMutationSizeIn, double initialPopScalingIn,
			double  NstaticGenerationsIn, bool verboseIn)
	{ Npop = NpopIn; eliteFrac = eliteFracIn; crossFrac = crossFracIn; eliteMutationFrac = eliteMutationFracIn;
	maxGenerations = maxGenerationsIn; mutationSize = mutationSizeIn; eliteMutationSize = eliteMutationSizeIn;
	 NstaticGenerations =  NstaticGenerationsIn; verbose = verboseIn;initialPopScaling = initialPopScalingIn;
	}

	void evaluatePopulationParallel( vector<vector<double> > &Xpop, vector <double> & F, vector <bool> & evaluateIndicator );

    // default constructor/destructor
	GeneticAlgorithmMPI()
	{
		Npop = 100;
		maxGenerations = 1000;
		eliteFrac = 0.1;
		crossFrac = 0.3;
		eliteMutationFrac = 0.2;
		mutationSize = 0.5;
		eliteMutationSize = 0.01;
		initialPopScaling = 0.5;
		NstaticGenerations = 50;
		verbose = 0;
	}
	~GeneticAlgorithmMPI(){}


};



#endif /* GENETICALGORITHMMPI_HPP_ */
