/*
 * GeneticAlgorithm.hpp
 *
 *  Created on: Oct 24, 2018
 *      Author: brian
 */

#ifndef GENETICALGORITHM_HPP_
#define GENETICALGORITHM_HPP_


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

void checkPopulationBoundsAndReplace( vector<vector<double> > &Xpop, std::vector <double> & Xlb, std::vector <double> & Xub, vector <bool> & evaluateIndicator );
void checkIndenticalChildAndReplace( vector<vector<double> > &Xpop, std::vector <double> & Xlb, std::vector <double> & Xub, vector <bool> & evaluateIndicator  );
void popSort( vector<vector<double> > &Xpop, vector <double> & F );


class GeneticAlgorithm : public AlgorithmBnd {

  private:

	// GA parameters
	int Npop;
	int maxGenerations;

	double eliteFrac, crossFrac, eliteMutationFrac;
	double mutationSize, eliteMutationSize;
	double initialPopScaling;
	double NstaticGenerations;

	bool verbose; // verbose output from optimization?
	bool graph;

  public:
	// main optimization function (instantiated algorithm function)
	void findMinBnd( std::vector <double> & X, std::vector <double> & Xlb, std::vector <double> & Xub, double & f0 , double & fOpt );

	// set the parameters
	void setGAParams( int NpopIn, int maxGenerationsIn, double eliteFracIn, double crossFracIn, double eliteMutationFracIn,
			 double mutationSizeIn, double eliteMutationSizeIn, double initialPopScalingIn,
			double  NstaticGenerationsIn, bool verboseIn, bool graphIn)
	{ Npop = NpopIn; eliteFrac = eliteFracIn; crossFrac = crossFracIn; eliteMutationFrac = eliteMutationFracIn;
	maxGenerations = maxGenerationsIn; mutationSize = mutationSizeIn; eliteMutationSize = eliteMutationSizeIn;
	 NstaticGenerations =  NstaticGenerationsIn; verbose = verboseIn;initialPopScaling = initialPopScalingIn;
	 graph = graphIn;
	}


	void evaluatePopulation( vector<vector<double> > &Xpop, vector <double> & F, vector <bool> & evaluateIndicator );

    // default constructor/destructor
	GeneticAlgorithm()
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
		graph = 0;
	}
	~GeneticAlgorithm(){}


};




#endif /* GENETICALGORITHM_HPP_ */
