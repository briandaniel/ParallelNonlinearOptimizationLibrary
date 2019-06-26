/*
 * GeneticAlgorithmMPI.cpp
 *
 *  Created on: Nov 6, 2018
 *      Author: brian
 */


#include "GeneticAlgorithmMPI.hpp"


void GeneticAlgorithmMPI::findMinBnd( std::vector <double> & X, std::vector <double> & Xlb, std::vector <double> & Xub, double & f0 , double & fOpt )
{

	// Get the number of processes and the current processor ID
	int Nprocs, procID;
	MPI_Comm_size(MPI_COMM_WORLD, &Nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &procID);

	// Problem size
	int Nparam = X.size();

	// Storage for population
	vector<vector<double> > Xpop(Npop, vector<double>(Nparam));
	vector<vector<double> > XpopNew(Npop, vector<double>(Nparam));

	// Storage for population evaluations
	vector <double> F( Npop, 0);
	vector <double> Fnew( Npop, 0);
	vector <bool> evaluateIndicator( Npop, true);
	vector <double> fitness( Npop, 0);


	// Calculate population sizes
	int Nelite, Ncross, NeliteMut, Nrand;
	Nelite = ceil(eliteFrac*Npop);
	NeliteMut = ceil(eliteMutationFrac*Npop);
	Ncross = ceil(crossFrac*Npop);
	Nrand = Npop - Nelite - NeliteMut - Ncross;
	if (Nrand <= 0 )
	{
		cout << "666 GA fractions set incorrectly. Sum must be less than 1 to avoid errors. Exiting..." << endl;
		exit(0);
	}
    if( verbose == true && procID == ROOT_ID )
    {
    	cout << endl << "------------------------------------------------------------" << endl ;

    	cout << "Computing genetic algorithm with population " << endl;
    	cout << "Nelite = " << Nelite << ", NeliteMut = " << NeliteMut << ", Ncross = " << Ncross << ", Nrand = " << Nrand << endl;

    	cout << "------------------------------------------------------------" << endl << endl;
    }

	// Create random initial population
	// Add randomness to other guesses
	srand((unsigned)time(0)); // have to seed the random number generator
	for(int i = 0; i < Nparam; i++)
	{
		Xpop[0][i] = X[i];
	}
	for( int i = 1; i < Npop; i++ )
	{
		for(int j = 0; j < Nparam; j++)
		{
			Xpop[i][j] = Xpop[0][j] + ( ( Xub[j] - Xlb[j] ) * timeRand() + Xlb[j] );
		}
	}

	// Check for identical children
	checkIndenticalChildAndReplace( XpopNew, Xlb, Xub, evaluateIndicator );

	// Check the bounds and replace guesses outside the bounds
	checkPopulationBoundsAndReplace( Xpop, Xlb, Xub, evaluateIndicator );

	// evaluate intial population
	evaluatePopulationParallel( Xpop, F, evaluateIndicator );
	f0 = F[0];

	// 0a. sort population
	popSort( Xpop, F );


	double FbestPrev = F[0];
	int Nstatic = 0;
	int iter = 0;
	while( iter < maxGenerations)
	{

		if( verbose == true && procID == ROOT_ID)
		{
			cout << "At generation = " << iter << " minimum of f = " << F[0] << "  at params:  ";
			print1DVector(Xpop[0]);
			cout << "Full population: ";
			print2DVector(Xpop);
			cout << "with F values "; print1DVector(F);
		}


		// 0b. Evaluate fitness as square difference (can be changed)
		for( int k = 0; k < Npop; k ++)
		{
			fitness[k] = pow( F[Npop-1] - F[k], 2 );
		}
		double maxFitness = fitness[0];


		// 1. Elite children
		for(int k = 0; k < Npop; k++){ evaluateIndicator[k] = true; }// Set to evaluate all
		int popIdx = 0;
		for( int k = 0; k < Nelite; k ++)
		{
			// save values
			for(int i = 0; i < Nparam; i++)
			{
				XpopNew[popIdx][i] = Xpop[popIdx][i];
			}
			Fnew[popIdx] = F[popIdx];

			// do not re-evaluate these
			evaluateIndicator[popIdx] = false;

			popIdx++;
		}


		// 2. Crossovers
		for( int k = 0; k < Ncross; k++ )
		{

			vector <int> indices( Nparam, 0);

			// Make selections for each "gene" based on fitness
			for(int i = 0; i < Nparam; i++)
			{
				while(indices[i] == 0)
				{
					int randomIndex = round( timeRand()*Npop );
					double selectValue = timeRand();
					if( selectValue <= fitness[randomIndex]/maxFitness )
					{
						indices[i] = randomIndex;
					}
				}
			}

			for(int i = 0; i < Nparam; i++)
			{
				XpopNew[popIdx][i] = Xpop[indices[i]][i];
			}

			popIdx++;
		}



		// 3. Compute random mutations
		// diminish spread size over generations
		double spreadRatio = mutationSize*(maxGenerations - iter )/maxGenerations;
		// double spreadRatio = mutationSize;

		// Compute mutations: starting values for the mutations are chosen
        // based on the fitness score
		for( int k = 0; k < Nrand; k++ )
		{

			int index = 0;
			for(int j = 0; j < Nparam; j++)
			{
				while (index == 0)
				{
					int randomIndex = round( timeRand()*Npop );
					double selectValue = timeRand();
					if( selectValue <= fitness[randomIndex]/maxFitness )
					{
						index = randomIndex;
					}
				}
			}


			for(int j = 0; j < Nparam; j++)
			{
				double mutation = spreadRatio*(Xub[j]-Xlb[j])*timeRand();
				XpopNew[popIdx][j] = Xpop[index][j] + mutation;
			}


			popIdx++;
		}



		// 4. Compute mutations of the elite children
		for( int k = 0; k < NeliteMut; k++ )
		{

			for(int j = 0; j < Nparam; j++)
			{
				int randomEliteIdx = round( timeRand()*Nelite );
				double mutation = eliteMutationSize*(Xub[j]-Xlb[j])*timeRand();
				XpopNew[popIdx][j] = Xpop[randomEliteIdx][j] + mutation;

			}

			popIdx++;
		}


		// 5a. Check for identical children
		checkIndenticalChildAndReplace( XpopNew, Xlb, Xub, evaluateIndicator  );

		// 5b. Check domain boundaries
		checkPopulationBoundsAndReplace( XpopNew, Xlb, Xub, evaluateIndicator  );

		// 6. Evaluate new population
		evaluatePopulationParallel( XpopNew, Fnew, evaluateIndicator );

		// 7. Sort population and copy to the old population
		popSort( XpopNew, Fnew );


		for( int i = 0; i < Npop; i++ )
		{
			F[i] = Fnew[i];
			for(int j = 0; j < Nparam; j++)
			{
				Xpop[i][j] = XpopNew[i][j];
			}
		}


		// Check for static generations
		double Fbest = F[0];
		{
			if( Fbest == FbestPrev )
			{
				Nstatic++;
			}
			else
			{
				Nstatic = 0;
			}
		}
		if(Nstatic > NstaticGenerations )
		{
			break;
		}
		FbestPrev = Fbest;
		iter++;

	}

	// store result
	fOpt = F[0];
	for(int i = 0; i < Nparam; i++)
	{
		X[i] = Xpop[0][i];
	}

	if( verbose == true && procID == ROOT_ID )
	{
		cout << endl << "-----------------------------------------------------------------------------------" << endl ;

		print2DVector(Xpop);
		print1DVector(F);
		cout << "Completed genetic algorithm." << endl;
		cout << "At generation = " << iter << " minimum of f = " << F[0] << "  at params:  ";
					print1DVector(Xpop[0]);

		cout << "-----------------------------------------------------------------------------------" << endl << endl ;
	}



}






void GeneticAlgorithmMPI::evaluatePopulationParallel( vector<vector<double> > &Xpop, vector <double> & F, vector <bool> & evaluateIndicator )
{


	// Get the number of processes and the current processor ID
	int Nprocs, procID;
	MPI_Comm_size(MPI_COMM_WORLD, &Nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &procID);

	int Nparam = Xpop[0].size();

	double * Xpop_i_local = new double[Nparam];
	double * Xpop_i = new double[Nparam];
	double * FarrayLocal = new double[Npop];
	double * Farray = new double[Npop];
	double * indicatorLocal = new double[Npop];
	double * indicator = new double[Npop];

	// Set population on all processors to that on ROOT_ID
	for( int i = 0; i < Npop; i++ )
	{
		if( procID != ROOT_ID )
		{
			for(int j = 0; j < Xpop[i].size(); j++)
			{
				Xpop[i][j] = 0;
			}
			F[i] = 0;
			evaluateIndicator[i] = 0;
		}
	}
	for( int i = 0; i < Npop; i++ )
	{

		for(int j = 0; j < Nparam; j++)
		{
			Xpop_i_local[j] = Xpop[i][j];
		}

		MPI_Allreduce(Xpop_i_local, Xpop_i, Nparam, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		for(int j = 0; j < Nparam; j++)
		{
			Xpop[i][j] = Xpop_i[j];
		}
		FarrayLocal[i] = F[i];
		indicatorLocal[i] = evaluateIndicator[i];
	}

	MPI_Allreduce(FarrayLocal, Farray, Npop, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(indicatorLocal , indicator, Npop, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	for( int i = 0; i < Npop; i++ )
	{
		F[i] = Farray[i];
		evaluateIndicator[i] = (bool) indicator[i];
	}



	// Compute load balance
	vector<int>procEvalID(Npop,ROOT_ID);
	int procIdx = 0;
	for(int i = 0; i < Npop; i++ )
	{
		if( procIdx > Nprocs - 1 )
		{
			procIdx = 0;
		}

		if( evaluateIndicator[i])
		{
			procEvalID[i] = procIdx;
			procIdx = procIdx+1;
		}
	}

	// Evaluate
	for( int i = 0; i < Npop; i++ )
	{
		if(  procEvalID[i] == procID ) // evaluate on correct processor
		{
			if( evaluateIndicator[i] )
			{
				F[i] = objPtr->objEval(Xpop[i]);
			}
		}
		else
		{
			for(int j = 0; j < Xpop[i].size(); j++)
			{
				Xpop[i][j] = 0;
			}
			F[i] = 0;
		}
	}



	// go through and sum components to all processors
	for( int i = 0; i < Npop; i++ )
	{

		// Need to do this to ensure that the population is the same on all processors
		for(int j = 0; j < Nparam; j++)
		{
			Xpop_i_local[j] = Xpop[i][j];
		}

		MPI_Allreduce(Xpop_i_local, Xpop_i, Nparam, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		for(int j = 0; j < Nparam; j++)
		{
			Xpop[i][j] = Xpop_i[j];
		}
		FarrayLocal[i] = F[i];
	}

	MPI_Allreduce(FarrayLocal, Farray, Npop, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	for( int i = 0; i < Npop; i++ )
	{
		F[i] = Farray[i];
	}

	// clean up
	delete [] Xpop_i;
	delete [] Xpop_i_local;
	delete [] FarrayLocal;
	delete [] Farray;
	delete [] indicatorLocal;
	delete [] indicator;
}








