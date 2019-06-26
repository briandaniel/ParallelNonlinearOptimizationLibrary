/*
 * GeneticAlgorithm.cpp
 *
 *  Created on: Oct 24, 2018
 *      Author: brian
 */


#include "GeneticAlgorithm.hpp"


void GeneticAlgorithm::findMinBnd( std::vector <double> & X, std::vector <double> & Xlb, std::vector <double> & Xub, double & f0 , double & fOpt )
{
	RenderContainerVTK rcVTK;
	double zscale = 0.1;
	if( graph )
	{
		rcVTK.renderWindow-> SetSize (1400, 1000) ;
	}


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

    if( verbose == true )
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
	evaluatePopulation( Xpop, F, evaluateIndicator );
	f0 = F[0];

	// 0a. sort population
	popSort( Xpop, F );


	double FbestPrev = F[0];
	int Nstatic = 0;
	int iter = 0;
	while( iter < maxGenerations)
	{

		if( verbose == true )
		{
			cout << "At generation = " << iter << " minimum of f = " << F[0] << "  at params:  ";
			print1DVector(Xpop[0]);
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
		evaluatePopulation( XpopNew, Fnew, evaluateIndicator );

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

		if( graph )
		{
			rcVTK.renderer->RemoveAllViewProps();

			plotParabolicSurf( zscale, rcVTK );

			for( int i = 0; i < Npop; i++ )
			{
				double colorIdx = i/(double)Npop;
				rcVTK.plotPoint(Xpop[i][0], Xpop[i][1], zscale*F[i], colorIdx, colorIdx, colorIdx, 8);
			}

			rcVTK.renderWindow->Render();
		}
	}

	// store result
	fOpt = F[0];
	for(int i = 0; i < Nparam; i++)
	{
		X[i] = Xpop[0][i];
	}

	if( verbose == true )
	{
		cout << endl << "-----------------------------------------------------------------------------------" << endl ;

		print2DVector(Xpop);
		print1DVector(F);
		cout << "Completed genetic algorithm." << endl;
		cout << "At generation = " << iter << " minimum of f = " << F[0] << "  at params:  ";
					print1DVector(Xpop[0]);

		cout << "-----------------------------------------------------------------------------------" << endl << endl ;
	}



	if( graph )
	{
		rcVTK.renderWindow->Render();
		// Interactor that pauses the program for interactions with the VTK renderer
		rcVTK.renderWindowInteractor->Start();

	}


}






void GeneticAlgorithm::evaluatePopulation( vector<vector<double> > &Xpop, vector <double> & F, vector <bool> & evaluateIndicator )
{
	for( int i = 0; i < Npop; i++ )
	{
		if( evaluateIndicator[i])
		{
			F[i] = objPtr->objEval(Xpop[i]);
		}
	}

}

void checkIndenticalChildAndReplace( vector<vector<double> > &Xpop, std::vector <double> & Xlb, std::vector <double> & Xub, vector <bool> & evaluateIndicator  )
{
	int Npop = Xpop.size();
	// Cycle through list
	for( int i = 0; i < Npop; i++ )
	{
		// Check the rest of the list for identical members
		for(int k = i+1; k < Npop; k++)
		{
			int Nsame = 0;
			for(int j = 0; j < Xpop[i].size(); j++)
			{
				if( Xpop[i][j] == Xpop[k][j])
				{
					Nsame ++;
				}

			}
			// if identical to the first member of the search, simply replace with a random entry
			if( Nsame == Xpop[i].size() )
			{
				for(int j = 0; j < Xpop[i].size(); j++)
				{
					Xpop[i][j] = Xlb[j] + ( Xub[j] - Xlb[j] )*timeRand() ;
				}
				evaluateIndicator[i] = true;
			}
		}
	}


}


void checkPopulationBoundsAndReplace(	vector<vector<double> > &Xpop, std::vector <double> & Xlb, std::vector <double> & Xub, vector <bool> & evaluateIndicator  )
{

	int Npop = Xpop.size();
	for( int i = 0; i < Npop; i++ )
	{
		for(int j = 0; j < Xlb.size(); j++)
		{
			// if outside bounds create a new random value for that variable inside the bounds
			if( Xpop[i][j] > Xub[j] || Xpop[i][j] < Xlb[j])
			{
				Xpop[i][j] = Xlb[j] + ( Xub[j] - Xlb[j] )*timeRand() ;
				evaluateIndicator[i] = true;
			}
		}
	}


}



// sorts the population based on the function values in F
void popSort( vector<vector<double> > &Xpop, vector <double> & F )
{

	int Npop = Xpop.size();
	int Nparam = Xpop[0].size() ;

    // Temporary variables
	vector<vector<double> >XpopTemp( Npop, vector<double>(Nparam) );
	vector <double> FTemp(Npop,0);



    // find largest value to be used later
    double FMax;
	int indexMax;
	int idxMin = 1;
    vectorMax ( F, F.size(), FMax, indexMax);

    // Sort
	for (int k = 0; k < Npop; k++)
	{
		double Fmin;
		vectorMin( F, F.size(), Fmin, idxMin);

		for (int j = 0; j <  Nparam; j++)
			XpopTemp[k][j] = Xpop[idxMin][j];

		FTemp[k] = F[idxMin];

		F[idxMin] = 2*FMax ;
	}


	for (int k = 0; k < Npop; k++)
	{
		F[k] = FTemp[k];
		for (int j = 0; j < Nparam; j++)
			Xpop[k][j] = XpopTemp[k][j];

	}


}
























