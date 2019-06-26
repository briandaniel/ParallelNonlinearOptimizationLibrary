/*
 * PNOL_Algorithm.hpp
 *
 *  Created on: Oct 22, 2018
 *      Author: brian
 */

#ifndef PNOL_ALGORITHM_HPP_
#define PNOL_ALGORITHM_HPP_

// Parallel commands
#ifndef ROOT_ID
#define ROOT_ID 0 // id of root process
#endif

// Standard includes
#include <vector>

// local includes
#include "PNOL_Objective.hpp"

class AlgorithmBnd{

  protected:
	// objective class used for objective evaluations
	Objective * objPtr;

  public:
	// Pure virtual minimizer
	virtual void findMinBnd( std::vector <double> & X, std::vector <double> & Xlb, std::vector <double> & Xub, double & f0 , double & fOpt ) = 0;

	// set the objective
	void setObjPtr( Objective & obj ){ objPtr = &obj; }

};

class Algorithm{

  protected:
	// objective class used for objective evaluations
	Objective * objPtr;

  public:
	// Pure virtual minimizer
	virtual void findMin( std::vector <double> & X, double & f0 , double & fOpt ) = 0;

	// set the objective
	void setObjPtr( Objective & obj ){ objPtr = &obj; }

};

class MultiAlgorithm{

  protected:
	// objective class used for objective evaluations
	MultiObjective * mObjPtr;

  public:
	// Pure virtual minimizer
	virtual void findMin( std::vector <double> & X, std::vector <double> & F0, std::vector <double> & F ) = 0;

	// set the objective
	void setObjPtr( MultiObjective & mObj ){ mObjPtr = &mObj; }

};



#endif /* PNOL_ALGORITHM_HPP_ */
