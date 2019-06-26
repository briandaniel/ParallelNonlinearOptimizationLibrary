/*
 * ExampleObjectives.hpp
 *
 *  Created on: Oct 22, 2018
 *      Author: brian
 */

#ifndef EXAMPLEOBJECTIVES_HPP_
#define EXAMPLEOBJECTIVES_HPP_

// standard includes
#include <math.h>

// local includes
#include "PNOL_Objective.hpp"



class GoldsteinFunction : public Objective {

  private:
	int evals;

  public:

	// main function (instantiated from virtual objective)
	double objEval(  vector <double> & X  )
	{
		evals++;
		double value;

		double x = X[0];
		double y = X[1];

		value = (1 + pow(x+y+1,2)*( 19-14*x + 3*pow(x,2) - 14*y + 6*x*y + 3*pow(y,2) ) )*
				( 30 + pow(2*x-3*y,2)*(18 - 32*x + 12*pow(x,2) + 48*y - 36*x*y + 27 *pow(y,2) ));

		return value;
	}

	GoldsteinFunction(){
		evals = 0;
	}

	double getEvals(){	return evals; }

};


class BoothFunction : public Objective {

  private:
	int evals;

  public:

	// main function (instantiated from virtual objective)
	double objEval(  vector <double> & X  )
	{
		evals++;
		double value;

		double x = X[0];
		double y = X[1];

		value = pow( x+2*y-7, 2) + pow(2*x + y - 5,2);

		return value;
	}

	BoothFunction(){
		evals = 0;
	}

	double getEvals(){	return evals; }

};

class RosenbrockObject : public Objective {

  private:
	int evals;

  public:

	// main function (instantiated from virtual objective)
	double objEval(  vector <double> & X  )
	{
		evals++;
		double value;

		value = 0;
		for(int k = 0; k <X.size()-1; k++)
		{
			value = value + ( 100.0*pow(X[k+1] - pow(X[k],2),2) + pow(1-X[k],2) ) ;
		}



		//cout << "X = "; print1DVector(X);
		//cout << ",    F(X) = " << value << endl;
		return value;
	}

	RosenbrockObject(){
		evals = 0;
	}

	double getEvals(){	return evals; }

};

class ExpCurveObjective : public MultiObjective {

  private:

	vector <double> xData;
	vector <double> yData;

  public:

	// main function (instantiated from virtual objective)
	void objEval(  vector <double> & X, vector <double> & F  )
	{

		for(int k = 0; k <xData.size(); k++)
		{
			double func = X[0]*exp( X[1]*xData[k] ) + X[2];
			F[k] = yData[k] - func;
		}

	}

	// get the data size
	int getDataSize(){ return xData.size(); }


    // default constructor/destructor
	ExpCurveObjective(){

		linspace(0,5,100,xData);

		yData.resize(xData.size());
		for(int k = 0; k < xData.size(); k++)
			yData[k] = 10.2*exp( 0.4*xData[k] ) + 0.1;// 0.2*(timeRand()-0.5);

	}
	~ExpCurveObjective(){
		xData.clear();
		yData.clear();
	}


};





class CubicObjective : public MultiObjective {

  private:

	vector <double> xData;
	vector <double> yData;

  public:

	// main function (instantiated from virtual objective)
	void objEval(  vector <double> & X, vector <double> & F  )
	{

		for(int k = 0; k <xData.size(); k++)
		{
			double func = X[0]*pow(xData[k],3) + X[1]*pow(xData[k],2) + X[2]*xData[k] + X[3];
			F[k] = yData[k] - func;
		}

	}

	// get the data size
	int getDataSize(){ return xData.size(); }


    // default constructor/destructor
	CubicObjective(){

		linspace(-5,5,100,xData);

		yData.resize(xData.size());
		for(int k = 0; k < xData.size(); k++)
			yData[k] = 0.3*pow(xData[k],3) + 1.1*pow(xData[k],2) - 4.3 *xData[k] + 7.3;// 0.2*(timeRand()-0.5);

	}
	~CubicObjective(){
		xData.clear();
		yData.clear();
	}


};




class PowerObject : public Objective {

  private:
	int power;

  public:

	// main function (instantiated from virtual objective)
	double objEval(  vector <double> & X  )
	{
		double value;

		value = 0;
		for(int k = 0; k <X.size(); k++)
		{
			value = value + pow(X[k],power);
		}
		return value;
	}

	void setPower( int powIn ){ power = powIn; }
	int getPower(){ return power; }

    // default constructor/destructor
	PowerObject(){ power = 2; }
	~PowerObject(){}


};



class PowerObjectSlow : public Objective {

  private:
	int power;

  public:

	// main function (instantiated from virtual objective)
	double objEval(  vector <double> & X  )
	{
		double value;

		// Slow computation for no reason
		int Ndummy = 1e5;
		double temp;
		for(int k = 0; k < Ndummy; k++)
			temp = pow(sin(k*2.3),1.1);

		value = 0;
		for(int k = 0; k <X.size(); k++)
		{
			value = value + pow(X[k],power);
		}
		return value;
	}

	void setPower( int powIn ){ power = powIn; }
	int getPower(){ return power; }

    // default constructor/destructor
	PowerObjectSlow(){ power = 2; }
	~PowerObjectSlow(){}

};





class ExpCurveObjectiveSingle : public Objective {

  private:

	vector <double> xData;
	vector <double> yData;

  public:

	// main function (instantiated from virtual objective)
	double objEval(  vector <double> & X  )
	{
		double Fnorm = 0;

		for(int k = 0; k <xData.size(); k++)
		{
			double func = X[0]*exp( X[1]*xData[k] ) + X[2];
			Fnorm = Fnorm + pow( yData[k] - func, 2);
		}

		return Fnorm;
	}

	// get the data size
	int getDataSize(){ return xData.size(); }


    // default constructor/destructor
	ExpCurveObjectiveSingle(){

		linspace(0,5,100,xData);

		yData.resize(xData.size());
		for(int k = 0; k < xData.size(); k++)
			yData[k] = 10.2*exp( 0.4*xData[k] ) + 0.1;// 0.2*(timeRand()-0.5);

	}
	~ExpCurveObjectiveSingle(){
		xData.clear();
		yData.clear();
	}


};

#endif /* EXAMPLEOBJECTIVES_HPP_ */
