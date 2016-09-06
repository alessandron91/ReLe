/*
 * GP_Policy.cpp
 *
 *  Created on: 04 set 2016
 *      Author: alessandro
 */


#include "rele/policy/q_policy/ActionValuePolicy.h"
#include "rele/approximators/Regressors.h"
#include "rele/approximators/regressors/others/GaussianProcess.h"
#include "rele/policy/q_policy/GP_Policy.h"



using namespace arma;
using namespace std;

namespace ReLe
{

GP_Policy::GP_Policy(GaussianProcess* gp,unsigned int nBins) :
    gp(gp)
{
		this->nBins=nBins;
}

GP_Policy::~GP_Policy()
{
}

arma::vec GP_Policy::operator()(const arma::vec& state)
{


		arma::vec actions=arma::linspace(-1.0,1.0,nBins);
		arma::vec testInput(state.size()+1);
		testInput.rows(0,state.size()-1)=state;
		arma::vec action={actions(0)};
		testInput(state.size())=action(0);
		auto& self=*gp;
		double max=self(testInput)(0);
    	for(int i=1;i<actions.size();i++)
    	{
    		testInput(state.size())=actions(i);
    		double output=self(testInput)(0);
    		if(output>max)
    		{
    			max=output;
    			action(0)=actions(i);
    		}

    	}


    return action;
}

double GP_Policy::operator() (typename state_type<DenseState>::const_type_ref state, typename action_type<DenseAction>::const_type_ref action)
{
	return 0;
}






}
