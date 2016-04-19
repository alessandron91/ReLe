#include "rele/policy/q_policy/e_Greedy.h"
#include "rele/utils/RandomGenerator.h"
#include "rele/policy/q_policy/e_GreedyMultipleRegressors.h"
#include "rele/approximators/Regressors.h"


#include <cassert>

using namespace arma;
using namespace std;

namespace ReLe
{
	e_GreedyMultipleRegressors::e_GreedyMultipleRegressors(unsigned int nactions,std::vector<BatchRegressor&> regressors)
	{
		this->nactions=nactions;
		for(int i=0;i<nactions;i++)
		{
			regressorsVector[i]=regressors[i];
		}



		eps=0.0;
	}

	e_GreedyMultipleRegressors::~e_GreedyMultipleRegressors()
	{

	}

/*void e_GreedyMultipleRegressors::setRegressor(unsigned int regressorIndex,BatchRegressor regressor)
	{
		this->regressorsVector[regressorIndex]=regressor;

	}*/


unsigned int e_GreedyMultipleRegressors::operator()(const arma::vec& state)
{
    unsigned int un;

    /*epsilon--greedy policy*/
    if (RandomGenerator::sampleEvent(this->eps))
        un = RandomGenerator::sampleUniformInt(0, nactions - 1);
    else
    {
        unsigned int nstates = state.size();
        vec regInput(nstates);
        regInput= state;
        un=0;
        vec&& qvalue0 = regressorsVector[un](regInput);
        double qmax=qvalue0[0];
        std::vector<int> optimal_actions;
        optimal_actions.push_back(un);
        for (unsigned int i = 1; i < regressorsVector.size(); ++i)
        {
            vec&& qvalue = regressorsVector[i](regInput);
            if (qmax < qvalue[0])
            {
                optimal_actions.clear();
                qmax = qvalue[0];
                un = i;
                optimal_actions.push_back(un);
            }
            else if (qmax == qvalue[0])
            {
                optimal_actions.push_back(i);
            }
        }
        unsigned int index = RandomGenerator::sampleUniformInt(0,
                             optimal_actions.size() - 1);
        un = optimal_actions[index];
        // un = optimal_actions[0];//--------------------- RIMUOVERE
    }

    return un;
}

double e_GreedyMultipleRegressors::operator()(const arma::vec& state, const unsigned int& action)
{
    //TODO [IMPORTANT] implement
    assert(false);
    return 0.0;
}









}
