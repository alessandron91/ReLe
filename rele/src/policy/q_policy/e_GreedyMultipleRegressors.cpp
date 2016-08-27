#include "rele/policy/q_policy/e_Greedy.h"
#include "rele/utils/RandomGenerator.h"
#include "rele/policy/q_policy/e_GreedyMultipleRegressors.h"
#include "rele/approximators/Regressors.h"
#include "rele/approximators/regressors/others/GaussianProcess.h"



#include <cassert>

using namespace arma;
using namespace std;

namespace ReLe
{

e_GreedyMultipleRegressors::e_GreedyMultipleRegressors(std::vector<std::vector<GaussianProcess*>>& regressors) :
    regressors(regressors),
    eps(eps)
{
}

e_GreedyMultipleRegressors::~e_GreedyMultipleRegressors()
{
}

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
        regInput = state;
        un = 0;

        double output = 0;
        for(unsigned int r = 0; r < regressors.size(); r++)
        {
            auto& self = *regressors[r][un];
            output += self(regInput)[0];
        }

        double qmax = output / regressors.size();
        std::vector<int> optimal_actions;
        optimal_actions.push_back(un);
        for(unsigned int i = 1; i < nactions; ++i)
        {
            double output = 0;
            for(unsigned int r = 0; r < regressors.size(); r++)
            {
                auto& self = *regressors[r][i];
                output += self(regInput)[0];
            }

            double qvalue = output / regressors.size();
            if (qmax < qvalue)
            {
                optimal_actions.clear();
                qmax = qvalue;
                un = i;
                optimal_actions.push_back(un);
            }
            else if (qmax == qvalue)
            {
                optimal_actions.push_back(i);
            }
        }
        unsigned int index = RandomGenerator::sampleUniformInt(0,
                             optimal_actions.size() - 1);
        un = optimal_actions[index];
    }

    return un;
}

double e_GreedyMultipleRegressors::operator()(const arma::vec& state, const unsigned int& action)
{
    //TODO [IMPORTANT] implement
    assert(false);
    return 0.0;
}

hyperparameters_map e_GreedyMultipleRegressors::getPolicyHyperparameters()
{
    hyperparameters_map hyperParameters;
    hyperParameters["eps"] = eps;
    return hyperParameters;
}
}
