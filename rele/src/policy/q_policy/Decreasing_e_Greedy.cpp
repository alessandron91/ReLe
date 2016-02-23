#include "rele/utils/RandomGenerator.h"
#include "rele/policy/q_policy/Decreasing_e_Greedy.h"

#include <sstream>
#include <cassert>

using namespace arma;
using namespace std;

namespace ReLe
{

Decreasing_e_Greedy::Decreasing_e_Greedy(int stateDim)
{
    Q = nullptr;
    eps = arma::vec(stateDim,arma::fill::ones);
    stateVisitCount = arma::vec(stateDim,arma::fill::zeros);

    testMode=false;

    nactions = 0;
}

Decreasing_e_Greedy::~Decreasing_e_Greedy()
{

}



unsigned int Decreasing_e_Greedy::operator()(const size_t& state)
{
    unsigned int un;
    int count=stateVisitCount(state)++;
    eps(state)=(double)(1/sqrt(count));

    const rowvec& Qx = Q->row(state);

    /*epsilon--greedy policy*/
    if (RandomGenerator::sampleEvent(this->eps(state)) && testMode==false)
        un = RandomGenerator::sampleUniformInt(0, Q->n_cols - 1);
    else
    {
        double qmax = Qx.max();
        uvec maxIndex = find(Qx == qmax);

        unsigned int index = RandomGenerator::sampleUniformInt(0,
                             maxIndex.n_elem - 1);
        un = maxIndex[index];
    }

    return un;
}
hyperparameters_map Decreasing_e_Greedy::getPolicyHyperparameters()
{
	hyperparameters_map map;

	for(unsigned int i = 0; i < eps.n_elem; i++)
	{
		map["eps"+to_string(i)]=eps(i);
	}


    return map;
}
double Decreasing_e_Greedy::operator()(const size_t& state, const unsigned int& action) /// A CHE SERVE???)
{
    const rowvec& Qx = Q->row(state);
    double qmax = Qx.max();
    uvec maxIndex = find(Qx == qmax);

    bool found = false;
    for (unsigned int i = 0; i < maxIndex.n_elem && !found; ++i)
    {
        if (maxIndex[i] == action)
            found = true;
    }
    if (found)
    {
        return 1.0 - eps(state) + eps(state) / Q->n_cols;
    }
    return eps(state) / Q->n_cols;
}







}
