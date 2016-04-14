#include "rele/algorithms/step_rules/StateActionLearningRate.h"
using namespace std;
using namespace arma;

namespace ReLe
{

StateActionLearningRate::StateActionLearningRate(double initialAlpha, double omega , double minAlpha ,unsigned int nstates, unsigned int nactions):
DecayingLearningRate(initialAlpha,omega,minAlpha)
{

	this->stateActionsUpdates=arma::mat(nstates,nactions,arma::fill::ones);
}

double StateActionLearningRate::operator() (FiniteState x,FiniteAction u)
		{
	 	 	int nUpdates=stateActionsUpdates(x,u);
	        double alpha = initialAlpha/std::pow(static_cast<double>(nUpdates), omega);
	        stateActionsUpdates(x,u)++;
	        return std::max(minAlpha, alpha);

		}
void StateActionLearningRate::reset()
{
	stateActionsUpdates.ones();


}










}
