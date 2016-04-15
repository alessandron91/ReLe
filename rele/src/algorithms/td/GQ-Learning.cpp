#include "rele/algorithms/td/GQ-Learning.h"
#include "rele/utils/RandomGenerator.h"
using namespace std;
using namespace arma;

namespace ReLe
{

GQ_Learning::GQ_Learning(ActionValuePolicy<FiniteState>& policy, LearningRate& alpha, unsigned int nSamples) :
    Q_Learning(policy, alpha),
	nSamples(nSamples)
{
}

void GQ_Learning::initEpisode(const FiniteState& state, FiniteAction& action)
{
    sampleAction(state, action);
}

void GQ_Learning::sampleAction(const FiniteState& state, FiniteAction& action)
{
    x = state.getStateN();
    u = policy(x);

    action.setActionN(u);
}

void GQ_Learning::step(const Reward& reward, const FiniteState& nextState,
                       FiniteAction& action)
{
    size_t xn = nextState.getStateN();
    double r = reward[0];

    computeWeights(xn);

    double W = arma::dot(Q.row(xn), weights.row(xn));

    double target = r + task.gamma * W;

    updateMeanAndSampleStdQ(target);

    x = xn;
    u = policy(xn);

    action.setActionN(u);
}

void GQ_Learning::endEpisode(const Reward& reward)
{
    double r = reward[0];
    double target = r;

    updateMeanAndSampleStdQ(target);
}

GQ_Learning::~GQ_Learning()
{
}

void GQ_Learning::init()
{
    FiniteTD::init();

    idxs = arma::mat(task.actionsNumber, task.actionsNumber - 1, arma::fill::zeros);
    arma::vec actions = arma::linspace(0, idxs.n_cols, idxs.n_rows);
    for(unsigned int i = 0; i < idxs.n_rows; i++)
        idxs.row(i) = actions(arma::find(actions != i)).t();

    sampleStdQ = Q + stdInfValue;
    Q2 = Q;
    weights=Q;
    weightsVar = Q;

    nUpdates = Q;
}

inline void GQ_Learning::updateMeanAndSampleStdQ(double target)
{
    double alpha = this->alpha(x, u);
    Q(x, u) = (1 - alpha) * Q(x, u) + alpha * target;
    Q2(x, u) = (1 - alpha) * Q2(x, u) + alpha * target * target;

    nUpdates(x, u)++;

    if(nUpdates(x, u) > 1)
    {
        weightsVar(x, u) = (1 - alpha) * (1 - alpha) * weightsVar(x, u) + alpha * alpha;
        double n = 1 / weightsVar(x, u);

        double var = (Q2(x, u) - Q(x, u) * Q(x, u)) / n;

        if(var >= stdZeroValue * stdZeroValue)
            sampleStdQ(x, u) = sqrt(var);
        else
            sampleStdQ(x, u) = stdZeroValue;
    }
}

void GQ_Learning::computeWeights(size_t xn)
{
	arma::mat qSamples=arma::mat(nSamples,Q.n_cols,arma::fill::zeros);
	arma::vec maxCounter=arma::vec(Q.n_cols,arma::fill::zeros);

	for(int u=0;u<Q.n_cols;u++)
	{
		double mean=Q(xn,u);
		double sigma=sampleStdQ(xn,u);
		for(int s=0;s<nSamples;s++)
		{
			/*
			if(sigma>1000)
			{
				sigma=100;
			}*/

			qSamples(s,u)=RandomGenerator::sampleNormal(mean,sigma);
		}

	}

	for(int s=0;s<nSamples;s++)
	{
		double qmax=qSamples.row(s).max();
		arma::vec samples=qSamples.row(s).t();
		arma::uvec maxIndex = find(samples == qmax);
		unsigned int index = RandomGenerator::sampleUniformInt(0,
		                         maxIndex.n_elem - 1);

		maxCounter(maxIndex(index))++;




	}

	for(unsigned u=0;u<Q.n_cols;u++)
	{
		weights(xn,u)=maxCounter(u)/nSamples;
	}









}

}
