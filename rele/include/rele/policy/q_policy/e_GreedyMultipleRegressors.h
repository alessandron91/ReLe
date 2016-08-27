#ifndef E_GREEDYMULTIPLEREGRESSORS_H_
#define E_GREEDYMULTIPLEREGRESSORS_H_

#include "rele/policy/q_policy/ActionValuePolicy.h"
#include "rele/approximators/Regressors.h"
#include "rele/approximators/regressors/others/GaussianProcess.h"


using namespace arma;

using namespace std;
namespace ReLe
{
class e_GreedyMultipleRegressors: public ActionValuePolicy<DenseState>
{
public:
    e_GreedyMultipleRegressors(std::vector<std::vector<GaussianProcess*>>& regressors);
    virtual ~e_GreedyMultipleRegressors();

    virtual unsigned int operator()(const arma::vec& state) override;
    virtual double operator()(const arma::vec& state, const unsigned int& action) override;

    virtual hyperparameters_map getPolicyHyperparameters() override;

    inline void setEpsilon(double eps)
    {
        this->eps = eps;
    }

    inline virtual std::string getPolicyName() override
    {
        return "MultipleRegressors e-Greedy";
    }

    inline void setRegressor(std::vector<std::vector<GaussianProcess*>>& regressors)
    {
        this->regressors = regressors;
    }

    inline double getEpsilon()
    {
        return this->eps;
    }

    virtual e_GreedyMultipleRegressors* clone() override
    {
        return new e_GreedyMultipleRegressors(*this);
    }

protected:
    double eps;
    std::vector<std::vector<GaussianProcess*>>& regressors;
};





}
#endif
