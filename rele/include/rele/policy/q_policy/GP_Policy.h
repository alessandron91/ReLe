#ifndef GP_POLICY_H_
#define GP_POLICY_H_

#include "rele/policy/Policy.h"
#include "rele/approximators/Regressors.h"
#include "rele/approximators/regressors/others/GaussianProcess.h"


using namespace arma;

using namespace std;
namespace ReLe
{
class GP_Policy: public NonParametricPolicy<DenseAction,DenseState>
{
public:
    GP_Policy(GaussianProcess* gp,unsigned int binsNumber);
    virtual ~GP_Policy();

    virtual arma::vec operator()(const arma::vec& state)override;



    virtual double operator() (typename state_type<DenseState>::const_type_ref state, typename action_type<DenseAction>::const_type_ref action)override;



    inline virtual std::string getPolicyName()override
    {
        return "GP_Policy";
    }

    inline void setRegressor(GaussianProcess* gp)
    {
        this->gp = gp;
    }



    virtual GP_Policy* clone() override
    {
        return this;
    }

    virtual std::string printPolicy()override
    {
    	return "";
    }

protected:
    GaussianProcess* gp;
    unsigned int nBins;
};





}
#endif

