#ifndef E_GREEDYMULTIPLEREGRESSORS_H_
#define E_GREEDYMULTIPLEREGRESSORS_H_

#include "rele/policy/q_policy/ActionValuePolicy.h"
#include "rele/approximators/Regressors.h"

using namespace arma;

using namespace std;
namespace ReLe
{
	class e_GreedyMultipleRegressors: public ActionValuePolicy<DenseState>
	{

			e_GreedyMultipleRegressors(unsigned int nGaussianRegressors,std::vector<BatchRegressor&>);
		    virtual ~e_GreedyMultipleRegressors();

		    virtual unsigned int operator()(const arma::vec& state) override;
		    virtual double operator()(const arma::vec& state, const unsigned int& action) override;

		    inline virtual std::string getPolicyName() override
		    {
		        return "MultipleRegressors e-Greedy";
		    }
		    //virtual hyperparameters_map getPolicyHyperparameters() override;

		    inline void setEpsilon(double eps)
		    {
		        this->eps = eps;
		    }

		    //inline void setRegressor(unsigned int regressorIndex,BatchRegressor& regr);

		    inline double getEpsilon()
		    {
		        return this->eps;
		    }

		/*    virtual e_GreedyMultipleRegressors* clone() override
		    {
		        return new e_GreedyMultipleRegressors(*this);
		    }*/

		protected:
		    double eps;
		    std::vector<BatchRegressor> regressorsVector;






	};








}
#endif
