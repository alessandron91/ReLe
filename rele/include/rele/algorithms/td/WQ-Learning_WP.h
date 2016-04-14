
#ifndef INCLUDE_RELE_ALGORITHMS_TD_WQ_LEARNING_WP_H_
#define INCLUDE_RELE_ALGORITHMS_TD_WQ_LEARNING_WP_H_

#include "rele/algorithms/td/WQ-Learning.h"


namespace ReLe
{

class WQ_Learning_WP:public WQ_Learning
{
public:
	WQ_Learning_WP(LearningRate& alpha);

	virtual void step(const Reward& reward, const FiniteState& nextState,
	                      FiniteAction& action) override;

private:
	class WPolicy: public ActionValuePolicy<FiniteState>
	{
	public:
	    WPolicy();
	    virtual ~WPolicy();

	    virtual unsigned int operator()(const size_t& state) override;
	    virtual double operator()(const size_t& state, const unsigned int& action) override;
	    virtual void setProbabilities(unsigned int state,arma::vec prob);

	    inline virtual std::string getPolicyName() override
	    {
	        return "WPolicy";
	    }

	    virtual WPolicy* clone() override
	    {
	        return new WPolicy(*this);
	    }
	private:

	    void initialization();

	protected:

	    arma::mat probabilities;
	    bool initialized;
	    bool testMode;

	};


	private:
	WPolicy policy;





};



}
#endif
