#ifndef FOREX_H_
#define FOREX_H_

#include "rele/core/Basics.h"
#include "rele/core/Environment.h"
#include <armadillo>
#include <set>
#include <ostream>
namespace ReLe
{

class Forex: public Environment<FiniteAction, FiniteState>
{
public:
	Forex(const arma::mat& dataset);
	virtual void step(const FiniteAction& action, FiniteState& nextState,
					  Reward& reward) override;
	virtual void getInitialState(FiniteState& state) override;

	//virtual void initStateIndexer();
	//virtual void initIndicatorsIndexer();
	virtual int getNextState(int action);
	virtual void setSettings(int stateDim,int actionDim);

	virtual void InitStatesIndexer();
	virtual int getRowMatrixIndex(arma::vec v,arma::mat b);
	//virtual int getStateIndex(arma::rowvec v);

	virtual void writeVecElemAwithTperiodInVecStateIndexer(arma::vec a,int col,int t);
	virtual ~Forex();

	const arma::mat& getStateIndexer() const {
		return stateIndexer;
	}

	double getProfit() const {
		return profit;
	}

	unsigned int getT() const {
		return t;
	}

	void setT(int t) {
		this->t = t;
	}

	void setProfit(double profit) {
		this->profit = profit;
	}

	const arma::rowvec& getRewards() const {
		return rewards;
	}

	bool isTestMode() const {
		return testMode;
	}

	void setTestMode(bool testMode) {
		this->testMode = testMode;
	}

	const arma::rowvec& getActions() const {
		return actions;
	}

	const arma::rowvec& getStates() const {
		return states;
	}

private:
	arma::mat dataset;
	int t;
	arma::vec indicatorSignal;
	arma::vec currentState;
	double currentPrice;
	double prevPrice;
	int prevAction;
	double profit;
	arma::mat indicatorsIndexer;
	arma::mat stateIndexer;
	arma::rowvec rewards;

	arma::rowvec states;
	arma::rowvec actions;
	bool testMode;


};

}

#endif
