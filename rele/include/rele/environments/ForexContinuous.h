/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo & Matteo Pirotta
 * Versione 1.0
 *
 * This file is part of rele.
 *
 * rele is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * rele is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with rele.  If not, see <http://www.gnu.org/licenses/>.
 */


/*
#ifndef FOREXCONTINUOUS_H_
#define FOREXCONTINUOUS_H_

#include "rele/core/ContinuousMDP.h"

namespace ReLe
{

class ForexContinuous: public ContinuousMDP
{
public:
    ForexContinuous()
    {
    }

    ForexContinuous(EnvironmentSettings* settings);

    ForexContinuous(std::size_t stateSize,const arma::mat& rawDataset,double budget,unsigned int priceCol,unsigned int horizon);

    virtual void step(const DenseAction& action, DenseState& nextState,
                          Reward& reward) override;


    virtual void getInitialState(DenseState& state) override;
    double getProfit() const;
    void setCurrentStateIdx(unsigned int currentStateIdx);


protected:
    DenseState getNextState(unsigned int action);


protected:
    DenseState currentState;
    arma::mat dataset;
    unsigned int currentStateIdx;
    unsigned int nIndicators;
    double currentPrice;
    double prevPrice;
    double profit;
    double prevAction;
    double budget;


};

}

#endif /* FOREX_CONTINUOUS_H_ */
