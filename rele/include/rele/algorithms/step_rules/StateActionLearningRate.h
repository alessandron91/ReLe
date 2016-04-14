/*
 * rele,
 *
 *
 * Copyright (C) 2016 Davide Tateo
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

#ifndef INCLUDE_RELE_ALGORITHMS_STEP_RULES_STATEACTIONLEARNINGRATE_H_
#define INCLUDE_RELE_ALGORITHMS_STEP_RULES_STATEACTIONLEARNINGRATE_H_

#include "rele/algorithms/step_rules/LearningRate.h"

#include <sstream>

namespace ReLe
{

class StateActionLearningRate: public DecayingLearningRate
{
public:

	StateActionLearningRate(double initialAlpha, double omega = 1.0, double minAlpha = 0.0,unsigned int nstates=1, unsigned int nactions=1);
    virtual double operator()(FiniteState x, FiniteAction u)override;

    /*!
     * Resets the learning rate to it's original value
     */
    virtual void reset()override;


    /*!
     * Destructor.
     */
    virtual ~StateActionLearningRate()
    {

    }
private:
    arma::mat stateActionsUpdates;
};
}

#endif /* INCLUDE_RELE_ALGORITHMS_STEP_RULES_STATEACTIONSLEARNINGRATE */
