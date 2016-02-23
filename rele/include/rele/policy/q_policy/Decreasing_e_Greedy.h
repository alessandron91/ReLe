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

#ifndef DECREASING_E_GREEDY_H_
#define DECREASING_E_GREEDY_H_

#include "rele/core/Basics.h"
#include "rele/policy/q_policy/ActionValuePolicy.h"

namespace ReLe
{

class Decreasing_e_Greedy: public ActionValuePolicy<FiniteState>
{
public:
    Decreasing_e_Greedy(int stateDim);
    virtual ~Decreasing_e_Greedy();

    virtual unsigned int operator()(const size_t& state) override;
    virtual double operator()(const size_t& state, const unsigned int& action) override;

    inline virtual std::string getPolicyName() override
    {
        return "Decreasing-e-Greedy";
    }
    virtual hyperparameters_map getPolicyHyperparameters() override;

    inline void setEpsilon(arma::vec eps)
    {
        this->eps = eps;
    }

    inline arma::vec getEpsilon()
    {
        return this->eps;
    }

    virtual Decreasing_e_Greedy* clone() override
    {
        return new Decreasing_e_Greedy(*this);
    }

	bool isTestMode() const {
		return testMode;
	}

	void setTestMode(bool testMode) {
		this->testMode = testMode;
	}


protected:
    arma::vec stateVisitCount;
    bool testMode;
    arma::vec eps;
};



}
#endif /* DECREASING_GREEDY_H_ */
