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
#include "rele/environments/ForexContinuous.h"

using namespace std;

namespace ReLe
{

	ForexContinuous::ForexContinuous(std::size_t stateSize,const arma::mat& rawDataset,unsigned int nIndicators,double budget,unsigned int priceCol,unsigned int horizon)
    	: ContinuousMDP(stateSize,1, false,false,0.85,horizon)

	{
		profit=0;
		prevAction=0;
		dataset=rawDataset;
		this->nIndicators=nIndicators;
		currentStateIdx=0;
		currentPrice=dataset(priceCol);
		this->budget=budget;

	}


	ForexContinuous::ForexContinuous(EnvironmentSettings *settings)
    	: ContinuousMDP(settings)
	{

	}

	ForexContinuous::step(const DenseAction& action, DenseState& nextState,
            Reward& reward)
	{




	}

    DenseState ForexContinuous::getNextState(unsigned int action)
    {
    	arma::vec indicatorsState=dataset(currentStateIdx,arma::span(0,nIndicators));



    }





}*/
