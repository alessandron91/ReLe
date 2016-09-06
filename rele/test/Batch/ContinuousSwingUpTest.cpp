/*
 * ContinuousSwingUpTest.cpp
 *
 *  Created on: 05 set 2016
 *      Author: alessandro
 */
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

#include "rele/environments/SwingPendulum.h"
#include "rele/core/Core.h"
#include "rele/algorithms/td/LinearSARSA.h"
#include "rele/approximators/basis/PolynomialFunction.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/approximators/basis/GaussianRbf.h"
#include "rele/approximators/basis/ConditionBasedFunction.h"
#include "rele/policy/nonparametric/RandomPolicy.h"
#include "rele/utils/FileManager.h"
#include "rele/core/PolicyEvalAgent.h"
#include "rele/environments/ContinuousSwingPendulum.h"
#include <fenv.h>

using namespace std;
using namespace ReLe;

int main(int argc, char *argv[])
{


    unsigned int episodes = 200;
    ContinuousSwingPendulum mdp;

    vector<Range> actionsVector;
    actionsVector.push_back(mdp.getConfig().actionRange);
    RandomPolicy<DenseState> policy(actionsVector);
    PolicyEvalAgent<DenseAction,DenseState> agent(policy);


    FileManager fm("ip", "randomPolicy"+std::to_string(episodes)+"Episodes");
    fm.createDir();
    fm.cleanDir();
    auto&& core = buildCore(mdp, agent);

    int nExperiments=100;
    for(int n=0;n<nExperiments;n++)
    {

    	std::string exp=std::to_string(n);


    core.getSettings().loggerStrategy = new WriteStrategy<DenseAction, DenseState>(fm.addPath("ip_"+exp+".log"));

    for (int i = 0; i < episodes; i++)
    {
        core.getSettings().episodeLength = 50;
        cout << "Starting episode: " << i << endl;
        core.runTestEpisode();
    }
    }


}




