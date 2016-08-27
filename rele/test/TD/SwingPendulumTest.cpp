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
#include "rele/policy/q_policy/e_Greedy.h"
#include "rele/utils/FileManager.h"
#include <fenv.h>

using namespace std;
using namespace ReLe;

int main(int argc, char *argv[])
{


    unsigned int episodes = 200;
    DiscreteActionSwingUp mdp;

    //BasisFunctions bVector = PolynomialFunction::generate(7, mdp.getSettings().statesNumber + 1);
    BasisFunctions b1 = GaussianRbf::generate(4, {-1, 1, -1, 1});
    BasisFunctions b2 = GaussianRbf::generate(5, {-1, 1, -1, 1});
    BasisFunctions b3 = GaussianRbf::generate(6, {-1, 1, -1, 1});
    std::vector<BasisFunction*> bVector;
    bVector.reserve(b1.size() + b2.size() + b3.size());
    bVector.insert(bVector.end(), b1.begin(), b1.end());
    bVector.insert(bVector.end(), b2.begin(), b2.end());
    bVector.insert(bVector.end(), b3.begin(), b3.end());
    BasisFunctions basis = AndConditionBasisFunction::generate(bVector, 2, mdp.getSettings().actionsNumber);

    DenseFeatures phi(basis);

    e_GreedyApproximate policy;
    policy.setEpsilon(1.0);
    ConstantLearningRateDense alpha(0.1);
    LinearGradientSARSA agent(phi, policy, alpha);

    agent.setLambda(0.8);

    FileManager fm("ip", "linearSarsa");
    fm.createDir();
    fm.cleanDir();
    auto&& core = buildCore(mdp, agent);

    int nExperiments=100;
    for(int n=0;n<nExperiments;n++)
    {

    	std::string exp=std::to_string(n);


    core.getSettings().loggerStrategy = new WriteStrategy<FiniteAction, DenseState>(fm.addPath("ip_"+exp+".log"));

    for (int i = 0; i < episodes; i++)
    {
        core.getSettings().episodeLength = 50;
        cout << "Starting episode: " << i << endl;
        core.runEpisode();
    }
    }


}
