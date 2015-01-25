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

#include "FiniteMDP.h"
#include "td/SARSA.h"
#include "td/Q-Learning.h"
#include "Core.h"

#include "grid_world/GridWorldGenerator.h"

#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
    /*const size_t statesNumber = 5;
    const size_t actionsNumber = 2;

    arma::cube R(actionsNumber, statesNumber, 2);
    arma::mat R0(statesNumber, 2);
    R0 << //
       0 << 0 << arma::endr //
       << 0 << 0 << arma::endr //
       << 1 << 0 << arma::endr //
       << 0 << 0 << arma::endr //
       << 0 << 0 << arma::endr;

    R.tube(arma::span(0), arma::span::all) = R0;
    R.tube(arma::span(1), arma::span::all) = R0;

    arma::cube P(actionsNumber, statesNumber, statesNumber);

    arma::mat P0(statesNumber, statesNumber);
    arma::mat P1(statesNumber, statesNumber);

    P0 << //
       0.2 << 0.8 << 0 << 0 << 0 << arma::endr //
       << 0 << 0.2 << 0.8 << 0 << 0 << arma::endr //
       << 0 << 0 << 0.2 << 0.8 << 0 << arma::endr //
       << 0 << 0 << 0 << 0.2 << 0.8 << arma::endr //
       << 0 << 0 << 0 << 0 << 1;

    P1 << //
       1 << 0 << 0 << 0 << 0 << arma::endr //
       << 0.8 << 0.2 << 0 << 0 << 0 << arma::endr //
       << 0 << 0.8 << 0.2 << 0 << 0 << arma::endr //
       << 0 << 0 << 0.8 << 0.2 << 0 << arma::endr //
       << 0 << 0 << 0 << 0.8 << 0.2;

    P.tube(arma::span(0), arma::span::all) = P0;
    P.tube(arma::span(1), arma::span::all) = P1;

    ReLe::FiniteMDP mdp(P, R, false, 0.9);
    ReLe::SARSA agent;
// 	ReLe::Q_Learning agent;
    ReLe::Core<ReLe::FiniteAction, ReLe::FiniteState> core(mdp, agent);

    core.getSettings().episodeLenght = 10000;
    core.getSettings().logTransitions = false;
    cout << "starting episode" << endl;
    core.runEpisode();*/

	if(argc > 1)
	{
		ReLe::GridWorldGenerator generator;
		generator.load(argv[1]);
	}
}