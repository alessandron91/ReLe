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

#include "rele/algorithms/td/WQ-Learning_WP.h"
#include "rele/utils/RandomGenerator.h"

#include <sstream>
#include <cassert>

using namespace arma;
using namespace std;

namespace ReLe {
WQ_Learning_WP::WQ_Learning_WP() :
		WQ_Learning(policy) {

}

void WQ_Learning_WP::step(const Reward& reward, const FiniteState& nextState,
		FiniteAction& action) {

auto state=this->x;
 WQ_Learning::step(reward,nextState,action);
 policy.setProbabilities(state,integrals);

}

WQ_Learning_WP::WPolicy::WPolicy() {
	initialized = false;
	Q = nullptr;
	//nactions = 0;
	testMode = false;
}

WQ_Learning_WP::WPolicy::~WPolicy() {

}

void WQ_Learning_WP::WPolicy::initialization() {
	if (!initialized) {
		int nstates = Q->n_rows;
		int actions = Q->n_cols;

		probabilities = arma::mat(nstates, actions, arma::fill::ones)
				* (double) 1 / actions;
		initialized = true;
	}

}

unsigned int WQ_Learning_WP::WPolicy::operator()(const size_t& state) {
	initialization();
	unsigned int un;

	// const rowvec& Qx = Q->row(state);

	int x = (unsigned int) state;
	std::vector<double> prob(nactions);
	for (int i = 0; i < nactions; i++) {
		prob[i] = this->probabilities(x, i);
	}
	un = RandomGenerator::sampleDiscrete(prob);

	return un;
}

double WQ_Learning_WP::WPolicy::operator()(const size_t& state,
		const unsigned int& action) {

	return probabilities(state, action);
}

void WQ_Learning_WP::WPolicy::setProbabilities(unsigned int state,
		arma::vec prob) {
	//this->probabilities(x,u)=prob(x,u);
	probabilities.row(state) = prob.t();

}

}
