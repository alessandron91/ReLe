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

#include "../../include/rele/environments/Rocky.h"

#include "Utils.h"

using namespace arma;
using namespace std;

namespace ReLe
{

Rocky::Rocky() :
    ContinuousMDP(STATESIZE, 3, 1, false, true, 0.9999), dt(0.01),
    maxOmega(M_PI), maxV(1), maxOmegar(M_PI-M_PI/12), maxVr(1), limitX(10), limitY(10), predictor(dt, limitX, limitY)
{
    //TODO parameter in the constructor
    vec2 spot;
    spot[0] = 5;
    spot[1] = 0;
    foodSpots.push_back(spot);
}

void Rocky::step(const DenseAction& action, DenseState& nextState,
                 Reward& reward)
{
    //action threshold
    double v = utils::threshold(action[0], maxV);
    double omega = utils::threshold(action[1], maxOmega);
    bool eat = (action[2] > 0 && v == 0 && omega == 0) ? true : false;

    //Compute rocky control using chicken pose prediction
    double omegar;
    double vr;
    computeRockyControl(vr, omegar);

    //Update rocky state
    double xrabs, yrabs;
    updateRockyPose(vr, omegar, xrabs, yrabs);

    //update chicken position
    updateChickenPose(v, omega);

    //update rocky relative position
    currentState[xr] = xrabs - currentState[x];
    currentState[yr] = yrabs - currentState[y];

    //Compute sensors
    computeSensors(eat);

    //Compute reward
    computeReward(reward);

    nextState = currentState;
}

void Rocky::getInitialState(DenseState& state)
{
    //chicken state
    currentState[x] = 0;
    currentState[y] = 0;
    currentState[theta] = 0;

    //sensors state
    currentState[energy] = 0;
    currentState[food] = 0;

    //rocky state
    currentState[xr] = 1;
    currentState[yr] = 1;
    currentState[thetar] = 0;

    //reset predictor state
    predictor.reset();

    currentState.setAbsorbing(false);

    state = currentState;
}

void Rocky::computeRockyControl(double& vr, double& omegar)
{
    //Predict chicken position
    double xhat, yhat, thetaDirhat;
    predictor.predict(currentState, xhat, yhat, thetaDirhat);

    //Compute rocky control signals
    double deltaTheta = utils::wrapToPi(thetaDirhat - currentState[thetar]);
    double omegarOpt = deltaTheta / dt;

    omegar = utils::threshold(omegarOpt, maxOmegar);

    if (abs(deltaTheta) > M_PI / 2)
    {
        vr = 0;
    }
    else if (abs(deltaTheta) > M_PI / 4)
    {
        vr = maxVr / 2;
    }
    else
    {
        vr = maxVr;
    }

    //FIXME levami
    /*vr = 0;
    omegar = 0;*/
}

void Rocky::updateRockyPose(double vr, double omegar, double& xrabs,
                            double& yrabs)
{
    vec2 chickenPosition = currentState.rows(span(x, y));
    vec2 rockyRelPosition = currentState.rows(span(xr, yr));

    double thetarM = (2 * currentState[thetar] + omegar * dt) / 2;
    currentState[thetar] = utils::wrapToPi(
                               currentState[thetar] + omegar * dt);
    xrabs = chickenPosition[0] + rockyRelPosition[0] + vr * cos(thetarM) * dt;
    yrabs = chickenPosition[1] + rockyRelPosition[1] + vr * sin(thetarM) * dt;

    //Anelastic walls
    xrabs = utils::threshold(xrabs, limitX);
    yrabs = utils::threshold(yrabs, limitY);
}

void Rocky::updateChickenPose(double v, double omega)
{
    double thetaM = (2 * currentState[theta] + omega * dt) / 2;
    currentState[x] += v * cos(thetaM) * dt;
    currentState[y] += v * sin(thetaM) * dt;

    //Anelastic walls
    currentState[x] = utils::threshold(currentState[x], limitX);
    currentState[y] = utils::threshold(currentState[y], limitY);

    currentState[theta] = utils::wrapToPi(
                              currentState[theta] + omega * dt);

    predictor.saveLastValues(thetaM, v);
}

void Rocky::computeSensors(bool eat)
{
    vec2 chickenPosition = currentState.rows(span(x, y));

    currentState[energy] = utils::threshold(currentState[energy] - 0.01, 0, 100);
    currentState[food] = 0;

    for (auto& spot : foodSpots)
    {
        if (norm(chickenPosition - spot) < 0.5)
        {
            currentState[food] = 1;

            if (eat)
            {
                currentState[energy] = utils::threshold(
                                           currentState[energy] + 5, 0, 100);
            }

            break;
        }
    }
}

void Rocky::computeReward(Reward& reward)
{
    vec2 chickenPosition = currentState.rows(span(x, y));
    vec2 rockyRelPosition = currentState.rows(span(xr, yr));

    if (norm(rockyRelPosition) < 0.05)
    {
        reward[0] = -100;
        currentState.setAbsorbing(true);
    }
    else if (norm(chickenPosition) < 0.4 && currentState[energy] > 0)
    {
        reward[0] = currentState[energy];
        currentState.setAbsorbing(true);
    }
    else
    {
        reward[0] = 0;
        currentState.setAbsorbing(false);
    }
}

Rocky::Predictor::Predictor(double dt, double limitX, double limitY) :
    dt(dt), limitX(limitX), limitY(limitY)
{
    reset();
}

void Rocky::Predictor::reset()
{
    thetaM = 0;
    v = 0;
}

void Rocky::Predictor::saveLastValues(double thetaM, double v)
{
    this->thetaM = thetaM;
    this->v = v;
}

void Rocky::Predictor::predict(const DenseState& state, double& xhat, double& yhat, double& thetaDirhat)
{
    xhat = state[x] + v * cos(thetaM) * dt;
    yhat = state[y] + v * sin(thetaM) * dt;

    //Anelastic walls
    xhat = utils::threshold(xhat, limitX);
    yhat = utils::threshold(yhat, limitY);

    thetaDirhat = utils::wrapToPi(atan2(yhat - (state[y] + state[yr]), xhat - (state[x] + state[xr])));
}

}
