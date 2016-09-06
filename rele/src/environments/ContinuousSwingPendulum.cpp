/*
 * ContinuousSwingPendulum.cpp
 *
 *  Created on: 05 set 2016
 *      Author: alessandro
 */




#include "rele/environments/ContinuousSwingPendulum.h"
#include "rele/environments/SwingPendulum.h"

#include "rele/utils/RandomGenerator.h"
#include <cassert>


using namespace std;

namespace ReLe
{



ContinuousSwingPendulum::ContinuousSwingPendulum() :
    ContinuousMDP(new SwingUpSettings(true)),  cleanConfig(true), config(static_cast<SwingUpSettings*>(settings))
{
    currentState.set_size(this->getSettings().stateDimensionality);

    //variable initialization
    previousTheta = cumulatedRotation = overRotatedTime = 0;
    overRotated = false;
    upTime = 0;
}

ContinuousSwingPendulum::ContinuousSwingPendulum(SwingUpSettings& config) :
   ContinuousMDP(&config), cleanConfig(false), config(&config)
{
    currentState.set_size(this->getSettings().stateDimensionality);

    //variable initialization
    previousTheta = cumulatedRotation = overRotatedTime = 0;
    overRotated = false;
    upTime = 0;
}

ContinuousSwingPendulum::ContinuousSwingPendulum(double initialPosition,bool randomStart):
    	    ContinuousMDP(new SwingUpSettings(true)),  cleanConfig(true), config(static_cast<SwingUpSettings*>(settings))

	{
		initialTheta=initialPosition;
		config->random_start=randomStart;

		currentState.set_size(this->getSettings().stateDimensionality);
	    //variable initialization
	    previousTheta = cumulatedRotation = overRotatedTime = 0;
	    overRotated = false;
	    upTime = 0;
	}




void ContinuousSwingPendulum::step(const DenseAction& action,
                                 DenseState& nextState, Reward& reward)
{
    const SwingUpSettings& swconfig = *config;




    //get current state
    double theta = currentState[0];
    double velocity = currentState[1];

    double uMax=5;
    double torque =config->actionRange.bound(action[0])*uMax;
    //std::cout << a.at() << std::endl;
    double thetaAcc = -swconfig.stepTime * velocity
                      + swconfig.mass * swconfig.g * swconfig.length * sin(theta) + torque;
    velocity = swconfig.velocityRange.bound(velocity + thetaAcc);
    theta += velocity * swconfig.stepTime;
    adjustTheta(theta);
    upTime = fabs(theta) > swconfig.upRange ? 0 : upTime + 1;

    //update current state
    currentState[0] = theta;
    currentState[1] = velocity;

    double signAngleDifference = std::atan2(std::sin(theta - previousTheta),
                                            std::cos(theta - previousTheta));

    cumulatedRotation += signAngleDifference;
    if (!overRotated && std::abs(cumulatedRotation) > 5.0f * M_PI)
        overRotated = true;
    if (overRotated)
        overRotatedTime += 1;
    previousTheta = theta;

    //###################### TERMINAL CONDITION REACHED ######################
    bool endepisode = false;
    if (swconfig.useOverRotated)
        // Reinforcement Learning in Continuous Time and Space (Kenji Doya)
        endepisode =
            (overRotated && (overRotatedTime > 1.0 / swconfig.stepTime)) ?
            true : false;
    //return upTime + 1 >= requiredUpTime / stepTime; // 1000 steps

    currentState.setAbsorbing(endepisode);
    nextState = currentState;


    //###################### REWARD ######################
    double noise=RandomGenerator::sampleNormal(0,1);

    if (swconfig.useOverRotated)
        // Reinforcement Learning in Continuous Time and Space (Kenji Doya)
        reward[0] = (!overRotated) ? cos(nextState[0])+noise : -1.0;
    else
    {

        reward[0] = cos(nextState[0])+noise;
    }



    nextState[0]=nextState[0]/M_PI;
    nextState[1]=nextState[1]/swconfig.velocityRange.hi();



}

void ContinuousSwingPendulum::getInitialState(DenseState& state)
{

    double theta;
    upTime = 0;
    if (config->random_start)
        theta = RandomGenerator::sampleUniform(config->thetaRange.lo(),
                                               config->thetaRange.hi());

    else
        theta = initialTheta;
    adjustTheta(theta);

    previousTheta = theta;
    cumulatedRotation = theta;
    overRotated = false;
    overRotatedTime = 0;
    currentState[0] = theta;
    currentState[1] = 0.0;

    currentState.setAbsorbing(false);

    state = currentState;


}

}
