/*
 * rele,
 *
 *
 * Copyright (C) 2015 Davide Tateo
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

#ifndef INCLUDE_RELE_ALGORITHMS_POLICY_SEARCH_GRADIENT_HIERARCHICALPOLICYGRADIENT_H_
#define INCLUDE_RELE_ALGORITHMS_POLICY_SEARCH_GRADIENT_HIERARCHICALPOLICYGRADIENT_H_

#include "HierarchicalAlgorithm.h"
#include "policy_search/gradient/onpolicy/GradientOutputData.h"
#include "policy_search/step_rules/StepRules.h"
#include "parametric/differentiable/DifferentiableOptions.h"
#include "RewardTransformation.h"

namespace ReLe
{

//TODO implement a real hierarchical policy gradient
template<class ActionC, class StateC>
class HierarchicalPolicyGradient: public HierarchicalAlgorithm<ActionC, StateC>
{

    using HierarchicalAlgorithm<ActionC, StateC>::stack;

public:
    HierarchicalPolicyGradient(Option<ActionC, StateC>& rootOption,
                               unsigned int nbEpisodes, StepRule& stepL, bool baseline = true,
                               int reward_obj = 0) :
        HierarchicalAlgorithm<ActionC, StateC>(rootOption),
        nbEpisodesToEvalPolicy(nbEpisodes), runCount(0), epiCount(0),
        df(1.0), Jep(0.0), rewardTr(new IndexRT(reward_obj)),
        cleanRT(true), useBaseline(baseline), output2LogReady(false),
        stepLength(stepL), currentItStats(nullptr)
    {
    }

    HierarchicalPolicyGradient(Option<ActionC, StateC>& rootOption,
                               unsigned int nbEpisodes, StepRule& stepL,
                               RewardTransformation& reward_tr, bool baseline = true) :
        HierarchicalAlgorithm<ActionC, StateC>(rootOption),
        nbEpisodesToEvalPolicy(nbEpisodes), runCount(0), epiCount(0),
        df(1.0), Jep(0.0), rewardTr(&reward_tr), cleanRT(false),
        useBaseline(baseline), output2LogReady(false),
        stepLength(stepL), currentItStats(nullptr)
    {
    }

    virtual ~HierarchicalPolicyGradient()
    {
        if (cleanRT)
        {
            delete rewardTr;
        }
    }

    // Agent interface
public:
    virtual void initEpisode(const StateC& state, ActionC& action)
    {
        df = 1.0;    //reset discount factor
        Jep = 0.0;    //reset J of current episode

        // Initialize variables
        initializeVariables();

        //--- set up agent output
        if (epiCount == 0)
        {
            currentItStats = new GradientIndividual();
            DifferentiablePolicy<FiniteAction, StateC>& policy = getPolicy(); //FIXME
            currentItStats->policy_parameters = policy.getParameters();
        }
        //---

        HierarchicalAlgorithm<ActionC, StateC>::sampleAction(state, action);

        // save state and action for late use
        currentState = state;
        currentAction = this->getPolicy()(state); //FIXME
    }

    virtual void initTestEpisode()
    {
    }

    virtual void step(const Reward& reward, const StateC& nextState,
                      ActionC& action)
    {

        updateStep(reward);

        //calculate current J value
        RewardTransformation& rTr = *rewardTr;
        Jep += df * rTr(reward);
        //update discount factor
        df *= this->task.gamma;

        HierarchicalAlgorithm<ActionC, StateC>::sampleAction(nextState, action);

        // save state and action for late use
        currentState = nextState;
        currentAction = this->getPolicy()(nextState); //FIXME
    }

    virtual void endEpisode(const Reward& reward)
    {
        updateStep(reward);

        //add last contribute
        RewardTransformation& rTr = *rewardTr;
        Jep += df * rTr(reward);

        //perform remaining operation
        this->endEpisode();

    }

    virtual void endEpisode()
    {

        //save policy value
        history_J[epiCount] = Jep;

        updateAtEpisodeEnd();

        //last episode is the number epiCount+1
        epiCount++;
        //check evaluation of actual policy
        if (epiCount == nbEpisodesToEvalPolicy)
        {
            updatePolicy();

            //reset counters and gradient
            epiCount = 0; //reset episode counter
            runCount++; //update run counter
            output2LogReady = true; //output must be ready for log
        }
    }

    virtual AgentOutputData* getAgentOutputDataEnd()
    {
        if (output2LogReady)
        {
            //output is ready, activate flag
            output2LogReady = false;
            return currentItStats;
        }
        return nullptr;
    }

protected:
    virtual void init()
    {
        history_J.assign(nbEpisodesToEvalPolicy, 0.0);
    }

    DifferentiablePolicy<FiniteAction, StateC>& getPolicy()
    {
        Option<ActionC, StateC>& option = *stack[0];

        return static_cast<DifferentiableOption<ActionC, StateC>&>(option).getPolicy();
    }

    virtual void initializeVariables() = 0;
    virtual void updateStep(const Reward& reward) = 0;
    virtual void updateAtEpisodeEnd() = 0;
    virtual void updatePolicy() = 0;

protected:
    unsigned int nbEpisodesToEvalPolicy;
    unsigned int runCount, epiCount;
    double df, Jep;
    StepRule& stepLength;
    RewardTransformation* rewardTr;
    bool cleanRT;

    std::vector<double> history_J;
    bool useBaseline, output2LogReady;
    GradientIndividual* currentItStats;

    FiniteAction currentAction; //FIXME
    StateC currentState;
};

#define USE_HPGA_MEMBERS                                           \
    typedef HierarchicalPolicyGradient<ActionC, StateC> Base;      \
    /*using Base::policy;*/                                            \
    using Base::nbEpisodesToEvalPolicy;                            \
    using Base::runCount;                                          \
    using Base::epiCount;                                          \
    using Base::df;                                                \
    using Base::Jep;                                               \
    using Base::stepLength;                                        \
    using Base::rewardTr;                                          \
    using Base::history_J;                                         \
    using Base::useBaseline;                                       \
    using Base::output2LogReady;                                   \
    using Base::currentItStats;                                    \
    using Base::currentAction;                                     \
    using Base::currentState;

}

#endif /* INCLUDE_RELE_ALGORITHMS_POLICY_SEARCH_GRADIENT_HIERARCHICALPOLICYGRADIENT_H_ */