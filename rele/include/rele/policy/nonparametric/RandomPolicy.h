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

#ifndef RANDOMPOLICY_H_
#define RANDOMPOLICY_H_

#include "Policy.h"
#include "Range.h"
#include "RandomGenerator.h"

namespace ReLe
{

/**
 * This agent selects a purely random action given any input state.
 * Each action value is selected within the given range, i.e., it is
 * drawn from an uniform bounded distribution:
 * \f[ \pi(a|s) = \mathcal{U}(\Omega)\f]
 * where \f$\Omega \subseteq R^{n}\f$.
 * @brief Purely continuous random policy
 */
template<class StateC>
class RandomPolicy : public Policy<DenseAction,StateC>
{

protected:

    std::vector<Range> mpRanges;
    DenseAction mpAction;

public:
    /**
     * Construct a RandomPolicy object, initializing the action range.
     * @brief Construct RandomPolicy
     * @param ranges the range of the action elements
     */
    RandomPolicy(std::vector<Range>& ranges)
        : mpRanges(ranges), mpAction(ranges->size())
    {
        for (int i = 0; i < ranges->size(); ++i)
        {
            mpAction[i] = 0.0;
        }
    }

    virtual ~RandomPolicy()
    {
        delete [] mpAction;
    }

    // Policy interface
public:
    std::string getPolicyName()
    {
        return std::string("RandomPolicy");
    }

    std::string getPolicyHyperparameters()
    {
        return std::string("");
    }

    std::string printPolicy()
    {
        return std::string("");
    }

    virtual DenseAction operator() (const StateC state)
    {
        for (int i = 0; i < mpRanges->dimension(); ++i)
        {
            double a = mpRanges[i].min();
            double b = mpRanges[i].max();
            double val = a + (b - a) * RandomGenerator::sampleUniform(0,1);
            mpAction[i] =  val;
        }
    }

    virtual double operator() (
        typename state_type<StateC>::const_type state,
        const DenseAction& action)
    {
        double prob = 1.0;
        std::vector<Range>::iterator it;
        for(it = mpRanges.begin(); it != mpRanges.end(); ++it)
        {
            prob *= 1.0 / (*it).width();
        }
        return prob;
    }

    /**
     * @copydoc Policy::pi()
     */
    double pi(const double* state, const double* action)
    {
        double prob = 1.0;
        typedef typename RLLib::Ranges<T>::iterator rit;
        for(rit it = mpRanges->begin(); it != mpRanges->end(); ++it)
        {
            prob *= 1.0 / (*it)->length();
        }
        return prob;
    }

};

/**
 * Random policy that produces action values according to a
 * discrete distribution, where each possible action has a predefined probability of being produced:
 * \f[\pi(a_i|s,D=[w_1,\dots,w_n]) = \frac{w_i}{\sum_{k=1}^{n} w_k}, \quad 1\leq i\leq n.\f]
 * The w's are a set of n non-negative individual weights set on construction (or using member param)
 * that defines the discrete action probability.
 * @brief Discrete random policy
 */
template<class ActionC, class StateC>
class StochasticDiscretePolicy: public virtual Policy<ActionC,StateC>
{
protected:
    std::vector<ActionC> mActions;
    double* distribution;
public:

    /**
     * Constructs a StochasticDiscretePolicy object, initializing it with the given ActionList. The distribution
     * is set uniformly over the action contained in the action list. Note that the ActionList in cloned (deep-copy).
     * All weights shall be non-negative values, and at least one of the values in the sequence must be positive.
     * @brief Construct a discrete policy
     * @param actions a pointer to an action list
     * @param actionDim the dimension of each action
     */
    StochasticDiscretePolicy(std::vector<ActionC> actions) :
        mActions(actions), distribution(new double[actions->size()])
    {
        int nbel = mActions.size();
        double p = 1/nbel;
        for (int i = 0; i < nbel; ++i)
        {
            distribution[i] = p;
        }
    }

    /**
     * Constructs a StochasticDiscretePolicy object, initializing it with the given ActionList and a set of weights
     * representing the distribution over actions.
     * All weights shall be non-negative values, and at least one of the values in the sequence must be positive.
     * They do not have to be 1-sum, normalization is performed in the construction phase.
     *
     * Note that the ActionList in cloned (deep-copy).
     * @brief Construct a discrete policy
     * @param actions a pointer to an action list
     * @param actionDim the dimension of each action
     * @param dist an array storing the discrete distribution
     */
    StochasticDiscretePolicy(std::vector<ActionC> actions, double* dist) :
        mActions(actions), distribution(new double[mActions->size()])
    {
        double tot = 0.0;
        for (int i = 0; i < mActions.size(); ++i)
        {
            tot += dist[i];
            distribution[i] = dist[i];
        }
        for (int i = 0; i < mActions.size(); ++i)
        {
            distribution[i] /= tot;
            std::cout << distribution[i] << " ";
        }
        std::cout << std::endl;
    }

    virtual ~StochasticDiscretePolicy()
    {
        delete [] distribution;
    }

    std::string getPolicyName()
    {
        return std::string("StochasticDiscretePolicy");
    }

    std::string getPolicyHyperparameters()
    {
        return std::string("");
    }

    std::string printPolicy()
    {
        return std::string("");
    }


    /**
     * @copydoc DiscreteActionPolicy::pi()
     */
    virtual double operator() (
        typename state_type<StateC>::const_type state,
        typename state_type<ActionC>::const_type action)
    {
        int idx = findAction(action);
        return distribution[idx];
    }

    /**
     * @copydoc DiscreteActionPolicy::sampleAction()
     */
    virtual typename action_type<ActionC>::type operator() (typename state_type<StateC>::const_type state)
    {
        double random = RandomGenerator::sampleUniform(0,1);
        double sum = 0.0;
        int i, ie;
        for (i = 0, ie = mActions.size(); i < ie; ++i)
        {
            sum += distribution[i];
            if (sum >= random)
                return mActions[i];
        }
        return mActions[ie-1];
    }

private:
    int findAction(ActionC& action)
    {
        for (int i = 0, ie = mActions.size(); i < ie; ++i)
        {
            if(action.isEqual(mActions[i]))
            {
                return i;
            }
        }
        std::cerr << "Error: unknown action" << std::endl;
        std::cerr << "Action:" << action << std::endl;
        abort();
        return;
    }
};

//TODO
template<class ActionC, class StateC>
class RandomDiscreteBiasPolicy: public StochasticDiscretePolicy<ActionC,StateC>
{
protected:
    ActionC& prev;
    int prevActionId;
    typedef StochasticDiscretePolicy<ActionC,StateC> Base;
public:
    RandomDiscreteBiasPolicy(std::vector<ActionC> actions) :
        StochasticDiscretePolicy<T>(actions), prev(actions[0]), prevActionId(0)
    { }

    virtual ~RandomDiscreteBiasPolicy()
    { }

    std::string getPolicyName()
    {
        return std::string("RandomDiscreteBiasPolicy");
    }

    std::string getPolicyHyperparameters()
    {
        return std::string("");
    }

    std::string printPolicy()
    {
        return std::string("");
    }

    /**
     * @copydoc StochasticDiscretePolicy::sampleAction()
     */
    virtual typename action_type<ActionC>::type operator() (typename state_type<StateC>::const_type state)
    {

        int i, ie;
        // 50% prev action
        if (Base::mActions.size() == 1)
            Base::distribution[0] = 1.0;
        else
        {
            for (i = 0, ie = mActions.size(); i < ie; ++i)
            {
                if (id == previd)
                    Base::distribution[id] = 0.5;
                else
                    Base::distribution[id] = 0.5 / (ie - 1);
            }
        }
        // chose an action
        double random = RandomGenerator::sampleUniform(0,1);
        double sum = 0.0;
        for (i = 0, ie = mActions.size(); i < ie; ++i)
        {
            sum += Base::distribution[i];
            if (sum >= random)
            {
                prev = Base::mActions[i];
                prevActionId = i;
                return prev;
            }
        }
        prev = Base::mActions[ie - 1];
        prevActionId = ie - 1;
        return prev;
    }

};

} // end namespace

#endif // RANDOMPOLICY_H_