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

#ifndef INCLUDE_RELE_ENVIRONMENTS_UNICYCLE_H_
#define INCLUDE_RELE_ENVIRONMENTS_UNICYCLE_H_

#include "ContinuousMDP.h"
#include <boost/numeric/odeint.hpp>
#include "ArmadilloOdeint.h"

namespace ReLe
{

class UnicyclePolarSettings : public EnvironmentSettings
{
public:
    UnicyclePolarSettings();
    static void defaultSettings(UnicyclePolarSettings& settings);
    virtual ~UnicyclePolarSettings();

public:
    double dt;

    virtual void WriteToStream(std::ostream& out) const;
    virtual void ReadFromStream(std::istream& in);
};

/**
 * References:
 * http://www.diva-portal.se/smash/get/diva2:662268/FULLTEXT01.pdf
 * http://cdn.intechopen.com/pdfs-wm/44029.pdf
 *
 * \hat{e}(t) = [x(t) - x_g; y(t) - y_g; \theta(t) - \theta_g]^{T}
 * e(t) = \begin{bmatrix} e_x(t)\\e_y(t)\\ e_\theta(t)\end{bmatrix}
 *      = \begin{bmatrix}  cos\theta_g & sin\theta_g & 0\\
 *                        -sin\theta_g & cos\theta_g & 0\\
 *                         0 & 0 & 1
 *        \end{bmatrix} \hat{e}(t)
 * \rho   = \sqrt(e_x^2+e_y^2)
 * \gamma = atan2(e_y,e_x) - e_\theta + \pi
 * \delta = \gamma + e_\theta
 *
 * Optimal control law:
 * v = k_1 \rho cos\gamma
 * w = k_2 \gamma + k_1 sin\gamma cos\gamma (gamma + k_3 \delta) / gamma
 */
class UnicyclePolar: public ContinuousMDP
{

    typedef arma::vec state_type;

private:
    //used in odeint
    class UnicyclePolarOde
    {

    public:
        double v, w;

        void operator()(const state_type& x, state_type& dx,
                        const double /* t */);

    };

public:

    UnicyclePolar();
    UnicyclePolar(UnicyclePolarSettings& config);

    virtual ~UnicyclePolar()
    {
        if (cleanConfig)
            delete unicycleConfig;
    }

    virtual void step(const DenseAction& action, DenseState& nextState,
                      Reward& reward);
    virtual void getInitialState(DenseState& state);

    inline const UnicyclePolarSettings& getSettings() const
    {
        return *unicycleConfig;
    }

private:
    UnicyclePolarSettings* unicycleConfig;
    UnicyclePolarOde unicycleode;
    bool cleanConfig;

    //[ define_adapt_stepper
    typedef boost::numeric::odeint::runge_kutta_cash_karp54< state_type > error_stepper_type;
    typedef boost::numeric::odeint::controlled_runge_kutta< error_stepper_type > controlled_stepper_type;
    controlled_stepper_type controlled_stepper;
    //]

};

}

#endif /* INCLUDE_RELE_ENVIRONMENTS_UNICYCLE_H_ */