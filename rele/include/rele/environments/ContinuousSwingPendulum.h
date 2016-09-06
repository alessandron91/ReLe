#ifndef CONTINUOUSSWINGPENDULUM_H_
#define CONTINUOUSSWINGPENDULUM_H_

#include "rele/core/ContinuousMDP.h"
#include "rele/utils/Range.h"
#include "rele/environments/SwingPendulum.h"


namespace ReLe
{

/*!
 * This class implements the Ship Steering problem.
 * The aim of this problem is to let a ship pass through
 * a gate when starting from a random position and moving
 * at constant speed.
 * For further information see <a href="http://people.cs.umass.edu/~mahadeva/papers/icml03-1.pdf">here</a>.
 *
 * References
 * ==========
 * [Ghavamzadeh, Mahadevan. Hierarchical Policy Gradient Algorithms. ICML 2013](http://people.cs.umass.edu/~mahadeva/papers/icml03-1.pdf)
 */
class ContinuousSwingPendulum : public ContinuousMDP
{
public:

    ContinuousSwingPendulum();

    /*!
     * \see Environment::step
     */

    ContinuousSwingPendulum(SwingUpSettings& config);
    ContinuousSwingPendulum(double initialPosition,bool randomStart);

    virtual void step(const DenseAction& action, DenseState& nextState,
                      Reward& reward) override;

    /*!
     * \see Environment::getInitialState
     */
    virtual void getInitialState(DenseState& state) override;

	const SwingUpSettings& getConfig() const {
		return *config;
	}

private:
    inline void adjustTheta(double& theta)
    {
        if (theta >= M_PI)
            theta -= 2.0 * M_PI;
        if (theta < -M_PI)
            theta += 2.0 * M_PI;
    }

private:
      double previousTheta, cumulatedRotation, overRotatedTime;
      double initialTheta=M_PI_2;
      bool overRotated;
      int upTime;
      //current state [theta, velocity]
      SwingUpSettings* config;
      bool cleanConfig;
};

}











#endif /*CONTINUOUSSWINGPENDULUM_H_ */
