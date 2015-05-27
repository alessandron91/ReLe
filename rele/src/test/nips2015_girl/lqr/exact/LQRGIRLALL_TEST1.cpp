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

#include "Core.h"
#include "parametric/differentiable/LinearPolicy.h"
#include "parametric/differentiable/NormalPolicy.h"
#include "features/SparseFeatures.h"
#include "DifferentiableNormals.h"
#include "basis/IdentityBasis.h"

#include "LQR.h"
#include "LQRsolver.h"
#include "PolicyEvalAgent.h"
#include "algorithms/GIRL.h"
#include "algorithms/PGIRL.h"

#include "ParametricRewardMDP.h"
#include "RandomGenerator.h"
#include "FileManager.h"

#include "policy_search/gradient/onpolicy/FunctionGradient.h"
#include "policy_search/gradient/PolicyGradientAlgorithm.h"

#include <boost/timer/timer.hpp>

using namespace boost::timer;
using namespace std;
using namespace ReLe;
using namespace arma;

class LQR_1D_WS : public IRLParametricReward<DenseAction, DenseState>,
    public RewardTransformation
{
public:

    LQR_1D_WS()
    {
        weights.set_size(2);
    }

    double operator()(DenseState& s, DenseAction& a, DenseState& ns)
    {
        return -(weights(0)*s(0)*s(0)+weights(1)*a(0)*a(0));
    }

    double operator()(const Reward& r)
    {
        return r[0];
    }

    arma::mat diff(DenseState& s, DenseAction& a, DenseState& ns)
    {
        arma::mat m(1,2);
        m(0) = -s(0)*s(0);
        m(1) = -a(0)*a(0);
        return m;
    }
};

class LQR_1D_R1 : public IRLParametricReward<DenseAction, DenseState>
{
public:
    double operator()(DenseState& s, DenseAction& a, DenseState& ns)
    {
        return -s(0)*s(0);
    }

    arma::mat diff(DenseState& s, DenseAction& a, DenseState& ns)
    {
        return arma::mat();
    }
};
class LQR_1D_R2 : public IRLParametricReward<DenseAction, DenseState>
{
public:
    double operator()(DenseState& s, DenseAction& a, DenseState& ns)
    {
        return -a(0)*a(0);
    }

    arma::mat diff(DenseState& s, DenseAction& a, DenseState& ns)
    {
        return arma::mat();
    }
};

class LQR_ND_WS : public IRLParametricReward<DenseAction, DenseState>
{
public:

    LQR_ND_WS(LQR& mdp)
        : lqr(mdp)
    {
        weights.set_size(lqr.getSettings().rewardDim);
    }

    double operator()(DenseState& s, DenseAction& a, DenseState& ns)
    {
        int dim = lqr.Q.size();
        double val = 0.0;
        arma::vec& x = s;
        arma::vec& u = a;
        for (int i = 0; i < dim; ++i)
        {
            arma::mat& R = lqr.R[i];
            arma::mat& Q = lqr.Q[i];
            arma::mat J = (x.t() * Q * x + u.t() * R * u) * weights(i);
            val -= J(0,0);
        }
        return val;
    }

    arma::mat diff(DenseState& s, DenseAction& a, DenseState& ns)
    {
        int dim = lqr.Q.size();
        arma::vec& x = s;
        arma::vec& u = a;
        arma::mat m(1,dim);
        for (int i = 0; i < dim; ++i)
        {
            arma::mat& R = lqr.R[i];
            arma::mat& Q = lqr.Q[i];
            arma::mat J = -(x.t() * Q * x + u.t() * R * u);
            m(0,i) = J(0,0);
        }
        return m;
    }

private:
    LQR& lqr;
};

class LQR_ND_R : public IRLParametricReward<DenseAction, DenseState>
{
public:
    LQR_ND_R(LQR& mdp, unsigned int idx)
        : lqr(mdp), idx(idx)
    {
    }

    double operator()(DenseState& s, DenseAction& a, DenseState& ns)
    {
        arma::vec& x = s;
        arma::vec& u = a;
        arma::mat& R = lqr.R[idx];
        arma::mat& Q = lqr.Q[idx];
        arma::mat J = -(x.t() * Q * x + u.t() * R * u);
        return J(0,0);
    }

    arma::mat diff(DenseState& s, DenseAction& a, DenseState& ns)
    {
        return arma::mat();
    }
private:
    unsigned int idx;
    LQR& lqr;
};


void help()
{
    cout << "lqr_GIRLNIPS_TEST1 seed episodes nbReward reward" << endl;
}

int main(int argc, char *argv[])
{
//        RandomGenerator::seed(4434224);

    /*** check inputs ***/
    vec eReward;
    int nbEpisodes;

    if (argc < 5)
    {
        help();
        abort();
    }

    long int seed = atol(argv[1]);
    RandomGenerator::seed(seed);
    std::cout << seed << std::endl;

    nbEpisodes = atoi(argv[2]);
    cout << "Episodes: " << nbEpisodes << endl;
    int nw = atoi(argv[3]);
    eReward.set_size(nw);
    for (int i = 0; i < nw; ++i)
        eReward(i) = atof(argv[4+i]);

    /******/

    FileManager fm("lqr", "GIRLNIPS_TEST1");
    fm.createDir();
    //    fm.cleanDir();
    std::cout << std::setprecision(OS_PRECISION);

    /* Learn lqr correct policy */

    int dim = eReward.n_elem;
    LQR mdp(dim, dim);
    mdp.startType = LQR::S0Type::RANDOM;

    BasisFunctions basis;
    for (int i = 0; i < dim; ++i)
    {
        basis.push_back(new IdentityBasis(i));
    }

    SparseFeatures phi;
    phi.setDiagonal(basis);

    MVNPolicy expertPolicy(phi);

    /*** solve the problem in exact way ***/
    LQRsolver solver(mdp,phi);
    solver.setRewardWeights(eReward);
    mat K = solver.computeOptSolution();
    arma::vec p = K.diag();
    expertPolicy.setParameters(p);

    std::cout << "Rewards: ";
    for (int i = 0; i < eReward.n_elem; ++i)
    {
        std::cout << eReward(i) << " ";
    }
    std::cout << "| Params: " << expertPolicy.getParameters().t() << std::endl;


    PolicyEvalAgent<DenseAction, DenseState> expert(expertPolicy);

    /* Generate LQR expert dataset */
    Core<DenseAction, DenseState> expertCore(mdp, expert);
    CollectorStrategy<DenseAction, DenseState> collection;
    expertCore.getSettings().loggerStrategy = &collection;
    expertCore.getSettings().episodeLenght = mdp.getSettings().horizon;
    expertCore.getSettings().testEpisodeN = nbEpisodes;
    expertCore.runTestEpisodes();


    //save data
    Dataset<DenseAction,DenseState>& data = collection.data;
    ofstream datafile(fm.addPath("data.log"), ios_base::out);
    datafile << std::setprecision(OS_PRECISION);
    data.writeToStream(datafile);
    datafile.close();


    char gtypestr[10];
    for (int i = 0; i < 5; ++i)
    {

        expertPolicy.setParameters(p);

        IRLGradType atype;
        if (i == 0)
        {
            cout << "GIRL REINFORCE";
            atype = IRLGradType::R;
            strcpy(gtypestr, "r");
        }
        else if (i == 1)
        {
            cout << "GIRL REINFORCE BASE";
            atype = IRLGradType::RB;
            strcpy(gtypestr, "rb");
        }
        else if (i == 2)
        {
            cout << "GIRL GPOMDP";
            atype = IRLGradType::G;
            strcpy(gtypestr, "g");
        }
        else if (i == 3)
        {
            cout << "GIRL GPOMDP BASE";
            atype = IRLGradType::GB;
            strcpy(gtypestr, "gb");
        }
        else if (i == 4)
        {
            cout << "GIRL ENAC";
            atype = IRLGradType::ENAC;
            strcpy(gtypestr, "enac");
        }
        else
        {
            std::cout << "Error unknown argument " << argv[1] << std::endl;
            help();
            exit(1);
        }

        /* Learn weight with GIRL */
        LQR_ND_WS rewardRegressor(mdp);
        assert(mdp.getSettings().gamma == 0.9);
        GIRL<DenseAction,DenseState> irlAlg(data, expertPolicy, rewardRegressor,
                                            mdp.getSettings().gamma, atype);


        char namet[100];
        sprintf(namet, "girl_time_%s.log", gtypestr);
        ofstream timefile(fm.addPath(namet));


        //Run GIRL
        cpu_timer timer;
        timer.start();
        irlAlg.run();
        timer.stop();
        arma::vec gnormw = irlAlg.getWeights();

        timefile << timer.format(10, "%w") << std::endl;

        cout << "Weights (gnorm): " << gnormw.t();

        char name[100];
        sprintf(name, "girl_gnorm_%s.log", gtypestr);
        ofstream outf(fm.addPath(name), std::ofstream::out);
        outf << std::setprecision(OS_PRECISION);
        for (int i = 0; i < gnormw.n_elem; ++i)
        {
            outf << gnormw[i] << " ";
        }
        outf.close();

        sprintf(name, "girl_gnorm_%s_neval.log", gtypestr);
        outf.open(fm.addPath(name), std::ofstream::out);
        outf << std::setprecision(OS_PRECISION);
        outf << irlAlg.getFunEvals();
        outf.close();


        std::vector<IRLParametricReward<DenseAction,DenseState>*> rewards;
        for (int i = 0; i < dim; ++i)
        {
            rewards.push_back(new LQR_ND_R(mdp, i));
        }

        //Run PLANE GIRL
        PlaneGIRL<DenseAction,DenseState> pgirl(data, expertPolicy, rewards,
                                                mdp.getSettings().gamma, atype);

        cpu_timer timer2;
        timer2.start();
        pgirl.run();
        timer2.stop();
        timefile << timer2.format(10, "%w") << std::endl;


        cout << "Weights (plane): " << pgirl.getWeights().t();

        sprintf(name, "girl_plane_%s.log", gtypestr);
        outf.open(fm.addPath(name), std::ofstream::out | std::ofstream::app);
        outf << std::setprecision(OS_PRECISION);
        arma::vec planew = pgirl.getWeights();
        for (int i = 0; i < planew.n_elem; ++i)
        {
            outf << planew[i] << " ";
        }

        outf.close();
        timefile.close();

        cout << endl;
    }

    return 0;
}