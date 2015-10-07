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

#ifndef PGIRL_H_
#define PGIRL_H_

#include "IRLAlgorithm.h"
#include "Policy.h"
#include "Transition.h"
#include "algorithms/GIRL.h" //TODO togliere, sola dipendenza gradient type
#include "ArmadilloExtensions.h"

#include <nlopt.hpp>
#include <cassert>

#include "feature_selection/PrincipalComponentAnalysis.h"

namespace ReLe
{

template<class ActionC, class StateC>
class PlaneGIRL : public IRLAlgorithm<ActionC, StateC>
{
public:

    PlaneGIRL(Dataset<ActionC,StateC>& dataset,
              DifferentiablePolicy<ActionC,StateC>& policy,
              DenseFeatures& phi,
              double gamma, IRLGradType aType)
        : policy(policy), data(dataset), phi(phi),
          gamma(gamma), atype(aType)
    {

    }

    virtual ~PlaneGIRL()
    {

    }

    virtual arma::vec getWeights() override
    {
        return weights;
    }

    virtual Policy<ActionC, StateC>* getPolicy() override
    {
        return &policy;
    }

    void setData(Dataset<ActionC,StateC>& dataset)
    {
        data = dataset;
    }


    arma::mat ReinforceGradient()
    {
        int dp  = policy.getParametersSize();
        arma::vec sumGradLog(dp), localg;
        arma::mat gradient_J(dp, dp, arma::fill::zeros);
        arma::vec Rew(dp);

        int nbEpisodes = data.size();
        for (int i = 0; i < nbEpisodes; ++i)
        {
            //core setup
            int nbSteps = data[i].size();


            // *** REINFORCE CORE *** //
            sumGradLog.zeros();
            double df = 1.0;
            Rew.zeros();
            // ********************** //

            //iterate the episode
            for (int t = 0; t < nbSteps; ++t)
            {
                Transition<ActionC, StateC>& tr = data[i][t];
                //            std::cout << tr.x << " " << tr.u << " " << tr.xn << " " << tr.r[0] << std::endl;

                // *** REINFORCE CORE *** //
                localg = policy.difflog(tr.x, tr.u);
                sumGradLog += localg;
                Rew += df * T *phi(vectorize(tr.x, tr.u, tr.xn));
                // ********************** //

                df *= gamma;

                if (tr.xn.isAbsorbing())
                {
                    assert(nbSteps == t+1);
                    break;
                }
            }

            // *** REINFORCE CORE *** //
            for(unsigned int i = 0; i < dp; i++)
                gradient_J.col(i) += Rew(i) * sumGradLog;

            // ********************** //

        }
        // compute mean values
        gradient_J /= nbEpisodes;

        return gradient_J;
    }

    arma::mat ReinforceBaseGradient()
    {
        int dp  = policy.getParametersSize();
        int nbEpisodes = data.size();

        arma::vec sumGradLog(dp), localg;
        arma::mat gradient_J(dp, dp, arma::fill::zeros);
        arma::vec Rew(dp);

        arma::mat baseline_J_num(dp, dp, arma::fill::zeros);
        arma::mat baseline_den(dp, dp, arma::fill::zeros);
        arma::mat return_J_ObjEp(dp, nbEpisodes);
        arma::mat sumGradLog_CompEp(dp,nbEpisodes);

        for (int ep = 0; ep < nbEpisodes; ++ep)
        {
            //core setup
            int nbSteps = data[ep].size();


            // *** REINFORCE CORE *** //
            sumGradLog.zeros();
            double df = 1.0;
            Rew.zeros();
            // ********************** //

            //iterate the episode
            for (int t = 0; t < nbSteps; ++t)
            {
                Transition<ActionC, StateC>& tr = data[ep][t];

                // *** REINFORCE CORE *** //
                localg = policy.difflog(tr.x, tr.u);
                for (int p = 0; p < dp; ++p)
                    assert(!isinf(localg(p)));
                sumGradLog += localg;
                Rew += df * T * phi(vectorize(tr.x, tr.u, tr.xn));
                // ********************** //

                df *= gamma;

                if (tr.xn.isAbsorbing())
                {
                    assert(nbSteps == t+1);
                    break;
                }
            }

            // *** REINFORCE BASE CORE *** //

            // store the basic elements used to compute the gradients

            return_J_ObjEp.col(ep) = Rew;

            sumGradLog_CompEp.col(ep) = sumGradLog;


            // compute the baselines
            for (int i = 0; i < dp; ++i)
            {
                baseline_J_num.col(i) += Rew(i) * sumGradLog % sumGradLog;
            }

            baseline_den = arma::repmat(sumGradLog % sumGradLog, 1, dp);

            // ********************** //

        }

        // *** REINFORCE BASE CORE *** //

        // compute the gradients
        for (int i = 0; i < dp; ++i)
        {

            arma::vec baseline_J(dp, arma::fill::zeros);
            arma::vec baseline_J_num_i = baseline_J_num.col(i);
            arma::vec baseline_den_i = baseline_den.col(i);
            arma::uvec nonZeros = arma::find(baseline_den_i != 0);

            baseline_J(nonZeros) = baseline_J_num_i(nonZeros) / baseline_den_i(nonZeros);


            for (int ep = 0; ep < nbEpisodes; ++ep)
            {
                gradient_J.col(i) += (return_J_ObjEp(i, ep) - baseline_J) % sumGradLog_CompEp.col(ep);
            }
        }

        // ********************** //

        // compute mean values
        gradient_J /= nbEpisodes;

        return gradient_J;
    }

    arma::mat GpomdpGradient()
    {
        int dp  = policy.getParametersSize();
        arma::vec sumGradLog(dp), localg;
        arma::mat gradient_J(dp, dp, arma::fill::zeros);

        int nbEpisodes = data.size();
        for (int i = 0; i < nbEpisodes; ++i)
        {
            //core setup
            int nbSteps = data[i].size();


            // *** GPOMDP CORE *** //
            sumGradLog.zeros();
            double df = 1.0;
            // ********************** //

            //iterate the episode
            for (int t = 0; t < nbSteps; ++t)
            {
                Transition<ActionC, StateC>& tr = data[i][t];

                // *** GPOMDP CORE *** //
                localg = policy.difflog(tr.x, tr.u);
                sumGradLog += localg;
                arma::vec creward = T*phi(vectorize(tr.x, tr.u, tr.xn));

                // compute the gradients
                for (int i = 0; i < dp; ++i)
                {
                    gradient_J.col(i) += df * creward(i) * sumGradLog;
                }
                // ********************** //

                df *= gamma;

                if (tr.xn.isAbsorbing())
                {
                    assert(nbSteps == t+1);
                    break;
                }
            }

        }
        // compute mean values
        gradient_J /= nbEpisodes;

        return gradient_J;
    }

    arma::mat GpomdpBaseGradient()
    {
        int dp  = policy.getParametersSize();
        int nbEpisodes = data.size();

        int maxSteps = 0;
        for (int i = 0; i < nbEpisodes; ++i)
        {
            int nbSteps = data[i].size();
            if (maxSteps < nbSteps)
                maxSteps = nbSteps;
        }

        arma::vec sumGradLog(dp), localg;
        arma::mat gradient_J(dp, dp, arma::fill::zeros);

        arma::cube baseline_J_num(dp, maxSteps, dp, arma::fill::zeros);
        arma::cube baseline_den(dp, maxSteps, dp, arma::fill::zeros);
        arma::cube reward_J_ObjEpStep(nbEpisodes, maxSteps, dp);
        arma::cube sumGradLog_CompEpStep(dp,nbEpisodes, maxSteps);
        arma::vec  maxsteps_Ep(nbEpisodes);

        for (int ep = 0; ep < nbEpisodes; ++ep)
        {
            //core setup
            int nbSteps = data[ep].size();


            // *** GPOMDP CORE *** //
            sumGradLog.zeros();
            double df = 1.0;
            // ********************** //

            //iterate the episode
            for (int t = 0; t < nbSteps; ++t)
            {
                Transition<ActionC, StateC>& tr = data[ep][t];

                // *** GPOMDP CORE *** //
                localg = policy.difflog(tr.x, tr.u);
                sumGradLog += localg;

                // store the basic elements used to compute the gradients
                arma::vec creward = T * phi(vectorize(tr.x, tr.u, tr.xn));
                reward_J_ObjEpStep.tube(ep,t) = df * creward;


                sumGradLog_CompEpStep.slice(t).col(ep) = sumGradLog;


                // compute the baselines
                for(unsigned int i = 0; i < dp; i++)
                {
                    baseline_J_num.slice(i).col(t) += df * creward(i) * sumGradLog % sumGradLog;
                    baseline_den.slice(i).col(t) += sumGradLog % sumGradLog;
                }

                // ********************** //

                df *= gamma;

                if (tr.xn.isAbsorbing())
                {
                    assert(nbSteps == t+1);
                    break;
                }
            }

            // store the actual length of the current episode (<= maxsteps)
            maxsteps_Ep(ep) = nbSteps;

        }

        // *** GPOMDP BASE CORE *** //

        // compute the gradients

        for (int ep = 0; ep < nbEpisodes; ++ep)
        {
            for (int t = 0; t < maxsteps_Ep(ep); ++t)
            {
                for(unsigned int i = 0; i < dp; i++)
                {
                    arma::vec baseline_J(dp, arma::fill::zeros);
                    arma::vec baseline_J_num_t = baseline_J_num.slice(i).col(t);
                    arma::vec baseline_den_t = baseline_den.slice(i).col(t);
                    arma::uvec nonZeros = arma::find(baseline_den_t != 0);

                    baseline_J(nonZeros) = baseline_J_num_t(nonZeros) / baseline_den_t(nonZeros);
                    gradient_J.col(i) += (reward_J_ObjEpStep(ep,t, i) - baseline_J) % sumGradLog_CompEpStep.slice(t).col(ep);
                }
            }
        }
        // ************************ //

        // compute mean values
        gradient_J /= nbEpisodes;

        return gradient_J;
    }

    arma::mat NaturalGradient()
    {
        int dp  = policy.getParametersSize();
        arma::vec localg;
        arma::mat fisher(dp,dp, arma::fill::zeros);

        int nbEpisodes = data.size();
        for (int i = 0; i < nbEpisodes; ++i)
        {
            //core setup
            int nbSteps = data[i].size();

            //iterate the episode
            for (int t = 0; t < nbSteps; ++t)
            {
                Transition<ActionC, StateC>& tr = data[i][t];

                // *** eNAC CORE *** //
                localg = policy.difflog(tr.x, tr.u);
                fisher += localg * localg.t();
                // ********************** //

                if (tr.xn.isAbsorbing())
                {
                    assert(nbSteps == t+1);
                    break;
                }
            }

        }
        fisher /= nbEpisodes;

        arma::mat gradient;
        if (atype == IRLGradType::NATR)
        {
            gradient = ReinforceGradient();
        }
        else if (atype == IRLGradType::NATRB)
        {
            gradient = ReinforceBaseGradient();
        }
        else if (atype == IRLGradType::NATG)
        {
            gradient = GpomdpGradient();
        }
        else if (atype == IRLGradType::NATGB)
        {
            gradient = GpomdpBaseGradient();
        }

        int rnk = arma::rank(fisher);
        arma::mat nat_grad(dp, dp);
        for(unsigned int i = 0; i < dp; i++)
        {
            if (rnk == fisher.n_rows)
            {
                nat_grad.col(i) = arma::solve(fisher, gradient.col(i));
            }
            else
            {
                std::cerr << "WARNING: Fisher Matrix is lower rank (rank = " << rnk << ")!!! Should be " << fisher.n_rows << std::endl;

                arma::mat H = arma::pinv(fisher);
                nat_grad.col(i) = H * gradient.col(i);
            }
        }

        return nat_grad;
    }

    arma::mat ENACGradient()
    {
        int dp  = policy.getParametersSize();
        arma::vec localg;
        arma::vec Rew(dp);
        arma::mat g(dp+1, dp, arma::fill::zeros);
        arma::vec phi(dp+1);
        arma::mat fisher(dp+1,dp+1, arma::fill::zeros);
        //        double Jpol = 0.0;

        int nbEpisodes = data.size();
        for (int i = 0; i < nbEpisodes; ++i)
        {
            //core setup
            int nbSteps = data[i].size();


            // *** eNAC CORE *** //
            double df = 1.0;
            Rew.zeros();
            phi.zeros();
            //    #ifdef AUGMENTED
            phi(dp) = 1.0;
            //    #endif
            // ********************** //

            //iterate the episode
            for (int t = 0; t < nbSteps; ++t)
            {
                Transition<ActionC, StateC>& tr = data[i][t];
                //            std::cout << tr.x << " " << tr.u << " " << tr.xn << " " << tr.r[0] << std::endl;

                // *** eNAC CORE *** //
                localg = policy.difflog(tr.x, tr.u);
                Rew += df * T * this->phi(vectorize(tr.x, tr.u, tr.xn));

                //Construct basis functions
                for (unsigned int i = 0; i < dp; ++i)
                    phi[i] += df * localg[i];
                // ********************** //

                df *= gamma;

                if (tr.xn.isAbsorbing())
                {
                    assert(nbSteps == t+1);
                    break;
                }
            }

            fisher += phi * phi.t();

            for(unsigned int i = 0; i < dp; i++)
                g.col(i) += Rew(i) * phi;

        }


        arma::mat nat_grad(dp+1, dp);
        int rnk = arma::rank(fisher);

        for(unsigned int i = 0; i < dp; i++)
        {
            if (rnk == fisher.n_rows)
            {
                nat_grad.col(i) = arma::solve(fisher, g.col(i));
            }
            else
            {
                std::cerr << "WARNING: Fisher Matrix is lower rank (rank = " << rnk << ")!!! Should be " << fisher.n_rows << std::endl;

                arma::mat H = arma::pinv(fisher);
                nat_grad.col(i) = H * g.col(i);
            }
        }

        return nat_grad.rows(0,dp-1);
    }

    arma::vec ENACBaseGradient(BasisFunction& rewardf)
    {
        int dp  = policy.getParametersSize();
        arma::vec localg;
        double Rew;
        arma::vec g(dp+1, arma::fill::zeros), eligibility(dp+1, arma::fill::zeros), phi(dp+1);
        arma::mat fisher(dp+1,dp+1, arma::fill::zeros);
        double Jpol = 0.0;

        int nbEpisodes = data.size();
        for (int i = 0; i < nbEpisodes; ++i)
        {
            //core setup
            int nbSteps = data[i].size();


            // *** eNAC CORE *** //
            double df = 1.0;
            Rew = 0.0;
            phi.zeros();
            //    #ifdef AUGMENTED
            phi(dp) = 1.0;
            //    #endif
            // ********************** //

            //iterate the episode
            for (int t = 0; t < nbSteps; ++t)
            {
                Transition<ActionC, StateC>& tr = data[i][t];
                //            std::cout << tr.x << " " << tr.u << " " << tr.xn << " " << tr.r[0] << std::endl;

                // *** eNAC CORE *** //
                localg = policy.difflog(tr.x, tr.u);
                double creward = rewardf(tr.x, tr.u, tr.xn);
                Rew += df * creward;

                //Construct basis functions
                for (unsigned int i = 0; i < dp; ++i)
                    phi[i] += df * localg[i];
                // ********************** //

                df *= gamma;

                if (tr.xn.isAbsorbing())
                {
                    assert(nbSteps == t+1);
                    break;
                }
            }

            Jpol += Rew;
            fisher += phi * phi.t();
            g += Rew * phi;
            eligibility += phi;

        }

        // compute mean value
        fisher /= nbEpisodes;
        g /= nbEpisodes;
        eligibility /= nbEpisodes;
        Jpol /= nbEpisodes;

        arma::vec nat_grad;
        int rnk = arma::rank(fisher);
        //        std::cout << rnk << " " << fisher << std::endl;
        if (rnk == fisher.n_rows)
        {
            arma::mat tmp = arma::solve(nbEpisodes * fisher - eligibility * eligibility.t(), eligibility);
            arma::mat Q = (1 + eligibility.t() * tmp) / nbEpisodes;
            arma::mat b = Q * (Jpol - eligibility.t() * arma::solve(fisher, g));
            arma::vec grad = g - eligibility * b;
            nat_grad = arma::solve(fisher, grad);
        }
        else
        {
            std::cerr << "WARNING: Fisher Matrix is lower rank (rank = " << rnk << ")!!! Should be " << fisher.n_rows << std::endl;

            arma::mat H = arma::pinv(fisher);
            arma::mat b = (1 + eligibility.t() * arma::pinv(nbEpisodes * fisher - eligibility * eligibility.t()) * eligibility)
                          * (Jpol - eligibility.t() * H * g)/ nbEpisodes;
            arma::vec grad = g - eligibility * b;
            nat_grad = H * (grad);
        }

        return nat_grad.rows(0,dp-1);
    }

    virtual void run() override
    {
        int dp = policy.getParametersSize();
        int dr = phi.rows();

        arma::mat phiBar = data.computeEpisodeFeatureExpectation(phi, gamma);

        //Principal component analysis
        if(phiBar.n_rows > dp)
        {
            PrincipalComponentAnalysis pca;
            pca.createFeatures(phiBar, dp, false);
            T = pca.getTransformation();
        }
        else
        {
            T = arma::mat(dp, dp, arma::fill::eye);
        }

        std::cout << "T" << std::endl << T << std::endl;


        arma::mat A = computeGradient();
        A.save("/tmp/ReLe/grad.log", arma::raw_ascii);

        std::cout << "Grads: \n" << A << std::endl;

        ////////////////////////////////////////////////
        /// PRE-PROCESSING
        ////////////////////////////////////////////////
        arma::mat Ared;         //reduced gradient matrix
        arma::uvec nonZeroIdx;  //nonzero elements of the reward weights
        int rnkG = rank(A);
        if ( rnkG < dr && A.n_rows >= A.n_cols )
        {
            //TODO FIX this....
            // select linearly independent columns
            arma::mat Asub;
            nonZeroIdx = rref(A, Asub);
            std::cout << "Asub: \n" << Asub << std::endl;
            std::cout << "idx: \n" << nonZeroIdx.t()  << std::endl;
            Ared = A.cols(nonZeroIdx);
            assert(rank(Ared) == Ared.n_cols);
        }
        else
        {
            Ared = A;
            nonZeroIdx.set_size(A.n_cols);
            std::iota (std::begin(nonZeroIdx), std::end(nonZeroIdx), 0);
        }


        if(nonZeroIdx.n_elem == 1)
        {
            arma::vec tmp = arma::zeros(dp);
            tmp(nonZeroIdx).ones();
            weights = T.t()*tmp;
            return;
        }


        Ared.save("/tmp/ReLe/gradRed.log", arma::raw_ascii);

        ////////////////////////////////////////////////
        /// GRAM MATRIX AND NORMAL
        ////////////////////////////////////////////////
        arma::mat gramMatrix = Ared.t() * Ared;
        unsigned int lastr = gramMatrix.n_rows;
        arma::mat X = gramMatrix.rows(0, lastr-2) - arma::repmat(gramMatrix.row(lastr-1), lastr-1, 1);
        X.save("/tmp/ReLe/GM.log", arma::raw_ascii);


        // COMPUTE NULL SPACE
        Y = null(X);
        std::cout << "Y: " << Y << std::endl;
        Y.save("/tmp/ReLe/NullS.log", arma::raw_ascii);


        // prepare the output
        weights = T.t()*Y;


        //Normalize (L1) weights
        weights /= arma::sum(weights);

    }

private:
	arma::mat computeGradient()
	{
		arma::mat A;

		switch (atype)
		{
			case R:
				A = ReinforceGradient();
				break;
			case RB:
				A = ReinforceBaseGradient();
				break;
			case G:
				A = GpomdpGradient();
				break;
			case GB:
				A = GpomdpBaseGradient();
				break;
			case NATR:
			case NATRB:
			case NATG:
			case NATGB:
				A = NaturalGradient();
				break;
			case ENAC:
				A = ENACGradient();
				break;
			default:
				std::cerr << "PGIRL ERROR" << std::endl;
				abort();
				break;
		}

		return A;
	}



protected:
    Dataset<ActionC,StateC>& data;
    DifferentiablePolicy<ActionC,StateC>& policy;
    DenseFeatures& phi;
    double gamma;
    arma::vec weights;
    IRLGradType atype;
    arma::mat Y;

    arma::mat T;

};


} //end namespace


#endif /* PGIRL_H_ */
