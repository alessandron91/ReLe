/*
 * GP-ContinuousFQIPolicyTest.cpp
 *
 *  Created on: 05 set 2016
 *      Author: alessandro
 */




/*
 * rele,
 *
 *
 * Copyright (C) 2016 Davide Tateo
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

#include "rele/core/Core.h"
#include "rele/environments/MountainCar.h"
#include "rele/core/PolicyEvalAgent.h"
#include "rele/policy/q_policy/e_Greedy.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/environments/ContinuousSwingPendulum.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/regressors/others/GaussianProcess.h"
#include "rele/utils/FileManager.h"
#include "rele/policy/q_policy/GP_Policy.h"
#include <iostream>

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[])
{

    ContinuousMDP* mdp;


    mdp = new ContinuousSwingPendulum;

    unsigned int stateDim = mdp->getSettings().stateDimensionality;
    unsigned int nActions = mdp->getSettings().actionsNumber+1;

    FileManager fm("testFqi");
    fm.createDir();
    fm.cleanDir();

    BasisFunctions bfs = IdentityBasis::generate(stateDim);
    DenseFeatures phi(bfs);

    unsigned int nExperiments =10;
    unsigned int nEpisodes = 50;
    unsigned int nBins=100;
    std::string pathCartella="/home/alessandro/Scrivania/AAAI/RELE AAAI/";
    std::string alg="wfqi";
    int nTestExp=36;
    arma::mat Jtest(nExperiments,nTestExp,arma::fill::zeros);




        for(unsigned int e = 0; e < nExperiments; e++)
        {
            //std::string testFileName = env + "-" + alg + "_" + std::to_string(e) + "Data.log";

            std::string loadPath = pathCartella + std::to_string(nEpisodes) + "Episodes/" + alg + "/";

            arma::mat hParams;
            hParams.load(loadPath + "hParams_" + std::to_string(e) + ".mat", arma::raw_ascii);
            arma::vec lengthScale = hParams.col(0);
            arma::vec rawSignalSigma = hParams.col(1);
            std::cout<<rawSignalSigma(arma::find(rawSignalSigma != arma::datum::inf))<<std::endl;

            double signalSigma = arma::as_scalar(rawSignalSigma(arma::find(rawSignalSigma != arma::datum::inf)));


            GaussianProcess *gps;

                arma::mat alpha;
                arma::mat activeSetMat;
                alpha.load(loadPath + "alphas_" + std::to_string(e) + ".mat", arma::raw_ascii);
                activeSetMat.load(loadPath + "activeSetVectors_" + std::to_string(e) + ".mat", arma::raw_ascii);

                arma::cube activeSet(activeSetMat.n_rows, activeSetMat.n_cols / nActions, nActions);
                std::cout<<"dopo"<<std::endl;

                for(unsigned int a = 0; a < nActions; a++)
                {
                	std::cout<<nActions<<std::endl;

                    activeSet.slice(a) = activeSetMat.cols(arma::span(stateDim * a, stateDim * a + stateDim - 1));
                }

                for(unsigned int i = 0; i < alpha.n_cols; i++)
                {
                    arma::vec rawAlphaVec = alpha.col(i);
                    arma::vec alphaVec = rawAlphaVec(arma::find(rawAlphaVec != arma::datum::inf));

                    GaussianProcess* gp = new GaussianProcess(phi);
                    gp->getHyperParameters().lengthScale = lengthScale;
                    gp->getHyperParameters().signalSigma = signalSigma;

                    gp->setAlpha(alphaVec);

                    arma::mat rawActiveSetMat = activeSet.slice(i);
                    arma::vec temp = rawActiveSetMat.col(0);
                    arma::mat activeSetMat = rawActiveSetMat.rows(arma::find(temp != arma::datum::inf));

                    gp->setFeatures(activeSetMat);

                }




            GP_Policy policy(gps,nBins);
            PolicyEvalAgent<DenseAction, DenseState> agent(policy);



            	arma::vec discRewards=arma::vec(nTestExp,arma::fill::zeros);
            	unsigned int counter=0;

            	for(int testExp=0;testExp<nTestExp;testExp++)
            	{





                std::string testFileNamePend =  alg + "_nepisodes:"+std::to_string(nEpisodes) + std::to_string(e)+"_"+std::to_string(testExp)+ "_Data.log";
                ContinuousSwingPendulum testMdp((2*M_PI/nTestExp) *testExp,false);
                auto&& core = buildCore(testMdp, agent);
                core.getSettings().episodeLength = 100;
                core.getSettings().loggerStrategy =
                new WriteStrategy<DenseAction, DenseState>(fm.addPath(testFileNamePend));

                core.runTestEpisode();

                arma::mat testEpisodes;

                testEpisodes.load(fm.addPath(testFileNamePend), arma::csv_ascii);

                arma::vec rewards=testEpisodes.col(5);
                //arma::vec cumulativeRewards=arma::cumsum(rewards);
                //Js(e,a)=arma::sum(rewards)/core.getSettings().episodeLength;


                Jtest(e,testExp)=arma::sum(rewards/core.getSettings().episodeLength);


               // std::cout<<"alg: "<<alg<<" exp: "<<e<<" test:"<< testExp<<std::endl;

               /* discRewards(counter) = 0;
                for(unsigned int k = 1; k < testEpisodes.n_rows - 1; k++)
                discRewards(counter) += pow(mdp->getSettings().gamma, k - 1) * rewards(k);

                Jtest(e,testExp,a)=discRewards(counter);
                counter++;*/


                std::cout<<"alg: "<<alg<<" exp: "<<e<<" test:"<< testExp<<std::endl;



            	}


                // calcolo reward medio per esperimento

        }





    delete mdp;


   // std::string saveFileName = "Js-" + ".txt";
    //Js.save(savePath + saveFileName, arma::raw_ascii);
    Jtest.save(pathCartella + std::to_string(nEpisodes) + "Episodes/FqiRewards.txt", arma::raw_ascii);



}
