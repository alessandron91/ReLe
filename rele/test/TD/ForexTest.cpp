#include "rele/core/Core.h"
#include "rele/core/PolicyEvalAgent.h"
#include "rele/policy/q_policy/e_Greedy.h"
#include "rele/algorithms/td/WQ-Learning.h"
#include "rele/algorithms/td/Q-Learning.h"

#include "rele/utils/FileManager.h"
#include "rele/core/FiniteMDP.h"
#include "rele/algorithms/td/DoubleQ-Learning.h"
#include "rele/environments/Forex.h"
#include "rele/algorithms/td/GQ-Learning.h"
#include "rele/policy/q_policy/Decreasing_e_Greedy.h"
#include "rele/algorithms/step_rules/StateActionLearningRate.h"
#include <armadillo>
#include <iostream>
#include <time.h>
//#include "rele/policy/q_policy/WPolicy.h"
#include <string>     // std::string, std::to_string
#define N_STATES 1944


using namespace std;
using namespace ReLe;

int main(int argc, char *argv[])
{


	clock_t time;
    bool acquireData = true;

    FileManager fm("Forex", "FQI");
    fm.createDir();
    fm.cleanDir();
    Dataset<FiniteAction, FiniteState> data;

	   ofstream out1("/home/alessandro/Scrivania/stati.txt");
  if (acquireData)
  {

	  int nSamples=1;
	  int nEpisodes=20;
	  std::string alg="provagq1000";

	  arma::mat profits=arma::mat(nEpisodes,nSamples);
	  arma::mat profitTest=arma::mat(nEpisodes,nSamples);
	  //arma::mat profitTestLearning=arma::mat(nEpisodes,nSamples);


	  arma::mat dataset;
	  dataset.load("/home/alessandro/Scrivania/Tesi/Data/TestReLe/dataset/trainingSet_d_GBP_9808_28.txt");
	  arma::mat testset;
	  testset.load("/home/alessandro/Scrivania/Tesi/Data/TestReLe/dataset/testSet_d_GBP_9808_28.txt");

	 //arma::cube cumulativeProfits=arma::cube(dataset.n_rows,nEpisodes,nSamples);

 	  arma::mat rewardsMatrix=arma::mat(nSamples,testset.n_rows);
 	  arma::mat actionsMatrix=arma::mat(nSamples,testset.n_rows);
 	  arma::mat statesMatrix=arma::mat(nSamples,testset.n_rows);

 	  time=clock();
	  for(int sample=0;sample<nSamples;sample++)
	  {


        Forex&& mdp =Forex(dataset);
       // arma::mat a=mdp.getStateIndexer();
        Decreasing_e_Greedy policy(1944);
        //WPolicy policy(1944,3);

       // ForexIndPolicy policy(dataset);

       StateActionLearningRate lrate=StateActionLearningRate(1,1,0.0,N_STATES,3);
       GQ_Learning agent(policy,lrate,20);
       //WQ_Learning agent(policy,lrate);



       //agent.setAlpha(0.1);

        auto&& core = buildCore(mdp, agent);

        core.getSettings().episodeLength = dataset.n_rows-1;
        CollectorStrategy<FiniteAction, FiniteState> collection;
         core.getSettings().loggerStrategy = &collection;

 	 	//ofstream rewFile("/home/alessandro/Scrivania/Tesi/Data/TestReLe/Rewards_Q_Daily",ios::out);


         for(unsigned int episode = 0; episode < nEpisodes; episode++)
        {

            cout << endl << "### Starting episode " << episode << "###" << endl;
            cout << endl << "Sample: " << sample << "###" << endl;

            /*policy.setCurrentInd(i);
            policy.setCurrentT(0);*/
            mdp.setT(0);
            mdp.setProfit(0);
        	core.runEpisode();
            cout<<"T: "<<mdp.getT()<<endl;
            cout<<"PROFIT::::"<<mdp.getProfit()<<endl;
            profits(episode,sample)=mdp.getProfit();

            ////   TEST   /////
            Forex&& mdpTest =Forex(testset);


           // WPolicy trainedPolicy=policy;
            e_Greedy trainedPolicy;
            trainedPolicy.setQ(policy.getQ());
            trainedPolicy.setEpsilon(0);

            //Decreasing_e_Greedy trainedPolicy = policy;
            //trainedPolicy.setTestMode(true);
            //trainedPolicy.setEpsilon(0);
            PolicyEvalAgent<FiniteAction,FiniteState> testAgent(trainedPolicy);
            auto&& coreTest = buildCore(mdpTest, testAgent);
            coreTest.getSettings().episodeLength = testset.n_rows-1;

            cout <<"TEST : "<<sample<<endl;
            mdpTest.setT(0);
            mdpTest.setProfit(0);
            coreTest.runTestEpisode();
            cout<<"PROFIT::::"<<mdpTest.getProfit()<<endl;
            profitTest(episode,sample)=mdpTest.getProfit();



            if(episode==nEpisodes-1){
            rewardsMatrix.row(sample)=mdpTest.getRewards();
            actionsMatrix.row(sample)=mdpTest.getActions();
            statesMatrix.row(sample)=mdpTest.getStates();

            }




        }

         //rewFile<<mdp.getRewards()<<endl;

/// HO AGGIUNTO UNO STATO A uN INDICATO




        if(sample==(int)nSamples/2)
        {
        	ofstream pfileMtrain("/home/alessandro/Scrivania/Tesi/Data/TestReLe/Risultati/Nuovi 11-04/Profitto/"+alg+"/TrainHalf",ios::out);
        	pfileMtrain<<profits<<endl;
        	ofstream pfileMtrain2("/home/alessandro/Scrivania/Tesi/Data/TestReLe/Risultati/Nuovi 11-04/Profitto/"+alg+"/TestHalf",ios::out);
        	pfileMtrain2<<profitTest<<endl;
        }

        std::string samp=std::to_string(sample);

	 	//ofstream policyFile("/home/alessandro/Scrivania/Tesi/Data/TestReLe/Risultati/Nuovi_newDataset_gamma0.8/Policy/"+alg+"/Policy_"+samp,ios::out);
	 	//policyFile<<agent.getQ()<<endl;


	  }
	 	ofstream pfile1("/home/alessandro/Scrivania/Tesi/Data/TestReLe/Risultati/Nuovi 11-04/Profitto/"+alg+"/Train",ios::out);
	 	ofstream pfile2("/home/alessandro/Scrivania/Tesi/Data/TestReLe/Risultati/Nuovi 11-04/Profitto/"+alg+"/Test",ios::out);


	 	pfile1<<profits<<endl;
	 	pfile2<<profitTest<<endl;



	 	ofstream actionsMatrixFile("/home/alessandro/Scrivania/Tesi/Data/TestReLe/Risultati/Nuovi 11-04/Rewards/"+alg+"/ActionsMatrix",ios::out);
	 	ofstream statesMatrixFile("/home/alessandro/Scrivania/Tesi/Data/TestReLe/Risultati/Nuovi 11-04/Rewards/"+alg+"/StatesMatrix",ios::out);

	 	ofstream rewardsMatrixFile("/home/alessandro/Scrivania/Tesi/Data/TestReLe/Risultati/Nuovi 11-04/Rewards/"+alg+"/RewardsMatrix",ios::out);
	 	rewardsMatrixFile<<rewardsMatrix<<endl;
	 	actionsMatrixFile<<actionsMatrix<<endl;
	 	statesMatrixFile<<statesMatrix<<endl;
  }

  cout<<"executionTimeGQ1000"<<endl;
  cout<<endl<<(clock()-time) /(double)CLOCKS_PER_SEC<<endl;
}
