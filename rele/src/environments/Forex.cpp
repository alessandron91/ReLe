#include "rele/environments/Forex.h"
#include <armadillo>
#define PRICE 7
#define N_INDICATORS 7
#define ACTION 7
#define SPREAD 0.0002
#define N_STATES 1944
#include <iostream>
#include <ostream>

using namespace std;

namespace ReLe
{


	Forex::Forex(const arma::mat& dataset)
	{
		this->dataset=dataset;
		currentPrice=this->dataset(0,PRICE);/// VERIFICAREEEEEEEEEE

		currentPrice=this->dataset(0,PRICE);/// VERIFICAREEEEEEEEEE
		prevAction=0;

		 profit=0;
		 stateIndexer=arma::mat(N_STATES,N_INDICATORS+1,arma::fill::zeros);
		 indicatorSignal=arma::vec(N_INDICATORS);
		 currentState=arma::vec(N_INDICATORS+1);
		 setSettings(N_STATES,3);
		 t=0;
		 rewards=arma::rowvec(dataset.n_rows);
		 actions=arma::rowvec(dataset.n_rows);
		 states=arma::rowvec(dataset.n_rows);

		 InitStatesIndexer();

	}

	Forex::~Forex(){}

	void Forex::step (const FiniteAction& action, FiniteState& nextState,
            Reward& reward)
	{


		///// INSERIRE LIMITE LUNGHEZZA DATASET
		double c=0;
		int act=action.getActionN();




		double diff=currentPrice-prevPrice;



		if(act!=0 && prevAction!=act)
		{
			c=SPREAD;
		}


		if(prevAction==0)
		{
			reward[0]=0-c;
		}

		if(prevAction==1)
		{
			reward[0]=diff-c;
		}

		if(prevAction==2)
		{
			reward[0]=-diff-c;
		}
		nextState=getNextState(act);
		profit=profit+reward[0];
		//cout<<"Reward:::"<<reward<<endl;
		//cout<<"PROFIT::::"<<profit<<endl
		if(t<=dataset.n_rows-1)
			rewards(t)=(double)reward[0];

		if(t==dataset.n_rows)  /// DA CANCELLARE
			{//profit=0;
			t=0;
			}

		if(testMode==true)
		{
			actions(t)=action.getActionN();
			states(t)=nextState.getStateN();
		}





	}

	void Forex::getInitialState(FiniteState& state)
	{
		state=getNextState(0);// in Nextstate t++

	}



	int Forex::getNextState(int action)
	{

		for(int i=0;i<N_INDICATORS;i++)
		{
			indicatorSignal(i)=dataset(t,i);
			currentState(i)=indicatorSignal(i);
		}

		currentState(N_INDICATORS)=action;
		prevPrice=currentPrice;
		currentPrice=dataset(t,PRICE);
		prevAction=action;

		t++;


		int index= getRowMatrixIndex(currentState,stateIndexer);

		/*currentState.print();
		cout<<index<<endl;*/
		return index;


	}


	int nextIndex(arma::vec a,int j)
	{

		if(j!=a.n_elem-1)
		{
		return j+1;
		}
		else
		{
		return 0;
		}


	}

	void Forex::writeVecElemAwithTperiodInVecStateIndexer(arma::vec a,int col,int t)
	{
		int j=0;
		int el=a(j);

		for(int i=0;i<stateIndexer.n_rows;i++)
		{

			stateIndexer(i,col)=(int)el;

			if(i%t==0 && i!=0)
			{

				j=nextIndex(a,j);

				el=a(j);
			}





		}


	}

	void Forex::InitStatesIndexer()
	{
		arma::vec macd={1,2};
		arma::vec mom={1,2};
		arma::vec maco={1,2};
		arma::vec rsi={0,1,2};
		arma::vec cci={0,1,2};
		arma::vec pcb={0,1,2};
		arma::vec stoc={0,1,2};
		arma::vec actions={0,1,2};

		macd.print();
		mom.print();



		int nrows=N_STATES;


		int t=nrows/macd.n_elem;
		writeVecElemAwithTperiodInVecStateIndexer(macd,0,t);
		t=t/mom.n_elem;
		//stateIndexer.print();

		writeVecElemAwithTperiodInVecStateIndexer(mom,1,t);
		//stateIndexer.print();
		t=t/maco.n_elem;
		writeVecElemAwithTperiodInVecStateIndexer(maco,2,t);
		t=t/rsi.n_elem;
		writeVecElemAwithTperiodInVecStateIndexer(rsi,3,t);
		t=t/cci.n_elem;
		writeVecElemAwithTperiodInVecStateIndexer(cci,4,t);
		t=t/pcb.n_elem;

		writeVecElemAwithTperiodInVecStateIndexer(pcb,5,t);
		t=t/stoc.n_elem;
		writeVecElemAwithTperiodInVecStateIndexer(stoc,6,t);
		t=t/actions.n_elem;

		writeVecElemAwithTperiodInVecStateIndexer(actions,7,t);

		ofstream stati("Scrivania/stati.txt");
		stati<<stateIndexer<<endl;








	}



	bool equal(arma::vec a,arma::rowvec b)
	{
		for(int i=0;i<a.n_elem;i++)
		{
			if(a(i)!=b(i))
			{
				return false;
			}
		}
		return true;


	}


	int Forex::getRowMatrixIndex(arma::vec v,arma::mat b)
	{
		for(int i=0;i<b.n_rows;i++)
		{
			if(equal(v,b.row(i)))
				return i;
		}
		return -1;

	}


	void Forex::setSettings(int stateDim,int actionDim)
	{
		   EnvironmentSettings& task = this->getWritableSettings();
		        task.isFiniteHorizon = true;
		        //task.horizon = horizon;
		        task.gamma = 0.8;
		        //task.isAverageReward = false;
		        task.isEpisodic = true;
		        task.statesNumber = stateDim;
		        task.actionsNumber=actionDim;
		        task.stateDimensionality = 1;
		        task.rewardDimensionality = 1;
	}

}
