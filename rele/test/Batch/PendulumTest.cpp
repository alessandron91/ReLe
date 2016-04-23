#include "rele/core/BatchCore.h"
#include "rele/core/PolicyEvalAgent.h"
#include "rele/policy/q_policy/e_Greedy.h"
#include "rele/algorithms/batch/td/DoubleFQI.h"
#include "rele/algorithms/batch/td/W-FQI.h"
#include "rele/approximators/features/DenseFeatures.h"
#include "rele/environments/SwingPendulum.h"
#include "rele/approximators/basis/IdentityBasis.h"
#include "rele/approximators/regressors/others/GaussianProcess.h"
#include "rele/approximators/regressors/others/LinearApproximator.h"
#include "rele/utils/FileManager.h"

#include <iostream>

using namespace std;
using namespace ReLe;
using namespace arma;



int main(int argc, char *argv[])
{



	arma::mat alphaMatrix;
	arma::cube activeSetCube;
	alphaMatrix.load(" ");
	activeSetCube.load(" ");
	std::vector<GaussianProcess> gp;
	for(int i=0;i<alphaMatrix.n_cols;i++) //iterate on actions
	{
		arma::vec alphaVec;
		int infIndex=alphaMatrix.n_rows-1;
		for(int j=0;j<alphaMatrix.n_rows;j++)
		{
			if (alphaMatrix(i,j)==inf)
			{
				infIndex=j;

			}
		}
		alphaVec=alphaMatrix.col(i).subvec(0,infIndex);
		gp[i].setAlpha(alphaVec);
	}

	for(int i=0;i<activeSetCube.n_slices;i++)    //// Scritto separatamente per maggiore leggibilità ( per ottimizzare codice e spazio potrebbe essere inserito nel for precedente cambiando l'ordine di azione e stato
	{
		arma::mat activeSetMat;
		int infIndex=activeSetCube.n_rows-1;
		for(int j=0;j<activeSetCube.n_rows;j++)
		{
			if(activeSetCube(j,0,i)==inf) //  indice 0 corretto?
			{
				infIndex=j;
			}
		}
		activeSetMat=activeSetCube.slice(i).submat(0,0,infIndex,activeSetCube.n_cols);
		BatchData<arma::Mat,true> dataset=activeSetMat; /// togliere assegnamento se funzion

		gp[i].setFeatures(dataset);





	}
















}


