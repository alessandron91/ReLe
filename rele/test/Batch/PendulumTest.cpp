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
#include <cmath>

using namespace std;
using namespace ReLe;
using namespace arma;

int main(int argc, char *argv[]) {

	std::string alg = "fqi";
	//std::string alg = "dfqi";


	std::vector<GaussianProcess> gp;
	std::vector<GaussianProcess> gpA; //vettori di n regressori per n azioni
	std::vector<GaussianProcess>gpB;

	if (alg == "fqi" || alg == "wfqi")
	{

		arma::mat alphaMatrix;
		arma::cube activeSetCube;
		alphaMatrix.load(" ");
		activeSetCube.load(" ");


		for (int j = 0; j < alphaMatrix.n_cols; j++) //iterate on actions
			{
				arma::vec alphaVec;
				int infIndex = alphaMatrix.n_rows - 1;
				bool foundInfIndex=false;
				for (int i = 0; i < alphaMatrix.n_rows && foundInfIndex==false; i++) //iterate on samples
				{
					if (alphaMatrix(i, j) == arma::datum::inf)
					{
						infIndex = i;
						foundInfIndex=true;
					}
				}
				alphaVec = alphaMatrix.col(j).subvec(0, infIndex);
				gp[j].setAlpha(alphaVec);
			}

		for (int z = 0; z < activeSetCube.n_slices; z++) //// iterate on actions -- Scritto separatamente per maggiore leggibilità ( per ottimizzare codice e spazio potrebbe essere inserito nel for precedente cambiando l'ordine di azione e stato
			{
				arma::mat activeSetMat;
				int infIndex = activeSetCube.n_rows - 1;
				bool foundInfIndex=false;

				for (int i = 0; i < activeSetCube.n_rows && foundInfIndex==false; i++)
				{
					if (activeSetCube(i, 0, z) == arma::datum::inf) //  indice 0 corretto?
					{
						infIndex = i;
						foundInfIndex=true;
					}
				}
				activeSetMat = activeSetCube.slice(z).submat(0, 0, infIndex,activeSetCube.n_cols);
				gp[z].setFeatures(activeSetMat);

			}
		}
	else if(alg=="dfqi")
	{
		arma::cube alphaCube;
		arma::cube activeSetCubeA;
		arma::cube activeSetCubeB;
		alphaCube.load(" ");
		activeSetCubeA.load(" ");
		activeSetCubeB.load(" ");

		for(int j=0;j<alphaCube.n_cols;j++)
		{
			arma::mat alphaMatA;
			arma::mat alphaMatB;
			arma::vec infIndexes=arma::vec(2);
			std::vector<bool> foundIndex;
			foundIndex[0]=false;
			foundIndex[1]=false;
			infIndexes(0) = alphaCube.n_rows - 1;
			infIndexes(1) = alphaCube.n_rows - 1;

			for(int i=0; i<alphaCube.n_rows ; i++)
			{
				for(int z=0;z<alphaCube.n_slices && foundIndex[z]==false;z++)
				{
					if(alphaCube(i,j,z)==arma::datum::inf)
					{
						foundIndex[z]=true;
						infIndexes(z)=i;
					}
				}
			}
			alphaMatA=alphaCube.slice(0).submat(0,0,infIndexes(0),alphaCube.n_cols);
			alphaMatB=alphaCube.slice(1).submat(0,0,infIndexes(1),alphaCube.n_cols);
			gpA[j].setAlpha(alphaMatA.col(j));
			gpB[j].setAlpha(alphaMatB.col(j));

		///
			arma::mat activeSetMatA;
			arma::mat activeSetMatB;
			int infActiveSetIndexA=activeSetCubeA.n_rows-1;
			bool foundActiveSetInfIndexA=false;
			for(int i=0; i < activeSetCubeA.n_rows && foundActiveSetInfIndexA==false; i++)
			{
				if(activeSetCubeA(i,0,j)==arma::datum::inf )
				{
					infActiveSetIndexA=i;
					foundActiveSetInfIndexA=true;
				}

			}

			int infActiveSetIndexB=activeSetCubeB.n_rows-1;
			bool foundActiveSetInfIndexB=false;
			for(int i=0; i < activeSetCubeB.n_rows && foundActiveSetInfIndexB==false; i++)
			{
				if(activeSetCubeB(i,0,j)==arma::datum::inf)
				{
					infActiveSetIndexB=i;
					foundActiveSetInfIndexB=true;
				}
			}

			activeSetMatA=activeSetCubeA.slice(0).submat(0,0,infActiveSetIndexA,infActiveSetIndexA);
			activeSetMatB=activeSetCubeB.slice(0).submat(0,0,infActiveSetIndexB,infActiveSetIndexB);
			gpA[j].setFeatures(activeSetMatA);
			gpB[j].setFeatures(activeSetMatB);

			}

		}









}

