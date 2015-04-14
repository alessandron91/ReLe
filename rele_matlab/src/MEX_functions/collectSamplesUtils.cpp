#include "collectSamplesUtils.h"

#include <DifferentiableNormals.h>
#include <Core.h>
#include <PolicyEvalAgent.h>
#include <parametric/differentiable/NormalPolicy.h>
#include <parametric/differentiable/GibbsPolicy.h>
#include <BasisFunctions.h>
#include <basis/PolynomialFunction.h>
#include <basis/ConditionBasedFunction.h>
#include <LQR.h>
#include <NLS.h>
#include <DeepSeaTreasure.h>

using namespace std;
using namespace ReLe;
using namespace arma;

///////////////////////////////////////////////////////////// USED FOR DEEP SEA TREASURE

class deep_2state_identity: public BasisFunction
{
    double operator()(const arma::vec& input)
    {
        return ((input[0] == 1) && (input[1] == 1))?1:0;
    }
    void writeOnStream(std::ostream& out)
    {
        out << "deep_2state" << endl;
    }
    void readFromStream(std::istream& in) {}
};

class deep_state_identity: public BasisFunction
{
    double operator()(const arma::vec& input)
    {
        return (input[0] == 1)?1:0;
    }
    void writeOnStream(std::ostream& out)
    {
        out << "deep_state" << endl;
    }
    void readFromStream(std::istream& in) {}
};
/////////////////////////////////////////////////////////////


#define SAMPLES_GATHERING(ActionC, StateC) \
        PolicyEvalAgent\
        <ActionC,StateC> agent(policy);\
        ReLe::Core<ActionC, StateC> oncore(mdp, agent);\
        MatlabCollectorStrategy<ActionC, StateC> strat = MatlabCollectorStrategy<ActionC, StateC>(gamma);\
        oncore.getSettings().loggerStrategy = &strat;\
        int horiz = mdp.getSettings().horizon;\
        oncore.getSettings().episodeLenght = horiz;\
        int nbTrajectories = nbEpisodes;\
        for (int n = 0; n < nbTrajectories; ++n)\
            oncore.runTestEpisode();\
        std::vector<MatlabCollectorStrategy<ActionC,StateC>::MatlabEpisode>& data = strat.data;\
        int ds = data[0].dx;\
        int da = data[0].du;\
        int dr = data[0].dr;\
        MEX_DATA_FIELDS(fieldnames);\
        SAMPLES = mxCreateStructMatrix(data.size(), 1, 5, fieldnames);\
        DRETURN = mxCreateDoubleMatrix(dr, data.size(), mxREAL);\
        double* Jptr = mxGetPr(DRETURN);\
        for (int i = 0, ie = data.size(); i < ie; ++i)\
        {\
            int steps = data[i].steps;\
            mxArray* state_vector      = mxCreateDoubleMatrix(ds, steps, mxREAL);\
            memcpy(mxGetPr(state_vector), data[i].states.memptr(), sizeof(double)*ds*steps);\
            mxArray* nextstate_vector  = mxCreateDoubleMatrix(ds, steps, mxREAL);\
            memcpy(mxGetPr(nextstate_vector), data[i].nextstates.memptr(), sizeof(double)*ds*steps);\
            mxArray* action_vector     = mxCreateDoubleMatrix(da, steps, mxREAL);\
            memcpy(mxGetPr(action_vector), data[i].actions.memptr(), sizeof(double)*da*steps);\
            mxArray* reward_vector     = mxCreateDoubleMatrix(dr, steps, mxREAL);\
            memcpy(mxGetPr(reward_vector), data[i].rewards.memptr(), sizeof(double)*dr*steps);\
            mxArray* absorb_vector     = mxCreateNumericMatrix(1, steps, mxINT32_CLASS, mxREAL);\
            memcpy(mxGetPr(absorb_vector), data[i].absorb.memptr(), sizeof(signed char)*steps);\
            mxSetFieldByNumber(SAMPLES, i, 0, state_vector);\
            mxSetFieldByNumber(SAMPLES, i, 1, action_vector);\
            mxSetFieldByNumber(SAMPLES, i, 2, reward_vector);\
            mxSetFieldByNumber(SAMPLES, i, 3, nextstate_vector);\
            mxSetFieldByNumber(SAMPLES, i, 4, absorb_vector);\
            for (int oo = 0; oo < dr; ++oo)\
                Jptr[i*dr+oo] = data[i].Jvalue[oo];\
        }


#define IN_DOMAIN     prhs[0]
#define IN_NBEPISODES prhs[1]
#define IN_MAXSTEPS   prhs[2]
#define IN_GAMMA      prhs[3]

#define SAMPLES  plhs[0]
#define DRETURN  plhs[1]

void
CollectSamplesInContinuousMDP(
    int nlhs, mxArray *plhs[], /* Output variables */
    int nrhs, const mxArray *prhs[] /* Input variables */
)
{

    char* domain_settings = mxArrayToString(IN_DOMAIN);
    int nbEpisodes = mxGetScalar(IN_NBEPISODES);
    int maxSteps   = mxGetScalar(IN_MAXSTEPS);
    double gamma   = mxGetScalar(IN_GAMMA);
    mexPrintf("%f\n", gamma);

    if (strcmp(domain_settings, "lqr") == 0)
    {
        LQR mdp(1,1);
        PolynomialFunction* pf = new PolynomialFunction(1,1);
        DenseBasisVector basis;
        basis.push_back(pf);
        LinearApproximator regressor(mdp.getSettings().continuosStateDim, basis);
        NormalPolicy policy(0.1, &regressor);

        SAMPLES_GATHERING(DenseAction, DenseState)
//         //////////////////////////////////////////////// METTERE IN UNA DEFINE
//
//         PolicyEvalAgent
//         <DenseAction,DenseState> agent(policy);
//
//         ReLe::Core<DenseAction, DenseState> oncore(mdp, agent);
//         MatlabCollectorStrategy<DenseAction, DenseState> strat = MatlabCollectorStrategy<DenseAction, DenseState>(gamma);
//         oncore.getSettings().loggerStrategy = &strat;
//
//         int horiz = mdp.getSettings().horizon;
//         oncore.getSettings().episodeLenght = horiz;
//
//         int nbTrajectories = nbEpisodes;
//         for (int n = 0; n < nbTrajectories; ++n)
//             oncore.runTestEpisode();
//
//         std::vector<MatlabCollectorStrategy<DenseAction,DenseState>::MatlabEpisode>& data = strat.data;
//
//         int ds = data[0].dx;
//         int da = data[0].du;
//         int dr = data[0].dr;
//
//         MEX_DATA_FIELDS(fieldnames);
//         // return samples
//         SAMPLES = mxCreateStructMatrix(data.size(), 1, 5, fieldnames);
//         DRETURN = mxCreateDoubleMatrix(dr, data.size(), mxREAL);
//         double* Jptr = mxGetPr(DRETURN);
//
//         for (int i = 0, ie = data.size(); i < ie; ++i)
//         {
//             int steps = data[i].steps;
//
//             mxArray* state_vector      = mxCreateDoubleMatrix(ds, steps, mxREAL);
//             memcpy(mxGetPr(state_vector), data[i].states.memptr(), sizeof(double)*ds*steps);
//
//
//             mxArray* nextstate_vector  = mxCreateDoubleMatrix(ds, steps, mxREAL);
//             memcpy(mxGetPr(nextstate_vector), data[i].nextstates.memptr(), sizeof(double)*ds*steps);
//
//             mxArray* action_vector     = mxCreateDoubleMatrix(da, steps, mxREAL);
//             memcpy(mxGetPr(action_vector), data[i].actions.memptr(), sizeof(double)*da*steps);
//
//             mxArray* reward_vector     = mxCreateDoubleMatrix(dr, steps, mxREAL);
//             memcpy(mxGetPr(reward_vector), data[i].rewards.memptr(), sizeof(double)*dr*steps);
//
//
//             mxArray* absorb_vector     = mxCreateNumericMatrix(1, steps, mxINT32_CLASS, mxREAL);
//             memcpy(mxGetPr(absorb_vector), data[i].absorb.memptr(), sizeof(signed char)*steps);
//
//
//             mxSetFieldByNumber(SAMPLES, i, 0, state_vector);
//             mxSetFieldByNumber(SAMPLES, i, 1, action_vector);
//             mxSetFieldByNumber(SAMPLES, i, 2, reward_vector);
//             mxSetFieldByNumber(SAMPLES, i, 3, nextstate_vector);
//             mxSetFieldByNumber(SAMPLES, i, 4, absorb_vector);
//
//             for (int oo = 0; oo < dr; ++oo)
//                 Jptr[i*dr+oo] = data[i].Jvalue[oo];
//         }
//         ////////////////////////////////////////////////
    }
    else if (strcmp(domain_settings, "nls") == 0)
    {
        NLS mdp;
        int dim = mdp.getSettings().continuosStateDim;

        //--- define policy
        DenseBasisVector basis;
        basis.generatePolynomialBasisFunctions(1,dim);
        delete basis.at(0);
        basis.erase(basis.begin());
        LinearApproximator meanRegressor(dim, basis);

        DenseBasisVector stdBasis;
        stdBasis.generatePolynomialBasisFunctions(1,dim);
        delete stdBasis.at(0);
        stdBasis.erase(stdBasis.begin());
        LinearApproximator stdRegressor(dim, stdBasis);
        arma::vec stdWeights(stdRegressor.getParametersSize());
        stdWeights.fill(0.5);
        stdRegressor.setParameters(stdWeights);


        NormalStateDependantStddevPolicy policy(&meanRegressor, &stdRegressor);

        arma::vec pp(2);
        pp(0) = -0.4;
        pp(1) = 0.4;
        meanRegressor.setParameters(pp);

        SAMPLES_GATHERING(DenseAction, DenseState)
    }
    else if (strcmp(domain_settings, "dam") == 0)
    {
        Dam mdp;

        GaussianRbf* gf1 = new GaussianRbf(0,50);
        GaussianRbf* gf2 = new GaussianRbf(50,20);
        GaussianRbf* gf3 = new GaussianRbf(120,40);
        GaussianRbf* gf4 = new GaussianRbf(160,50);
        DenseBasisVector basis;
        basis.push_back(gf1);
        basis.push_back(gf2);
        basis.push_back(gf3);
        basis.push_back(gf4);
        cout << basis << endl;
        LinearApproximator regressor(mdp.getSettings().continuosStateDim, basis);
        vec p(5);
        p(0) = 50;
        p(1) = -50;
        p(2) = 0;
        p(3) = 0;
        p(4) = 50;
        MVNLogisticPolicy policy(&regressor, 50*ones<vec>(p.n_elem), p);

        SAMPLES_GATHERING(DenseAction, DenseState)
    }
    else
    {
        mexErrMsgTxt("CollectSamplesInContinuousMDP: Unknown settings!\n");
    }



    mxFree(domain_settings);
}

void
CollectSamplesInDenseMDP(
    int nlhs, mxArray *plhs[], /* Output variables */
    int nrhs, const mxArray *prhs[] /* Input variables */
)
{

    char* domain_settings = mxArrayToString(IN_DOMAIN);
    int nbEpisodes = mxGetScalar(IN_NBEPISODES);
    int maxSteps   = mxGetScalar(IN_MAXSTEPS);
    double gamma   = mxGetScalar(IN_GAMMA);

    if (strcmp(domain_settings, "deep") == 0)
    {
        DeepSeaTreasure mdp;
        vector<FiniteAction> actions;
        for (int i = 0; i < mdp.getSettings().finiteActionDim; ++i)
            actions.push_back(FiniteAction(i));

        //--- policy setup
        PolynomialFunction* pf0 = new PolynomialFunction(2,0);
        vector<unsigned int> dim = {0,1};
        vector<unsigned int> deg = {1,0};
        PolynomialFunction* pfs1 = new PolynomialFunction(dim,deg);
        deg = {0,1};
        PolynomialFunction* pfs2 = new PolynomialFunction(dim,deg);
        deg = {1,1};
        PolynomialFunction* pfs1s2 = new PolynomialFunction(dim, deg);
        deep_2state_identity* d2si = new deep_2state_identity();
        deep_state_identity* dsi = new deep_state_identity();

        DenseBasisVector basis;
        for (int i = 0; i < actions.size() -1; ++i)
        {
            basis.push_back(new AndConditionBasisFunction(pf0,2,i));
            basis.push_back(new AndConditionBasisFunction(pfs1,2,i));
            basis.push_back(new AndConditionBasisFunction(pfs2,2,i));
            basis.push_back(new AndConditionBasisFunction(pfs1s2,2,i));
            basis.push_back(new AndConditionBasisFunction(d2si,2,i));
            basis.push_back(new AndConditionBasisFunction(dsi,2,i));
        }

        LinearApproximator regressor(mdp.getSettings().continuosStateDim + 1, basis);
        ParametricGibbsPolicy<DenseState> policy(actions, &regressor, 1);

        SAMPLES_GATHERING(FiniteAction, DenseState)
    }
    else
    {
        mexErrMsgTxt("CollectSamplesInDenseMDP: Unknown settings!\n");
    }



    mxFree(domain_settings);
}
