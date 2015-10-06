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

#ifndef INCLUDE_RELE_IRL_FEATURE_SELECTION_PRINCIPALCOMPONENTANALYSIS_H_
#define INCLUDE_RELE_IRL_FEATURE_SELECTION_PRINCIPALCOMPONENTANALYSIS_H_

class PrincipalComponentAnalysis
{
public:
    void createFeatures(const arma::mat& features, unsigned int k, bool useCorrelation = true)
    {
        arma::mat Sigma;

        if(useCorrelation)
        {
            //compute correlation of features
            Sigma = arma::cor(features.t());
            Sigma = (Sigma + Sigma.t())/2;
        }
        else
        {
            //compute covariance of features
            Sigma = arma::cov(features.t());
        }

        std::cout << "Sigma" << std::endl << Sigma << std::endl;

        //compute eigenvalues and eigenvectors
        arma::vec s;
        arma::mat A;

        arma::eig_sym(s, A, Sigma);
        s = arma::sort(s, "descend");
        A = fliplr(A);

        T = A.cols(0, k-1).t();

        std::cout << "eigvals = " << s.rows(0, k-1).t() << std::endl;

        arma::mat normalizedFeatures = features;
        normalizedFeatures.each_col() -= arma::sum(features, 1)/features.n_cols;


        newFeatures = T*normalizedFeatures;

    }

    inline arma::mat getTransformation()
    {
        return T;
    }

    inline arma::mat getNewFeatures()
    {
        return newFeatures;
    }

private:
    arma::mat newFeatures;
    arma::mat T;

};


#endif /* INCLUDE_RELE_IRL_FEATURE_SELECTION_PRINCIPALCOMPONENTANALYSIS_H_ */
