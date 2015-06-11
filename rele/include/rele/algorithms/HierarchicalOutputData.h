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

#ifndef INCLUDE_RELE_ALGORITHMS_HIERARCHICALOUTPUTDATA_H_
#define INCLUDE_RELE_ALGORITHMS_HIERARCHICALOUTPUTDATA_H_

#include "Basics.h"

namespace ReLe
{

class HierarchicalOutputData : virtual public AgentOutputData
{

public: //FIXME implement this
    HierarchicalOutputData()
    {
        traces.resize(1);
    }

    virtual void writeData(std::ostream& os)
    {

    }

    virtual void writeDecoratedData(std::ostream& os)
    {

    }

    void addNewTrace()
    {
        traces.resize(traces.size() + 1);
    }

    std::vector<std::vector<int>> traces;
};

}


#endif /* INCLUDE_RELE_ALGORITHMS_HIERARCHICALOUTPUTDATA_H_ */
