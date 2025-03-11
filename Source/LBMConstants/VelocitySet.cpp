#include "VelocitySet.h"

#include <iostream>

int32_t get_VelocitySet_dimention(VelocitySet velocity_set)
{
    switch (velocity_set) {
    case D2Q9:
        return 2;
    case D3Q15:
        return 3;
    case D3Q19:
        return 3;
    case D3Q27:
        return 3;
    }

    std::cout << "[LBM Error] get_VelocitySet_dimention() is called but given velocity_set is not supported" << std::endl;
}

int32_t get_VelocitySet_velocity_count(VelocitySet velocity_set)
{
    switch (velocity_set) {
    case D2Q9:
        return 9;
    case D3Q15:
        return 15;
    case D3Q19:
        return 19;
    case D3Q27:
        return 27;
    }

    std::cout << "[LBM Error] get_VelocitySet_velocity_count() is called but given velocity_set is not supported" << std::endl;
}
