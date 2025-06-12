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

int32_t get_VelocitySet_vector_count(VelocitySet velocity_set)
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

    std::cout << "[LBM Error] get_VelocitySet_vector_count() is called but given velocity_set is not supported" << std::endl;
}

std::string get_VelocitySet_to_macro(VelocitySet velocity_set)
{
    switch (velocity_set) {
    case D2Q9:
        return "D2Q9";
    case D3Q15:
        return "D3Q15";
    case D3Q19:
        return "D3Q19";
    case D3Q27:
        return "D3Q27";
    }

    std::cout << "[LBM Error] get_VelocitySet_to_macro() is called but given velocity_set is not supported" << std::endl;
}

std::vector<glm::vec4> get_velosity_vectors(VelocitySet velocity_set)
{
    switch (velocity_set) {
    case D2Q9:
        return{
            glm::vec4( 0,  0,  0,  4.0 / 9),
            glm::vec4( 1,  0,  0,  1.0 / 9),
            glm::vec4(-1,  0,  0,  1.0 / 9),
            glm::vec4( 0,  1,  0,  1.0 / 9),
            glm::vec4( 0, -1,  0,  1.0 / 9),
            glm::vec4( 1,  1,  0,  1.0 / 36),
            glm::vec4(-1, -1,  0,  1.0 / 36),
            glm::vec4(-1,  1,  0,  1.0 / 36),
            glm::vec4( 1, -1,  0,  1.0 / 36)
        };
    case D3Q15:
        return{
            glm::vec4( 0,  0,  0,  2.0 / 9),
            glm::vec4( 1,  0,  0,  1.0 / 9),
            glm::vec4(-1,  0,  0,  1.0 / 9),
            glm::vec4( 0,  1,  0,  1.0 / 9),
            glm::vec4( 0, -1,  0,  1.0 / 9),
            glm::vec4( 0,  0,  1,  1.0 / 9),
            glm::vec4( 0,  0, -1,  1.0 / 9),
            glm::vec4( 1,  1,  1,  1.0 / 72),
            glm::vec4(-1, -1, -1,  1.0 / 72),
            glm::vec4( 1,  1, -1,  1.0 / 72),
            glm::vec4(-1, -1,  1,  1.0 / 72),
            glm::vec4( 1, -1,  1,  1.0 / 72),
            glm::vec4(-1,  1, -1,  1.0 / 72),
            glm::vec4(-1,  1,  1,  1.0 / 72),
            glm::vec4( 1, -1, -1,  1.0 / 72)
        };
    case D3Q19:
        return{
            glm::vec4( 0,  0,  0,  1.0 / 3),
            glm::vec4( 1,  0,  0,  1.0 / 18),
            glm::vec4(-1,  0,  0,  1.0 / 18),
            glm::vec4( 0,  1,  0,  1.0 / 18),
            glm::vec4( 0, -1,  0,  1.0 / 18),
            glm::vec4( 0,  0,  1,  1.0 / 18),
            glm::vec4( 0,  0, -1,  1.0 / 18),
            glm::vec4( 1,  1,  0,  1.0 / 36),
            glm::vec4(-1, -1,  0,  1.0 / 36),
            glm::vec4( 1, -1,  0,  1.0 / 36),
            glm::vec4(-1,  1,  0,  1.0 / 36),
            glm::vec4( 1,  0,  1,  1.0 / 36),
            glm::vec4(-1,  0, -1,  1.0 / 36),
            glm::vec4( 1,  0, -1,  1.0 / 36),
            glm::vec4(-1,  0,  1,  1.0 / 36),
            glm::vec4( 0,  1,  1,  1.0 / 36),
            glm::vec4( 0, -1, -1,  1.0 / 36),
            glm::vec4( 0,  1, -1,  1.0 / 36),
            glm::vec4( 0, -1,  1,  1.0 / 36),
        };
    case D3Q27:
        return{
            glm::vec4( 0,  0,  0,  8.0 / 27),
            glm::vec4( 1,  0,  0,  2.0 / 27),
            glm::vec4(-1,  0,  0,  2.0 / 27),
            glm::vec4( 0,  1,  0,  2.0 / 27),
            glm::vec4( 0, -1,  0,  2.0 / 27),
            glm::vec4( 0,  0,  1,  2.0 / 27),
            glm::vec4( 0,  0, -1,  2.0 / 27),
            glm::vec4( 1,  1,  0,  1.0 / 54),
            glm::vec4(-1, -1,  0,  1.0 / 54),
            glm::vec4( 1,  0,  1,  1.0 / 54),
            glm::vec4(-1,  0, -1,  1.0 / 54),
            glm::vec4( 0,  1,  1,  1.0 / 54),
            glm::vec4( 0, -1, -1,  1.0 / 54),
            glm::vec4( 1, -1,  0,  1.0 / 54),
            glm::vec4(-1,  1,  0,  1.0 / 54),
            glm::vec4( 1,  0, -1,  1.0 / 54),
            glm::vec4(-1,  0,  1,  1.0 / 54),
            glm::vec4( 0,  1, -1,  1.0 / 54),
            glm::vec4( 0, -1,  1,  1.0 / 54),
            glm::vec4( 1,  1,  1,  1.0 / 216),
            glm::vec4(-1, -1, -1,  1.0 / 216),
            glm::vec4( 1,  1, -1,  1.0 / 216),
            glm::vec4(-1, -1,  1,  1.0 / 216),
            glm::vec4( 1, -1,  1,  1.0 / 216),
            glm::vec4(-1,  1, -1,  1.0 / 216),
            glm::vec4(-1,  1,  1,  1.0 / 216),
            glm::vec4( 1, -1, -1,  1.0 / 216)
        };
    }

    std::cout << "[LBM Error] get_velosity_vectors() is called but given velocity_set is not supported" << std::endl;

}

int32_t get_SimplifiedVelocitySet_dimention(SimplifiedVelocitySet velocity_set)
{
    switch (velocity_set) {
    case D2Q5:
        return 2;
    case D3Q7:
        return 3;
    }

    std::cout << "[LBM Error] get_SimplifiedVelocitySet_dimention() is called but given velocity_set is not supported" << std::endl;
}

int32_t get_SimplifiedVelocitySet_vector_count(SimplifiedVelocitySet velocity_set)
{
    switch (velocity_set) {
    case D2Q5:
        return 5;
    case D3Q7:
        return 7;
    }

    std::cout << "[LBM Error] get_SimplifiedVelocitySet_vector_count() is called but given velocity_set is not supported" << std::endl;
}

std::string get_SimplifiedVelocitySet_to_macro(SimplifiedVelocitySet velocity_set)
{
    switch (velocity_set) {
    case D2Q5:
        return "D2Q5";
    case D3Q7:
        return "D3Q7";
    }

    std::cout << "[LBM Error] get_SimplifiedVelocitySet_to_macro() is called but given velocity_set is not supported" << std::endl;
}

std::vector<glm::vec4> get_velosity_vectors(SimplifiedVelocitySet velocity_set)
{
     switch (velocity_set) {
    case D2Q5:
        return{
            glm::vec4( 0,  0,  0,  1.0 / 3),
            glm::vec4( 1,  0,  0,  1.0 / 6),
            glm::vec4(-1,  0,  0,  1.0 / 6),
            glm::vec4( 0,  1,  0,  1.0 / 6),
            glm::vec4( 0, -1,  0,  1.0 / 6),
        };
    case D3Q7:
        return{
            glm::vec4( 0,  0,  0,  1.0 / 4),
            glm::vec4( 1,  0,  0,  1.0 / 8),
            glm::vec4(-1,  0,  0,  1.0 / 8),
            glm::vec4( 0,  1,  0,  1.0 / 8),
            glm::vec4( 0, -1,  0,  1.0 / 8),
            glm::vec4( 0,  0,  1,  1.0 / 8),
            glm::vec4( 0,  0, -1,  1.0 / 8),
        };
    }

    std::cout << "[LBM Error] get_velosity_vectors() is called but given velocity_set is not supported" << std::endl;
}
