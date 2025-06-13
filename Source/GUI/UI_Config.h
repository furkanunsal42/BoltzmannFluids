#ifndef UI_CONFIG_H
#define UI_CONFIG_H

namespace {
    constexpr unsigned int constexpr_max(unsigned int a, unsigned int b) {
        return (a > b) ? a : b;
    }
}

// Panel sizes
constexpr unsigned int main_window_width    = 1280;
constexpr unsigned int main_window_height   = 720;
constexpr unsigned int right_panel_width    = 335;
constexpr unsigned int left_panel_width     = 335;
constexpr unsigned int middle_panel_width   = constexpr_max(500, (main_window_width - left_panel_width - right_panel_width));

constexpr float GRAVITY_MIN = -100.f;
constexpr float GRAVITY_MAX =  100.f;

constexpr float INITIAL_VELOCITY_MIN = -100.f;
constexpr float INITIAL_VELOCITY_MAX =  100.f;

constexpr float INITIAL_TEMPRATURE_MIN =  -273.15f;
constexpr float INITIAL_TEMPRATURE_MAX =   10000.f;

#endif // UI_CONFIG_H
