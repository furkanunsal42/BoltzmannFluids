#ifndef UI_CONFIG_H
#define UI_CONFIG_H

#define CONFIG_INCLUDE_LIMITS
#ifdef CONFIG_INCLUDE_LIMITS
#include <limits>
#endif

namespace {
    constexpr unsigned int constexpr_max(unsigned int a, unsigned int b) {
        return (a > b) ? a : b;
    }
}

// Panel sizes
constexpr unsigned int main_window_width    = 1280;
constexpr unsigned int main_window_height   = 720;

constexpr unsigned int right_panel_width    = 520;
constexpr unsigned int left_panel_width     = 300;
constexpr unsigned int middle_panel_width   = constexpr_max(500, (main_window_width - left_panel_width - right_panel_width));

constexpr unsigned int render_box_height            = 500;
constexpr unsigned int timeline_height              = 80;
constexpr unsigned int application_output_height    = 150;

// Initial Conditions
constexpr float DECIMAL_COUNT = 10;
constexpr float GRAVITY_MIN = std::numeric_limits<float>::lowest();
constexpr float GRAVITY_MAX = std::numeric_limits<float>::max();

constexpr float INITIAL_VELOCITY_MIN = std::numeric_limits<float>::lowest();
constexpr float INITIAL_VELOCITY_MAX = std::numeric_limits<float>::max();

constexpr float INITIAL_TEMPRATURE_MIN = std::numeric_limits<float>::lowest();
constexpr float INITIAL_TEMPRATURE_MAX = std::numeric_limits<float>::max();

// Item Properties
constexpr float POSITION_MIN = std::numeric_limits<float>::lowest();
constexpr float POSITION_MAX = std::numeric_limits<float>::max();

#endif // UI_CONFIG_H
