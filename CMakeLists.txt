cmake_minimum_required(VERSION 3.19)

project(BoltzmannFluids_UI
VERSION 1.0.0
LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)

# GraphicsCortex

set(CORTEX_DEPENDENCIES_DIR GraphicsCortex/Dependencies/)

file(GLOB_RECURSE GraphicsCortex_Sources       
    GraphicsCortex/Source/*.cpp 
    GraphicsCortex/Source/*.h 
    GraphicsCortex/Dependencies/stb_image/*.cpp 
    GraphicsCortex/Dependencies/stb_image/*.h 
    GraphicsCortex/Dependencies/physx/snippetcommon/*.cpp 
    GraphicsCortex/Dependencies/physx/snippetcommon/*.j 
    GraphicsCortex/Dependencies/imgui/*.h 
    GraphicsCortex/Dependencies/imgui/*.cpp 
    )
    
file(COPY "GraphicsCortex/Source/GLSL/" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/GraphicsCortex/Source/GLSL")

add_library(GraphicsCortex STATIC ${GraphicsCortex_Sources})
target_compile_options(GraphicsCortex PUBLIC /sdl /openmp)

target_include_directories(GraphicsCortex PUBLIC 
    GraphicsCortex/Source/Graphics/RenderPasses
    GraphicsCortex/Source/System
    GraphicsCortex/Source/Physics
    GraphicsCortex/Source/Graphics/Windowing
    GraphicsCortex/Source/Graphics/Textures
    GraphicsCortex/Source/Graphics/Text
    GraphicsCortex/Source/Graphics/Shaders
    GraphicsCortex/Source/Graphics/Rendering
    GraphicsCortex/Source/Graphics/Meshes
    GraphicsCortex/Source/Graphics/Math
    GraphicsCortex/Source/Graphics/Lights
    GraphicsCortex/Source/Graphics/GUI
    GraphicsCortex/Source/Graphics/Defaults
    GraphicsCortex/Source/Graphics/Buffers
    GraphicsCortex/Source/Graphics
    GraphicsCortex/Source
    ${CORTEX_DEPENDENCIES_DIR}libtiff/include
    ${CORTEX_DEPENDENCIES_DIR}assimp/include
    ${CORTEX_DEPENDENCIES_DIR}stb_image
    ${CORTEX_DEPENDENCIES_DIR}GLEW/include
    ${CORTEX_DEPENDENCIES_DIR}imgui
    ${CORTEX_DEPENDENCIES_DIR}glm
    ${CORTEX_DEPENDENCIES_DIR}GLFWx64/include
    ${CORTEX_DEPENDENCIES_DIR}physx/include
    ${CORTEX_DEPENDENCIES_DIR}physx/snippetutils
    ${CORTEX_DEPENDENCIES_DIR}physx/snippetcommon
    ${CORTEX_DEPENDENCIES_DIR}hiredis/include
    ${CORTEX_DEPENDENCIES_DIR}freetype/include
    )


target_link_directories(GraphicsCortex PUBLIC
    ${CORTEX_DEPENDENCIES_DIR}libtiff/lib
    ${CORTEX_DEPENDENCIES_DIR}assimp/lib
    ${CORTEX_DEPENDENCIES_DIR}stb_image
    ${CORTEX_DEPENDENCIES_DIR}glm
    ${CORTEX_DEPENDENCIES_DIR}GLEW/lib/Release/x64
    ${CORTEX_DEPENDENCIES_DIR}GLFWx64/lib-vc2022
    ${CORTEX_DEPENDENCIES_DIR}physx/lib_checked
    ${CORTEX_DEPENDENCIES_DIR}hiredis/libx64
    ${CORTEX_DEPENDENCIES_DIR}freetype
)

target_link_libraries(GraphicsCortex PUBLIC
    tiff.lib
    freetype.lib
    hiredis.lib
    SnippetUtils_static_64.lib
    SnippetRender_static_64.lib
    SimulationController_static_64.lib
    SceneQuery_static_64.lib
    PhysXVehicle_static_64.lib
    PhysXTask_static_64.lib
    PhysXPvdSDK_static_64.lib
    PhysXFoundation_64.lib
    PhysXExtensions_static_64.lib
    PhysXCooking_64.lib
    PhysXCommon_64.lib
    PhysXCharacterKinematic_static_64.lib
    PhysX_64.lib
    LowLevelDynamics_static_64.lib
    LowLevelAABB_static_64.lib
    LowLevel_static_64.lib
    assimp-vc143-mt.lib
    zlibstatic.lib
    IrrXML.lib
    opengl32.lib
    glew32s.lib
    glfw3.lib
    kernel32.lib
    user32.lib
    gdi32.lib
    winspool.lib
    comdlg32.lib
    advapi32.lib
    shell32.lib
    ole32.lib
    oleaut32.lib
    uuid.lib
    odbc32.lib
    odbccp32.lib
)

add_compile_definitions(
    GLEW_STATIC
    WIN32
    NDEBUG
    _CONSOLE
)

# BoltzmannFluids

file(GLOB_RECURSE BoltzmannFluids_Sources       
    BoltzmannFluids/Source/*.cpp 
    BoltzmannFluids/Source/*.h 
    )

file(COPY "BoltzmannFluids/Source/GLSL/" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/BoltzmannFluids/Source/GLSL")

#add_library(BoltzmannFluids STATIC ${BoltzmannFluids_Sources} ${BoltzmannFluids_Data})
add_library(BoltzmannFluids STATIC ${BoltzmannFluids_Sources})

target_include_directories(BoltzmannFluids PUBLIC     
    BoltzmannFluids/Source
    )


target_link_directories(BoltzmannFluids PUBLIC

)

target_link_libraries(BoltzmannFluids PUBLIC GraphicsCortex)

#BoltzmannFLuids_Standalone

file(GLOB_RECURSE BoltzmannFluids_Standalone_Sources       
    BoltzmannFluids_Standalone/Source/*.cpp 
    BoltzmannFluids_Standalone/Source/*.h 
    BoltzmannFluids_Standalone/Mains/*.cpp
    BoltzmannFluids_Standalone/Mains/*.h
    )

    file(COPY "BoltzmannFluids_Standalone/Source/GLSL/" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/BoltzmannFluids_Standalone/Source/GLSL")

add_executable(BoltzmannFluids_Standalone ${BoltzmannFluids_Standalone_Sources})

target_include_directories(BoltzmannFluids_Standalone PUBLIC     
    BoltzmannFluids_Standalone/Source
    BoltzmannFluids_Standalone/Mains
    )

target_link_libraries(BoltzmannFluids_Standalone PUBLIC BoltzmannFluids)

# BoltzmannFluids_UI

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} COMPONENTS Widgets OpenGLWidgets REQUIRED)

file(COPY "BoltzmannFluids_UI/Models/" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/BoltzmannFluids_UI/Models/")
file(COPY "BoltzmannFluids_UI/Source/GLSL/" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/BoltzmannFluids_UI/Source/GLSL/")

file(GLOB_RECURSE PROJECT_SOURCES
  BoltzmannFluids_UI/*.cpp
  BoltzmannFluids_UI/*.h
  *.qrc
  )

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(BoltzmannFluids_UI
      MANUAL_FINALIZATION
      ${PROJECT_SOURCES}
    )
# Define target properties for Android with Qt 6 as:
#    set_property(TARGET BoltzmannFluids_UI APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
#                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
# For more information, see https://doc.qt.io/qt-6/qt-add-executable.html#target-creation
else()
    if(ANDROID)
        add_library(BoltzmannFluids_UI SHARED
            ${PROJECT_SOURCES}
        )
# Define properties for Android with Qt 5 after find_package() calls as:
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else()
        add_executable(BoltzmannFluids_UI
            ${PROJECT_SOURCES}
        )
    endif()
endif()

target_link_libraries(BoltzmannFluids_UI PRIVATE Qt${QT_VERSION_MAJOR}::Widgets Qt${QT_VERSION_MAJOR}::OpenGLWidgets)

# Qt for iOS sets MACOSX_BUNDLE_GUI_IDENTIFIER automatically since Qt 6.1.
# If you are developing for iOS or macOS you should consider setting an
# explicit, fixed bundle identifier manually though.
if(${QT_VERSION} VERSION_LESS 6.1.0)
  set(BUNDLE_ID_OPTION MACOSX_BUNDLE_GUI_IDENTIFIER com.example.BoltzmannFluids_UI)
endif()
set_target_properties(BoltzmannFluids_UI PROPERTIES
    ${BUNDLE_ID_OPTION}
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

include(GNUInstallDirs)
install(TARGETS BoltzmannFluids_UI
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(BoltzmannFluids_UI)
endif()

target_link_libraries(BoltzmannFluids_UI PUBLIC BoltzmannFluids)

add_custom_command(TARGET BoltzmannFluids_UI POST_BUILD
    COMMAND Qt6::windeployqt
    ARGS $<TARGET_FILE:BoltzmannFluids_UI>
)
