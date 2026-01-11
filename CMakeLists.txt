cmake_minimum_required(VERSION 3.28)
project(DataScrapper)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# --- Global RPATH Settings (These are fine at the top) ---
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "/lib/x86_64-linux-gnu")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Dependencies ---
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libs/libtorch")
find_package(Torch REQUIRED)
find_package(CURL REQUIRED)
find_package(fmt REQUIRED)
find_package(mongocxx REQUIRED)
find_package(bsoncxx REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
file(GLOB_RECURSE UTIL_SOURCES  
    "libs/*.cpp"
    "libs/*.h" 
    "libs/*.hpp"
)

# --- Library ---
add_library(Utils STATIC ${UTIL_SOURCES})

if(APPLE)
    message(STATUS "Configuring Utils with Metal/MPS support for macOS")
    # 1. Find the necessary Apple Frameworks
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(QUARTZ_CORE_FRAMEWORK QuartzCore REQUIRED)
    find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)
    find_library(MPS_GRAPH_FRAMEWORK MetalPerformanceShadersGraph REQUIRED)

    # 2. Add them to your Utils library
    target_link_libraries(Utils PUBLIC 
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${QUARTZ_CORE_FRAMEWORK}
        ${MPS_FRAMEWORK}
        ${MPS_GRAPH_FRAMEWORK}
    )

    # 3. Enable Objective-C++ if you plan to use .mm files for Metal
    enable_language(OBJCXX)
endif()
target_include_directories(Utils PUBLIC include)
target_link_libraries(Utils PUBLIC 
    CURL::libcurl 
    mongo::mongocxx_shared
    mongo::bsoncxx_shared
    fmt::fmt
    ${TORCH_LIBRARIES}
)

# --- Executable ---
add_executable(a.out src/main.cpp)
target_link_libraries(a.out PUBLIC Utils)
target_compile_features(a.out PRIVATE cxx_std_17)

# --- ADD IT HERE (After the target is created) ---
target_link_options(a.out PRIVATE "-Wl,-rpath,/lib/x86_64-linux-gnu")
