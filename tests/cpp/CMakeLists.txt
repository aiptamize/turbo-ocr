cmake_minimum_required(VERSION 3.20)
project(turbo_ocr_tests_standalone LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc)

# Build turbo_ocr_common library (subset needed for tests)
set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
add_library(turbo_ocr_common STATIC
    ${PROJECT_ROOT}/src/detection/det_postprocess.cpp
    ${PROJECT_ROOT}/src/recognition/crop_utils.cpp
    ${PROJECT_ROOT}/src/recognition/ctc_decode.cpp
    ${PROJECT_ROOT}/src/decode/fast_png_decoder.cpp
    ${PROJECT_ROOT}/src/layout/reading_order.cpp
    ${PROJECT_ROOT}/third_party/clipper/clipper.cpp
    ${PROJECT_ROOT}/third_party/simdutf/simdutf.cpp
)
target_include_directories(turbo_ocr_common PUBLIC
    ${PROJECT_ROOT}/include
    ${PROJECT_ROOT}/third_party
    ${PROJECT_ROOT}/third_party/simdutf
)
target_link_libraries(turbo_ocr_common PUBLIC ${OpenCV_LIBS})
target_compile_options(turbo_ocr_common PRIVATE -O2 -Wall -Wextra)

# Test executable
add_executable(turbo_ocr_tests
    test_box.cpp
    test_encoding.cpp
    test_serialization.cpp
    test_ctc_decode.cpp
    test_perspective.cpp
    test_det_postprocess.cpp
    test_xy_cut.cpp
    test_logger_ratelimit.cpp
    test_image_dims.cpp
    ${PROJECT_ROOT}/third_party/catch2/catch_amalgamated.cpp
)
target_link_libraries(turbo_ocr_tests PRIVATE turbo_ocr_common)
target_include_directories(turbo_ocr_tests PRIVATE
    ${PROJECT_ROOT}/third_party/catch2
)
target_compile_options(turbo_ocr_tests PRIVATE -O2 -Wall -Wextra)
