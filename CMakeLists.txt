cmake_minimum_required(VERSION 3.14)
project(llm-batch CXX)
set(CMAKE_CXX_STANDARD 17)

find_library(CURL_LIB NAMES curl libcurl PATHS "C:/Users/Matthew/vcpkg/installed/x64-windows/lib" NO_DEFAULT_PATH)
set(CURL_INC "C:/Users/Matthew/vcpkg/installed/x64-windows/include")

find_package(Threads REQUIRED)

foreach(EXAMPLE basic_batch file_batch checkpoint_batch resumable_batch)
    add_executable(${EXAMPLE} examples/${EXAMPLE}.cpp)
    target_include_directories(${EXAMPLE} PRIVATE include ${CURL_INC})
    target_link_libraries(${EXAMPLE} PRIVATE ${CURL_LIB} Threads::Threads)
    target_compile_options(${EXAMPLE} PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/W4>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>
    )
endforeach()
