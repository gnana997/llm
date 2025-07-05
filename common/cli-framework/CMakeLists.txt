# CLI Framework Library
add_library(llama_cli_framework STATIC
    core/command.cpp
    core/application.cpp
    utils/config.cpp
    commands/generate_command.cpp
    commands/gpu_info_command.cpp
    main_wrapper.cpp
)

target_include_directories(llama_cli_framework PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

target_link_libraries(llama_cli_framework
    common
    llama
)

# Set C++ standard
target_compile_features(llama_cli_framework PUBLIC cxx_std_17)

# Platform-specific settings
if(WIN32)
    target_compile_definitions(llama_cli_framework PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# Export headers
set(LLAMA_CLI_FRAMEWORK_HEADERS
    core/command.h
    core/registry.h
    core/application.h
    utils/config.h
    commands/generate_command.h
    commands/gpu_info_command.h
)

# Install headers
install(FILES ${LLAMA_CLI_FRAMEWORK_HEADERS}
        DESTINATION include/llama/cli-framework)