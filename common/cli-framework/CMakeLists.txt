# CLI Framework Library
set(CLI_FRAMEWORK_SOURCES
    core/command.cpp
    core/application.cpp
    utils/config.cpp
    commands/generate_command.cpp
    commands/gpu_info_command.cpp
    main_wrapper.cpp
)

# Add CUDA sources if CUDA is enabled
if(GGML_CUDA)
    list(APPEND CLI_FRAMEWORK_SOURCES commands/gpu_info_cuda.cu)
endif()

add_library(llama_cli_framework STATIC ${CLI_FRAMEWORK_SOURCES})

target_include_directories(llama_cli_framework PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/src
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

# Configure CUDA compilation if enabled
if(GGML_CUDA)
    # Enable CUDA language
    enable_language(CUDA)
    
    # Set CUDA properties for the library
    set_target_properties(llama_cli_framework PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    
    # Add CUDA include directories
    target_include_directories(llama_cli_framework PRIVATE ${CUDA_INCLUDE_DIRS})
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