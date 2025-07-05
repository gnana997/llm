# Unified LLM CLI tool
set(TARGET llm)

add_executable(${TARGET} main.cpp)

target_link_libraries(${TARGET} PRIVATE 
    common 
    llama 
    llama_cli_framework
    ${CMAKE_THREAD_LIBS_INIT}
)

target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Platform-specific settings
if(WIN32)
    target_compile_definitions(${TARGET} PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# Install the executable
install(TARGETS ${TARGET} RUNTIME DESTINATION bin)

# Create symlinks for backward compatibility
if(NOT WIN32)
    install(CODE "
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E create_symlink llm llama
            WORKING_DIRECTORY \${CMAKE_INSTALL_PREFIX}/bin
        )
    ")
endif()