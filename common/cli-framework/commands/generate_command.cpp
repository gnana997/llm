#include "generate_command.h"
#include "arg.h"
#include "sampling.h"
#include "log.h"
#include "../../src/log/log-ex.h"
#include <iostream>
#include <vector>

// Forward declare the main generation function from the existing code
extern int llama_main(common_params& params);

namespace llama {
namespace cli {

int GenerateCommand::execute(CommandContext& ctx) {
    // Start performance timer for the entire generation
    PERF_TIMER(PERF_EVENT_CUSTOM, "generate_command");
    
    LOG_CAT(COMMON_LOG_CAT_INFERENCE, GGML_LOG_LEVEL_INFO, "Starting text generation\n");
    
    // Parse parameters
    common_params params = parseParams(ctx);
    
    // Log generation configuration
    LOG_PERF("Generation config: batch_size=%d, ctx_size=%d, n_predict=%d\n", 
             params.n_batch, params.n_ctx, params.n_predict);
    
    // Check if model path is provided
    if (params.model.path.empty()) {
        std::string model_path = parseModelPath(ctx);
        if (model_path.empty()) {
            std::cerr << "Error: Model path is required. Use --model <path> or set in config.\n";
            return 1;
        }
        params.model.path = model_path;
    }
    
    LOG_CAT(COMMON_LOG_CAT_MODEL, GGML_LOG_LEVEL_INFO, "Using model: %s\n", 
            params.model.path.c_str());
    
    // Log model loading performance
    LOG_PERF_START(PERF_EVENT_MODEL_LOAD_START, params.model.path);
    
    // Call the existing main function
    // In the actual implementation, we would refactor main.cpp into a library
    // For now, we'll need to expose the core functionality
    int result = llama_main(params);
    
    LOG_PERF_END(PERF_EVENT_MODEL_LOAD_END, params.model.path);
    
    // Print performance summary if enabled
    if (ctx.hasOption("log-perf") && ctx.getOption("log-perf") == "true") {
        common_log_ex::instance().print_performance_summary();
    }
    
    return result;
}

common_params GenerateCommand::parseParams(const CommandContext& ctx) {
    common_params params;
    
    // First, apply config file settings
    const auto& config = ctx.config();
    
    // Model settings from config
    if (config.count("model.path")) {
        params.model.path = config.at("model.path");
    }
    if (config.count("model.gpu_layers")) {
        params.n_gpu_layers = std::stoi(config.at("model.gpu_layers"));
    }
    
    // Generation settings from config
    if (config.count("generation.temperature")) {
        params.sampling.temp = std::stof(config.at("generation.temperature"));
    }
    if (config.count("generation.top_k")) {
        params.sampling.top_k = std::stoi(config.at("generation.top_k"));
    }
    if (config.count("generation.top_p")) {
        params.sampling.top_p = std::stof(config.at("generation.top_p"));
    }
    
    // Override with command line options
    if (ctx.hasOption("model")) {
        params.model.path = ctx.getOption("model");
    }
    if (ctx.hasOption("prompt")) {
        params.prompt = ctx.getOption("prompt");
    }
    if (ctx.hasOption("n-predict")) {
        params.n_predict = std::stoi(ctx.getOption("n-predict", "-1"));
    }
    if (ctx.hasOption("ctx-size")) {
        params.n_ctx = std::stoi(ctx.getOption("ctx-size", "2048"));
    }
    if (ctx.hasOption("batch-size")) {
        params.n_batch = std::stoi(ctx.getOption("batch-size", "2048"));
    }
    if (ctx.hasOption("threads")) {
        params.cpuparams.n_threads = std::stoi(ctx.getOption("threads", "-1"));
    }
    if (ctx.hasOption("temperature")) {
        params.sampling.temp = std::stof(ctx.getOption("temperature", "0.8"));
    }
    
    // Interactive mode
    if (ctx.hasOption("interactive")) {
        params.interactive = true;
    }
    
    // Handle positional arguments (prompt from args if not from option)
    if (params.prompt.empty() && !ctx.args().empty()) {
        params.prompt = ctx.args()[0];
    }
    
    return params;
}

std::string GenerateCommand::parseModelPath(const CommandContext& ctx) {
    // Priority order:
    // 1. Command line --model option
    // 2. Config file model.path
    // 3. Environment variable LLM_MODEL_PATH
    
    if (ctx.hasOption("model")) {
        return ctx.getOption("model");
    }
    
    if (ctx.config().count("model.path")) {
        return ctx.config().at("model.path");
    }
    
    const char* env_model = std::getenv("LLM_MODEL_PATH");
    if (env_model) {
        return env_model;
    }
    
    return "";
}

} // namespace cli
} // namespace llama