#include "common/cli-framework/core/application.h"
#include "common/cli-framework/core/registry.h"
#include "common/cli-framework/commands/generate_command.h"
#include "common/cli-framework/commands/gpu_info_command.h"
#include "common.h"
#include "common/log.h"
#include "src/log/log-ex.h"
#include <sstream>
#include <algorithm>

#ifdef _WIN32
#include <io.h>
#include <stdio.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif

using namespace llama::cli;

// Build info is available through common.h
extern int LLAMA_BUILD_NUMBER;
extern const char * LLAMA_COMMIT;

// Version string
#define LLAMA_VERSION "0.1.0"

// Initialize logging based on command line options
static void initializeLogging(const CommandContext& ctx) {
    auto& logger = common_log_ex::instance();
    
    // Set log level - need to set both verbosity thresholds
    std::string log_level = ctx.getOption("log-level", "info");
    int verbosity = 0;
    if (log_level == "debug") {
        verbosity = LOG_DEFAULT_DEBUG;  // LOG_DEFAULT_DEBUG = 1
    } else if (log_level == "perf") {
        verbosity = 2;
    } else if (log_level == "trace") {
        verbosity = 3;
    }
    
    // Set both verbosity thresholds
    common_log_verbosity_thold = verbosity;
    common_log_ex_verbosity_thold = verbosity;
    
    // Set log file if specified
    std::string log_file = ctx.getOption("log-file", "");
    if (!log_file.empty()) {
        // Configure rotation
        common_log_rotation_config rotation;
        rotation.base_path = log_file;
        rotation.max_file_size = std::stoi(ctx.getOption("log-rotate-size", "100")) * 1024 * 1024;
        rotation.max_files = std::stoi(ctx.getOption("log-rotate-count", "5"));
        logger.set_rotation_config(rotation);
    }
    
    // Set log format
    std::string format = ctx.getOption("log-format", "text");
    if (format == "json") {
        logger.set_format(COMMON_LOG_FORMAT_JSON);
    } else {
        logger.set_format(COMMON_LOG_FORMAT_TEXT);
    }
    
    // Enable performance logging
    std::string perf = ctx.getOption("log-perf", "false");
    if (perf == "true" || perf == "1" || perf == "yes") {
        logger.enable_performance_summary(true);
    }
    
    // Set category filter
    std::string categories = ctx.getOption("log-categories", "");
    if (!categories.empty()) {
        uint32_t cat_filter = 0;
        std::stringstream ss(categories);
        std::string cat;
        while (std::getline(ss, cat, ',')) {
            // Trim whitespace
            cat.erase(0, cat.find_first_not_of(" \t"));
            cat.erase(cat.find_last_not_of(" \t") + 1);
            
            if (cat == "general") cat_filter |= COMMON_LOG_CAT_GENERAL;
            else if (cat == "perf") cat_filter |= COMMON_LOG_CAT_PERF;
            else if (cat == "memory") cat_filter |= COMMON_LOG_CAT_MEMORY;
            else if (cat == "gpu") cat_filter |= COMMON_LOG_CAT_GPU;
            else if (cat == "inference") cat_filter |= COMMON_LOG_CAT_INFERENCE;
            else if (cat == "model") cat_filter |= COMMON_LOG_CAT_MODEL;
            else if (cat == "batch") cat_filter |= COMMON_LOG_CAT_BATCH;
            else if (cat == "cache") cat_filter |= COMMON_LOG_CAT_CACHE;
            else if (cat == "network") cat_filter |= COMMON_LOG_CAT_NETWORK;
        }
        if (cat_filter != 0) {
            logger.set_category_filter(cat_filter);
        }
    }
    
    // Set regex filter
    std::string filter = ctx.getOption("log-filter", "");
    if (!filter.empty()) {
        logger.add_regex_filter(filter, true);
    }
}

// Version command
class VersionCommand : public Command {
public:
    std::string name() const override { return "version"; }
    std::string description() const override { return "Show version information"; }
    std::string usage() const override { return "version"; }
    
    int execute(CommandContext& ctx) override {
        (void)ctx; // Unused parameter
        std::cout << "llama.cpp unified CLI\n";
        std::cout << "Version: " << LLAMA_VERSION << "\n";
        std::cout << "Build: " << LLAMA_BUILD_NUMBER << " (" << LLAMA_COMMIT << ")\n";
        std::cout << "Built with " << LLAMA_COMPILER << " for " << LLAMA_BUILD_TARGET << "\n";
        return 0;
    }
};

// Server command (placeholder)
class ServerCommand : public Command {
public:
    std::string name() const override { return "server"; }
    std::string description() const override { return "Start the API server"; }
    std::string usage() const override { return "server [options]"; }
    std::string category() const override { return "service"; }
    
    int execute(CommandContext& ctx) override {
        (void)ctx; // Unused parameter
        std::cout << "Starting server...\n";
        // TODO: Integrate server implementation
        return 0;
    }
};

// Quantize command (placeholder)
class QuantizeCommand : public Command {
public:
    std::string name() const override { return "quantize"; }
    std::string description() const override { return "Quantize a model to reduce size"; }
    std::string usage() const override { return "quantize <input> <output> <type>"; }
    std::string category() const override { return "model management"; }
    
    int execute(CommandContext& ctx) override {
        if (ctx.args().size() < 3) {
            std::cerr << "Error: quantize requires input, output, and type arguments\n";
            return 1;
        }
        std::cout << "Quantizing model...\n";
        // TODO: Integrate quantize implementation
        return 0;
    }
};

// Benchmark command (placeholder)
class BenchmarkCommand : public Command {
public:
    std::string name() const override { return "benchmark"; }
    std::string description() const override { return "Run performance benchmarks"; }
    std::string usage() const override { return "benchmark [options]"; }
    std::vector<std::string> aliases() const override { return {"bench"}; }
    std::string category() const override { return "testing"; }
    
    int execute(CommandContext& ctx) override {
        (void)ctx; // Unused parameter
        std::cout << "Running benchmarks...\n";
        // TODO: Integrate benchmark implementation
        return 0;
    }
};

// Convert command (placeholder)
class ConvertCommand : public Command {
public:
    std::string name() const override { return "convert"; }
    std::string description() const override { return "Convert models between formats"; }
    std::string usage() const override { return "convert <input> <output> [options]"; }
    std::string category() const override { return "model management"; }
    
    int execute(CommandContext& ctx) override {
        if (ctx.args().size() < 2) {
            std::cerr << "Error: convert requires input and output arguments\n";
            return 1;
        }
        std::cout << "Converting model...\n";
        // TODO: Integrate convert implementation
        return 0;
    }
};

// Parse command line arguments for logging options early
static void parseLoggingOptions(int argc, char** argv, CommandContext& ctx) {
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (arg[0] == '-' && arg[1] == '-') {
            std::string opt(arg + 2);
            size_t eq_pos = opt.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = opt.substr(0, eq_pos);
                std::string value = opt.substr(eq_pos + 1);
                // Only parse logging-related options
                if (key.find("log") == 0) {
                    ctx.setOption(key, value);
                }
            } else {
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    // Only parse logging-related options
                    if (opt.find("log") == 0) {
                        ctx.setOption(opt, argv[i + 1]);
                        i++;
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    // Parse logging options early
    CommandContext early_ctx(argc, argv);
    parseLoggingOptions(argc, argv, early_ctx);
    initializeLogging(early_ctx);
    
    // Create application
    Application app("llama", LLAMA_VERSION);
    app.setDescription("Unified CLI for LLaMA.cpp - Production-grade local LLM inference");
    
    // Add global options
    app.addGlobalOption("config", "Path to configuration file");
    app.addGlobalOption("log-level", "Set log level (debug, info, warn, error, perf, trace)", "info");
    app.addGlobalOption("log-file", "Write logs to file");
    app.addGlobalOption("log-perf", "Enable performance logging", "false");
    app.addGlobalOption("log-categories", "Log categories to enable (comma-separated: general,perf,memory,gpu,inference,model,batch,cache,network)");
    app.addGlobalOption("log-format", "Log output format (text, json)", "text");
    app.addGlobalOption("log-rotate-size", "Max log file size in MB before rotation", "100");
    app.addGlobalOption("log-rotate-count", "Number of rotated log files to keep", "5");
    app.addGlobalOption("log-filter", "Regex pattern to filter log messages");
    
    // Register commands
    app.addCommand(std::make_unique<GenerateCommand>());
    app.addCommand(std::make_unique<ServerCommand>());
    app.addCommand(std::make_unique<QuantizeCommand>());
    app.addCommand(std::make_unique<BenchmarkCommand>());
    app.addCommand(std::make_unique<ConvertCommand>());
    app.addCommand(std::make_unique<VersionCommand>());
    app.addCommand(std::make_unique<GpuInfoCommand>());
    
    // Run application
    return app.run(argc, argv);
}