#pragma once

#include "../core/command.h"
#include "common.h"

namespace llama {
namespace cli {

class GenerateCommand : public Command {
public:
    std::string name() const override { return "generate"; }
    
    std::string description() const override { 
        return "Generate text using a language model";
    }
    
    std::string usage() const override {
        return "generate [options] --model <path> [--prompt <text>]";
    }
    
    std::vector<std::string> aliases() const override {
        return {"gen", "g"};
    }
    
    std::string category() const override {
        return "inference";
    }
    
    int execute(CommandContext& ctx) override;

private:
    // Convert CLI framework context to llama.cpp common_params
    common_params parseParams(const CommandContext& ctx);
    
    // Helper to parse model path
    std::string parseModelPath(const CommandContext& ctx);
};

} // namespace cli
} // namespace llama