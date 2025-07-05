#pragma once

#include "../core/command.h"
#include <memory>

namespace llama {
namespace cli {

class GpuInfoCommand : public Command {
public:
    // Command metadata
    std::string name() const override { return "gpu-info"; }
    std::string description() const override { return "Display GPU information and capabilities"; }
    std::string usage() const override { return "gpu-info [options]"; }
    std::vector<std::string> aliases() const override { return {"gpuinfo"}; }
    std::string category() const override { return "system"; }
    
    // Command execution
    int execute(CommandContext& ctx) override;
    
private:
    // Helper methods
    void displayBasicInfo();
    void displayVerboseInfo();
    void displayJsonInfo();
    void displayTopology();
    
    // Options
    bool verbose_ = false;
    bool json_ = false;
    bool benchmark_ = false;
};

} // namespace cli
} // namespace llama