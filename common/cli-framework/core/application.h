#pragma once

#include "command.h"
#include "registry.h"
#include "../utils/config.h"
#include <iostream>
#include <iomanip>

namespace llama {
namespace cli {

class Application {
public:
    Application(const std::string& name, const std::string& version)
        : name_(name), version_(version) {}

    // Application metadata
    void setDescription(const std::string& desc) { description_ = desc; }
    
    // Run the application
    int run(int argc, char** argv);

    // Add commands
    void addCommand(CommandPtr cmd) {
        root_command_.addSubcommand(std::move(cmd));
    }

    // Global options
    void addGlobalOption(const std::string& name, const std::string& description,
                        const std::string& default_value = "") {
        global_options_[name] = {description, default_value};
    }

private:
    // Parse command line arguments
    int parseAndExecute(CommandContext& ctx);
    
    // Show help
    void showHelp(std::ostream& out = std::cout) const;
    void showVersion(std::ostream& out = std::cout) const;
    void showCommandHelp(const Command& cmd, std::ostream& out = std::cout) const;
    
    // Parse global options
    bool parseGlobalOptions(CommandContext& ctx);

    // Root command (contains all top-level commands)
    class RootCommand : public Command {
    public:
        std::string name() const override { return "llama"; }
        std::string description() const override { return "LLaMA CLI"; }
        std::string usage() const override { return "llama [options] <command> [<args>]"; }
        int execute(CommandContext& ctx) override { 
            (void)ctx; // Unused parameter
            return 0; 
        }
    };

private:
    std::string name_;
    std::string version_;
    std::string description_;
    RootCommand root_command_;
    
    struct OptionInfo {
        std::string description;
        std::string default_value;
    };
    std::map<std::string, OptionInfo> global_options_;
};

} // namespace cli
} // namespace llama