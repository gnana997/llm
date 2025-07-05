#include "application.h"
#include <algorithm>
#include <sstream>
#include <cstring>

namespace llama {
namespace cli {

int Application::run(int argc, char** argv) {
    CommandContext ctx(argc, argv);
    
    // Load configuration file if it exists
    Config config;
    config.load();
    ctx.setConfig(config.values());
    
    return parseAndExecute(ctx);
}

int Application::parseAndExecute(CommandContext& ctx) {
    if (ctx.argc() < 1) {
        showHelp(std::cerr);
        return 1;
    }

    // Parse arguments
    std::vector<std::string> args;
    bool options_done = false;
    
    for (int i = 1; i < ctx.argc(); ++i) {
        const char* arg = ctx.argv()[i];
        
        if (!options_done && arg[0] == '-') {
            if (strcmp(arg, "--") == 0) {
                options_done = true;
                continue;
            }
            
            // Handle global options
            if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
                showHelp();
                return 0;
            }
            
            if (strcmp(arg, "-v") == 0 || strcmp(arg, "--version") == 0) {
                showVersion();
                return 0;
            }
            
            // Parse other global options
            std::string opt_name;
            std::string opt_value;
            
            if (arg[1] == '-') {
                // Long option
                std::string opt(arg + 2);
                size_t eq_pos = opt.find('=');
                if (eq_pos != std::string::npos) {
                    opt_name = opt.substr(0, eq_pos);
                    opt_value = opt.substr(eq_pos + 1);
                } else {
                    opt_name = opt;
                    if (i + 1 < ctx.argc() && ctx.argv()[i + 1][0] != '-') {
                        opt_value = ctx.argv()[++i];
                    }
                }
            } else {
                // Short option
                opt_name = std::string(arg + 1);
                if (i + 1 < ctx.argc() && ctx.argv()[i + 1][0] != '-') {
                    opt_value = ctx.argv()[++i];
                }
            }
            
            ctx.setOption(opt_name, opt_value);
        } else {
            args.push_back(arg);
        }
    }
    
    ctx.setArgs(args);
    
    // No command specified
    if (args.empty()) {
        showHelp();
        return 0;
    }
    
    // Find and execute command
    const std::string& cmd_name = args[0];
    Command* cmd = root_command_.findSubcommand(cmd_name);
    
    if (!cmd) {
        std::cerr << "Error: Unknown command '" << cmd_name << "'\n\n";
        showHelp(std::cerr);
        return 1;
    }
    
    // Remove command name from args
    args.erase(args.begin());
    ctx.setArgs(args);
    
    // Handle subcommands
    Command* current = cmd;
    while (!args.empty() && current->hasSubcommands()) {
        Command* subcmd = current->findSubcommand(args[0]);
        if (subcmd) {
            current = subcmd;
            args.erase(args.begin());
            ctx.setArgs(args);
        } else {
            break;
        }
    }
    
    // Check for help on specific command
    if (ctx.hasOption("help")) {
        showCommandHelp(*current);
        return 0;
    }
    
    // Execute the command
    return current->execute(ctx);
}

void Application::showHelp(std::ostream& out) const {
    out << name_ << " " << version_ << "\n";
    if (!description_.empty()) {
        out << "\n" << description_ << "\n";
    }
    
    out << "\nUsage: " << root_command_.usage() << "\n";
    
    // Group commands by category
    std::map<std::string, std::vector<const Command*>> categories;
    for (const auto& [name, cmd] : root_command_.subcommands()) {
        if (!cmd->hidden()) {
            categories[cmd->category()].push_back(cmd.get());
        }
    }
    
    out << "\nCommands:\n";
    for (const auto& [category, commands] : categories) {
        if (category != "general") {
            out << "\n  " << category << ":\n";
        }
        
        for (const auto* cmd : commands) {
            out << "    " << std::left << std::setw(20) << cmd->name() 
                << " " << cmd->description() << "\n";
        }
    }
    
    out << "\nGlobal Options:\n";
    out << "    " << std::left << std::setw(20) << "-h, --help" 
        << " Show this help message\n";
    out << "    " << std::left << std::setw(20) << "-v, --version" 
        << " Show version information\n";
    
    for (const auto& [name, info] : global_options_) {
        out << "    " << std::left << std::setw(20) << "--" + name 
            << " " << info.description;
        if (!info.default_value.empty()) {
            out << " (default: " << info.default_value << ")";
        }
        out << "\n";
    }
    
    out << "\nRun '" << name_ << " <command> --help' for more information on a command.\n";
}

void Application::showVersion(std::ostream& out) const {
    out << name_ << " version " << version_ << "\n";
}

void Application::showCommandHelp(const Command& cmd, std::ostream& out) const {
    out << "Usage: " << name_ << " " << cmd.usage() << "\n\n";
    out << cmd.description() << "\n";
    
    if (cmd.hasSubcommands()) {
        out << "\nSubcommands:\n";
        for (const auto& [name, subcmd] : cmd.subcommands()) {
            if (!subcmd->hidden()) {
                out << "  " << std::left << std::setw(20) << name 
                    << " " << subcmd->description() << "\n";
            }
        }
    }
}

} // namespace cli
} // namespace llama