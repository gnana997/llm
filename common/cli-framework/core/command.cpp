#include "command.h"
#include <algorithm>

namespace llama {
namespace cli {

void Command::addSubcommand(CommandPtr cmd) {
    if (cmd) {
        const std::string& name = cmd->name();
        
        // Register aliases before moving the command
        for (const auto& alias : cmd->aliases()) {
            aliases_[alias] = name;
        }
        
        // Now move the command
        subcommands_[name] = std::move(cmd);
    }
}

Command* Command::findSubcommand(const std::string& name) {
    // First check direct match
    auto it = subcommands_.find(name);
    if (it != subcommands_.end()) {
        return it->second.get();
    }
    
    // Check aliases
    auto alias_it = aliases_.find(name);
    if (alias_it != aliases_.end()) {
        auto cmd_it = subcommands_.find(alias_it->second);
        if (cmd_it != subcommands_.end()) {
            return cmd_it->second.get();
        }
    }
    
    // Try to find by prefix (for abbreviated commands)
    std::vector<std::string> matches;
    for (const auto& [cmd_name, cmd] : subcommands_) {
        if (cmd_name.find(name) == 0) {
            matches.push_back(cmd_name);
        }
    }
    
    // Only return if there's exactly one match
    if (matches.size() == 1) {
        return subcommands_[matches[0]].get();
    }
    
    return nullptr;
}

} // namespace cli
} // namespace llama