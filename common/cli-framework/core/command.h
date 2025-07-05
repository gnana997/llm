#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <functional>

namespace llama {
namespace cli {

// Forward declarations
class CommandContext;
class Command;

// Type definitions
using CommandPtr = std::unique_ptr<Command>;
using CommandMap = std::map<std::string, CommandPtr>;
using AliasMap = std::map<std::string, std::string>;
using ArgumentList = std::vector<std::string>;

// Base command interface
class Command {
public:
    virtual ~Command() = default;

    // Command metadata
    virtual std::string name() const = 0;
    virtual std::string description() const = 0;
    virtual std::string usage() const = 0;
    
    // Optional metadata
    virtual std::vector<std::string> aliases() const { return {}; }
    virtual std::string category() const { return "general"; }
    virtual bool hidden() const { return false; }

    // Command execution
    virtual int execute(CommandContext& ctx) = 0;

    // Subcommand support
    virtual bool hasSubcommands() const { return !subcommands_.empty(); }
    virtual const CommandMap& subcommands() const { return subcommands_; }
    virtual void addSubcommand(CommandPtr cmd);
    virtual Command* findSubcommand(const std::string& name);

protected:
    CommandMap subcommands_;
    AliasMap aliases_;
};

// Command execution context
class CommandContext {
public:
    CommandContext(int argc, char** argv) 
        : argc_(argc), argv_(argv), current_index_(0) {}

    // Argument access
    int argc() const { return argc_; }
    char** argv() const { return argv_; }
    
    // Parsed arguments
    const ArgumentList& args() const { return args_; }
    void setArgs(ArgumentList args) { args_ = std::move(args); }
    
    // Global options
    const std::map<std::string, std::string>& options() const { return options_; }
    void setOption(const std::string& key, const std::string& value) {
        options_[key] = value;
    }
    
    bool hasOption(const std::string& key) const {
        return options_.find(key) != options_.end();
    }
    
    std::string getOption(const std::string& key, const std::string& default_value = "") const {
        auto it = options_.find(key);
        return it != options_.end() ? it->second : default_value;
    }

    // Environment and configuration
    const std::map<std::string, std::string>& config() const { return config_; }
    void setConfig(std::map<std::string, std::string> cfg) { config_ = std::move(cfg); }

    // Error handling
    void setError(const std::string& error) { error_ = error; }
    const std::string& error() const { return error_; }
    bool hasError() const { return !error_.empty(); }

private:
    int argc_;
    char** argv_;
    int current_index_;
    ArgumentList args_;
    std::map<std::string, std::string> options_;
    std::map<std::string, std::string> config_;
    std::string error_;
};

// Command builder helper
template<typename T>
class CommandBuilder {
public:
    static CommandPtr create() {
        return std::make_unique<T>();
    }
};

} // namespace cli
} // namespace llama