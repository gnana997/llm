#pragma once

#include "command.h"
#include <memory>
#include <functional>
#include <mutex>

namespace llama {
namespace cli {

// Command factory function type
using CommandFactory = std::function<CommandPtr()>;

// Command registry singleton
class CommandRegistry {
public:
    static CommandRegistry& instance() {
        static CommandRegistry registry;
        return registry;
    }

    // Register a command factory
    void registerCommand(const std::string& name, CommandFactory factory) {
        std::lock_guard<std::mutex> lock(mutex_);
        factories_[name] = factory;
    }

    // Register a command type
    template<typename T>
    void registerCommand(const std::string& name) {
        registerCommand(name, []() { return std::make_unique<T>(); });
    }

    // Create a command instance
    CommandPtr createCommand(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = factories_.find(name);
        if (it != factories_.end()) {
            return it->second();
        }
        return nullptr;
    }

    // Get all registered command names
    std::vector<std::string> getCommandNames() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::string> names;
        names.reserve(factories_.size());
        for (const auto& [name, _] : factories_) {
            names.push_back(name);
        }
        return names;
    }

    // Check if a command is registered
    bool hasCommand(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return factories_.find(name) != factories_.end();
    }

private:
    CommandRegistry() = default;
    CommandRegistry(const CommandRegistry&) = delete;
    CommandRegistry& operator=(const CommandRegistry&) = delete;

    mutable std::mutex mutex_;
    std::map<std::string, CommandFactory> factories_;
};

// Helper macro for command registration
#define REGISTER_COMMAND(name, type) \
    static bool _registered_##type = []() { \
        CommandRegistry::instance().registerCommand<type>(name); \
        return true; \
    }();

// Auto-registration helper
template<typename T>
class AutoRegisterCommand {
public:
    AutoRegisterCommand(const std::string& name) {
        CommandRegistry::instance().registerCommand<T>(name);
    }
};

} // namespace cli
} // namespace llama