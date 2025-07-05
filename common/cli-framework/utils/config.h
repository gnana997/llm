#pragma once

#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <filesystem>

namespace llama {
namespace cli {

class Config {
public:
    Config() = default;

    // Load configuration from file
    bool load(const std::string& path = "");
    
    // Save configuration to file
    bool save(const std::string& path) const;
    
    // Get/set values
    std::string get(const std::string& key, const std::string& default_value = "") const;
    void set(const std::string& key, const std::string& value);
    
    bool has(const std::string& key) const;
    void remove(const std::string& key);
    
    // Get all values
    const std::map<std::string, std::string>& values() const { return values_; }
    
    // Merge another config
    void merge(const Config& other);
    
    // Clear all values
    void clear() { values_.clear(); }

    // Static methods for finding config files
    static std::vector<std::string> getConfigPaths();
    static std::string findConfigFile();

private:
    bool loadJson(const std::string& path);
    bool loadYaml(const std::string& path);
    bool saveJson(const std::string& path) const;
    bool saveYaml(const std::string& path) const;
    
    void parseEnvOverrides();
    
    std::map<std::string, std::string> values_;
};

} // namespace cli
} // namespace llama