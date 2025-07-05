#include "config.h"
#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#include <cstring>
#else
#include <unistd.h>
#include <pwd.h>
#endif

// POSIX environ declaration must be outside namespace
#ifndef _WIN32
extern char** environ;
#endif

namespace llama {
namespace cli {

namespace fs = std::filesystem;

std::vector<std::string> Config::getConfigPaths() {
    std::vector<std::string> paths;
    
    // Current directory
    paths.push_back(".llmrc");
    paths.push_back(".llmrc.json");
    paths.push_back(".llmrc.yaml");
    paths.push_back(".llmrc.yml");
    
    // User home directory
    std::string home;
#ifdef _WIN32
    const char* userprofile = std::getenv("USERPROFILE");
    if (userprofile) {
        home = userprofile;
    } else {
        const char* homedrive = std::getenv("HOMEDRIVE");
        const char* homepath = std::getenv("HOMEPATH");
        if (homedrive && homepath) {
            home = std::string(homedrive) + std::string(homepath);
        }
    }
#else
    const char* home_env = std::getenv("HOME");
    if (home_env) {
        home = home_env;
    } else {
        struct passwd* pw = getpwuid(getuid());
        if (pw) {
            home = pw->pw_dir;
        }
    }
#endif
    
    if (!home.empty()) {
        paths.push_back(home + "/.llmrc");
        paths.push_back(home + "/.llmrc.json");
        paths.push_back(home + "/.llmrc.yaml");
        paths.push_back(home + "/.llmrc.yml");
        paths.push_back(home + "/.config/llm/config");
        paths.push_back(home + "/.config/llm/config.json");
        paths.push_back(home + "/.config/llm/config.yaml");
        paths.push_back(home + "/.config/llm/config.yml");
    }
    
    // System-wide config
#ifdef _WIN32
    paths.push_back("C:\\ProgramData\\llm\\config");
    paths.push_back("C:\\ProgramData\\llm\\config.json");
#else
    paths.push_back("/etc/llm/config");
    paths.push_back("/etc/llm/config.json");
    paths.push_back("/etc/llm/config.yaml");
#endif
    
    return paths;
}

std::string Config::findConfigFile() {
    auto paths = getConfigPaths();
    for (const auto& path : paths) {
        if (fs::exists(path) && fs::is_regular_file(path)) {
            return path;
        }
    }
    return "";
}

bool Config::load(const std::string& path) {
    std::string config_path = path;
    
    // If no path specified, find config file
    if (config_path.empty()) {
        config_path = findConfigFile();
        if (config_path.empty()) {
            // No config file found, load env overrides only
            parseEnvOverrides();
            return true;
        }
    }
    
    // Check if file exists
    if (!fs::exists(config_path)) {
        return false;
    }
    
    // Determine format by extension
    bool success = false;
    size_t path_len = config_path.length();
    if (path_len >= 5 && config_path.substr(path_len - 5) == ".json") {
        success = loadJson(config_path);
    } else if ((path_len >= 5 && config_path.substr(path_len - 5) == ".yaml") || 
               (path_len >= 4 && config_path.substr(path_len - 4) == ".yml")) {
        success = loadYaml(config_path);
    } else {
        // Try to detect format
        std::ifstream file(config_path);
        if (file) {
            char first_char;
            file >> std::ws >> first_char;
            file.close();
            
            if (first_char == '{' || first_char == '[') {
                success = loadJson(config_path);
            } else {
                success = loadYaml(config_path);
            }
        }
    }
    
    if (success) {
        // Apply environment variable overrides
        parseEnvOverrides();
    }
    
    return success;
}

bool Config::loadJson(const std::string& path) {
    // Simple JSON parser - in production, we'd use a proper JSON library
    std::ifstream file(path);
    if (!file) {
        return false;
    }
    
    std::string content((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    
    // Very basic JSON parsing - just extract key-value pairs
    // This is a placeholder - in production, use nlohmann/json or similar
    size_t pos = 0;
    while ((pos = content.find("\"", pos)) != std::string::npos) {
        size_t key_start = pos + 1;
        size_t key_end = content.find("\"", key_start);
        if (key_end == std::string::npos) break;
        
        std::string key = content.substr(key_start, key_end - key_start);
        
        size_t colon = content.find(":", key_end);
        if (colon == std::string::npos) break;
        
        size_t value_start = content.find("\"", colon);
        if (value_start == std::string::npos) break;
        value_start++;
        
        size_t value_end = content.find("\"", value_start);
        if (value_end == std::string::npos) break;
        
        std::string value = content.substr(value_start, value_end - value_start);
        
        values_[key] = value;
        pos = value_end + 1;
    }
    
    return true;
}

bool Config::loadYaml(const std::string& path) {
    // Simple YAML parser - in production, use yaml-cpp
    std::ifstream file(path);
    if (!file) {
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Find key-value separator
        size_t sep = line.find(':');
        if (sep == std::string::npos) {
            continue;
        }
        
        // Extract key and value
        std::string key = line.substr(0, sep);
        std::string value = line.substr(sep + 1);
        
        // Trim whitespace
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        
        // Remove quotes if present
        if (value.size() >= 2 && 
            ((value.front() == '"' && value.back() == '"') ||
             (value.front() == '\'' && value.back() == '\''))) {
            value = value.substr(1, value.size() - 2);
        }
        
        values_[key] = value;
    }
    
    return true;
}

void Config::parseEnvOverrides() {
    // Look for environment variables starting with LLM_
#ifdef _WIN32
    // Windows doesn't have environ, iterate through environment
    char* env_block = GetEnvironmentStrings();
    if (!env_block) return;
    
    char* env = env_block;
    while (*env) {
        std::string env_str(env);
#else
    if (!environ) return;
    
    for (char** env = environ; *env != nullptr; ++env) {
        std::string env_str(*env);
#endif
        if (env_str.substr(0, 4) == "LLM_") {
            size_t eq_pos = env_str.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = env_str.substr(4, eq_pos - 4); // Skip "LLM_"
                std::string value = env_str.substr(eq_pos + 1);
                
                // Convert to lowercase and replace _ with .
                std::transform(key.begin(), key.end(), key.begin(), ::tolower);
                std::replace(key.begin(), key.end(), '_', '.');
                
                values_[key] = value;
            }
        }
#ifdef _WIN32
        env += strlen(env) + 1;
    }
    FreeEnvironmentStrings(env_block);
#else
    }
#endif
}

std::string Config::get(const std::string& key, const std::string& default_value) const {
    auto it = values_.find(key);
    return it != values_.end() ? it->second : default_value;
}

void Config::set(const std::string& key, const std::string& value) {
    values_[key] = value;
}

bool Config::has(const std::string& key) const {
    return values_.find(key) != values_.end();
}

void Config::remove(const std::string& key) {
    values_.erase(key);
}

void Config::merge(const Config& other) {
    for (const auto& [key, value] : other.values_) {
        values_[key] = value;
    }
}

bool Config::save(const std::string& path) const {
    size_t path_len = path.length();
    if (path_len >= 5 && path.substr(path_len - 5) == ".json") {
        return saveJson(path);
    } else if ((path_len >= 5 && path.substr(path_len - 5) == ".yaml") || 
               (path_len >= 4 && path.substr(path_len - 4) == ".yml")) {
        return saveYaml(path);
    }
    // Default to JSON
    return saveJson(path);
}

bool Config::saveJson(const std::string& path) const {
    std::ofstream file(path);
    if (!file) {
        return false;
    }
    
    file << "{\n";
    bool first = true;
    for (const auto& [key, value] : values_) {
        if (!first) {
            file << ",\n";
        }
        file << "  \"" << key << "\": \"" << value << "\"";
        first = false;
    }
    file << "\n}\n";
    
    return true;
}

bool Config::saveYaml(const std::string& path) const {
    std::ofstream file(path);
    if (!file) {
        return false;
    }
    
    for (const auto& [key, value] : values_) {
        file << key << ": " << value << "\n";
    }
    
    return true;
}

} // namespace cli
} // namespace llama