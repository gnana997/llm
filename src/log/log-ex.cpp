#include "log-ex.h"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cstdarg>
#include <ctime>
#include <cstring>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

class common_log_ex::impl {
public:
    impl() : format(COMMON_LOG_FORMAT_TEXT), 
             category_filter(COMMON_LOG_CAT_ALL),
             enable_perf_summary(false),
             current_file_size(0) {
        // Initialize performance tracking
        perf_start_times.reserve(100);
        perf_events.reserve(1000);
    }
    
    ~impl() {
        if (log_file.is_open()) {
            if (enable_perf_summary) {
                write_performance_summary();
            }
            log_file.close();
        }
    }
    
    void log_formatted(enum ggml_log_level level, const char* category_str, const char* fmt, va_list args) {
        char buffer[4096];
        vsnprintf(buffer, sizeof(buffer), fmt, args);
        
        if (format == COMMON_LOG_FORMAT_JSON) {
            log_json(level, category_str, buffer);
        } else {
            log_text(level, category_str, buffer);
        }
        
        // Also log to common_log for compatibility
        common_log_add(common_log_main(), level, "%s", buffer);
    }
    
    void log_json(enum ggml_log_level level, const char* category, const char* message) {
        std::lock_guard<std::mutex> lock(mutex);
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream json;
        json << "{";
        json << "\"timestamp\":\"" << std::put_time(std::localtime(&time_t), "%Y-%m-%dT%H:%M:%S");
        json << "." << std::setfill('0') << std::setw(3) << ms.count() << "Z\",";
        json << "\"level\":\"" << level_to_string(level) << "\",";
        if (category && strlen(category) > 0) {
            json << "\"category\":\"" << category << "\",";
        }
        json << "\"message\":\"" << escape_json(message) << "\"";
        json << "}\n";
        
        write_to_file(json.str());
    }
    
    void log_text(enum ggml_log_level level, const char* category, const char* message) {
        (void)level; // Unused parameter - level is handled by common_log
        std::lock_guard<std::mutex> lock(mutex);
        
        if (category && strlen(category) > 0) {
            std::string formatted = "[" + std::string(category) + "] " + message;
            write_to_file(formatted);
        } else {
            write_to_file(message);
        }
    }
    
    void log_perf_event(const common_perf_event& event) {
        std::lock_guard<std::mutex> lock(mutex);
        perf_events.push_back(event);
        
        if (format == COMMON_LOG_FORMAT_JSON) {
            std::stringstream json;
            json << "{\"type\":\"perf_event\",";
            json << "\"event_type\":\"" << perf_event_type_to_string(event.type) << "\",";
            json << "\"timestamp\":" << std::chrono::duration_cast<std::chrono::microseconds>(
                event.timestamp.time_since_epoch()).count() << ",";
            if (event.duration_us > 0) {
                json << "\"duration_us\":" << event.duration_us << ",";
            }
            if (event.size > 0) {
                json << "\"size\":" << event.size << ",";
            }
            if (!event.name.empty()) {
                json << "\"name\":\"" << escape_json(event.name) << "\",";
            }
            if (!event.metadata.empty()) {
                json << "\"metadata\":{";
                bool first = true;
                for (const auto& [key, value] : event.metadata) {
                    if (!first) json << ",";
                    json << "\"" << escape_json(key) << "\":\"" << escape_json(value) << "\"";
                    first = false;
                }
                json << "},";
            }
            // Remove trailing comma
            std::string str = json.str();
            if (str.back() == ',') str.pop_back();
            str += "}\n";
            write_to_file(str);
        }
    }
    
    void start_perf_event(common_perf_event_type type, const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex);
        std::string key = std::to_string(static_cast<int>(type)) + "_" + name;
        perf_start_times[key] = std::chrono::high_resolution_clock::now();
    }
    
    void end_perf_event(common_perf_event_type type, const std::string& name) {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::lock_guard<std::mutex> lock(mutex);
        
        std::string key = std::to_string(static_cast<int>(type)) + "_" + name;
        auto it = perf_start_times.find(key);
        if (it != perf_start_times.end()) {
            common_perf_event event;
            event.type = type;
            event.timestamp = it->second;
            event.duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - it->second).count();
            event.name = name;
            
            perf_events.push_back(event);
            perf_start_times.erase(it);
            
            // Log the event
            log_perf_event(event);
        }
    }
    
    bool should_log(const std::string& message) {
        // Check regex filters
        for (const auto& [pattern, include] : regex_filters) {
            if (std::regex_search(message, pattern)) {
                return include;
            }
        }
        return true;
    }
    
    void write_to_file(const std::string& message) {
        if (rotation_config.base_path.empty()) {
            return;
        }
        
        if (!log_file.is_open()) {
            open_log_file();
        }
        
        log_file << message;
        log_file.flush();
        
        current_file_size += message.size();
        rotate_if_needed_internal();
    }
    
    void open_log_file() {
        if (rotation_config.base_path.empty()) {
            return;
        }
        
        current_log_path = rotation_config.base_path;
        log_file.open(current_log_path, std::ios::app);
        
        if (log_file.is_open()) {
            current_file_size = fs::file_size(current_log_path);
        }
    }
    
    void rotate_if_needed_internal() {
        if (current_file_size >= rotation_config.max_file_size) {
            rotate_log_file();
        }
    }
    
    void rotate_log_file() {
        if (log_file.is_open()) {
            log_file.close();
        }
        
        // Rename existing files
        for (int i = rotation_config.max_files - 1; i > 0; --i) {
            std::string old_name = rotation_config.base_path + "." + std::to_string(i);
            std::string new_name = rotation_config.base_path + "." + std::to_string(i + 1);
            
            if (fs::exists(old_name)) {
                if (i == rotation_config.max_files - 1) {
                    fs::remove(old_name);
                } else {
                    fs::rename(old_name, new_name);
                }
            }
        }
        
        // Rename current file
        if (fs::exists(current_log_path)) {
            fs::rename(current_log_path, rotation_config.base_path + ".1");
        }
        
        // Open new file
        open_log_file();
    }
    
    void write_performance_summary() {
        // Calculate performance statistics
        std::unordered_map<common_perf_event_type, std::vector<int64_t>> durations;
        
        for (const auto& event : perf_events) {
            if (event.duration_us > 0) {
                durations[event.type].push_back(event.duration_us);
            }
        }
        
        std::stringstream summary;
        summary << "\n=== Performance Summary ===\n";
        
        for (const auto& [type, times] : durations) {
            if (times.empty()) continue;
            
            std::vector<int64_t> sorted_times = times;
            std::sort(sorted_times.begin(), sorted_times.end());
            int64_t sum = 0;
            for (auto t : sorted_times) sum += t;
            
            int64_t avg = sum / sorted_times.size();
            int64_t median = sorted_times[sorted_times.size() / 2];
            int64_t p95 = sorted_times[static_cast<size_t>(sorted_times.size() * 0.95)];
            int64_t p99 = sorted_times[static_cast<size_t>(sorted_times.size() * 0.99)];
            
            summary << perf_event_type_to_string(type) << ":\n";
            summary << "  Count: " << sorted_times.size() << "\n";
            summary << "  Avg: " << avg << " us\n";
            summary << "  Median: " << median << " us\n";
            summary << "  P95: " << p95 << " us\n";
            summary << "  P99: " << p99 << " us\n";
        }
        
        write_to_file(summary.str());
    }
    
    std::string escape_json(const std::string& str) {
        std::stringstream ss;
        for (char c : str) {
            switch (c) {
                case '"': ss << "\\\""; break;
                case '\\': ss << "\\\\"; break;
                case '\b': ss << "\\b"; break;
                case '\f': ss << "\\f"; break;
                case '\n': ss << "\\n"; break;
                case '\r': ss << "\\r"; break;
                case '\t': ss << "\\t"; break;
                default:
                    if (c < 0x20) {
                        ss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
                    } else {
                        ss << c;
                    }
            }
        }
        return ss.str();
    }
    
    const char* level_to_string(enum ggml_log_level level) {
        switch (level) {
            case GGML_LOG_LEVEL_NONE: return "NONE";
            case GGML_LOG_LEVEL_DEBUG: return "DEBUG";
            case GGML_LOG_LEVEL_INFO: return "INFO";
            case GGML_LOG_LEVEL_WARN: return "WARN";
            case GGML_LOG_LEVEL_ERROR: return "ERROR";
            case GGML_LOG_LEVEL_CONT: return "CONT";
            default: return "UNKNOWN";
        }
    }
    
    const char* perf_event_type_to_string(common_perf_event_type type) {
        switch (type) {
            case PERF_EVENT_MODEL_LOAD_START: return "MODEL_LOAD_START";
            case PERF_EVENT_MODEL_LOAD_END: return "MODEL_LOAD_END";
            case PERF_EVENT_BATCH_START: return "BATCH_START";
            case PERF_EVENT_BATCH_END: return "BATCH_END";
            case PERF_EVENT_TOKEN_START: return "TOKEN_START";
            case PERF_EVENT_TOKEN_END: return "TOKEN_END";
            case PERF_EVENT_GPU_ALLOC: return "GPU_ALLOC";
            case PERF_EVENT_GPU_FREE: return "GPU_FREE";
            case PERF_EVENT_GPU_COPY_TO: return "GPU_COPY_TO";
            case PERF_EVENT_GPU_COPY_FROM: return "GPU_COPY_FROM";
            case PERF_EVENT_KERNEL_LAUNCH: return "KERNEL_LAUNCH";
            case PERF_EVENT_CACHE_HIT: return "CACHE_HIT";
            case PERF_EVENT_CACHE_MISS: return "CACHE_MISS";
            case PERF_EVENT_CUSTOM: return "CUSTOM";
            default: return "UNKNOWN";
        }
    }
    
    // Member variables
    std::mutex mutex;
    common_log_format format;
    uint32_t category_filter;
    std::vector<std::pair<std::regex, bool>> regex_filters;
    common_log_rotation_config rotation_config;
    bool enable_perf_summary;
    
    std::ofstream log_file;
    std::string current_log_path;
    size_t current_file_size;
    
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> perf_start_times;
    std::vector<common_perf_event> perf_events;
};

// Singleton instance
common_log_ex& common_log_ex::instance() {
    static common_log_ex instance;
    return instance;
}

common_log_ex::common_log_ex() : pimpl(std::make_unique<impl>()) {}
common_log_ex::~common_log_ex() = default;

void common_log_ex::log_perf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    pimpl->log_formatted(GGML_LOG_LEVEL_INFO, "PERF", fmt, args);
    va_end(args);
}

void common_log_ex::log_trace(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    pimpl->log_formatted(GGML_LOG_LEVEL_DEBUG, "TRACE", fmt, args);
    va_end(args);
}

void common_log_ex::log_perf_v(int verbosity, const char* fmt, ...) {
    if (verbosity <= common_log_verbosity_thold) {
        va_list args;
        va_start(args, fmt);
        pimpl->log_formatted(GGML_LOG_LEVEL_INFO, "PERF", fmt, args);
        va_end(args);
    }
}

void common_log_ex::log_trace_v(int verbosity, const char* fmt, ...) {
    if (verbosity <= common_log_verbosity_thold) {
        va_list args;
        va_start(args, fmt);
        pimpl->log_formatted(GGML_LOG_LEVEL_DEBUG, "TRACE", fmt, args);
        va_end(args);
    }
}

void common_log_ex::log_perf_event(const common_perf_event& event) {
    pimpl->log_perf_event(event);
}

void common_log_ex::log_perf_event_start(common_perf_event_type type, const std::string& name) {
    pimpl->start_perf_event(type, name);
}

void common_log_ex::log_perf_event_end(common_perf_event_type type, const std::string& name) {
    pimpl->end_perf_event(type, name);
}

void common_log_ex::log_with_category(common_log_category category, enum ggml_log_level level, const char* fmt, ...) {
    if (!(category & pimpl->category_filter)) {
        return;
    }
    
    va_list args;
    va_start(args, fmt);
    
    const char* cat_str = "";
    switch (category) {
        case COMMON_LOG_CAT_GENERAL: cat_str = "GENERAL"; break;
        case COMMON_LOG_CAT_PERF: cat_str = "PERF"; break;
        case COMMON_LOG_CAT_MEMORY: cat_str = "MEMORY"; break;
        case COMMON_LOG_CAT_GPU: cat_str = "GPU"; break;
        case COMMON_LOG_CAT_INFERENCE: cat_str = "INFERENCE"; break;
        case COMMON_LOG_CAT_MODEL: cat_str = "MODEL"; break;
        case COMMON_LOG_CAT_BATCH: cat_str = "BATCH"; break;
        case COMMON_LOG_CAT_CACHE: cat_str = "CACHE"; break;
        case COMMON_LOG_CAT_NETWORK: cat_str = "NETWORK"; break;
        default: cat_str = "UNKNOWN"; break;
    }
    
    pimpl->log_formatted(level, cat_str, fmt, args);
    va_end(args);
}

void common_log_ex::set_format(common_log_format format) {
    pimpl->format = format;
}

void common_log_ex::set_category_filter(uint32_t categories) {
    pimpl->category_filter = categories;
}

void common_log_ex::add_regex_filter(const std::string& pattern, bool include) {
    pimpl->regex_filters.emplace_back(std::regex(pattern), include);
}

void common_log_ex::set_rotation_config(const common_log_rotation_config& config) {
    pimpl->rotation_config = config;
    if (!config.base_path.empty()) {
        pimpl->open_log_file();
    }
}

void common_log_ex::enable_performance_summary(bool enable) {
    pimpl->enable_perf_summary = enable;
}

void common_log_ex::rotate_if_needed() {
    pimpl->rotate_if_needed_internal();
}

void common_log_ex::force_rotate() {
    pimpl->rotate_log_file();
}

void common_log_ex::print_performance_summary() {
    pimpl->write_performance_summary();
}

std::string common_log_ex::get_performance_summary_json() {
    std::stringstream json;
    json << "{\"performance_summary\":{";
    
    // Calculate statistics
    std::unordered_map<common_perf_event_type, std::vector<int64_t>> durations;
    
    for (const auto& event : pimpl->perf_events) {
        if (event.duration_us > 0) {
            durations[event.type].push_back(event.duration_us);
        }
    }
    
    bool first_type = true;
    for (const auto& [type, times] : durations) {
        if (times.empty()) continue;
        
        if (!first_type) json << ",";
        first_type = false;
        
        std::vector<int64_t> sorted_times = times;
        std::sort(sorted_times.begin(), sorted_times.end());
        
        int64_t sum = 0;
        for (auto t : sorted_times) sum += t;
        
        json << "\"" << pimpl->perf_event_type_to_string(type) << "\":{";
        json << "\"count\":" << sorted_times.size() << ",";
        json << "\"avg\":" << (sum / sorted_times.size()) << ",";
        json << "\"median\":" << sorted_times[sorted_times.size() / 2] << ",";
        json << "\"p95\":" << sorted_times[static_cast<size_t>(sorted_times.size() * 0.95)] << ",";
        json << "\"p99\":" << sorted_times[static_cast<size_t>(sorted_times.size() * 0.99)];
        json << "}";
    }
    
    json << "}}";
    return json.str();
}

void common_log_ex::flush() {
    if (pimpl->log_file.is_open()) {
        pimpl->log_file.flush();
    }
}