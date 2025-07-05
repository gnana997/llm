#pragma once

#include "ggml.h" // for ggml_log_level
#include <string>
#include <unordered_map>
#include <regex>
#include <memory>
#include <chrono>
#include <mutex>
#include <atomic>

// Verbosity threshold for log-ex (similar to common_log_verbosity_thold)
extern int common_log_ex_verbosity_thold;

// Extended log levels for performance and tracing
enum common_log_level_ex : int {
    COMMON_LOG_LEVEL_PERF  = 10,  // Performance metrics
    COMMON_LOG_LEVEL_TRACE = 11,  // Detailed trace logging
};

// Log categories for filtering
enum common_log_category : uint32_t {
    COMMON_LOG_CAT_GENERAL    = 1 << 0,
    COMMON_LOG_CAT_PERF       = 1 << 1,
    COMMON_LOG_CAT_MEMORY     = 1 << 2,
    COMMON_LOG_CAT_GPU        = 1 << 3,
    COMMON_LOG_CAT_INFERENCE  = 1 << 4,
    COMMON_LOG_CAT_MODEL      = 1 << 5,
    COMMON_LOG_CAT_BATCH      = 1 << 6,
    COMMON_LOG_CAT_CACHE      = 1 << 7,
    COMMON_LOG_CAT_NETWORK    = 1 << 8,
    COMMON_LOG_CAT_ALL        = 0xFFFFFFFF
};

// Output format options
enum common_log_format {
    COMMON_LOG_FORMAT_TEXT,
    COMMON_LOG_FORMAT_JSON
};

// Performance event types
enum common_perf_event_type {
    PERF_EVENT_MODEL_LOAD_START,
    PERF_EVENT_MODEL_LOAD_END,
    PERF_EVENT_BATCH_START,
    PERF_EVENT_BATCH_END,
    PERF_EVENT_TOKEN_START,
    PERF_EVENT_TOKEN_END,
    PERF_EVENT_GPU_ALLOC,
    PERF_EVENT_GPU_FREE,
    PERF_EVENT_GPU_COPY_TO,
    PERF_EVENT_GPU_COPY_FROM,
    PERF_EVENT_KERNEL_LAUNCH,
    PERF_EVENT_CACHE_HIT,
    PERF_EVENT_CACHE_MISS,
    PERF_EVENT_CUSTOM
};

// Performance event data
struct common_perf_event {
    common_perf_event_type type;
    std::chrono::high_resolution_clock::time_point timestamp;
    int64_t duration_us;  // For end events
    size_t size;          // For memory/data operations
    std::string name;     // For custom events or details
    std::unordered_map<std::string, std::string> metadata;
};

// Log rotation configuration
struct common_log_rotation_config {
    size_t max_file_size = 100 * 1024 * 1024;  // 100MB default
    int max_files = 5;
    bool compress_old_files = false;
    std::string base_path;
};

// Extended logging interface
class common_log_ex {
public:
    static common_log_ex& instance();
    
    // Extended logging functions
    void log_perf(const char* fmt, ...);
    void log_trace(const char* fmt, ...);
    void log_perf_v(int verbosity, const char* fmt, ...);
    void log_trace_v(int verbosity, const char* fmt, ...);
    
    // Performance event logging
    void log_perf_event(const common_perf_event& event);
    void log_perf_event_start(common_perf_event_type type, const std::string& name = "");
    void log_perf_event_end(common_perf_event_type type, const std::string& name = "");
    
    // Category-based logging
    void log_with_category(common_log_category category, enum ggml_log_level level, const char* fmt, ...);
    
    // Configuration
    void set_format(common_log_format format);
    void set_category_filter(uint32_t categories);
    void add_regex_filter(const std::string& pattern, bool include);
    void set_rotation_config(const common_log_rotation_config& config);
    void enable_performance_summary(bool enable);
    void set_verbosity_thold(int verbosity); // Set the verbosity threshold
    
    // Log rotation
    void rotate_if_needed();
    void force_rotate();
    
    // Performance summary
    void print_performance_summary();
    std::string get_performance_summary_json();
    
    // Utility
    void flush();
    
private:
    common_log_ex();
    ~common_log_ex();
    
    // Internal implementation
    class impl;
    std::unique_ptr<impl> pimpl;
};

// Convenience macros for extended logging
#define LOG_PERF(...) \
    do { \
        if (2 <= common_log_ex_verbosity_thold) { \
            common_log_ex::instance().log_perf(__VA_ARGS__); \
        } \
    } while (0)

#define LOG_TRACE(...) \
    do { \
        if (3 <= common_log_ex_verbosity_thold) { \
            common_log_ex::instance().log_trace(__VA_ARGS__); \
        } \
    } while (0)

#define LOG_PERFV(verbosity, ...) \
    do { \
        if ((verbosity) <= common_log_ex_verbosity_thold) { \
            common_log_ex::instance().log_perf_v(verbosity, __VA_ARGS__); \
        } \
    } while (0)

#define LOG_TRACEV(verbosity, ...) \
    do { \
        if ((verbosity) <= common_log_ex_verbosity_thold) { \
            common_log_ex::instance().log_trace_v(verbosity, __VA_ARGS__); \
        } \
    } while (0)

#define LOG_CAT(category, level, ...) \
    common_log_ex::instance().log_with_category(category, level, __VA_ARGS__)

// Performance event macros
#define LOG_PERF_START(type, name) \
    common_log_ex::instance().log_perf_event_start(type, name)

#define LOG_PERF_END(type, name) \
    common_log_ex::instance().log_perf_event_end(type, name)

// RAII performance timer
class perf_timer {
public:
    perf_timer(common_perf_event_type type, const std::string& name = "")
        : type_(type), name_(name) {
        LOG_PERF_START(type_, name_);
    }
    
    ~perf_timer() {
        LOG_PERF_END(type_, name_);
    }
    
private:
    common_perf_event_type type_;
    std::string name_;
};

#define PERF_TIMER(type, name) perf_timer _perf_timer_##__LINE__(type, name)