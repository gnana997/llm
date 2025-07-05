#include "orchestration_logger.h"
#include "llama-impl.h"

namespace llama {
namespace orchestration {

// Static initialization to ensure logger is set up properly
static bool ensure_orchestration_logger_initialized() {
    static bool initialized = false;
    if (!initialized) {
        // Ensure the extended logging system is initialized
        auto& log_ex = common_log_ex::instance();
        
        // Set default format for orchestration logs
        log_ex.set_format(COMMON_LOG_FORMAT_TEXT);
        
        // Enable performance tracking by default
        log_ex.enable_performance_summary(true);
        
        initialized = true;
    }
    return true;
}

// Force initialization at startup
static bool logger_initialized = ensure_orchestration_logger_initialized();

} // namespace orchestration
} // namespace llama