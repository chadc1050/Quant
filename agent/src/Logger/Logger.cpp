#include "Logger/Logger.hpp"

#ifndef LOG_LEVEL
#define LOG_LEVEL DEBUG
#endif

std::mutex logMutex;

void trace(const std::string &msg) { log(msg, TRACE); }

void debug(const std::string &msg) { log(msg, DEBUG); }

void info(const std::string &msg) { log(msg, INFO); }

void warn(const std::string &msg) { log(msg, WARN); }

void error(const std::string &msg) { log(msg, ERROR); }

void log(const std::string &msg, const LogLevel level) {

    if (level < LOG_LEVEL) {
        return;
    }

    const std::string levelStr = getLevelStr(level);

    const std::time_t now = std::time(nullptr);
    const std::tm* localTime = std::localtime(&now);

#if defined(_WIN32)
    DWORD processId = GetCurrentProcessId();
#else
    pid_t processId = getpid();
#endif

    std::lock_guard guard(logMutex);

    std::cout << "[" << levelStr << "][" << std::put_time(localTime, "%Y-%m-%dT%H:%M:%S") << "][" << processId << "] " << msg << std::endl;
}

std::string getLevelStr(const LogLevel level) {
    switch (level) {
        case TRACE:
            return "\033[37mTRACE\033[0m";
        case DEBUG:
            return "\033[34mDEBUG\033[0m";
        case INFO:
            return "\033[32mINFO\033[0m";
        case WARN:
            return "\033[33mWARN\033[0m";
        case ERROR:
            return "\033[31mERROR\033[0m";
        default:
            return "UNKNOWN";
    }
}