#pragma once

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>


enum LogLevel { TRACE, DEBUG, INFO, WARN, ERROR };

void trace(const std::string &msg);
void debug(const std::string &msg);
void info(const std::string &msg);
void warn(const std::string &msg);
void error(const std::string &msg);
void log(const std::string &msg, LogLevel level);
std::string getLevelStr(LogLevel level);