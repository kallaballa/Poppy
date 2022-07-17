#include "terminal.hpp"

#include <iostream>

Terminal::Terminal() {
}

Terminal::~Terminal() {
}

string Terminal::makeColor(const string& s, const COLORS& c) {
  return "\033[0;" + std::to_string(c) + "m" + s + "\033[0;39m";
}

string Terminal::makeBold(const string& s) {
  return "\033[1m" + s + "\033[0m";
}

