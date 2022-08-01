#include "terminal.hpp"

#include <iostream>

Terminal::Terminal() {
}

Terminal::~Terminal() {
}

string Terminal::makeColor(const string& s, const COLORS& c) {
#ifndef _WASM
	return "\033[0;" + std::to_string(c) + "m" + s + "\033[0;39m";
#else
	return s;
#endif
}

string Terminal::makeBold(const string& s) {
#ifndef _WASM
	return "\033[1m" + s + "\033[0m";
#else
	return s;
#endif
}

