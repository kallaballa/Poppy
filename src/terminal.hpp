#ifndef SRC_TERMINAL_HPP_
#define SRC_TERMINAL_HPP_

#include <string>

using std::string;

class Terminal {
public:
	enum COLORS {
	  BLACK=30,
	  RED=31,
	  GREEN=32,
	  YELLOW=33,
	  BLUE=34,
	  PINK=35,
	  CYAN=36,
	  WHITE=37,
	  NORMAL=39
	};

	string makeColor(const string& s, const COLORS& c);
	string makeBold(const string& s);

	Terminal();
	virtual ~Terminal();

};

#endif /* SRC_TERMINAL_HPP_ */
