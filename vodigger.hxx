#ifndef VODIGGER_HEADER
#define VODIGGER_HEADER

#include <string>

namespace vodigger {


	enum Phase {TRAIN=0, TEST, GUESS};

	enum Shape {BATCH=0, CHANNEL, HEIGHT, WIDTH};

	std::string phaseToString(Phase phase);

}



#endif
