//
#include"rnFCnetwork.h"

using namespace std;

int main()
{
	auto rnactivator = [](const double rnvalue){return 1/(1 + std::exp(-rnvalue));};
	rnFullConnectedNetwork rnfcN(5, 5, rnactivator);
	rnfcN.rncheck();
}
