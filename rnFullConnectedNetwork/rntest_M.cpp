//
#include"rnmat.hpp"
using namespace std;
int main()
{
	rnmatrix<double> rnm(5,5,1);
	rnm.display();
	rnm[1][1] = 10;
	cout<<endl;
	rnm.display();
	rnmatrix<double> rnm2(5,5,-2);
	cout<<endl;
	(rnm+rnm2).display();
	cout<<endl;
	(rnm2-rnm).display();
	cout<<endl;
	(rnm*100).display();
	cout<<endl;
	(rnm*rnm2).display();
	cout<<endl;
	(rnmatrix<double>().dotTime(rnm,rnm2)).display();
}
