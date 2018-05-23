#include<iostream>
#include"rnlstm.h"
using namespace std;
//predict a simple sequence
//training set: 0.01,0.02,0.03...0.17
//predict set: 0.18,0.19,0.2
//verification set: none
int main()
{
	//rnmatrix<double> rnm1(2,2,1);
	//rnmatrix<double> rnm2(2,2,5);
	rnLstmLayer rnlstm_layer(3, 3, 0.03);
	const unsigned rnsize = 20;
	std::vector<rnmatrix<double>> rnvvec;
	for(unsigned rncnt = 0; rncnt < rnsize; ++rncnt)
		rnvvec.push_back(rnmatrix<double>(3, 1, 0.01*(rncnt+1)));
	rnlstm_layer.training(rnvvec, 5000, 150, 17);
	for(unsigned rncnt = 0; rncnt < rnsize - 3; ++rncnt)
	{
		rnlstm_layer.forward(rnvvec[rncnt]);
		rnmatrix<double> rntemp = rnlstm_layer.getOutput();
		rntemp.display();
		cout<<endl;
	}
	rnmatrix<double> rntemp = rnlstm_layer.getOutput();
	for(unsigned rncnt = 0; rncnt < 3; ++rncnt)
	{
		rnlstm_layer.forward(rntemp);
		rntemp = rnlstm_layer.getOutput();
		rntemp.display();
		cout<<endl;
	}
	cout<<"training error: "<<rnlstm_layer.getError()<<endl;
} 
