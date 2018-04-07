#include<iostream>
#include"rnlstm.h"
using namespace std;

int main()
{
	//rnmatrix<double> rnm1(2,2,1);
	//rnmatrix<double> rnm2(2,2,5);
	rnLstmLayer rnlstm_layer(3, 3, 0.05);
	const unsigned rnsize = 20;
	std::vector<rnmatrix<double>> rnvvec;
	for(unsigned rncnt = 0; rncnt < rnsize; ++rncnt)
		rnvvec.push_back(rnmatrix<double>(3, 1, 0.01*(rncnt+1)));
	rnlstm_layer.training(rnvvec, 2000, 150, 17);
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
	cout<<rnlstm_layer.getError()<<endl;
	//rnlstm_layer.check();
//	rnSigmoid rnsig;
//	rnTanh rnth;
//	const rnActivator& rnac = rnsig;
//	const rnActivator& rnac2 = rnth;
//	for(int rncnt = -50; rncnt < 50; ++rncnt)
//		cout<<rnac.getForward()(rncnt)<<" ";
//	cout<<endl;
//	for(int rncnt = -50; rncnt < 50; ++rncnt)
//		cout<<rnac2.getForward()(rncnt)<<" ";
//	cout<<endl;

} 
