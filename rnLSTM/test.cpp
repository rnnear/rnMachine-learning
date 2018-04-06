#include<iostream>
#include"rnlstm.h"
using namespace std;

int main()
{
	//rnmatrix<double> rnm1(2,2,1);
	//rnmatrix<double> rnm2(2,2,5);
//	rnLstmLayer rnlstm_layer;
//	for(int rncnt = 0; rncnt < 100; ++rncnt)
//		cout<<rnlstm_layer.rnrandom()<<endl;
	rnSigmoid rnsig;
	rnTanh rnth;
	const rnActivator& rnac = rnsig;
	const rnActivator& rnac2 = rnth;
	for(int rncnt = -50; rncnt < 50; ++rncnt)
		cout<<rnac.getForward()(rncnt)<<" ";
	cout<<endl;
	for(int rncnt = -50; rncnt < 50; ++rncnt)
		cout<<rnac2.getForward()(rncnt)<<" ";
	cout<<endl;

} 
