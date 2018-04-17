#include<iostream>
#include"rnFCnetwork.h"

using namespace std;

int main()
{
	//rnmatrix<double> rnm1(2,2,1);
	//rnmatrix<double> rnm2(2,2,5);
	auto rnactivator1 = [](const double rnvalue){return 1/(1 + std::exp(-rnvalue));};
	//rnLstmLayer rnlstm_layer(3, 3, 0.05);
	rnFullConnectedNetwork rnfcn(3,3,rnactivator1);
	
	const unsigned rnsize = 20;
	vector<vector<double>> rnvvec;
	vector<vector<double>> rnvvec_label;
	for(unsigned rncnt = 0; rncnt < rnsize; ++rncnt)
	{
		rnvvec.push_back(vector<double>(3, 0.01*(rncnt+1)));
		rnvvec_label.push_back(vector<double>(3, 0.02*(rncnt+1)));
	}
	rnfcn.training(rnvvec, rnvvec_label, 0.05, 1000);
	for(unsigned rncnt = 0; rncnt < rnsize; ++rncnt)
	{
		rnmatrix<double> rntemp = rnfcn.predict(rnmatrix<double>(rnvvec[rncnt], 0));
		rntemp.display();
		cout<<endl;
	}
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
