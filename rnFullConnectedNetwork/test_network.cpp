#include<iostream>
#include"rnFCnetwork.h"
//simple classfied test case
//maybe overfit but verified correction

using namespace std;

int main()
{
	rnFullConnectedNetwork rnfcn(3,3);
	const unsigned rnsize = 15;
	vector<vector<double>> rnvvec;
	vector<vector<double>> rnvvec_label;
	for(unsigned rncnt = 0; rncnt < rnsize; ++rncnt)
	{
		rnvvec.push_back(vector<double>(3, 0.01*(rncnt+1)));
		rnvvec_label.push_back(vector<double>(3, 0.02*(rncnt+1)));
	}
	rnfcn.training(rnvvec, rnvvec_label, 0.3, 10000);
	for(unsigned rncnt = 0; rncnt < rnsize; ++rncnt)
	{
		rnmatrix<double> rntemp = rnfcn.predict(rnmatrix<double>(rnvvec[rncnt], 0));
		rntemp.display();
		cout<<endl;
	}
} 
