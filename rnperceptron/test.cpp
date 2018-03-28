#include"rnPerceptron.h"
using namespace std;
int main()
{
	auto rnactivator = [](const double rnnum){return rnnum>0.0?1.0:0.0;};
	vector<vector<double>> rnvvec_data(4, vector<double>(2, 0));
	rnvvec_data[0] = {1,1};
	rnvvec_data[1] = {1,0};
	rnvvec_data[2] = {0,1};
	rnvvec_data[3] = {0,0};
	vector<double> rnvec_label = {1,0,0,0};
	rnPerceptron rnp(2, rnactivator);
	double rnlearning_rate = 0.1;
	rnp.training(rnvvec_data, rnvec_label, 10, rnlearning_rate);
	rnp.display();
	cout<<"1 and 1 = "<<rnp.predict(rnvvec_data[0])<<endl;
	cout<<"1 and 0 = "<<rnp.predict(rnvvec_data[1])<<endl;
	cout<<"0 and 1 = "<<rnp.predict(rnvvec_data[2])<<endl;
	cout<<"0 and 0 = "<<rnp.predict(rnvvec_data[3])<<endl;
}
