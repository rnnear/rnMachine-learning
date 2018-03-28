//single perceptron
//rnPerceptron.cpp
#include"rnPerceptron.h"
rnPerceptron::data_type rnPerceptron::inner_product(const std::vector<data_type>& rnvec) const
{
	data_type rntemp = 0;
	for(size_type rncnt = 0; rncnt < rnvec.size(); ++rncnt)
		rntemp += rnvec[rncnt]*vecWeight[rncnt + 1];
	rntemp+=vecWeight[0];
	return rntemp;
}
void rnPerceptron::training(const std::vector<std::vector<data_type>> rnvvec_data, const std::vector<data_type> rnvec_label,
		const size_type rniter_times, const data_type rnlearning_rate)
{
	if(rnvvec_data.size() != rnvec_label.size())
		throw std::runtime_error("training data size doesn't match label size");
	for(size_type rncnt_iter_times = 0; rncnt_iter_times < rniter_times; ++rncnt_iter_times)
	{
		for(size_type rncnt = 0; rncnt < rnvvec_data.size(); ++rncnt)
		{
			if(rncnt_iter_times == 0&&rnvvec_data[rncnt].size() != input_dim - 1)
				throw std::runtime_error("input data size doesn't match preset size");
			data_type rnpredict_result = fActivator(inner_product(rnvvec_data[rncnt]));
			data_type rntemp = rnlearning_rate*(rnvec_label[rncnt] - rnpredict_result);
			for(size_type rncnt_temp = 1; rncnt_temp < vecWeight.size(); ++rncnt_temp)
				vecWeight[rncnt_temp] += rntemp*rnvvec_data[rncnt][rncnt_temp - 1];
			vecWeight[0] += rntemp;
		}
	}
	learning_rate = rnlearning_rate;
}
const rnPerceptron::data_type rnPerceptron::predict(const std::vector<data_type>& rnvec) const
{return fActivator(inner_product(rnvec));}
void rnPerceptron::display() const
{
	std::cout<<"weight: [";
	for(size_type rncnt = 1; rncnt < vecWeight.size(); ++rncnt)
		std::cout<<vecWeight[rncnt]<<" ";
	std::cout<<'\b'<<"]"<<std::endl;
	std::cout<<"bias: "<<vecWeight[0]<<std::endl;
}
