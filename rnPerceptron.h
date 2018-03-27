//single perceptron class
#include<iostream>
#include<vector>
#include<stdexcept>
#include<cstddef>
#include<functional>

#ifndef RNPERCEPTRON_H
#define RNPERCEPTRON_H
class rnPerceptron
{
public:
	typedef double data_type;
	typedef std::size_t size_type;
	rnPerceptron(const size_type rninput_num, const std::function<data_type(const data_type)>& rnfun): input_dim(rninput_num+1),
		fActivator(rnfun), vecWeight(input_dim, 0){}
	void training(const std::vector<std::vector<data_type>>rnvvec_data,const std::vector<data_type> rnvec_label, const size_type rniter_times,
			const data_type rnlearning_rate);
	const data_type predict(const std::vector<data_type>& rnvec) const;
	void display() const;
private:
	data_type inner_product(const std::vector<data_type>& rnvec) const;
	size_type input_dim; // include bias b
	std::function<data_type(const data_type)> fActivator;
	std::vector<data_type> vecWeight;
	data_type learning_rate;
};
#endif

