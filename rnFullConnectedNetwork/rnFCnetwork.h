//rnFCnetwork.h
//0328 finished
//0417 update comment

//activator must be sigmoid
/* This program has realized basic full connected network.
 * There are two class types about FC and one class type about matrix operation
 * This FC was implemented using sigmoid function as activator
 * Only supported C++11 and higher versions
 */
#include"rnMat.hpp"
#include<iostream>
#include<vector>
#include<cstddef>
#include<functional>
#include<stdexcept>
#include<cmath>
#include<cstdlib>
#include<ctime>

#ifndef RNFCNETWORK_H
#define RNFCNETWORK_H
#define RNDEBUG
//forword declaration
class rnFullConnectedNetwork;

//single hidden layer class
class rnFullConnectedLayer
{
	friend class rnFullConnectedNetwork;
public:
	typedef double data_type;
	typedef std::size_t size_type;

private:
	rnFullConnectedLayer() = default;
	//construction func, arguments mean input dimension, output dimension and activitor(must be sigmoid) from left to right each.
	rnFullConnectedLayer(const size_type rnin_dim, const size_type rnout_dim, const std::function<const data_type(const data_type)> rnf);
	
	//single layer output and delta calculation.
	rnmatrix<data_type> getOutput(const rnmatrix<data_type>& rnmat);
	rnmatrix<data_type> getNextDelta(const rnmatrix<data_type>& rnmat);

	//update weight matrix.
	void update(const double rnlearning_rate);

	//get random value and range from -init_parameter to init_parameter.
	data_type rnrandom() const;

	rnmatrix<data_type> activate_each_element(const rnmatrix<data_type>& rnmat)const;
	size_type inputDim;
	size_type outputDim;
	std::function<const data_type(const data_type)> activator;
	rnmatrix<data_type> inputMat;
	rnmatrix<data_type> weightMat;
	rnmatrix<data_type> biasMat;
	rnmatrix<data_type> inputDeltaMat;
	//initialize parameter
	static constexpr data_type init_parameter = 1e-5; 
};

//full connected network interface class.
class rnFullConnectedNetwork
{
public:
	typedef double data_type;
	typedef std::size_t size_type;

//construction func, arguments mean input dimension, output dimension, hidden layer number except input layer and output layer,
//hidden layer dimension and activator that must be sigmoid from left to right.
	rnFullConnectedNetwork(const size_type rninput_dim, const size_type rnoutput_dim,
			const size_type rnhidden_layer_num = 1, const size_type rnhidden_layer_dim = 0,
			const std::function<const data_type(const data_type)> rnactivator = [](const data_type rnvalue){return 1/(1 + std::exp(-rnvalue));});
	//calculate output result.
	rnmatrix<data_type> predict(const rnmatrix<data_type>& rninput);
	//training func, arguments mean data that is a vector<vector>, label that is also a vector<vector>, learning rate and iterator times.
	void training(const std::vector<std::vector<data_type>>& rnvvec_data, const std::vector<std::vector<data_type>>& rnvvec_label,
			const double rnlearning_rate, const size_type rniter_times = 1);
	//calculate error between one data vector and related label.
	data_type getSingleSampleError(const rnmatrix<data_type>& rnpred_data, const rnmatrix<data_type>& rnlabel)const;
	//check gradient that is used to verify correction.
	void rncheck();
private:
	//calculate output layer delta
	rnmatrix<data_type> get_output_layer_delta(const rnmatrix<data_type>& rnpredict_output, const rnmatrix<data_type>& rnlabel)const;
	//change weight parameter according to every layer delta
	void updateWeight(const rnmatrix<data_type>& rnout_layer_delta, const double rnlearning_rate);
	size_type inputDim;
	size_type outputDim;
	std::function<const data_type(const data_type)> activator; 
	size_type hiddenLayerNum;
	size_type hiddenLayerDim;
	std::vector<rnFullConnectedLayer> vecNetwork;
};
#endif
