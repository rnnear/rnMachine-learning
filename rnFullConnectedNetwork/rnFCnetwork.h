//full connected network
//0328
//activator must be sigmoid
#include"rnMat.hpp"
#include<iostream>
#include<vector>
#include<cstddef>
#include<functional>
#include<stdexcept>
#include<cmath>

#define THROW throw

#ifndef RNFCNETWORK_H
#define RNFCNETWORK_H
#define RNDEBUG
class rnFullConnectedNetwork;
class rnFullConnectedLayer
{
	friend class rnFullConnectedNetwork;
public:
	typedef double data_type;
	typedef std::size_t size_type;
	rnFullConnectedLayer(){}
	rnFullConnectedLayer(const size_type rnin_dim, const size_type rnout_dim, const std::function<const data_type(const data_type)> rnf):
		inputDim(rnin_dim), outputDim(rnout_dim),activator(rnf), inputMat(rnin_dim,1,0), weightMat(rnout_dim, rnin_dim), 
		biasMat(rnout_dim, 1, 0), inputDeltaMat(rnout_dim, 1, 0){} 
	rnmatrix<data_type> getOutput(const rnmatrix<data_type>& rnmat);
	rnmatrix<data_type> getNextDelta(const rnmatrix<data_type>& rnmat);
	void update(const double rnlearning_rate);
private:
	rnmatrix<data_type> activate_each_element(const rnmatrix<data_type>& rnmat)const;
	size_type inputDim;
	size_type outputDim;
	std::function<const data_type(const data_type)> activator;
	rnmatrix<data_type> inputMat; //it's input data for a layer forword
	rnmatrix<data_type> weightMat;
	rnmatrix<data_type> biasMat;
	rnmatrix<data_type> inputDeltaMat;
};
class rnFullConnectedNetwork
{
public:
	typedef double data_type;
	typedef std::size_t size_type;

	rnFullConnectedNetwork(const size_type rninput_dim, const size_type rnoutput_dim, const std::function<const data_type(const data_type)> rnactivator,
			const size_type rnhidden_layer_num = 1, const size_type rnhidden_layer_dim = 0);
	rnmatrix<data_type> predict(const rnmatrix<data_type>& rninput);
	void training(const std::vector<std::vector<data_type>>& rnvvec_data, const std::vector<std::vector<data_type>>& rnvvec_label,
			const double rnlearning_rate, const size_type rniter_times = 1);
	data_type getSingleSampleError(const rnmatrix<data_type>& rnpred_data, const rnmatrix<data_type>& rnlabel)const;
	void rncheck()
	{
		data_type rndelta = 0.0001;
		rnmatrix<data_type> rndata(inputDim, 1, 10);
		std::vector<data_type> rnvec_data(inputDim, 10);
		std::vector<std::vector<data_type>> rnvvec_data(1, rnvec_data); 

		rnmatrix<data_type> rnlabel(outputDim, 1, 0.9);
		std::vector<data_type> rnvec_label(outputDim, 0.9);
		std::vector<std::vector<data_type>> rnvvec_label(1, rnvec_label); 
#ifndef RNDEBUG
#endif
		vecNetwork[1].weightMat[0][0] += rndelta;
		auto rntemp = predict(rndata);
		data_type rnd1 = getSingleSampleError(rntemp, rnlabel);
		vecNetwork[1].weightMat[0][0] -= 2*rndelta;
#ifndef RNDEBUG
#endif

		auto rntemp2 = predict(rndata);
		data_type rnd2 = getSingleSampleError(rntemp2, rnlabel);
		std::cout<<"definion result: "<<(rnd1-rnd2)/(2*rndelta)<<std::endl;
		vecNetwork[1].weightMat[0][0] += rndelta;
		data_type rnold_data = vecNetwork[1].weightMat[0][0];

		training(rnvvec_data, rnvvec_label, 1, 1);
		std::cout<<"training result: "<<rnold_data - vecNetwork[1].weightMat[0][0]<<std::endl;
	}
	
private:
	rnmatrix<data_type> get_output_layer_delta(const rnmatrix<data_type>& rnpredict_output, const rnmatrix<data_type>& rnlabel)const;
	void updateWeight(const rnmatrix<data_type>& rnout_layer_delta, const double rnlearning_rate);
	size_type inputDim;
	size_type outputDim;
	std::function<const data_type(const data_type)> activator; 
	size_type hiddenLayerNum;
	size_type hiddenLayerDim;
	std::vector<rnFullConnectedLayer> vecNetwork;
};
#endif
