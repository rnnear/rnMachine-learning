//full connected network
#include"rnFCnetwork.h"
//rnFullConnectedLayer
rnFullConnectedLayer::rnFullConnectedLayer(const size_type rnin_dim, const size_type rnout_dim, const std::function<const data_type(const data_type)> rnf):
	inputDim(rnin_dim), outputDim(rnout_dim),activator(rnf), inputMat(rnin_dim,1,0), weightMat(rnout_dim, rnin_dim), 
	biasMat(rnout_dim, 1, 0), inputDeltaMat(rnout_dim, 1, 0) 
{
	for(size_type rnrow = 0; rnrow < weightMat.getRowDim(); ++rnrow)
		for(size_type rncol = 0; rncol < weightMat.getColDim(); ++rncol)
			weightMat[rnrow][rncol] = rnrandom();
}
rnFullConnectedLayer::data_type rnFullConnectedLayer::rnrandom() const
{
	static unsigned rntemp = 0; 
	if(rntemp == 0)
	{
		rntemp = static_cast<unsigned>(std::time(NULL)); 
		std::srand(rntemp);
	}
	return (std::rand()/static_cast<data_type>(RAND_MAX) - 0.5)*2*init_parameter;
}
rnmatrix<rnFullConnectedLayer::data_type> rnFullConnectedLayer::activate_each_element(const rnmatrix<data_type>& rnmat)const
{
	rnmatrix<data_type> rnmat_temp(rnmat.getRowDim(), rnmat.getColDim(),0);
	for(size_type rnrow = 0; rnrow < rnmat.getRowDim(); ++rnrow)
		for(size_type rncol = 0; rncol < rnmat.getColDim(); ++rncol)
			rnmat_temp[rnrow][rncol] = activator(rnmat[rnrow][rncol]);
	return rnmat_temp;
}

rnmatrix<rnFullConnectedLayer::data_type> rnFullConnectedLayer::getOutput(const rnmatrix<data_type>& rnmat)
{
	inputMat = rnmat;
	rnmatrix<data_type> rnmat_temp = weightMat*inputMat + biasMat;
	return activate_each_element(rnmat_temp);
}
rnmatrix<rnFullConnectedLayer::data_type> rnFullConnectedLayer::getNextDelta(const rnmatrix<data_type>& rnmat)
{
	inputDeltaMat = rnmat;
	auto rnmat_temp = rnmat.transposition(weightMat)*rnmat;
	auto rnmat_temp2 = inputMat - rnmatrix<data_type>().dotTime(inputMat, inputMat);
	return rnmatrix<data_type>().dotTime(rnmat_temp2,rnmat_temp);
}
void rnFullConnectedLayer::update(const double rnlearning_rate)
{
	for(size_type rnrow = 0; rnrow < weightMat.getRowDim(); ++rnrow)
	{
		auto rnmat_temp = inputMat*(inputDeltaMat[rnrow][0]*rnlearning_rate);
		for(size_type rncol = 0; rncol < weightMat.getColDim(); ++rncol)
			weightMat[rnrow][rncol] += rnmat_temp[rncol][0];
	}
	biasMat = biasMat + inputDeltaMat*rnlearning_rate;
}
//rnFullConnectedNetwork
rnFullConnectedNetwork::rnFullConnectedNetwork(const size_type rninput_dim, const size_type rnoutput_dim,
		const size_type rnhidden_layer_num, const size_type rnhidden_layer_dim,
		const std::function<const data_type(const data_type)> rnactivator):
		inputDim(rninput_dim), outputDim(rnoutput_dim),activator(rnactivator),hiddenLayerNum(rnhidden_layer_num),
			hiddenLayerDim(rnhidden_layer_dim == 0?size_type(std::sqrt(rninput_dim*rnoutput_dim)):rnhidden_layer_dim), vecNetwork(hiddenLayerNum + 1)
{
	if(hiddenLayerNum == 0)
		vecNetwork[0] = rnFullConnectedLayer(inputDim, outputDim, activator);
	else
	{
		vecNetwork[0] = rnFullConnectedLayer(inputDim, hiddenLayerDim, activator);
		vecNetwork[vecNetwork.size() - 1] = rnFullConnectedLayer(hiddenLayerDim, outputDim, activator);
		for(size_type rncnt = 1; rncnt < vecNetwork.size() - 1; ++rncnt)
			vecNetwork[rncnt] = rnFullConnectedLayer(hiddenLayerDim, hiddenLayerDim, activator);
	}
}
rnmatrix<rnFullConnectedNetwork::data_type> rnFullConnectedNetwork::predict(const rnmatrix<data_type>& rninput)
{
	rnmatrix<data_type> rnoutput(rninput);
	for(size_type rncnt = 0; rncnt < vecNetwork.size();++rncnt)
		rnoutput = vecNetwork[rncnt].getOutput(rnoutput);
	return rnoutput;
}
rnmatrix<rnFullConnectedNetwork::data_type> 
rnFullConnectedNetwork::get_output_layer_delta(const rnmatrix<data_type>& rnpredict_output, const rnmatrix<data_type>& rnlabel)const
{
	return rnmatrix<data_type>().dotTime(rnpredict_output-(rnmatrix<data_type>().dotTime(rnpredict_output, rnpredict_output)), rnlabel - rnpredict_output);
}
void rnFullConnectedNetwork::updateWeight(const rnmatrix<data_type>& rnout_layer_delta, const double rnlearning_rate)
{
	rnmatrix<data_type> rnmat_temp(rnout_layer_delta);
	for(size_type rncnt = vecNetwork.size(); 0 < rncnt; --rncnt)
		rnmat_temp = vecNetwork[rncnt - 1].getNextDelta(rnmat_temp);
	for(size_type rncnt = 0; rncnt < vecNetwork.size(); ++rncnt)
		vecNetwork[rncnt].update(rnlearning_rate);
}
void rnFullConnectedNetwork::training(const std::vector<std::vector<data_type>>& rnvvec_data, const std::vector<std::vector<data_type>>& rnvvec_label,
		const double rnlearning_rate, const size_type rniter_times)
{
	for(size_type rncnt = 0; rncnt < rniter_times; ++rncnt)
		for(size_type rnpos = 0; rnpos < rnvvec_data.size(); ++rnpos)
		{
			rnmatrix<data_type> rntemp = predict(rnmatrix<data_type>(rnvvec_data[rnpos], 0));
			rnmatrix<data_type> rntemp2 = get_output_layer_delta(rntemp, rnmatrix<data_type>(rnvvec_label[rnpos], 0));
			updateWeight(rntemp2, rnlearning_rate);
		}
}
rnFullConnectedNetwork::data_type rnFullConnectedNetwork::getSingleSampleError(const rnmatrix<data_type>& rnpred_data, const rnmatrix<data_type>& rnlabel)const
{
	data_type rnsum = 0;
	auto rntemp = rnpred_data - rnlabel;
	auto rntemp2 = rnmatrix<data_type>().dotTime(rntemp, rntemp);
	for(size_type rncnt = 0; rncnt < rntemp2.getRowDim(); ++rncnt)
		rnsum += rntemp2[rncnt][0];
	return rnsum/2;
}
void rnFullConnectedNetwork::rncheck()
{
	data_type rndelta = 0.001;
	rnmatrix<data_type> rndata(inputDim, 1, 0.1);
	std::vector<data_type> rnvec_data(inputDim, 0.1);
	std::vector<std::vector<data_type>> rnvvec_data(1, rnvec_data); 

	rnmatrix<data_type> rnlabel(outputDim, 1, 0.9);
	std::vector<data_type> rnvec_label(outputDim, 0.9);
	std::vector<std::vector<data_type>> rnvvec_label(1, rnvec_label); 
	size_type rnrow = 1;
	size_type rncol = 2;
	size_type p = 0;
	vecNetwork[p].weightMat[rnrow][rncol] += rndelta;
	auto rntemp = predict(rndata);
	data_type rnd1 = getSingleSampleError(rntemp, rnlabel);
	vecNetwork[p].weightMat[rnrow][rncol] -= 2*rndelta;
	auto rntemp2 = predict(rndata);
	data_type rnd2 = getSingleSampleError(rntemp2, rnlabel);
	std::cout<<"definition result: "<<(rnd1-rnd2)/(2*rndelta)<<std::endl;
	vecNetwork[p].weightMat[rnrow][rncol] += rndelta;
	data_type rnold_data = vecNetwork[p].weightMat[rnrow][rncol];

	training(rnvvec_data, rnvvec_label, 1, 1);
	std::cout<<"training result: "<<rnold_data - vecNetwork[p].weightMat[rnrow][rncol]<<std::endl;
}
