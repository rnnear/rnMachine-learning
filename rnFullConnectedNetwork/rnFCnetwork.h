//full connected network
//0328
#include"rnMat.hpp"
#include<iostream>
#include<vector>
#include<cstddef>
#include<functional>
#include<stdexcept>

#define THROW throw

#ifndef RNFCNETWORK_H
#ifndef RNFCNETWORK_H
class rnFullConnectedLayer
{
public:
	typedef double data_type;
	typedef std::size_t size_type;
	rnFullConnectedLayer(const size_type rnin_dim, const size_type rnout_dim, const std::function<const data_type(const data_type)> rnf):
		inputDim(rnin_dim), outputDim(rnout_dim),activator(rnf), inputMat(rnin_dim + 1,1,0), weightMat(rnout_dim, rnin_dim+1){} 
	rnmatrix getOutput(const rnmatrix& rnmat);
private:
	size_type inputDim;
	size_type outputDim;
	std::function<const data_type(const data_type)> activator;
	rnmatrix<data_type> inputMat; //it's input data for a layer
	rnmatrix<data_type> weightMat;
};
#endif
rnmatrix rnFullConnectedLayer::getOutput(const rnmatrix& rnmat)
{
	inputMat.partAssign(rnmat, 1);
	auto rnmat_temp = weightMat*inputMat;
	for(size_type rnrow = 0; rnrow < rnmat_temp.getRowDim(); ++rnrow)
		for(size_type rncol = 0; rncol < rnmat_temp.getColDim(); ++rncol)
			rnmat_temp[rnrow][rncol] = activator(rnmat_temp[rnrow][rncol]);
	return rnmat_temp;
}
