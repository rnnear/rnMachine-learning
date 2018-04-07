//lstm
//2018.04.06 realize main part
//2018.04.07 add interface
#include<cmath>
#include"rnmat.hpp"
#include<cstdlib>
#include<ctime>

#ifndef RNLSTM_H 
#define RNLSTM_H
class rnActivator
{
public:
	typedef double data_type;
	typedef unsigned long int size_type;
	virtual const std::function<data_type(const data_type)>& getForward()const=0;
	virtual const std::function<data_type(const data_type)>& getBackward()const=0;
	std::function<data_type(const data_type)> forward;
	std::function<data_type(const data_type)> backward;
};
class rnSigmoid: public rnActivator
{
public:
	rnSigmoid()
	{
		forward = [](const data_type rnvalue){return 1.0/(1.0+std::exp(-rnvalue));};
		backward = [](const data_type rnvalue){return rnvalue*(1-rnvalue);};
	}
	virtual const std::function<data_type(const data_type)>& getForward()const override{return forward;}
	virtual const std::function<data_type(const data_type)>& getBackward()const override{return backward;}
};
class rnTanh: public rnActivator
{
public:
	rnTanh()
	{
		forward = [](const data_type rnvalue){return 1-2/(1+std::exp(2*rnvalue));};
		backward = [](const data_type rnvalue){return 1-rnvalue*rnvalue;};
	}
	virtual const std::function<data_type(const data_type)>& getForward()const override{return forward;}
	virtual const std::function<data_type(const data_type)>& getBackward()const override{return backward;}
};
class rnLstmLayer
{
public:
	typedef double data_type;
	typedef unsigned long int size_type;
	rnLstmLayer() = default;
	rnLstmLayer(const size_type rninput_dim, const size_type rnoutput_dim, const data_type rnrate = 0.02);
	void forward(const rnmatrix<data_type>& rnx);
	void backward(const rnmatrix<data_type>& rnx, const rnmatrix<data_type>& rndelta);
	void update();
	void resetState();
	rnmatrix<data_type> getOutput() const{return vecHt[time];}
	void training(const std::vector<rnmatrix<data_type>>& rnvvec_data, const size_type rniter_times, const data_type rntime_limit, const size_type rntrain_size);
	data_type test(const std::vector<rnmatrix<data_type>>& rnvvec_data, const size_type rnpos);
	rnmatrix<data_type> getLastDelta(const rnmatrix<data_type>& rny, const rnmatrix<data_type>& rnt) const;
	data_type getError()const{return error;}

	void check(); 
private:
	data_type rnrandom()const;
	void init_weight(rnmatrix<data_type>& rnm);
	void init_delta(std::vector<rnmatrix<data_type>>& rnvec);
	void init_grad();
	rnmatrix<data_type> calc_gate(const rnmatrix<data_type>& rnx,
			const rnmatrix<data_type>& rnwh, const rnmatrix<data_type>& rnwx,
			const rnmatrix<data_type>& rnb, const rnActivator& rnfun)const;
	void calc_delta(const size_type rnpos);
	void calc_gradient(const rnmatrix<data_type>& rnx);
	data_type calc_error(const rnmatrix<data_type>&, const rnmatrix<data_type>&)const;


	size_type inputDim;
	size_type outputDim; //state dimension equal to output dimension
	data_type learning_rate;
	
	rnmatrix<data_type> wfh, wfx, bf;//forget gate weight and bias
	rnmatrix<data_type> wih, wix, bi;//input gate weight and bias
	rnmatrix<data_type> woh, wox, bo;//output gate weight and bias
	rnmatrix<data_type> wcth, wctx, bct;//temporary cell state gate weight and bias

	std::vector<rnmatrix<data_type>> vecFt;
	std::vector<rnmatrix<data_type>> vecIt;
	std::vector<rnmatrix<data_type>> vecOt;
	std::vector<rnmatrix<data_type>> vecCtt;
	std::vector<rnmatrix<data_type>> vecCt;
	std::vector<rnmatrix<data_type>> vecHt;

	std::vector<rnmatrix<data_type>> vecDelta; //cell state error
	std::vector<rnmatrix<data_type>> vecFtDelta; //forget gate error
	std::vector<rnmatrix<data_type>> vecItDelta; //input gate error
	std::vector<rnmatrix<data_type>> vecOtDelta; //output gate error
	std::vector<rnmatrix<data_type>> vecCttDelta; //temporary cell state error

	rnmatrix<data_type> wfh_grad, wfx_grad, bf_grad;
	rnmatrix<data_type> wih_grad, wix_grad, bi_grad;
	rnmatrix<data_type> woh_grad, wox_grad, bo_grad;
	rnmatrix<data_type> wcth_grad, wctx_grad, bct_grad;

	size_type time = 0;
	rnSigmoid gateActivator;
	rnTanh outputActivator;
	data_type error = 0;
	static constexpr data_type init_parameter = 1e-4;  
};
#endif
