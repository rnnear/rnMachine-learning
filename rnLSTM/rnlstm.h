//long short term
//2018.04.06 realize main part
//2018.04.07 add interface
//2018.04.23 add comment

/* C++11 code
 * using BP training and activators are sigmoid and tanh for gate and output
 * Only one layer of space dimension
 * error model of output layer is E=0.5*sum((_yi - ti)^2), _y means predicte vector and t means label vector
 */
#include"rnmat.hpp"
#include<cmath>
#include<cstdlib>
#include<ctime>

#ifndef RNLSTM_H 
#define RNLSTM_H
//base class for dynamic binding
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
//gate activator
class rnSigmoid: public rnActivator
{
public:
	rnSigmoid()
	{
		//function definition
		forward = [](const data_type rnvalue){return 1.0/(1.0+std::exp(-rnvalue));};
		//function derivate
		backward = [](const data_type rnvalue){return rnvalue*(1-rnvalue);};
	}
	virtual const std::function<data_type(const data_type)>& getForward()const override{return forward;}
	virtual const std::function<data_type(const data_type)>& getBackward()const override{return backward;}
};
//output activator
class rnTanh: public rnActivator
{
public:
	rnTanh()
	{
		//function definition
		forward = [](const data_type rnvalue){return 1-2/(1+std::exp(2*rnvalue));};
		//function derivate
		backward = [](const data_type rnvalue){return 1-rnvalue*rnvalue;};
	}
	virtual const std::function<data_type(const data_type)>& getForward()const override{return forward;}
	virtual const std::function<data_type(const data_type)>& getBackward()const override{return backward;}
};
//single lstm layer
class rnLstmLayer
{
public:
	typedef double data_type;
	typedef unsigned long int size_type;

	rnLstmLayer() = default;
	//construction func, arguments mean input dimension, output dimension, learning rate each from left to right
	rnLstmLayer(const size_type rninput_dim, const size_type rnoutput_dim, const data_type rnrate = 0.02);

	//calculate next step cell state
	void forward(const rnmatrix<data_type>& rnx);

	//calculate every step delta and gradient of all parameters  
	void backward(const rnmatrix<data_type>& rnx, const rnmatrix<data_type>& rndelta);

	//update all the parameters after backward
	void update();
	
	//reset every step cell state for check
	void resetState();

	//predict result
	rnmatrix<data_type> getOutput() const{return vecHt[time];}

	//training func, arguments mean training data, iterator times, cost time limit, the number of part to train(left part to validate) 
	void training(const std::vector<rnmatrix<data_type>>& rnvvec_data, const size_type rniter_times, const data_type rntime_limit, const size_type rntrain_size);

	//calculate the delta of ouput layer
	rnmatrix<data_type> getLastDelta(const rnmatrix<data_type>& rny, const rnmatrix<data_type>& rnt) const;

	//get training error
	data_type getError()const{return error;}

	//exam the correction of code
	void check(); 

private:
	//validate function used in training function
	data_type test(const std::vector<rnmatrix<data_type>>& rnvvec_data, const size_type rnpos);
	//get random double value range from -init_parameter to init_parameter
	data_type rnrandom()const;
	//initialize
	void init_weight(rnmatrix<data_type>& rnm);
	void init_delta(std::vector<rnmatrix<data_type>>& rnvec);
	void init_grad();

	//tool functions for train
	rnmatrix<data_type> calc_gate(const rnmatrix<data_type>& rnx,
			const rnmatrix<data_type>& rnwh, const rnmatrix<data_type>& rnwx,
			const rnmatrix<data_type>& rnb, const rnActivator& rnfun)const;
	void calc_delta(const size_type rnpos);
	void calc_gradient(const rnmatrix<data_type>& rnx);

	//calculate single error between prediction and label
	data_type calc_error(const rnmatrix<data_type>&, const rnmatrix<data_type>&)const;

	size_type inputDim;
	size_type outputDim; //state dimension equal to output dimension
	data_type learning_rate;
	
	rnmatrix<data_type> wfh, wfx, bf;//forget gate weight and bias
	rnmatrix<data_type> wih, wix, bi;//input gate weight and bias
	rnmatrix<data_type> woh, wox, bo;//output gate weight and bias
	rnmatrix<data_type> wcth, wctx, bct;//temporary cell state gate weight and bias

	std::vector<rnmatrix<data_type>> vecFt; //forget gate value every step
	std::vector<rnmatrix<data_type>> vecIt; //input gate value every step
	std::vector<rnmatrix<data_type>> vecOt; //output gate value every step
	std::vector<rnmatrix<data_type>> vecCtt;//temporary cell state every step
	std::vector<rnmatrix<data_type>> vecCt; //cell state every step
	std::vector<rnmatrix<data_type>> vecHt; //output value every step

	std::vector<rnmatrix<data_type>> vecDelta; //cell state error
	std::vector<rnmatrix<data_type>> vecFtDelta; //forget gate error
	std::vector<rnmatrix<data_type>> vecItDelta; //input gate error
	std::vector<rnmatrix<data_type>> vecOtDelta; //output gate error
	std::vector<rnmatrix<data_type>> vecCttDelta; //temporary cell state error
	
	//gradient of all parameters
	rnmatrix<data_type> wfh_grad, wfx_grad, bf_grad; 
	rnmatrix<data_type> wih_grad, wix_grad, bi_grad;
	rnmatrix<data_type> woh_grad, wox_grad, bo_grad;
	rnmatrix<data_type> wcth_grad, wctx_grad, bct_grad;

	//step count
	size_type time = 0;

	//gate activator function
	rnSigmoid gateActivator;
	//output activator function
	rnTanh outputActivator;
	//training error
	data_type error = 0;
	//initialization range
	static constexpr data_type init_parameter = 1e-4;  
};
#endif
