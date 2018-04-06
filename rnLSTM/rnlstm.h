//lstm
//2018.04.06
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
	rnLstmLayer(const size_type rninput_dim, const size_type rnoutput_dim, const data_type rnrate = 0.2);
	void forward(const rnmatrix<data_type>& rnx);
	void backward(const rnmatrix<data_type>& rnx, const rnmatrix<data_type>& rndelta);
	void update();
private:
	data_type rnrandom()const;
	void init_weight(rnmatrix<data_type>& rnm);
	void init_delta(std::vector<rnmatrix<data_type>& rnvec);
	void init_grad();
	rnmatrix<data_type> calc_gate(const rnmatrix<data_type>& rnx,
			const rnmatrix<data_type>& rnwh, const rnmatrix<data_type>& rnwx,
			const rnmatrix<data_type>& rnb, const rnActivator& rnfun)const;
	void calc_delta(const size_type rnpos);
	void calc_gradient(const rnmatrix<data_type>& rnx);

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
	static constexpr data_type init_parameter = 1e-4;  
};
#endif

void rnLstmLayer::update()
{
	wfh = wfh - wfh_grad*learning_rate;
	wih = wih - wih_grad*learning_rate;
	woh = woh - woh_grad*learning_rate;
	wcth = wcth - wcth_grad*learning_rate;

	wfx = wfx - wfx_grad*learning_rate;
	wix = wix - wix_grad*learning_rate;
	wox = wox - wox_grad*learning_rate;
	wctx = wctx - wctx_grad*learning_rate;

	bf = bf - bf_grad*learning_rate;
	bi = bi - bi_grad*learning_rate;
	bo = bo - bo_grad*learning_rate;
	bct = bct - bct_grad*learning_rate;
}
void rnLstmLayer::calc_gradient(const rnmatrix<data_type>& rnx)
{
	init_grad();
	rnmatrix<data_type> rnmat;
	for(size_type rncnt = time; 0 < rncnt; --rncnt)
	{
		rnmatrix<data_type> rnh_prev = rnmat.transposition(vecHt[rncnt - 1]);
		wfh_grad = wfh_grad + (vecFtDelta[rncnt]*rnh_prev);
		wih_grad = wih_grad + (vecItDelta[rncnt]*rnh_prev);
		woh_grad = woh_grad + (vecOtDelta[rncnt]*rnh_prev);
		wcth_grad = wcth_grad + (vecCttDelta[rncnt]*rnh_prev);

		bf_grad = bf_grad + vecFtDelta[rncnt];
		bi_grad = bi_grad + vecItDelta[rncnt];
		bo_grad = bo_grad + vecOtDelta[rncnt];
		bct_grad = bct_grad + vecCttDelta[rncnt];
	}
	rnmatrix<data_type> rnx_t = rnmat.transposition(rnx);
	wfx_grad = vecFtDelta[time]*rnx_t;
	wix_grad = vecItDelta[time]*rnx_t;
	wox_grad = vecOtDelta[time]*rnx_t;
	wctx_grad = vecCttDelta[time]*rnx_t;

}
void rnLstmLayer::init_grad()
{
	wfh(outputDim, outputDim, 0), wfx(outputDim, inputDim, 0), bf(outputDim, 1, 0),

	wfh_grad = rnmatrix<data_type>(outputDim, outputDim, 0);
	wih_grad = rnmatrix<data_type>(outputDim, outputDim, 0);
	woh_grad = rnmatrix<data_type>(outputDim, outputDim, 0);
	wcth_grad = rnmatrix<data_type>(outputDim, outputDim, 0);

	wfx_grad = rnmatrix<data_type>(outputDim, inputDim, 0);
	wix_grad = rnmatrix<data_type>(outputDim, inputDim, 0);
	wox_grad = rnmatrix<data_type>(outputDim, inputDim, 0);
	wctx_grad = rnmatrix<data_type>(outputDim, inputDim, 0);

	bf_grad = rnmatrix<data_type>(outputDim, 1, 0);
	bi_grad = rnmatrix<data_type>(outputDim, 1, 0);
	bo_grad = rnmatrix<data_type>(outputDim, 1, 0);
	bct_grad = rnmatrix<data_type>(outputDim, 1, 0);
}
void rnLstmLayer::calc_delta(const size_type rnpos)
{
	rnmatrix<data_type> rnf = vecFt[rnpos];
	rnmatrix<data_type> rni = vecIt[rnpos];
	rnmatrix<data_type> rni_temp = rni;
	rni_temp.calcEach(gateActivator.getBackward());

	rnmatrix<data_type> rno = vecOt[rnpos];
	rnmatrix<data_type> rno_temp = rno;
	rno_temp.calcEach(gateActivator.getBackward());

	rnmatrix<data_type> rnctt = vecCtt[rnpos];
	rnmatrix<data_type> rnc = vecCt[rnpos];
	rnmatrix<data_type> rnc_prev = vecCt[rnpos-1];
	rnmatrix<data_type> rndelta = vecDelta[rnpos];

	rnc.calcEach(outputActivator.getForward()); //th(c)
	rnf.calcEach(gateActivator.getBackward());  //Dsigmoid(f)

	rnmatrix<data_type> rnod = rndelta^rnc^rno_temp;
	rnc.calcEach(outputActivator.getBackward()); //Dth(c)

	rnmatrix<data_type> rntemp_d = rndelta^rno^rnc; 
	rnmatrix<data_type> rnfd = rntemp_d^rnc_prev^rnf;
	rnmatrix<data_type> rnid = rntemp_d^rnctt^rni_temp;
	rnctt.calcEach(outputActivator.getBackward()); //Dth(ctt)
	rnmatrix<data_type> rncttd = rntemp_d^rni^rnctt;

	rnmatrix<data_type> rnmat;
	rnmatrix<data_type> rndelta_prev = rnmat.transposition(rnod)*woh+
									   rnmat.transposition(rnid)*wih+
									   rnmat.transposition(rnfd)*wfh+
									   rnmat.transposition(rncttd)*wcth;
	vecDelta[rnpos-1] = rnmat.transposition(rndelta_prev);
	vecFtDelta[rnpos] = rnfd;
	vecItDelta[rnpos] = rnid;
	vecOtDelta[rnpos] = rnod;
	vecCttDelta[rnpos] = rncttd;
}

void rnLstmLayer::init_delta(std::vector<rnmatrix<data_type>& rnvec)
{
	size_type rntemp = (time+1 > rnvec.size())?(time+1-rnvec.size()):0;
	for(size_type rncnt = 0; rncnt < rntemp; ++rncnt)
		rnvec.push_back(rnmatrix<data_type>(outputDim, 1, 0));
}
void rnLstmLayer::backward(const rnmatrix<data_type>& rnx, const rnmatrix<data_type>& rndelta)
{
	//init delta vector
	init_delta(vecDelta);
	init_delta(vecFtDelta);
	init_delta(vecItDelta);
	init_delta(vecOtDelta);
	init_delta(vecCttDelta);

	vecDelta[time] = rndelta;  //input delta according to label
	for(size_type rncnt = time; 0 < rncnt; --rncnt)
		calc_delta(rncnt);
	calc_gradient(rnx);////
}
rnmatrix<rnLstmLayer::data_type>
rnLstmLayer::calc_gate(const rnmatrix<data_type>& rnx,
			const rnmatrix<data_type>& rnwh, const rnmatrix<data_type>& rnwx,
			const rnmatrix<data_type>& rnb, const rnActivator& rnfun)const
{
	rnmatrix<data_type> rntemp = rnwh*vecHt[time-1]+rnwx*rnx+rnb;
	rntemp.calcEach(rnfun.getForward());
	return rntemp;
}

void rnLstmLayer::forward(const rnmatrix<data_type>& rnx)
{
	++time;
	rnmatrix<data_type> rnf = calc_gate(rnx, wfh, wfx, bf, gateActivator);
	vecFt.push_back(rnf);

	rnmatrix<data_type> rni = calc_gate(rnx, wih, wix, bi, gateActivator);
	vecIt.push_back(rni);

	rnmatrix<data_type> rno = calc_gate(rnx, woh, wox, bo, gateActivator);
	vecOt.push_back(rno);

	rnmatrix<data_type> rnctt = calc_gate(rnx, wcth, wctx, bct, outputActivator);
	vecCtt.push_back(rnctt);

	rnmatrix<data_type> rnc = (rnf^vecCt[time-1])+(rni^rnctt); 
	vecCt.push_back(rnc);

	rnc.calcEach(outputActivator.getForward());
	rnmatrix<data_type> rnh =  rno^rnc;
	vecHt.push_back(rnh);
}
rnLstmLayer::rnLstmLayer(const size_type rninput_dim, const size_type rnoutput_dim, const data_type rnrate):
	inputDim(rninput_dim), outputDim(rnoutput_dim), learning_rate(rnrate),
	wfh(outputDim, outputDim, 0), wfx(outputDim, inputDim, 0), bf(outputDim, 1, 0),
	wih(outputDim, outputDim, 0), wix(outputDim, inputDim, 0), bi(outputDim, 1, 0),
	woh(outputDim, outputDim, 0), wox(outputDim, inputDim, 0), bo(outputDim, 1, 0),
	wcth(outputDim, outputDim, 0),wctx(outputDim, inputDim, 0),bct(outputDim, 1, 0),
	vecFt(1, rnmatrix<data_type>(outputDim, 1, 0)),
	vecIt(1, rnmatrix<data_type>(outputDim, 1, 0)),
	vecOt(1, rnmatrix<data_type>(outputDim, 1, 0)),
	vecCtt(1, rnmatrix<data_type>(outputDim, 1, 0)),
	vecCt(1, rnmatrix<data_type>(outputDim, 1, 0)),
	vecHt(1, rnmatrix<data_type>(outputDim, 1, 0)),
	vecDelta(1, rnmatrix<data_type>(outputDim, 1, 0)),
	vecFtDelta(1, rnmatrix<data_type>(outputDim, 1, 0)),
	vecItDelta(1, rnmatrix<data_type>(outputDim, 1, 0)),
	vecOtDelta(1, rnmatrix<data_type>(outputDim, 1, 0)),
	vecCttDelta(1, rnmatrix<data_type>(outputDim, 1, 0)), time(0)
{
	init_weight(wfh);
	init_weight(wfx);

	init_weight(wih);
	init_weight(wix);

	init_weight(woh);
	init_weight(wox);

	init_weight(wcth);
	init_weight(wctx);
}
rnLstmLayer::data_type rnLstmLayer::rnrandom() const
{
	static unsigned rntemp = 0; 
	if(rntemp == 0)
	{
		rntemp = static_cast<unsigned>(std::time(NULL)); 
		std::srand(rntemp);
	}
	return (std::rand()/static_cast<data_type>(RAND_MAX) - 0.5)*2*init_parameter;
}
void rnLstmLayer::init_weight(rnmatrix<data_type>& rnm)
{
	for(size_type rnrow = 0; rnrow < rnm.getRowDim(); ++rnrow)
		for(size_type rncol = 0; rncol < rnm.getColDim(); ++rncol)
			rnm[rnrow][rncol] = rnrandom();
}
