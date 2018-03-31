// simple matrix operation
#include<vector>
#include<cstddef>
#include<stdexcept>
#include<iostream>

#ifndef RNMATRIX_H
#define RNMATRIX_H
#define THROW throw
template<typename T>
class rnmatrix
{
public:
	typedef T data_type;
	typedef std::size_t size_type;
	static rnmatrix dotTime(const rnmatrix& rnleft, const rnmatrix& rnright);
	rnmatrix(){}
	rnmatrix(const size_type rnrow_num, const size_type rncol_num, const data_type rninit_num = 0):rowDim(rnrow_num), 
		colDim(rncol_num), mat(rnrow_num, std::vector<data_type>(rncol_num, rninit_num)){}
	rnmatrix(const std::vector<data_type>& rnvec): rowDim(1), colDim(rnvec.size()), mat(1, rnvec){}
	rnmatrix(const std::vector<std::vector<data_type> >& rnvvec): rowDim(rnvvec.size()), colDim((rnvvec.size() == 0?0:rnvvec[0].size())),
		mat(rnvvec){} //exception
	rnmatrix(const std::vector<data_type>& rnvec, int); //vector matrix
	const size_type getRowDim() const {return rowDim;}
	const size_type getColDim() const {return colDim;}
	std::vector<data_type>& operator[](const size_type rnindex){return mat[rnindex];}
	const std::vector<data_type>& operator[](const size_type rnindex)const {return mat[rnindex];}
	rnmatrix operator *(const rnmatrix& rnright) const;
	rnmatrix operator *(const data_type rnright) const;
	rnmatrix operator +(const rnmatrix& rnright) const;
	rnmatrix operator -(const rnmatrix& rnright) const;
	void partAssign(const rnmatrix& rnmat, const data_type rndata = 0);
	rnmatrix transposition(const rnmatrix& rnmat)const;
	void display()const;

private:
	static rnmatrix arithmetic(const rnmatrix& rnleft, const rnmatrix& rnright, const char rnc);
	size_type rowDim;
	size_type colDim;
	std::vector<std::vector<data_type> > mat;
};
template<typename T>
rnmatrix<T>::rnmatrix(const std::vector<data_type>& rnvec, int): rowDim(rnvec.size()), colDim(1), mat(rowDim, std::vector<data_type>(1,0)) //vector matrix
{
	for(size_type rncnt = 0; rncnt < rnvec.size(); ++rncnt)
		mat[rncnt][0] = rnvec[rncnt];
}
template<typename T>
rnmatrix<T> rnmatrix<T>::operator *(const rnmatrix<T>& rnright) const
{
	if(colDim != rnright.getRowDim())
		THROW std::runtime_error("operator*: Dimension mismatch");
	rnmatrix<T> rnret(rowDim, rnright.getColDim());
	for(size_type rnrow = 0; rnrow < rowDim; ++rnrow)
		for(size_type rncol = 0; rncol < rnright.getColDim(); ++rncol)
		{
			data_type rntemp = 0;
			for(size_type rncnt = 0; rncnt < colDim; ++rncnt)
				rntemp += mat[rnrow][rncnt]*rnright[rncnt][rncol];
			rnret[rnrow][rncol] = rntemp;
		}
	return rnret;
}
template<typename T>
rnmatrix<T> rnmatrix<T>::operator *(const data_type rnright) const
{return arithmetic(*this, rnmatrix<T>(rowDim, colDim, rnright),'*');}
template<typename T>
rnmatrix<T> rnmatrix<T>::arithmetic(const rnmatrix<T>& rnleft, const rnmatrix<T>& rnright, const char rnc)
{
	if(rnleft.getRowDim() != rnright.getRowDim()||rnleft.getColDim() != rnright.getColDim())
		THROW std::runtime_error("arithmetic: Dimension mismatch");
	rnmatrix<T> rnret(rnleft.getRowDim(), rnleft.getColDim());
	for(size_type rnrow = 0; rnrow < rnleft.getRowDim(); ++rnrow)
		for(size_type rncol = 0; rncol < rnleft.getColDim(); ++rncol)
		{
			if(rnc == '+')
				rnret[rnrow][rncol] = rnleft[rnrow][rncol] + rnright[rnrow][rncol];
			else if(rnc == '-')
				rnret[rnrow][rncol] = rnleft[rnrow][rncol] - rnright[rnrow][rncol];
			else if(rnc == '*')
				rnret[rnrow][rncol] = rnleft[rnrow][rncol] * rnright[rnrow][rncol];
			else
				THROW std::runtime_error("operator error: Can't recognize operator");
		}
	return rnret;
}
template<typename T>
rnmatrix<T> rnmatrix<T>::operator +(const rnmatrix<T>& rnright) const
{return arithmetic(*this, rnright, '+');}
template<typename T>
rnmatrix<T> rnmatrix<T>::operator -(const rnmatrix<T>& rnright) const
{return arithmetic(*this, rnright, '-');}
template<typename T>
void rnmatrix<T>::display()const
{
	for(size_type rnrow = 0; rnrow < rowDim; ++rnrow)
	{
		for(size_type rncol = 0; rncol < colDim; ++rncol)
			std::cout<<mat[rnrow][rncol]<<"\t";
		std::cout<<std::endl;
	}
}
template<typename T>
void rnmatrix<T>::partAssign(const rnmatrix<T>& rnmat, const T rndata)
{
	if(rowDim < rnmat.getRowDim() || colDim < rnmat.getColDim())
		THROW std::runtime_error("partAssign: dimension mismatch");
	rnmatrix<T> rnmat_temp(rowDim, colDim, 0);
	mat = arithmetic(mat,rnmat_temp, '*') + rnmatrix(rowDim, colDim, rndata);
	for(size_type rnrow = 0; rnrow < rnmat.getRowDim(); ++rnrow)
		for(size_type rncol = 0; rncol < rnmat.getColDim(); ++rncol)
			mat[rnrow][rncol] = rnmat[rnrow][rncol];
}
template<typename T>
rnmatrix<T> rnmatrix<T>::transposition(const rnmatrix& rnmat)const
{
	rnmatrix<T> rnmat_temp(rnmat.getColDim(), rnmat.getRowDim(), 0);
	for(size_type rnrow = 0; rnrow < rnmat.getRowDim();++rnrow)
		for(size_type rncol = 0; rncol < rnmat.getColDim(); ++rncol)
			rnmat_temp[rncol][rnrow] = rnmat[rnrow][rncol];
	return rnmat_temp;
}
template<typename T>
rnmatrix<T> rnmatrix<T>::dotTime(const rnmatrix& rnleft, const rnmatrix& rnright)
{
	return arithmetic(rnleft, rnright,'*');
}
#endif
