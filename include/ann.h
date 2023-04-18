#ifndef __ANN_H__
#define __ANN_H__
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <string.h>
#include <ctime>
#include <cstdlib>
#include <omp.h>
#include <random>


/***********************************************************************
 * NAME : 			double AF_LeakyReLU(z)
 * 
 * DESCRIPTION : 	Applies the leaky ReLU function to a value.
 * 
 * ********************************************************************/
double AF_LeakyReLU(double z);

/***********************************************************************
 * NAME : 			double AF_LeakyReLUGradient(z)
 * 
 * DESCRIPTION : 	Returns the gradient of the leaky ReLU function.
 * 
 * ********************************************************************/
double AF_LeakyReLUGradient(double z);

/***********************************************************************
 * NAME : 			double AF_InverseLeakyReLU(z)
 * 
 * DESCRIPTION : 	Calculates the inverse of the leaky ReLU function.
 * 
 * ********************************************************************/
double AF_InverseLeakyReLU(double a);

/***********************************************************************
 * NAME : 			double AF_InverseLeakyReLUGradient(z)
 * 
 * DESCRIPTION : 	Calculates the gradient of the inverse of the 
 * 					leaky ReLU function.
 * 
 * ********************************************************************/
double AF_InverseLeakyReLUGradient(double a);

/***********************************************************************
 * NAME : 			double AF_Linear(z)
 * 
 * DESCRIPTION : 	Applies the Linear function to a value.
 * 
 * ********************************************************************/
double AF_Linear(double z);

/***********************************************************************
 * NAME : 			double AF_LinearGradient(z)
 * 
 * DESCRIPTION : 	Returns the gradient of the linear function.
 * 
 * ********************************************************************/
double AF_LinearGradient(double z);

/***********************************************************************
 * NAME : 			double AF_ReLU(z)
 * 
 * DESCRIPTION : 	Applies the ReLU function to a value.
 * 
 * ********************************************************************/
double AF_ReLU(double z);

/***********************************************************************
 * NAME : 			double AF_ReLUGradient(z)
 * 
 * DESCRIPTION : 	Returns the gradient of the ReLU function.
 * 
 * ********************************************************************/
double AF_ReLUGradient(double z);

/***********************************************************************
 * NAME : 			double AF_Sigmoid(z)
 * 
 * DESCRIPTION : 	Applies the Sigmoid function to a value.
 * 
 * ********************************************************************/
double AF_Sigmoid(double z);

/***********************************************************************
 * NAME : 			double AF_SigmoidGradient(z)
 * 
 * DESCRIPTION : 	Returns the gradient of the sigmoid function.
 * 
 * ********************************************************************/
double AF_SigmoidGradient(double z);

/***********************************************************************
 * NAME : 			double AF_InverseSigmoid(z)
 * 
 * DESCRIPTION : 	Calculates the inverse of the sigmoid function.
 * 
 * ********************************************************************/
double AF_InverseSigmoid(double a);

/***********************************************************************
 * NAME : 			double AF_InverseSigmoidGradient(z)
 * 
 * DESCRIPTION : 	Calculates the gradient of the inverse of the 
 * 					sigmoid function.
 * 
 * ********************************************************************/
double AF_InverseSigmoidGradient(double a);

/***********************************************************************
 * NAME : 			double AF_Softplus(z)
 * 
 * DESCRIPTION : 	Applies the softplus function to a value.
 * 
 * ********************************************************************/
double AF_Softplus(double z);

/***********************************************************************
 * NAME : 			double AF_SoftplusGradient(z)
 * 
 * DESCRIPTION : 	Returns the gradient of the softplus function.
 * 
 * ********************************************************************/
double AF_SoftplusGradient(double z);

/***********************************************************************
 * NAME : 			double AF_InverseSigmoid(z)
 * 
 * DESCRIPTION : 	Calculates the inverse of the sigmoid function.
 * 
 * ********************************************************************/
double AF_InverseSoftplus(double a);

/***********************************************************************
 * NAME : 			double AF_InverseSoftplusGradient(z)
 * 
 * DESCRIPTION : 	Calculates the gradient of the inverse of the 
 * 					softplus function.
 * 
 * ********************************************************************/
double AF_InverseSoftplusGradient(double a);

/***********************************************************************
 * NAME : 			double AF_Tanh(z)
 * 
 * DESCRIPTION : 	Applies the tanh function to a value.
 * 
 * ********************************************************************/
double AF_Tanh(double z);

/***********************************************************************
 * NAME : 			double AF_TanhGradient(z)
 * 
 * DESCRIPTION : 	Returns the gradient of the tanh function.
 * 
 * ********************************************************************/
double AF_TanhGradient(double z);

/***********************************************************************
 * NAME : 			double AF_InverseTanh(z)
 * 
 * DESCRIPTION : 	Calculates the inverse of the tanh function.
 * 
 * ********************************************************************/
double AF_InverseTanh(double a);

/***********************************************************************
 * NAME : 			double AF_InverseTanhGradient(z)
 * 
 * DESCRIPTION : 	Calculates the gradient of the inverse of the 
 * 					tanh function.
 * 
 * ********************************************************************/
double AF_InverseTanhGradient(double a);

typedef double (*ActFunc)(double);

/***********************************************************************
 * NAME : 			ActFunc AFFromString(str)
 * 
 * DESCRIPTION : 	Returns a pointer to the activation function which
 * 					matches the input string.
 * 
 * INPUTS :
 * 		const char *str		String naming the type of activation 
 * 							function to use, can be any of the 
 * 							following: 'leaky_relu'|'relu'|'linear'|
 * 							'softplus'|'sigmoid'|'tanh'
 * 
 * RETURNS :
 * 		ActFunc		Pointer to the activation function
 * 
 * ********************************************************************/
ActFunc AFFromString(const char *str);



/***********************************************************************
 * NAME	: 		T* createArray(a,n)
 * 
 * DESCRIPTION : 	Template function to create an array of length n
 * 					given a pointer *a.
 * 
 * INPUTS : 
 * 		T	*a		Pointer to new array.
 * 		int	n		Length of the array.
 * 
 * RETURNS : 
 * 		T	*a		Pointer to the new array
 * 
 * ********************************************************************/
template <class T> T* createArray(T *a, int n);

/***********************************************************************
 * NAME	: 		void destroyArray(a)
 * 
 * DESCRIPTION : 	Template function to destroy a 1D array
 * 
 * INPUTS : 
 * 		T	*a		Pointer to the array
 * 
 * ********************************************************************/
template <class T> void destroyArray(T *a);

/***********************************************************************
 * NAME	: 		T* create2DArray(a,n0,n1)
 * 
 * DESCRIPTION : 	Template function to create an array with dimensions
 * 					(n0,n1) given a pointer **a.
 * 
 * INPUTS : 
 * 		T	**a		Pointer to new array.
 * 		int	n0		Length of the 1st dimension of the array.
 * 		int	n1		Length of the 2nd dimension of the array.
 * 
 * RETURNS : 
 * 		T	**a		Pointer to the new array
 * 
 * ********************************************************************/
template <class T> T** create2DArray(T **a, int n0, int n1);

/***********************************************************************
 * NAME	: 		void destroy2DArray(a,n0)
 * 
 * DESCRIPTION : 	Template function to destroy a 2D array
 * 
 * INPUTS : 
 * 		T	**a		Pointer to the array
 * 		int	n0		Length of the 1st dimension of the array.
 * 
 * ********************************************************************/
template <class T> void destroy2DArray(T **a, int n0);


/***********************************************************************
 * NAME	: 		T* appendToArray(a,na,b,nb,n)
 * 
 * DESCRIPTION : 	Template function to append two arrays.
 * 
 * INPUTS : 
 * 		T	*a		Pointer to first array (second will be appended to 
 * 					this).
 * 		int	na		Length of the 1st array.
 * 		T	*b		Pointer to second array
 * 		int	nb		Length of the 2nd array.
 * 
 * OUTPUTS :
 * 		int	n		Total array length.
 * 
 * RETURNS : 
 * 		T	*a		Pointer to the new array
 * 
 * ********************************************************************/
template <class T> T* appendToArray(T *a, int na, T *b, int nb, int *n);


/***********************************************************************
 * NAME	: 		T* arrayCopy(in,out,n)
 * 
 * DESCRIPTION : 	Template function to copy one array to another.
 * 
 * INPUTS : 
 * 		T	*in		Pointer to input array
 * 		T	*out	Pointer to output array
 * 		int	n		Length of arrays
 * 
 * 
 * ********************************************************************/
template <typename T> void arrayCopy(T *in, T *out, int n);



/***********************************************************************
 * NAME : Matrix
 * 
 * DESCRIPTION : The purpose of this object is to store a matrix as a 2D
 * 				array. It can be filled with zeros (default) or with the
 * 				values from another 2D array. Simple operations can be 
 * 				performed using the member functions of the object; more
 * 				advanced operations can be done be passing the object to 
 * 				one of the functions in matrixmath.h.
 * 
 * ********************************************************************/
class Matrix {
	public:
		/* Initialize matrix with zeros */
		Matrix(int*);
		Matrix(int,int);
		
		/* initialize matrix with existing data */
		Matrix(int*,double**);
		Matrix(int,int,double**);
		
		/* copy constructor */
		Matrix(const Matrix &obj);
	
		/* destructor */
		~Matrix();
		
		/* fill the entire matrix with zeros */
		void FillZeros();
		
		/* multiply each elements by some number */
		void TimesScalar(double);
		
		/* divide each element by a scalar */
		void DivideScalar(double);
		
		/* add each element to a scalar */
		void AddScalar(double);
		
		/* subtract a scalar from each elements */
		void SubtractScalar(double);
		
		/* subtract each element from a scalar */
		void SubtractFromScalar(double);
		
		/* Don't use this with massive arrays*/
		void PrintMatrix();
		void PrintMatrix(const char *);
		
		/* copy another matrix into the current one */
		void CopyMatrix(Matrix&);
		
		/* fill this matrix with the data from another of the same size*/
		void FillMatrix(double**);
		void FillMatrix(float**);
		
		/* return the data from this matrix into a 2D array */
		void ReturnMatrix(double**);
		void ReturnMatrix(float**);
		
		/* this is the data array with its shape and total size*/
		int shape[2];
		int size;
		double **data = NULL;
	private:
		bool DeleteData;
		
};


/***********************************************************************
 * NAME : Matrix
 * 
 * DESCRIPTION : The purpose of this object is to store a an array of
 * 				matrix objects. This object is particularly useful for
 * 				storing the weight/bias matrices of a neural network and
 * 				for use as temporary propagation matrices.
 * 
 * ********************************************************************/
class MatrixArray {
	public:
		/* constructor of the array */
		MatrixArray(int,int*);
		MatrixArray(unsigned char **memstart);
		
		/* copy constructor */
		MatrixArray(const MatrixArray &obj);
		
		/* destructor */
		~MatrixArray();
		
		/* random initialization of the array contained in this object */
		void RandomInit(float);
		
		/* this is the array of matrices contained within this object */
		int n;
		Matrix **matrix;
	private:

};



/***********************************************************************
 * NAME : 			MatrixMultiply(a,b,aT,bT,out)
 * 
 * DESCRIPTION : 	This will multiply two matrices (element-wise).
 * 
 * INPUTS : 
 * 			Matrix	&a	First matrix
 * 			Matrix	&b	Second matrix
 * 			bool	aT	True if we are to transpose matrix a first
 * 			bool	bT	True if we are to transpose matrix b first
 *
 * OUTPUTS : 
 * 			Matrix 	&out	Output matrix
 *
 * ********************************************************************/
void MatrixMultiply(Matrix &a, Matrix &b, bool aT, bool bT, Matrix &out);

/***********************************************************************
 * NAME : 			MatrixDot(a,b,aT,bT,out)
 * 
 * DESCRIPTION : 	This will multiply two matrices a x b.
 * 
 * INPUTS : 
 * 			Matrix	&a	First matrix 
 * 			Matrix	&b	Second matrix
 * 			bool	aT	True if we are to transpose matrix a first
 * 			bool	bT	True if we are to transpose matrix b first
 *
 * OUTPUTS : 
 * 			Matrix 	&out	Output matrix = a x b
 *
 * ********************************************************************/
void MatrixDot(Matrix &a, Matrix &b, bool aT, bool bT, Matrix &out);

/***********************************************************************
 * NAME : 			MatrixAdd(a,b,out)
 * 
 * DESCRIPTION : 	This will add matrices a and b.
 * 
 * INPUTS : 
 * 			Matrix	&a	First matrix
 * 			Matrix	&b	Second matrix
 * 			bool	aT	True if we are to transpose matrix a first
 * 			bool	bT	True if we are to transpose matrix b first
 *
 * OUTPUTS : 
 * 			Matrix 	&out	Output matrix = a + b
 *
 * ********************************************************************/
void MatrixAdd(Matrix &a, Matrix &b, bool aT, bool bT, Matrix &out);

/***********************************************************************
 * NAME : 			MatrixSubtract(a,b,aT,bT,out)
 * 
 * DESCRIPTION : 	This will subtract matrix b from matrix a.
 * 
 * INPUTS : 
 * 			Matrix	&a	First matrix
 * 			Matrix	&b	Second matrix
 * 			bool	aT	True if we are to transpose matrix a first
 * 			bool	bT	True if we are to transpose matrix b first
 *
 * OUTPUTS : 
 * 			Matrix 	&out	Output matrix = a - b
 *
 * ********************************************************************/
void MatrixSubtract(Matrix &a, Matrix &b, bool aT, bool bT, Matrix &out);

/***********************************************************************
 * NAME : 			ApplyFunctionToMatrix(a,AF,o)
 * 
 * DESCRIPTION : 	This will apply an activation function to each 
 * 					element within a matrix.
 * 
 * INPUTS : 
 * 			Matrix	&a		Input matrix - shape (n,m)
 * 			ActFunc	AF		Activation function 
 * 							(see activationfunctions.h)
 *
 * OUTPUTS : 
 * 			Matrix 	&o		Output matrix = AF(a) - shape (n,m)
 *
 * ********************************************************************/
void ApplyFunctionToMatrix(Matrix &a, ActFunc AF, Matrix &o);

/***********************************************************************
 * NAME : 			ApplyFunctionToMatrix(a,AF)
 * 
 * DESCRIPTION : 	This will apply an activation function to each 
 * 					element within a matrix.
 * 
 * INPUTS : 
 * 			Matrix	&a		Input AND output matrix - shape (n,m)
 * 							a = AF(a)
 * 			ActFunc	AF		Activation function 
 * 							(see activationfunctions.h)
 *
 *
 * ********************************************************************/
void ApplyFunctionToMatrix(Matrix &a, ActFunc AF);


/***********************************************************************
 * NAME : 			AddBiasVectorToMatrix(a,b)
 * 
 * DESCRIPTION : 	Adds a bias vector of shape (1,m) to a matrix of 
 * 					shape (n,m).
 * 
 * INPUTS : 
 * 			Matrix	&a		Input AND output matrix - shape (n,m)
 * 							a = a + b
 * 			Matrix	&b		Bias vector (Matrix) shape (1,m)
 *
 *
 * ********************************************************************/
void AddBiasVectorToMatrix(Matrix &a, Matrix &b);

/***********************************************************************
 * NAME : 	void BackPropagate(w,b,Deltas,a,y,AFGrad,CFDelta,L1,L2,wGrad,bGrad)
 * 
 * DESCRIPTION :  	Propagates errors backwards through the network in
 * 					order to calculate the gradients required for 
 * 					training weights and biases.
 * 
 * INPUTS : 
 * 		MatrixArray		&w		Weight matrices
 * 		MatrixArray		&b		Bias matrices
 * 		MatrixArray		&Deltas	Deltas calculated by cost functions 
 * 								(these are calucalted in function, the 
 * 								array is provided to save repeatedly 
 * 								recreating the array)
 * 		MatrixArray		&a		The outputs of each layer
 * 		Matrix			y		The network training output
 * 		ActFunc			*AFGrad	Array of functions providing gradients 
 * 								to activation functions.
 * 		CostFuncDelta	*CFDelta	Cost function-specific delta function
 * 		double			L1			L1 regularization parameter
 * 		double			L2 			L2 regularization parameter
 * 
 * OUTPUTS : 
 * 		MatrixArray		&wGrad	Array of weight matrix gradients
 * 		MatrixArray		&bGrad	Array of bias matrix gradients
 * 
 * ********************************************************************/
void BackPropagate(MatrixArray &w, MatrixArray &b, 
	MatrixArray &Deltas, MatrixArray &a, Matrix y, ActFunc *AFgrad, 
	CostFuncDelta *CFDelta, double L1, double L2, 
	MatrixArray &wGrad, MatrixArray &bGrad);

/***********************************************************************
 * NAME : 	void _BackPropDeltas(dlin,w,AFGrad,a,dlout)
 * 
 * DESCRIPTION :  	Propagates errors backwards through the network in
 * 					order to calculate the gradients required for 
 * 					training weights and biases.
 * 
 * INPUTS : 
 * 		Matrix		&dlin	Deltas from layer
 * 		Matrix		&w		Weight matrix
 * 		Matrix		&a		The outputs of this layer
 * 		ActFunc		AFGrad	Gradient of this layers activation function
 * 
 * OUTPUTS : 
 * 		Matrix		&dlout	Next set of deltas further down the matrix 
 * 					(backwards)
 * 
 * ********************************************************************/
void _BackPropDeltas(Matrix &dlin, Matrix &w, ActFunc AFGrad, Matrix &a, Matrix &dlout);



/***********************************************************************
 * NAME : 	BoxCox
 * 
 * DESCRIPTION : 	Box-Cox transforms data.
 * 
 * INPUTS : 
 * 		int 	n		The number of elements
 * 		float 	*x		The input array to be transformed
 * 		float 	lambda	The power of the transform
 * 		float 	shift	The shift applied to x
 * 		float 	mu		Mean value to transform by
 * 		float 	sig		Standard deviation
 * 		
 * OUTPUTS :
 * 		float 	*xt		Transformed array
 * 
 * ********************************************************************/
void BoxCox(int n, float *x, float lambda, float shift, float mu, float sig, float *xt);

/***********************************************************************
 * NAME : 	float BoxCox(x,lambda,shift,mu,sig)
 * 
 * DESCRIPTION : 	Box-Cox transforms data.
 * 
 * INPUTS : 
 * 		float 	x		The input value to be transformed
 * 		float 	lambda	The power of the transform
 * 		float 	shift	The shift applied to x
 * 		float 	mu		Mean value to transform by
 * 		float 	sig		Standard deviation
 * 		
 * RETURNS :
 * 		float 	*xt		Transformed value
 * 
 * ********************************************************************/
float BoxCox(float x, float lambda, float shift, float mu, float sig);


/***********************************************************************
 * NAME :  void ReverseBoxCox(n,xt,lambda,shift,mu,sig,x)
 * 
 * DESCRIPTION : 	Reverses the Box-Cox transform on data.
 * 
 * INPUTS : 
 * 		int 	n		The number of elements
 * 		float 	*xt		The input array to be transformed back
 * 		float 	lambda	The power of the transform
 * 		float 	shift	The shift applied to x
 * 		float 	mu		Mean value to transform by
 * 		float 	sig		Standard deviation
 * 		
 * OUTPUTS :
 * 		float 	*x		Transformed array
 * 
 * ********************************************************************/
void ReverseBoxCox(int n, float *xt, float lambda, float shift, float mu, float sig, float *x);

/***********************************************************************
 * NAME :  float ReverseBoxCox(xt,lambda,shift,mu,sig)
 * 
 * DESCRIPTION : 	Reverses the Box-Cox transform on data.
 * 
 * INPUTS : 
 * 		float 	xt		The input value to be transformed back
 * 		float 	lambda	The power of the transform
 * 		float 	shift	The shift applied to x
 * 		float 	mu		Mean value to transform by
 * 		float 	sig		Standard deviation
 * 		
 * RETURNS :
 * 		float 	x		Reverse-transformed value.
 * 
 * ********************************************************************/
float ReverseBoxCox(float xt, float lambda, float shift, float mu, float sig);

/***********************************************************************
 * NAME : 	double cliplog(x,min)
 * 
 * DESCRIPTION : This function will clip values of x before passing to 
 * 					the log function so that there are not -inf's, a 
 * 					good value for min is 1e-40
 * 
 * INPUTS :
 * 		double	x	The value to be loged
 * 		double 	mn	Minimum value
 * 
 * RETURNS :
 * 		double	log(x)	Log of x
 * 	
 * ********************************************************************/
double cliplog(double x, double mn);


/*******************************************************************
 * NAME : 		double crossEntropyCost(h,y,w,L1,L2)
 * 
 * DESCRIPTION : 	This function will calculate the cross-entropy cost 
 * 					function.
 * 
 * INPUTS :
 * 		Matrix		&h 		Matrix containing the ouputs of the neural 
 * 							network, shape (m,K), where m = number of 
 * 							samples, K = number of output nodes.
 *		Matrix		&y 		Matrix containing the one-hot target values 
 * 							for h, shape (m,K).
 *		MatrixArray	&w	 	MatrixArray object containing network 
 * 							weights.
 * 		double 		L1 		L1 regularization parameter.
 * 		double		L2 		L2 regularization parameter.
 * 
 * RETURNS :
 * 		double	J 	cost (set L1=0.0 and L2=0.0 for classification cost). 
 * 
 ******************************************************************/
double crossEntropyCost(Matrix &h, Matrix &y, MatrixArray &w, double L1, double L2);

/*******************************************************************
 * NAME : 		void crossEntropyDelta(h,y,InvAFGrad,Deltas)
 * 
 * DESCRIPTION : 	This function will calculate the deltas of the 
 * 					network.
 * 
 * INPUTS :
 * 		Matrix		&h 		Matrix containing the ouputs of the neural 
 * 							network, shape (m,K), where m = number of 
 * 							samples, K = number of output nodes.
 *		Matrix		&y 		Matrix containing the one-hot target values 
 * 							for h, shape (m,K).
 *		ActFunc		InvAFGrad	Gradient of the inverse of the activation
 * 								function.
 * 
 * OUTPUTS :
 * 		Matrix	Deltas 	The deltas calculatedfrom the results.
 * 
 ******************************************************************/
void crossEntropyDelta(Matrix &h, Matrix &y, ActFunc InvAFGrad, Matrix &Deltas);

/*******************************************************************
 * NAME : 		double meanSquaredCost(h,y,w,L1,L2)
 * 
 * DESCRIPTION : 	This function will calculate the mean-squared cost 
 * 					function.
 * 
 * INPUTS :
 * 		Matrix		&h 		Matrix containing the ouputs of the neural 
 * 							network, shape (m,K), where m = number of 
 * 							samples, K = number of output nodes.
 *		Matrix		&y 		Matrix containing the one-hot target values 
 * 							for h, shape (m,K).
 *		MatrixArray	&w	 	MatrixArray object containing network 
 * 							weights.
 * 		double 		L1 		L1 regularization parameter.
 * 		double		L2 		L2 regularization parameter.
 * 
 * RETURNS :
 * 		double	J 	cost (set L1=0.0 and L2=0.0 for classification cost). 
 * 
 ******************************************************************/
double meanSquaredCost(Matrix &h, Matrix &y, MatrixArray &w, double L1, double L2);

/*******************************************************************
 * NAME : 		void meanSquaredDelta(h,y,InvAFGrad,Deltas)
 * 
 * DESCRIPTION : 	This function will calculate the deltas of the 
 * 					network.
 * 
 * INPUTS :
 * 		Matrix		&h 		Matrix containing the ouputs of the neural 
 * 							network, shape (m,K), where m = number of 
 * 							samples, K = number of output nodes.
 *		Matrix		&y 		Matrix containing the one-hot target values 
 * 							for h, shape (m,K).
 *		ActFunc		InvAFGrad	Gradient of the inverse of the activation
 * 								function.
 * 
 * OUTPUTS :
 * 		Matrix	Deltas 	The deltas calculatedfrom the results.
 * 
 ******************************************************************/
void meanSquaredDelta(Matrix &h, Matrix &y, ActFunc InvAFGrad, Matrix &Deltas);

/* these typedefs will be used so that we can switch cost functions */
typedef double (*CostFunc)(Matrix&,Matrix&,MatrixArray&,double, double);
typedef double (*CostFuncDelta)(Matrix&,Matrix&,ActFunc,Matrix&);


class Network {
	public:
	
		/* Network object constructors */
		/* Start a new network */
		Network(int,int*,float,float);
		/* load from a stored file */
		Network(const char*);
		/* load from a memory address */
		Network(unsigned char *);
		
		/* Copy constructor */
		Network(const Network &obj);
		
		/* Destructor */
		~Network();
		
		/*Changing network*/
		void ResetWeights();
		void ResetNetwork();
		void ChangeNetworkArchitecture();
		void InsertHiddenLayer();
		void UseBestWeights();	
		
		/* edit the activation functions with an integer code*/
		void SetActivationFunction(int,int);
		
		/* set the training algorithm */
		void SetTrainingAlgorithm(int);
		
		/* set the cost function */
		void SetCostFunction(int);
		
		void SetL1();
		void SetL2();
		
		/*Input data*/
		void InputTrainingData(int*,double*,int,int*,bool);
		void InputCrossValidationData(int*,double*,int,int*,bool);
		void InputTestData(int*,double*,int,int*,bool);
		
		/*Training network*/
		void Train();
		
		/*Classification*/
		void ClassifyData(int*,float*,int,int*);
		void ClassifyData(int*,float*,int,int*,float*);

		/*Save network to file*/
		void Save(const char*);

		/*Return parameters*/
		int GetnSteps();
		void GetTrainingProgress(float*,float*,float*,float*,float*,float*,float*);
		void GetTrainingAccuracy(float*,float*);
		void GetCrossValidationAccuracy(float*,float*);
		void GetTestAccuracy(float*,float*);
		int GetL();
		void Gets(int*);
	private:
		/*Network architecture*/
        int L; //number of layers
        int *s; //number of units in each layer
        bool Trained;
        
        /*Regularization parameters*/
        double L1, L2;
        
        /*Weight matrices*/
        MatrixArray *ThetaW, *ThetaWGrad;
        
        /*Bias matrices*/
        MatrixArray *ThetaB, *ThetaBGrad;
        
        /*Delta matrices - these will hopefully speed up training if they're only created once for each training set*/
        MatrixArray *Delta;
        
        /*Activation function lists*/
        int *AFCodes; //integer code corresponding to the type of neuron
        ActFunc *AF; //Actual activation functions will be stored here
        ActFunc *AFgrad; //These will be used to calculate the gradient during back propagation (and therefore need ot be inverse)

        /*Cost function stuff*/
        int CFcode; //integer corresponding to the type of cost function to use
        CostFunc *CF; //The actual cost function pointer
        CostFuncDelta *CFDelta; //Pointer to the object which calculates the deltas

        /*Training data*/
        Matrix *Xt; //Training data matrix, shape (mt,s[0])
        int *yt0 = NULL; //integer class labels, shape (mt,)
        Matrix *yt; //one-hot matrix, shape (mt,s[-1])
        bool TData; //True if data exists
        int mt; //number of samples
        MatrixArray *at, *zt; //Training set propagation arrays

        /*Cross-validation data*/
        Matrix *Xc; 
        int *yc0 = NULL; 
        Matrix *yc; 
        bool CData; 
        int mc;
        MatrixArray *ac, *zc;

        /*Test data*/
        Matrix *Xtest;
        int *ytest0 = NULL; 
        Matrix *ytest;
        bool TestData; 
        int mtest; 
        MatrixArray *atest, *ztest; 
        
        /*Storing Best Weights (based on CV accuracy)*/
        MatrixArray *BestThetaW, *BestThetaB; //stored best weights
        bool Best; // whether or not the best weights are being stored
        float BestAccuracy; //best CV accuracy saved during training

		/*Training progress arrays*/
		float *Jt, *JtClass, Jc, Jtest; //Cost function
		float *At, *Ac, *Atest; //Percentage classification accuracy  
		int nSteps;

		/*Private functions to write*/
		void _GetOneHotClassLabels(int*,Matrix&);
		void _CreatePropagationMatrices(MatrixArray&,MatrixArray&);
		void _InitDeltaMatrices();
		void _CalculateStepGD();
		void _CalculateStepNesterov();
		void _CalculateStepRMSProp();
		void _CalculateStepRProp();
		void _CheckCostGradient();
		void _PopulateActivationFunctions();
		void _SetCostFunction();
		void _TrainNetwork();
		void _LoadNetwork();
		void _AppendToArray(float*);
		void _InitWeights();
};	



/***********************************************************************
 * NAME : NetworkFunc
 * 
 * DESCRIPTION : An untrainable artificial neural network object. The 
 * 				point of this is to load an ANN from memory and use it
 * 				purely for prediction.
 * 
 * ********************************************************************/
class NetworkFunc {
	public:
		/* this object must be loaded from memory address */
		NetworkFunc(unsigned char*, const char*, const char*, const char*);
		
		/*or file*/
		//NetworkFunc(const char*,  const char*, const char*, const char*);
		
		/* or a blank one*/
		NetworkFunc(int, int*, const char*,const char *, const char*);
		
		
		/* copy constructor */
		NetworkFunc(const NetworkFunc &obj);
		
		/*Destructor */
		~NetworkFunc();
		
		/*Predict the output based on an input matrix*/
		void Predict(int, float **, float **);

        /*Weights and biases*/
        MatrixArray *W_, *B_;

		/* number of layers/nodes per layer */
		int L_;
		int *s_;
		
		/* rescaling parameters */
		float *scale0_, *scale1_;
		
	private:

		
		
        /*Activation function lists*/
        int HiddenAFCode_; //integer code corresponding to the type of neuron
        int OutputAFCode_; //integer code corresponding to the type of neuron
        ActFunc *AF_; //Actual activation functions will be stored here

        /*Cost function stuff*/
        CostFunc CF_; //The actual cost function pointer
        

        
        /*store the training and validation costs*/
        int nEpoch_;
        float *Jt_, *Jc_;
        
        /* private functions*/
        
        /* a function to populate the activation functions */
        void _PopulateActivationFunctions(const char *, const char *);
        
        /* another to set the cost function */
        void _SetCostFunction(const char *);
	
		/* A functionw hich creates a matrix array that allows us to
		propagate data through the network */
		MatrixArray* _CreatePropagationMatrices(int);

		/* This function will rescale the output matrix so that produces
		the expected values */
		void _RescaleOut(Matrix &);
};



/***********************************************************************
 * NAME : 	void SeedRandom()
 * 
 * DESCRIPTION : 	Seeds the random number generator using the time.
 * 
 * ********************************************************************/
void SeedRandom();

/***********************************************************************
 * NAME : 	int	RandomNumber(Range)
 * 
 * DESCRIPTION : 	Provides random integers within a specified range.
 * 
 * INPUTS : 
 * 		int		*Range		2-element array defining the upper and lower
 * 							limits for the output.
 * 
 * RETURNS : 
 * 		int		RandomNumber	Single random integer
 * 
 * ********************************************************************/
int RandomNumber(int *Range);

/***********************************************************************
 * NAME : 	int	RandomNumber(R0,R1)
 * 
 * DESCRIPTION : 	Provides random integers within a specified range.
 * 
 * INPUTS : 
 * 		int		R0		Lower limit of the output range
 * 		int		R1		Upper limit of the output range
 * 
 * RETURNS : 
 * 		int		RandomNumber	Single random integer
 * 
 * ********************************************************************/
int RandomNumber(int R0, int R1);

/***********************************************************************
 * NAME : 	float RandomNumber(Range)
 * 
 * DESCRIPTION : 	Provides random floats within a specified range.
 * 
 * INPUTS : 
 * 		float 	*Range		2-element array defining the upper and lower
 * 							limits for the output.
 * 
 * RETURNS : 
 * 		float		RandomNumber	Single random float
 * 
 * ********************************************************************/
float RandomNumber(float *Range);

/***********************************************************************
 * NAME : 	float	RandomNumber(R0,R1)
 * 
 * DESCRIPTION : 	Provides random floats within a specified range.
 * 
 * INPUTS : 
 * 		float		R0		Lower limit of the output range
 * 		float		R1		Upper limit of the output range
 * 
 * RETURNS : 
 * 		float		RandomNumber	Single random float
 * 
 * ********************************************************************/
float RandomNumber(float R0, float R1);

/***********************************************************************
 * NAME : 	float	RandomNumber(R0,R1)
 * 
 * DESCRIPTION : 	Provides random floats within a specified 
 * 					logarithmic range.
 * 
 * INPUTS : 
 * 		float		R0		Lower limit of the output range
 * 		float		R1		Upper limit of the output range
 * 
 * RETURNS : 
 * 		float		RandomNumber	Single random float
 * 
 * ********************************************************************/
float RandomLogRange(float R0, float R1);

/***********************************************************************
 * NAME : 	float	RandomNumber(R0,R1)
 * 
 * DESCRIPTION : 	Provides random floats within a specified 
 * 					logarithmic range.
 * 
 * INPUTS : 
 * 		float 	*Range		2-element array defining the upper and lower
 * 							limits for the output.
 * 
 * RETURNS : 
 * 		float		RandomNumber	Single random float
 * 
 * ********************************************************************/
float RandomLogRange(float *Range);


/***********************************************************************
 * NAME : 	unsigned char * readArray(p,v,n)
 * 
 * DESCRIPTION : Reads in a 1D floating point array from memory given a 
 * 				pointer.
 * 
 * INPUTS : 
 * 		unsigned char	*p	Pointer to the memory address to start from
 * 		
 * OUTPUTS : 
 * 		float 			*v 	This will contain the array of floats
 * 		int				*n	The number of elements within v
 * 
 * RETURNS : 
 * 		unsigned char	*p	New pointer address
 * 
 * ********************************************************************/
unsigned char * readArray(unsigned char *p, float **v, int *n);


/***********************************************************************
 * NAME : 	unsigned char * readArray(p,v,n)
 * 
 * DESCRIPTION : Reads in a 1D floating point array from memory given a 
 * 				pointer.
 * 
 * INPUTS : 
 * 		unsigned char	*p	Pointer to the memory address to start from
 * 		
 * OUTPUTS : 
 * 		double 			*v 	This will contain the array of floats
 * 		int				*n	The number of elements within v
 * 
 * RETURNS : 
 * 		unsigned char	*p	New pointer address
 * 
 * ********************************************************************/
unsigned char * readArray(unsigned char *p, double **v, int *n);


/***********************************************************************
 * NAME : 	unsigned char * readArray(p,v,n)
 * 
 * DESCRIPTION : Reads in a 1D integer array from memory given a 
 * 				pointer.
 * 
 * INPUTS : 
 * 		unsigned char	*p	Pointer to the memory address to start from
 * 		
 * OUTPUTS : 
 * 		int 			*v 	This will contain the array of floats
 * 		int				*n	The number of elements within v
 * 
 * RETURNS : 
 * 		unsigned char	*p	New pointer address
 *  
 * ********************************************************************/
unsigned char * readArray(unsigned char *p, int **v, int *n);

/***********************************************************************
 * NAME : 	unsigned char * readArray(p,v,shape)
 * 
 * DESCRIPTION : Reads in a 2D floating point array from memory given a 
 * 				pointer.
 * 
 * INPUTS : 
 * 		unsigned char	*p	Pointer to the memory address to start from
 * 		
 * OUTPUTS : 
 * 		float 			**v 	This will contain the array of floats
 * 		int				*shape	The number of elements within each 
 * 								dimension of v
 * 
 * RETURNS : 
 * 		unsigned char	*p	New pointer address
 *  
 * ********************************************************************/
unsigned char * readArray(unsigned char *p, float ***v, int *shape);

/***********************************************************************
 * NAME : 	float L1Regularization(w,L1,m)
 * 
 * DESCRIPTION : Calculates the L1 regularization.
 * 
 * INPUTS : 
 * 		MatrixArray	&w	Network weight matrices
 * 		float		L1	L1 regularization parameter
 * 		int			m	Number of samples.
 * 
 * RETURNS : 
 * 		float 	The additional cost due to regularization.
 * 
 * ********************************************************************/
float L1Regularization(MatrixArray &w, float L1, int m);

/***********************************************************************
 * NAME : 	float L2Regularization(w,L1,m)
 * 
 * DESCRIPTION : Calculates the L2 regularization.
 * 
 * INPUTS : 
 * 		MatrixArray	&w	Network weight matrices
 * 		float		L2	L2 regularization parameter
 * 		int			m	Number of samples.
 * 
 * RETURNS : 
 * 		float 	The additional cost due to regularization.
 * 
 * ********************************************************************/
float L2Regularization(MatrixArray &w, float L2, int m);


/***********************************************************************
 * NAME : 	ApplyRegGradToMatrix(w,wGrad,L1,L2,m)
 * 
 * DESCRIPTION : Applies regularization to weight gradient matrix.
 * 
 * INPUTS : 
 * 		MatrixArray	&w		The weight matrices.
 * 		MatrixArray	&wGrad	The gradients of the weights from the 
 * 							back-propagation algorithm.
 * 		double		L1		L1 regularization parameter.
 * 		double		L2		L2 regularization parameter.
 * 		int			m		The number of samples.
 * 
 * ********************************************************************/
void ApplyRegGradToMatrix(MatrixArray &w, MatrixArray &wGrad,double L1, double L2, int m);

/***********************************************************************
 * NAME : 	void softmax(z,sm)
 * 
 * DESCRIPTION : Calcualtes the softmax function for the last layer of
 * 					a neural network.
 * 
 * INPUTS : 	
 * 		Matrix	z	The output of the final neural network layer before
 * 					any activation function has been applied to it.
 * 
 * OUTPUTS : 
 * 		Matrix	sm	The softmax function output, effectively a 
 * 					probability.
 * 
 * ********************************************************************/
void softmax(Matrix z, Matrix &sm);

#endif
