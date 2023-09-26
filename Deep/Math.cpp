#include "pch.h"
#include "Math.h"

namespace Math
{
	float Identity(float z)
	{
		return z;
	}
	float IdentityDerivative(float z)
	{
		return 1;
	}
	float Sigmoid(float z)
	{
		float expo = std::exp(-z);
		float result = 1 / (1 + std::exp(-z));
		return result;
	}

	float SigmoidDerivative(float z)
	{
		float result = Sigmoid(z) * (1 - Sigmoid(z));
		return result;
	}

	float CrossEntropy(float a, float y)
	{
		float result;
		if ( ((a==1) && (y==1)) || ((a == 0) && (y == 0)) )
		{
			
			result = 0;
		}
		//else if (((a == 1) && (y == 0)) || ((a ==0) && (y == 1)))
		//{
		//	result = std::numeric_limits<float>::max();
		//}
		else {
			result = -((y * std::log(a)) + ((1 - y) * std::log(1 - a)));
		}
		
		return result;
	}
	//Derivative with respect to a
	float CrossEntropyCostDerivative(float z, float a, float y)
	{
		return(a - y);
	}










	void MultiplyMatrix(const float* A,const float* B, float* target, size_t nA, size_t nB, size_t m)
	{
		size_t colChunk = 0;
		while (colChunk < nB)
		{

			size_t row = 0;
			while (row < nA)
			{
				size_t tile = 0;
				while (tile < nB)
				{
					size_t tileRow = 0;
					while (tileRow < 16)
					{
						size_t index = 0;
						target[row * nB + colChunk + index] = 0;
						while (index < 16)
						{
							target[row * nB + colChunk + index] +=
								A[row*m+ colChunk +index] *
								B[tile * m +tileRow*nB+ colChunk] ;

							index++;
						}
						tileRow++;
					}
					tile += 16;
				}
				row++;
			}
			colChunk += 16;
		}
		

	}




}
