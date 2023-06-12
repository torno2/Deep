#pragma once

#include <assert.h>
#include <iostream>

#define pr(x) std::cout << x <<'\n'


namespace Math{

	template <class T>
	class Matrix
	{
	public:
		Matrix() = delete;

		Matrix(T* a, int rowc, int colc) : Arr(a), RowCount(rowc), ColCount(colc) { }
		

		Matrix(const Matrix<T>& other) : RowCount(other.RowCount), ColCount(other.ColCount) 
		{
			T* temp = new T[RowCount * ColCount];
			unsigned i = 0;
			while (i < RowCount * ColCount)
			{
				
				

				temp[i] = other.Arr[i];
				i++;
			}
			Arr = temp;
		}


		~Matrix()
		{
			delete[] Arr;
		}




		T& operator[](const unsigned i)
		{
			return Arr[i];
		}


		
		const Matrix<T>& operator=(const Matrix<T>& other) const
		{
			return Matrix<T>(other);
		}


		const Matrix<T> operator+ (const Matrix<T>& other) const
		{
			assert(other.RowCount == RowCount && other.ColCount == ColCount);

			T* data = new T[RowCount * ColCount];
			unsigned i = 0;
			while (i < RowCount * ColCount)
			{
				data[i] = Arr[i] + other.Arr[i];
				i++;
			}
			;
			return Matrix<T>(data, RowCount, ColCount);
		}

		void operator+= (const Matrix<T>& other)
		{
			assert(other.RowCount == RowCount && other.ColCount == ColCount);
			unsigned i = 0;
			while (i < RowCount * ColCount)
			{
				Arr[i] += other.Arr[i];
				i++;
			}
		}


		const Matrix<T> operator* (const Matrix<T>& other) const
		{
			assert(other.ColCount == RowCount);

			T* temp = new T[other.RowCount * ColCount];
			unsigned i = 0;
			while (i < ColCount)
			{
				unsigned j = 0;
				while (j < other.RowCount)
				{
					temp[j + i * other.RowCount] = 0;
					unsigned k = 0;
					while (k < RowCount)
					{
						temp[j + i * other.RowCount] += Arr[k + i * RowCount] * other.Arr[j + k * other.ColCount];
						k++;
					}
					j++;
				}
				i++;
			}

			return Matrix<T>(temp, other.RowCount, ColCount);
		}

		void operator*= (const Matrix<T>& other)
		{
			assert(other.ColCount == RowCount);

			T* temp = new T[other.RowCount * ColCount];
			unsigned i = 0;
			while (i < ColCount)
			{
			unsigned j = 0;
				while (j < other.RowCount)
				{
					temp[j + i * other.RowCount] = 0;
					unsigned k = 0;
					while (k < RowCount)
					{
						temp[j + i * other.RowCount] += Arr[k + i * RowCount] * other.Arr[j + k * other.ColCount];
						k++;
					}
					j++;
				}
				i++;
			}

			RowCount = other.RowCount;

			delete[] Arr;
			Arr = temp;
		}


		const Matrix<T> GetTranspose()
		{	
			T* temp = new T[RowCount * ColCount];
			unsigned i = 0;
			while (i < ColCount)
			{
				unsigned j = i;
				while (j < RowCount)
				{
					temp[i + j * RowCount] = Arr[j + i * ColCount];
					j++;
				}
				i++;
			}
			return Matrix<T>(temp, ColCount, RowCount);
		}


		//Usefull for converting a row vector into a column vector and vice versa
		void Flip()
		{
			unsigned temp = RowCount;
			RowCount = ColCount;
			ColCount = temp;
		}


		//You can also strip rows and columns from the right and bottom using this.
		void Pad(unsigned rowCount , unsigned colCount)
		{
			T* temp = new T[rowCount * colCount];
			
			unsigned j = 0;
			while (j < colCount)
			{
				unsigned i = 0;
				while (i < rowCount)
				{
					if (((RowCount - i) > 0) && ((ColCount - j ) > 0))
					{
						temp[i + j * rowCount] = Arr[i + j * RowCount];
					}
					else 
					{
						temp[i + j * rowCount] = 0;
					}

					i++;
				}
				j++;
			}

			delete[] Arr;
			Arr = temp;
			RowCount = rowCount;
			ColCount = colCount;
		}




	public:
		T* Arr;
		unsigned RowCount, ColCount;




	public:
		friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& m)
		{

			unsigned j = 0;
			while (j < m.ColCount)
			{
				unsigned i = 0;
				while (i < m.RowCount)
				{
					os << m.Arr[i + j * m.RowCount] << ' ' << ' ';
					i++;

				}
				os << '\n' << '\n';

				j++;

			}
			return os;
		}


	};



}



