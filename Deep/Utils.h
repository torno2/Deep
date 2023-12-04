#pragma once
#include <iostream>

#define pr(x) std::cout << x <<'\n'

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
	os << "Vector Start" << std::endl;
	for (int i = 0; i < v.size(); i++)
	{
		os << i << " : " << v[i] << std::endl;
	}
	os << "Vector End";
	return os;
}

void PrintWeights(unsigned* layout, unsigned layoutSize, float* weights,unsigned layer = 0);

template <typename T>
void PArr(T* arr, unsigned count)
{
	for (int i = 0; i < count; i++)
	{
		pr(arr[i]);
	}
}

template <typename T>
unsigned ArrayThresholdViolationCheck(T* arr, unsigned count, T lower, T upper)
{
	unsigned counter = 0;
	for (unsigned i = 0; i < count; i++)
	{
		if (arr[i] > upper || arr[i] < lower)
		{
			counter++;
		}
	}

	return counter;
}

template <typename T>
unsigned ArrayMatchCheck(T* arr, unsigned count, T target)
{
	unsigned counter = 0;
	for (unsigned i = 0; i < count; i++)
	{
		if (arr[i] == target)
		{
			counter++;
		}
	}

	return counter;
}


class std::chrono::steady_clock::time_point;
struct Timer
{
private:
	std::chrono::steady_clock::time_point start;
public:
	void Start();
	float Stop();
};

void PrintImg(float* img, unsigned width, unsigned height);



void ThreadWorkloadDividerUtils(unsigned& start, unsigned& stop, unsigned workCount, unsigned thread, unsigned threadCount);

void ThreadWorkloadDividerWithPaddingUtils(unsigned& start, unsigned& stop, unsigned workCount, unsigned thread, unsigned threadCount , unsigned padding);