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

template <class T>
void PArr(T* arr, unsigned count)
{
	for (int i = 0; i < count; i++)
	{
		pr(arr[i]);
	}
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