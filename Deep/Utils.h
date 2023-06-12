#pragma once

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

