#include "pch.h"
#include "Utils.h"

void PrintWeights(unsigned* layout, unsigned layoutSize, float* weights,unsigned layer)
{
	if (layer == 0)
	{
		unsigned weightsStart = 0;
		for (unsigned layoutIndex = 1; layoutIndex < layoutSize; layoutIndex++)
		{
			
			for (unsigned layerIndex = 0; layerIndex < layout[layoutIndex]; layerIndex++)
			{

				for (unsigned prevLayer = 0; prevLayer < layout[layoutIndex - 1]; prevLayer++)
				{

					std::cout << " |" << prevLayer + layout[layoutIndex - 1] * layerIndex << ": " << weights[weightsStart+prevLayer + layout[layoutIndex - 1] * layerIndex];

				}

				std::cout << std::endl;
			}
			weightsStart += layout[layoutIndex - 1] * layout[layoutIndex];
		}
	}
	else
	{
		unsigned weightsStart = 0;
		for (unsigned layoutIndex = 1; layoutIndex < layoutSize; layoutIndex++)
		{
			if (layer == layoutIndex)
			{
				for (unsigned layerIndex = 0; layerIndex < layout[layoutIndex]; layerIndex++)
				{

					for (unsigned prevLayer = 0; prevLayer < layout[layoutIndex - 1]; prevLayer++)
					{

						std::cout << " |" << prevLayer + layout[layoutIndex - 1] * layerIndex << ": " << weights[weightsStart + prevLayer + layout[layoutIndex - 1] * layerIndex];

					}

					std::cout << std::endl;
				}
			}
			weightsStart += layout[layoutIndex - 1] * layout[layoutIndex];
		}
	}
}

void PrintImg(float* img, unsigned width, unsigned height)
{
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			if (img[j * width + i] == 0)
			{
				std::cout << " ";
			}
			else
			{
				std::cout << "x";
			}
			
		}
		pr("");
	}


}

void Timer::Start()
{
	start = std::chrono::high_resolution_clock::now();
}

float Timer::Stop()
{
	auto stop = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float>  time = stop - start;
	return time.count();
}
