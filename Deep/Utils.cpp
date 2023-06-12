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
