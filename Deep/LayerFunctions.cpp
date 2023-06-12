#pragma once
#include "pch.h"
#include "LayerFunctions.h"

namespace TNNT
{
	namespace LayerFunctions
	{

		void FullyConnectedFeedForward(NetworkPrototype* n)
		{
			unsigned layerIndex = 0;
			while (layerIndex < n->m_LayerLayout[n->m_PositionData.Layer].Nodes)
			{
				float weightedSum = 0;

				unsigned prevIndex = 0;
				while (prevIndex < n->m_LayerLayout[n->m_PositionData.Layer -1].Nodes)
				{
					weightedSum +=
						n->m_ABuffer[ (n->m_PositionData.A - n->m_LayerLayout[n->m_PositionData.Layer-1].Nodes) + prevIndex] *
						n->m_Weights[n->m_PositionData.Weights + n->m_LayerLayout[n->m_PositionData.Layer -1].Nodes * layerIndex + prevIndex];

					prevIndex++;
				}

				n->m_ZBuffer[layerIndex+n->m_PositionData.Z] = weightedSum + n->m_Biases[layerIndex+ n->m_PositionData.Biases];

				n->m_ABuffer[layerIndex+ n->m_PositionData.A] = n->m_Functions.NeuronFunctions[n->m_PositionData.Layer-1].f(n->m_ZBuffer[layerIndex + n->m_PositionData.Z]);

				layerIndex++;
			}

		}

		void FullyConnectedBackpropegateZ(NetworkPrototype* n)
		{
			
			const unsigned latterAPos = n->m_PositionData.A + n->m_LayerLayout[n->m_PositionData.Layer].Nodes;
			const unsigned latterZPos = n->m_PositionData.Z + n->m_LayerLayout[n->m_PositionData.Layer].Nodes;

			const unsigned latterBiases = n->m_PositionData.Biases + n->m_LayerLayout[n->m_PositionData.Layer].Biases;
			const unsigned latterWeights = n->m_PositionData.Weights + n->m_LayerLayout[n->m_PositionData.Layer].Weights;

			unsigned layerIndex = 0;
			while (layerIndex < n->m_LayerLayout[n->m_PositionData.Layer].Nodes)
			{
				float errorSum = 0;
				unsigned latterLayerIndex = 0;
				while (latterLayerIndex < n->m_LayerLayout[n->m_PositionData.Layer+1].Nodes)
				{
					errorSum += n->m_WeightsTranspose[latterWeights + n->m_LayerLayout[n->m_PositionData.Layer + 1].Nodes * layerIndex + latterLayerIndex] * n->m_DeltaZ[latterZPos + latterLayerIndex];
					latterLayerIndex++;
				}

				float z = n->m_ZBuffer[n->m_PositionData.Z + layerIndex];
				n->m_DeltaZ[n->m_PositionData.Z + layerIndex] = errorSum * n->m_Functions.NeuronFunctionsDerivatives[n->m_PositionData.Layer - 1].f(z);

				layerIndex++;
			}

		}

		void FullyConnectedBackpropegateBW(NetworkPrototype* n)
		{

			const unsigned prevAPos = n->m_PositionData.A - n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes;

			unsigned layerIndex = 0;
			while (layerIndex < n->m_LayerLayout[n->m_PositionData.Layer].Nodes)
			{

				const float dz = n->m_DeltaZ[n->m_PositionData.Z + layerIndex];
				n->m_DeltaBiases[n->m_PositionData.Biases + layerIndex] = dz;


				unsigned prevLayerIndex = 0;
				while (prevLayerIndex < n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes)
				{
					float a = n->m_ABuffer[prevAPos + prevLayerIndex];
					n->m_DeltaWeights[n->m_PositionData.Weights + n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes * layerIndex + prevLayerIndex] = a * dz;




					prevLayerIndex++;
				}

				layerIndex++;
			}
		
		}


		void ConvolutionLayerFeedForward(NetworkPrototype* n)
		{

			const unsigned subLayers = 3;
			const unsigned subLayerCount = n->m_LayerLayout[n->m_PositionData.Layer].Nodes / subLayers;

			const unsigned subLayerBiasesCount = n->m_LayerLayout[n->m_PositionData.Layer].Biases / subLayers;
			const unsigned subLayerWeightsCount = n->m_LayerLayout[n->m_PositionData.Layer].Weights / subLayers;

			const unsigned prevSubLayers = 1;
			const unsigned prevSubLayerCount = n->m_LayerLayout[n->m_PositionData.Layer-1].Nodes / prevSubLayers;

			const unsigned receptiveWidth = 5;
			const unsigned receptiveHeight = 5;

			const unsigned horizontalStride = 1;
			const unsigned verticalStride = 1;

			const unsigned imgWidth = sqrtf(n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes/ prevSubLayers);
			const unsigned imgHeight = sqrtf(n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes/ prevSubLayers);
			
			
			const unsigned prevLayerAPos = n->m_PositionData.A - n->m_LayerLayout[n->m_PositionData.Layer-1].Nodes;
			



			const unsigned startA = 0;

			unsigned subLayer = 0;
			while (subLayer < subLayers)
			{

				unsigned subLayerIndex = 0;
				while (subLayerIndex < subLayerCount)
				{
					//Beware "Edge"cases (The hint tells you to take a look at what happens when a receptive field goes outside of the image)
				
					const unsigned startA = ((subLayerIndex * verticalStride) / (imgHeight - receptiveHeight + 1))* imgWidth + (subLayerIndex * horizontalStride) % (imgWidth - receptiveWidth + 1);


					float weightedSum = 0;

					unsigned height = 0;
					while (height < receptiveHeight)
					{
						unsigned width = 0;
						while (width < receptiveWidth)
						{

							weightedSum +=
								n->m_Weights[n->m_PositionData.Weights + subLayerIndex * subLayerWeightsCount + height * receptiveWidth + width] *
								n->m_ABuffer[prevLayerAPos + startA + height * imgWidth + width ];


							width++;
						}
						height++;
					}

					n->m_ABuffer[n->m_PositionData.A + subLayerIndex + subLayer* subLayerCount] = n->m_Functions.NeuronFunctions[n->m_PositionData.Layer - 1].f(weightedSum + n->m_Biases[n->m_PositionData.Biases + subLayerIndex* subLayerBiasesCount]);

					subLayerIndex++;
				}
				subLayer++;
			}
		}

		void ConvolutionLayerBackpropegateZ(NetworkPrototype* n)
		{
			const unsigned subLayers = 3;
			const unsigned subLayerCount = n->m_LayerLayout[n->m_PositionData.Layer].Nodes / subLayers;

			const unsigned subLayerBiasesCount = n->m_LayerLayout[n->m_PositionData.Layer].Biases / subLayers;
			const unsigned subLayerWeightsCount = n->m_LayerLayout[n->m_PositionData.Layer].Weights / subLayers;
		}

  		void ConvolutionLayerBackpropegateBW(NetworkPrototype* n)
		{
			const unsigned subLayers = 3;
			const unsigned subLayerCount = n->m_LayerLayout[n->m_PositionData.Layer].Nodes / subLayers;

			const unsigned subLayerBiasesCount = n->m_LayerLayout[n->m_PositionData.Layer].Biases / subLayers;
			const unsigned subLayerWeightsCount = n->m_LayerLayout[n->m_PositionData.Layer].Weights / subLayers;

			const unsigned prevSubLayers = 1;
			const unsigned prevSubLayerCount = n->m_LayerLayout[n->m_PositionData.Layer].Nodes / prevSubLayers;

			const unsigned receptiveWidth = 5;
			const unsigned receptiveHeight = 5;

			const unsigned horizontalStride = 1;
			const unsigned verticalStride = 1;

			const unsigned imgWidth = sqrtf(n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes / prevSubLayers);
			const unsigned imgHeight = sqrtf(n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes / prevSubLayers);


			const unsigned prevLayerAPos = n->m_PositionData.A - n->m_LayerLayout[n->m_PositionData.Layer].Nodes;

			unsigned subLayer = 0;
			while (subLayer < subLayers)
			{
				unsigned weightIndex = 0;
				while (weightIndex < subLayerWeightsCount)
				{
					//TODO: COMPLETE THIS
					n->m_DeltaWeights[n->m_PositionData.Weights] =2 ;
				}

				subLayer++;
			}

		}


		void PoolingLayerFeedForward(NetworkPrototype* n)
		{
			const unsigned subLayers = 3;
			const unsigned subLayerCount = n->m_LayerLayout[n->m_PositionData.Layer].Nodes / subLayers;

			const unsigned subLayerBiases = n->m_LayerLayout[n->m_PositionData.Layer].Biases / subLayers;
			const unsigned subLayerWeights = n->m_LayerLayout[n->m_PositionData.Layer].Weights / subLayers;

			const unsigned prevSubLayers = 3;
			const unsigned prevSubLayerCount = n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes / prevSubLayers;


			const unsigned receptiveWidth = 2;
			const unsigned receptiveHeight = 2;

			const unsigned horizontalStride = 2;
			const unsigned verticalStride = 2;

			const unsigned imgWidth = sqrtf(n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes/ prevSubLayerCount);
			const unsigned imgHeight = sqrtf(n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes/ prevSubLayerCount);


			const unsigned prevLayerAPos = n->m_PositionData.A - n->m_LayerLayout[n->m_PositionData.Layer].Nodes;

			assert(prevSubLayers == subLayers);

			float check;
			float champ;

			unsigned subLayer = 0;
			while (subLayer < subLayers)
			{

				unsigned subLayerIndex = 0;
				while (subLayerIndex < subLayerCount)
				{
					//Beware "Edge"cases (The hint tells you to take a look at what happens when a receptive field goes outside of the image)
					const unsigned startA = ((subLayerIndex * verticalStride) / (imgHeight - receptiveHeight + 1)) * imgWidth + (subLayerIndex * horizontalStride) % (imgWidth - receptiveWidth + 1);


					
					champ = n->m_ABuffer[prevLayerAPos + subLayer * prevSubLayerCount + startA];

					unsigned height = 0;
					while (height < receptiveHeight)
					{
						unsigned width = 0;
						while (width < receptiveWidth)
						{
								
								check = n->m_ABuffer[prevLayerAPos+ subLayer* prevSubLayerCount + startA + width + height * imgWidth];
								if (check > champ)
								{
									champ = check;
								}


							width++;
						}
						height++;
					}

					n->m_ZBuffer[n->m_PositionData.Z + subLayerIndex + subLayer * subLayerCount] = champ;
					n->m_ABuffer[n->m_PositionData.A + subLayerIndex + subLayer * subLayerCount] = n->m_Functions.NeuronFunctions[n->m_PositionData.Layer-1].f(champ);

					subLayerIndex++;
				}
				subLayer++;
			}
		}

		void PoolingLayerBackpropegateZ(NetworkPrototype* n)
		{

			const unsigned subLayers = 3;
			const unsigned subLayerCount = n->m_LayerLayout[n->m_PositionData.Layer].Nodes / subLayers;

			const unsigned latterSubLayers = 3;
			const unsigned latterSubLayerCount = n->m_LayerLayout[n->m_PositionData.Layer + 1].Nodes / latterSubLayers;


			const unsigned receptiveWidth = 2;
			const unsigned receptiveHeight = 2;

			const unsigned horizontalStride = 2;
			const unsigned verticalStride = 2;

			const unsigned imgWidth = sqrtf(n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes / subLayerCount);
			const unsigned imgHeight = sqrtf(n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes / subLayerCount);


			const unsigned latterLayerZPos = n->m_PositionData.Z + n->m_LayerLayout[n->m_PositionData.Layer].Nodes;
			const unsigned latterLayerAPos = n->m_PositionData.A + n->m_LayerLayout[n->m_PositionData.Layer].Nodes;

			assert(latterSubLayers == subLayers);

			unsigned subLayer = 0;
			while (subLayer < latterSubLayers)
			{

				unsigned subLayerIndex = 0;
				while (subLayerIndex < latterSubLayerCount)
				{
					//TODO: Beware "Edge"cases (This hint tells you to take a look at what happens when a receptive field goes outside of the image. Also what do you do when the field skips entries entierly)
					const unsigned startA = ((subLayerIndex * verticalStride) / (imgHeight - receptiveHeight + 1)) * imgWidth + (subLayerIndex * horizontalStride) % (imgWidth - receptiveWidth + 1);


					float dz = n->m_DeltaZ[latterLayerZPos + subLayer * latterSubLayerCount + subLayerIndex];
					 

					unsigned height = 0;
					while (height < receptiveHeight)
					{
						unsigned width = 0;
						while (width < receptiveWidth)
						{	
							//TODO: There is some funky shit that happens here when a z is observed in more than one receptive field
							if (n->m_ABuffer[n->m_PositionData.A + subLayer * subLayerCount + startA + height*imgWidth + width] 
								== n->m_ABuffer[latterLayerAPos + subLayer * latterSubLayerCount + subLayerIndex] )
							{
								n->m_DeltaZ[n->m_PositionData.Z + subLayer * subLayerCount + startA + height * imgWidth + width] =
									dz * n->m_Functions.NeuronFunctionsDerivatives[n->m_PositionData.Layer-1].f(n->m_ZBuffer[n->m_PositionData.Z]);
							}
							else {
								n->m_DeltaZ[n->m_PositionData.Z + subLayer * subLayerCount + startA + height * imgWidth + width] = 0;
							} 
							width++;
						}
						height++;
					}

					subLayerIndex++;
				}
				subLayer++;
			}


		}

		void PoolingLayerBackpropegateBW(NetworkPrototype* n)
		{
			//This layer isnt supposed to have any weights or biases
			return;
		}


	}


	namespace CostFunctions
	{

		void CrossEntropy(NetworkPrototype* n)
		{
			float cost = 0;
			unsigned layerIndex = 0;
			while (layerIndex < n->m_LayerLayout[n->m_LayerLayoutCount - 1].Nodes)
			{
				cost += Math::CrossEntropy(n->m_ABuffer[layerIndex + n->m_PositionData.A], n->m_TargetBuffer[layerIndex]);
				layerIndex++;
			}
			n->m_CostBuffer = cost;
		}

		void CrossEntropyDerivative(NetworkPrototype* n)
		{


			unsigned layerIndex = 0;
			while (layerIndex < n->m_LayerLayout[n->m_PositionData.Layer].Nodes)
			{
				//Right here we need the a from this layer, and we therfore have to add the length of the previous layer to get to the start of this one.
				float z = n->m_ZBuffer[n->m_PositionData.Z + layerIndex];
				float a = n->m_ABuffer[n->m_PositionData.A + layerIndex];
				float y = n->m_TargetBuffer[layerIndex];

				float dz = Math::CrossEntropyCostDerivative(z, a, y);

				n->m_DeltaZ[n->m_PositionData.Z + layerIndex] = dz;
				

				layerIndex++;
			}


		}
	}

	namespace TrainingFunctions {

		void L2Regularization(NetworkPrototype* n)
		{
			unsigned index = 0;
			while (index < n->m_WeightsCount)
			{
				
				n->m_WeightsBuffer[index] *= (1 - (n->m_HyperParameters.LearningRate * n->m_HyperParameters.RegularizationConstant / ((float)n->m_Data->TrainingCount)));

				index++;
			}
		}

		void GradientDecent(NetworkPrototype* n)
		{

			unsigned i = 0;
			while (i < n->m_WeightsCount)
			{


				n->m_WeightsBuffer[i] -= (n->m_HyperParameters.LearningRate / ((float)n->m_HyperParameters.BatchCount)) * n->m_DeltaWeights[i];

				i++;
			}

			i = 0;
			while (i < n->m_BiasesCount)
			{
				n->m_BiasesBuffer[i] -= (n->m_HyperParameters.LearningRate / ((float)n->m_HyperParameters.BatchCount)) * n->m_DeltaBiases[i];
				i++;
			}
		}
	}
}


