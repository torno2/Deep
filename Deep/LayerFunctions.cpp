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
				while (prevIndex < n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes)
				{
					weightedSum +=
						n->m_A[(n->m_PositionData.A - n->m_LayerLayout[n->m_PositionData.Layer-1].Nodes) + prevIndex] *
						n->m_Weights[n->m_PositionData.Weights + n->m_LayerLayout[n->m_PositionData.Layer -1].Nodes * layerIndex + prevIndex];

					prevIndex++;
				}

				n->m_Z[layerIndex + n->m_PositionData.Z] = weightedSum + n->m_Biases[layerIndex + n->m_PositionData.Biases];
				float z = weightedSum + n->m_Biases[layerIndex + n->m_PositionData.Biases];
				n->m_A[layerIndex + n->m_PositionData.A] = n->m_Functions.NeuronFunctions[n->m_PositionData.Layer - 1].f(n->m_Z[layerIndex + n->m_PositionData.Z]);
				float a = n->m_Functions.NeuronFunctions[n->m_PositionData.Layer - 1].f(n->m_Z[layerIndex + n->m_PositionData.Z]);

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
				while (latterLayerIndex < n->m_LayerLayout[n->m_PositionData.Layer + 1].Nodes)
				{
					errorSum += n->m_Weights[latterWeights + n->m_LayerLayout[n->m_PositionData.Layer].Nodes * latterLayerIndex + layerIndex] * n->m_DeltaZ[latterZPos + latterLayerIndex];
					latterLayerIndex++;
				}

				float dz = n->m_Functions.NeuronFunctionsDerivatives[n->m_PositionData.Layer - 1].f(n->m_Z[n->m_PositionData.Z + layerIndex]);
				n->m_DeltaZ[n->m_PositionData.Z + layerIndex] = errorSum * dz;

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
					float a = n->m_A[prevAPos + prevLayerIndex];
					n->m_DeltaWeights[n->m_PositionData.Weights + n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes * layerIndex + prevLayerIndex] = a * dz;




					prevLayerIndex++;
				}

				layerIndex++;
			}
		
		}


		void ConvolutionLayerFeedForward(NetworkPrototype* n)
		{
	
			//Some of these constants may be usless.

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


			const unsigned prevLayerAPos = n->m_PositionData.A - n->m_LayerLayout[n->m_PositionData.Layer-1].Nodes;
			const unsigned prevLayerACount = n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes;

			const unsigned horizontalSteps = (imgWidth - receptiveWidth ) / (horizontalStride) + 1;
			const unsigned verticalSteps = (imgHeight - receptiveHeight ) / (verticalStride) + 1;

			assert(imgWidth >= receptiveWidth && imgHeight >= receptiveHeight);

			

			unsigned subLayer = 0;
			while (subLayer < subLayers)
			{

				unsigned subLayerIndex = 0;
				while (subLayerIndex < subLayerCount)
				{
					
				
					const unsigned leftUpperCornerA = ((subLayerIndex % horizontalSteps) * horizontalStride) + ((subLayerIndex / verticalSteps) * verticalStride) * imgWidth ;



					float weightedSum = 0;

					unsigned height = 0;
					while (height < receptiveHeight)
					{
						unsigned width = 0;
						while (width < receptiveWidth)
						{

							weightedSum +=
								n->m_Weights[n->m_PositionData.Weights + subLayerIndex * subLayerWeightsCount + height * receptiveWidth + width] *
								n->m_A[prevLayerAPos  + leftUpperCornerA + height * imgWidth + width ];


							width++;
						}
						height++;
					}

					float z = weightedSum + n->m_Biases[n->m_PositionData.Biases + subLayerIndex * subLayerBiasesCount];

					n->m_Z[n->m_PositionData.Z + subLayerIndex + subLayer * subLayerCount] = z;
					n->m_A[n->m_PositionData.A + subLayerIndex + subLayer * subLayerCount] = n->m_Functions.NeuronFunctions[n->m_PositionData.Layer - 1].f(z);

					subLayerIndex++;
				}
				subLayer++;
			}
		}

		void ConvolutionLayerBackpropegateZ(NetworkPrototype* n)
		{
			//Some of these constants may be usless.

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
			const unsigned prevLayerACount = n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes;

			const unsigned horizontalSteps = (imgWidth - receptiveWidth) / (horizontalStride)+1;
			const unsigned verticalSteps = (imgHeight - receptiveHeight) / (verticalStride)+1;

			assert(imgWidth >= receptiveWidth && imgHeight >= receptiveHeight);
		}

  		void ConvolutionLayerBackpropegateBW(NetworkPrototype* n)
		{
			//Some of these constants may be usless.

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

			const unsigned imgWidth  = sqrtf(n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes / prevSubLayers);
			const unsigned imgHeight = sqrtf(n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes / prevSubLayers);


			const unsigned prevLayerAPos = n->m_PositionData.A - n->m_LayerLayout[n->m_PositionData.Layer-1].Nodes;
			const unsigned prevLayerACount = n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes;

			const unsigned horizontalSteps = (imgWidth - receptiveWidth) / (horizontalStride)+1;
			const unsigned verticalSteps = (imgHeight - receptiveHeight) / (verticalStride)+1;

			assert(imgWidth >= receptiveWidth && imgHeight >= receptiveHeight);

			unsigned subLayer = 0;
			while (subLayer < subLayers)
			{
				unsigned subLayerIndex = 0;
				while (subLayerIndex < subLayerCount)
				{

					const unsigned leftUpperCornerA = ((subLayerIndex % horizontalSteps) * horizontalStride) + ((subLayerIndex / verticalSteps) * verticalStride) * imgWidth;

					unsigned height = 0;
					while (height < receptiveHeight)
					{

						unsigned width = 0;
						while(width < receptiveWidth)
						{


							


								n->m_DeltaWeights[n->m_PositionData.Weights + subLayerWeightsCount * subLayer + height* receptiveWidth + width] =
									n->m_DeltaZ[n->m_PositionData.Z + subLayer * subLayerCount + subLayerIndex] *
									n->m_A[prevLayerAPos + leftUpperCornerA + height * prevLayerACount + width];

							


							width++;
						}

						height++;
					}

					n->m_Biases[n->m_PositionData.Biases + subLayerBiasesCount * subLayer + subLayerIndex] = n->m_DeltaZ[n->m_PositionData.Z + subLayer * subLayerCount + subLayerIndex];

				subLayerIndex++;
				}
			subLayer++;
			}

		}


		void PoolingLayerFeedForward(NetworkPrototype* n)
		{
			const unsigned subLayers = 1;
			const unsigned subLayerCount = n->m_LayerLayout[n->m_PositionData.Layer].Nodes / subLayers;

			const unsigned subLayerBiasesCount = n->m_LayerLayout[n->m_PositionData.Layer].Biases / subLayers;
			const unsigned subLayerWeightsCount = n->m_LayerLayout[n->m_PositionData.Layer].Weights / subLayers;

			const unsigned prevSubLayers = 1;
			const unsigned prevSubLayerCount = n->m_LayerLayout[n->m_PositionData.Layer-1].Nodes / prevSubLayers;

			const unsigned receptiveWidth = 2;
			const unsigned receptiveHeight = 2;

			const unsigned horizontalStride = 2;
			const unsigned verticalStride = 2;

			const unsigned imgWidth = sqrtf(n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes / prevSubLayers);
			const unsigned imgHeight = sqrtf(n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes / prevSubLayers);


			const unsigned prevLayerAPos = n->m_PositionData.A - n->m_LayerLayout[n->m_PositionData.Layer-1].Nodes;
			const unsigned prevLayerACount = n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes;

			const unsigned horizontalSteps = (imgWidth - receptiveWidth ) / (horizontalStride) + 1;
			const unsigned verticalSteps = (imgHeight - receptiveHeight ) / (verticalStride) + 1;

			assert(imgWidth >= receptiveWidth && imgHeight >= receptiveHeight);

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
					const unsigned leftUpperCornerA = ((subLayerIndex % horizontalSteps) * horizontalStride) + ((subLayerIndex / verticalSteps) * verticalStride) * imgWidth;


					
					champ = n->m_A[prevLayerAPos + subLayer * prevSubLayerCount + leftUpperCornerA];

					unsigned height = 0;
					while (height < receptiveHeight)
					{
						unsigned width = 0;
						while (width < receptiveWidth)
						{
								
								check = n->m_A[prevLayerAPos+ subLayer* prevSubLayerCount + leftUpperCornerA + width + height * imgWidth];
								if (check > champ)
								{
									champ = check;
								}


							width++;
						}
						height++;
					}

					n->m_Z[n->m_PositionData.Z + subLayerIndex + subLayer * subLayerCount] = champ;
					n->m_A[n->m_PositionData.A + subLayerIndex + subLayer * subLayerCount] = n->m_Functions.NeuronFunctions[n->m_PositionData.Layer-1].f(champ);

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

			const unsigned imgWidth = sqrtf(subLayerCount);
			const unsigned imgHeight = sqrtf(subLayerCount);


			const unsigned latterLayerZPos = n->m_PositionData.Z + n->m_LayerLayout[n->m_PositionData.Layer].Nodes;
			const unsigned latterLayerAPos = n->m_PositionData.A + n->m_LayerLayout[n->m_PositionData.Layer].Nodes;


			const unsigned horizontalSteps = (imgWidth - receptiveWidth) / (horizontalStride) + 1;
			const unsigned verticalSteps = (imgHeight - receptiveHeight) / (verticalStride)   + 1;

			assert(imgWidth >= receptiveWidth && imgHeight >= receptiveHeight);


			assert(latterSubLayers == subLayers);

			unsigned subLayer = 0;
			while (subLayer < subLayers)
			{

				unsigned latterSubLayerIndex = 0;
				while (latterSubLayerIndex < latterSubLayerCount)
				{

					//TODO: Beware "Edge"cases (This hint tells you to take a look at what happens when a receptive field goes outside of the image. Also what do you do when the field skips entries entierly)
					
					const unsigned leftUpperCornerA = ((latterSubLayerIndex % horizontalSteps) * horizontalStride) + ((latterSubLayerIndex / verticalSteps) * verticalStride) * imgWidth;






					const float dz = n->m_DeltaZ[latterLayerZPos + subLayer * latterSubLayerCount + latterSubLayerIndex];


					const float aLatter = n->m_A[latterLayerAPos + subLayer * latterSubLayerCount + latterSubLayerIndex];

					unsigned height = 0;
					while (height < receptiveHeight)
					{
						unsigned width = 0;
						while (width < receptiveWidth)
						{
							const float a = n->m_A[n->m_PositionData.A + subLayer * subLayerCount + leftUpperCornerA + height * imgWidth + width];

							//TODO: There is some funky shit that happens here when a z is observed in more than one receptive field. Edit: Think you should add the changes made for each receptive field it appears in.
							if ( a == aLatter)
							{
								n->m_DeltaZ[n->m_PositionData.Z + subLayer * subLayerCount + leftUpperCornerA + height * imgWidth + width] =
									dz * n->m_Functions.NeuronFunctionsDerivatives[n->m_PositionData.Layer-1].f(n->m_Z[n->m_PositionData.Z+ latterLayerZPos + subLayer * latterSubLayerCount + latterSubLayerIndex]); // f(n->m_Z[n->m_PositionData.Z]) does this need to be fixed?
							}
							else {
								n->m_DeltaZ[n->m_PositionData.Z + subLayer * subLayerCount + leftUpperCornerA + height * imgWidth + width] = 0;
							} 
							width++;
						}
						height++;
					}

					latterSubLayerIndex++;
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

			unsigned startAPos = n->m_ACount - n->m_LayerLayout[n->m_LayerLayoutCount - 1].Nodes;




			unsigned layerIndex = 0;
			while (layerIndex < n->m_OutputBufferCount)
			{
				
				float a = n->m_OutputBuffer[layerIndex];
				float y = n->m_TargetBuffer[layerIndex];

				float cost = Math::CrossEntropy(a, y);

				n->m_CostBuffer += cost;


				layerIndex++;
			}
		}

		void CrossEntropyDerivative(NetworkPrototype* n)
		{


			unsigned layerIndex = 0;
			while (layerIndex < n->m_OutputBufferCount)
			{
				
				float z = n->m_Z[n->m_PositionData.Z + layerIndex];
				float a = n->m_OutputBuffer[layerIndex];
				float y = n->m_TargetBuffer[layerIndex];

				float dz = Math::CrossEntropyCostDerivative(z, a, y);

				n->m_DeltaZ[n->m_PositionData.Z + layerIndex] = dz;
				

				layerIndex++;
			}


		}
	}


	//Old
#if OLD
	namespace TrainingFunctions
	{

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
#endif

	//New
#if NEW

	namespace TrainingFunctions 
	{

		void L2Regularization(NetworkPrototype* n)
		{
			

			unsigned index = 0;
			while (index < n->m_LayerLayout[n->m_PositionData.Layer].Weights)
			{
				
				n->m_WeightsBuffer[n->m_PositionData.Weights+index] *= (1 - (n->m_HyperParameters.LearningRate * n->m_HyperParameters.RegularizationConstant / ((float)n->m_Data->TrainingCount)));

				index++;
			}
		}

		void GradientDecent(NetworkPrototype* n)
		{

			unsigned index = 0;
			while (index < n->m_LayerLayout[n->m_PositionData.Layer].Weights)
			{

				if (index < n->m_LayerLayout[n->m_PositionData.Layer].Biases)
				{
					n->m_BiasesBuffer[n->m_PositionData.Biases + index] -= (n->m_HyperParameters.LearningRate / ((float)n->m_HyperParameters.BatchCount)) * n->m_DeltaBiases[index];
				}

				n->m_WeightsBuffer[n->m_PositionData.Weights + index] -= (n->m_HyperParameters.LearningRate / ((float)n->m_HyperParameters.BatchCount)) * n->m_DeltaWeights[index];

				index++;
			}

		}
	}
#endif
}


