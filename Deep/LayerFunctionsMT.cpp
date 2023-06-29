#include "pch.h"
#include "LayerFunctionsMT.h"

namespace TNNT
{

	namespace LayerFunctionsMT
	{

		void FullyConnectedFeedForward(NetworkPrototypeMT* n, unsigned thread)
		{

			unsigned start, stop;

			n->ThreadWorkloadDivider(start, stop, n->m_LayerLayout[n->m_PositionData.Layer].Nodes, thread);

			unsigned layerIndex = start;
			while (layerIndex < stop)
			{
				float weightedSum = 0;

				unsigned prevIndex = 0;
				while (prevIndex < n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes)
				{
					weightedSum +=
						n->m_ABuffer[(n->m_PositionData.A - n->m_LayerLayout[n->m_PositionData.Layer-1].Nodes) + prevIndex] *
						n->m_Weights[n->m_PositionData.Weights + n->m_LayerLayout[n->m_PositionData.Layer -1].Nodes * layerIndex + prevIndex];

					prevIndex++;
				}

				n->m_ZBuffer[layerIndex + n->m_PositionData.Z] = weightedSum + n->m_Biases[layerIndex + n->m_PositionData.Biases];
				n->m_ABuffer[layerIndex + n->m_PositionData.A] = n->m_Functions.NeuronFunctions[n->m_PositionData.Layer - 1].f(n->m_ZBuffer[layerIndex + n->m_PositionData.Z]);

				layerIndex++;
			}

		}

		void FullyConnectedBackpropegateZ(NetworkPrototypeMT* n, unsigned thread)
		{

			unsigned start, stop;

			n->ThreadWorkloadDivider(start, stop, n->m_LayerLayout[n->m_PositionData.Layer].Nodes, thread);

			const unsigned latterAPos = n->m_PositionData.A + n->m_LayerLayout[n->m_PositionData.Layer].Nodes;
			const unsigned latterZPos = n->m_PositionData.Z + n->m_LayerLayout[n->m_PositionData.Layer].Nodes;

			const unsigned latterBiases = n->m_PositionData.Biases + n->m_LayerLayout[n->m_PositionData.Layer].Biases;
			const unsigned latterWeights = n->m_PositionData.Weights + n->m_LayerLayout[n->m_PositionData.Layer].Weights;

			unsigned layerIndex = start;
			while (layerIndex < stop)
			{
				float errorSum = 0;
				unsigned latterLayerIndex = 0;
				while (latterLayerIndex < n->m_LayerLayout[n->m_PositionData.Layer + 1].Nodes)
				{
					errorSum += n->m_Weights[latterWeights + n->m_LayerLayout[n->m_PositionData.Layer].Nodes * latterLayerIndex + layerIndex] * n->m_DeltaZ[latterZPos + latterLayerIndex];
					latterLayerIndex++;
				}

				float dz = n->m_Functions.NeuronFunctionsDerivatives[n->m_PositionData.Layer - 1].f(n->m_ZBuffer[n->m_PositionData.Z + layerIndex]);
				n->m_DeltaZ[n->m_PositionData.Z + layerIndex] = errorSum * dz;

				layerIndex++;
			}

		}

		void FullyConnectedBackpropegateBW(NetworkPrototypeMT* n, unsigned thread)
		{
			unsigned start, stop;	
			n->ThreadWorkloadDivider(start, stop, n->m_LayerLayout[n->m_PositionData.Layer].Nodes, thread);

			const unsigned prevAPos = n->m_PositionData.A - n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes;

			unsigned layerIndex = start;
			while (layerIndex < stop)
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


	}



	namespace CostFunctionsMT
	{
		void CrossEntropy(NetworkPrototypeMT* n, unsigned thread)
		{

			unsigned startAPos = n->m_ABufferCount - n->m_LayerLayout[n->m_LayerLayoutCount - 1].Nodes;

			unsigned start, stop;
			n->ThreadWorkloadDivider(start, stop, n->m_LayerLayout[n->m_LayerLayoutCount - 1].Nodes, thread);

			unsigned layerIndex = start;
			while (layerIndex < stop)
			{
				
				float a = n->m_ABuffer[startAPos + layerIndex];
				float y = n->m_TargetBuffer[layerIndex];

				float cost = Math::CrossEntropy(a, y);

				n->m_CostBuffer[thread] += cost;


				layerIndex++;
			}
		}

		void CrossEntropyDerivative(NetworkPrototypeMT* n, unsigned thread)
		{

			unsigned start, stop;

			n->ThreadWorkloadDivider(start, stop, n->m_LayerLayout[n->m_PositionData.Layer].Nodes, thread);


			unsigned layerIndex = start;
			while (layerIndex < stop)
			{
				
				float z = n->m_ZBuffer[n->m_PositionData.Z + layerIndex];
				float a = n->m_ABuffer[n->m_PositionData.A + layerIndex];
				float y = n->m_TargetBuffer[layerIndex];

				float dz = Math::CrossEntropyCostDerivative(z, a, y);

				n->m_DeltaZ[n->m_PositionData.Z + layerIndex] = dz;


				layerIndex++;
			}

		}


	}


	namespace TrainingFunctionsMT
	{

		void L2Regularization(NetworkPrototypeMT* n, unsigned thread)
		{
			unsigned start, stop;
			n->ThreadWorkloadDivider(start, stop, n->m_WeightsCount, thread);

			unsigned index = start;
			while (index < stop)
			{

				n->m_WeightsBuffer[index] *= (1 - (n->m_HyperParameters.LearningRate * n->m_HyperParameters.RegularizationConstant / ((float)n->m_Data->TrainingCount)));
				 
				index++;
			}

		}


		void GradientDecent(NetworkPrototypeMT* n, unsigned thread)
		{
			unsigned start, stop;
			n->ThreadWorkloadDivider(start, stop, n->m_WeightsCount, thread);
			
			unsigned i = start;
			while (i < stop)
			{

				n->m_WeightsBuffer[i] -= (n->m_HyperParameters.LearningRate / ((float)n->m_HyperParameters.BatchCount)) * n->m_DeltaWeights[i];

				i++;
			}

			n->ThreadWorkloadDivider(start, stop, n->m_BiasesCount, thread);

			i = start;
			while (i < stop)
			{

				n->m_BiasesBuffer[i] -= (n->m_HyperParameters.LearningRate / ((float)n->m_HyperParameters.BatchCount)) * n->m_DeltaBiases[i];

				i++;
			}

		}


	}


}