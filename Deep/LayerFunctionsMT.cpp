#include "pch.h"
#include "LayerFunctionsMT.h"

namespace TNNT
{

	namespace LayerFunctionsMT
	{

		void FullyConnectedFeedForward(NetworkPrototypeMT* n, unsigned thread)
		{

			

			unsigned startNodes = n->m_PaddingData.Nodes[(2*thread) + (2 * n->m_SlaveThreadCount * n->m_PositionData.Layer)];
			unsigned stopNodes = n->m_PaddingData.Nodes[(2*thread + 1) + (2 * n->m_SlaveThreadCount * n->m_PositionData.Layer)];

			unsigned formerNodesPos = n->m_PositionData.A - (n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes + n->m_PaddingData.FloatPaddingPerLayer);


			unsigned weightPadding = 0;

			unsigned stopWeightsPos = 1 + (2 * n->m_SlaveThreadCount * (n->m_PositionData.Layer -1));
			unsigned stopWeights = n->m_PaddingData.Weights[stopWeightsPos];

			unsigned layerIndexNoPadding = startNodes - n->m_PaddingData.FloatPadding * thread;
			unsigned layerIndex = startNodes;
			while (layerIndex < stopNodes)
			{
				float weightedSum = 0;
				

				unsigned prevNodePadding = 0;

				unsigned stopPrevNodesPos = 1 + (2 * n->m_SlaveThreadCount * (n->m_PositionData.Layer - 1));
				unsigned stopPrevNodes = n->m_PaddingData.Nodes[stopPrevNodesPos];
				
				

				unsigned prevIndex = 0;
				while (prevIndex < n->m_LayerLayout[n->m_PositionData.Layer-1].Nodes)
				{
					

					if ( prevIndex + prevNodePadding >= stopPrevNodes)
					{

						prevNodePadding += n->m_PaddingData.FloatPadding;

						stopPrevNodesPos += 2;
						stopPrevNodes = n->m_PaddingData.Nodes[stopPrevNodesPos];
					}

					if (n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes * layerIndexNoPadding + prevIndex + weightPadding >= stopWeights)
					{
						

						weightPadding += n->m_PaddingData.FloatPadding;

						stopWeightsPos += 2;
						stopWeights = n->m_PaddingData.Weights[stopWeightsPos];
					}

					float A = n->m_A[formerNodesPos + prevIndex + prevNodePadding];
					float weight = n->m_Weights[n->m_PositionData.Weights + n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes * layerIndexNoPadding + prevIndex + weightPadding];

					weightedSum += A * weight;

					prevIndex++;
					


				}


				
				n->m_Z[layerIndex + n->m_PositionData.Z] = weightedSum + n->m_Biases[layerIndex + n->m_PositionData.Biases];
				n->m_A[layerIndex + n->m_PositionData.A] = n->m_Functions.NeuronFunctions[n->m_PositionData.Layer - 1].f(n->m_Z[layerIndex + n->m_PositionData.Z]);
				
				layerIndexNoPadding++;
				layerIndex++;
			}

		}

		void FullyConnectedBackpropegateZ(NetworkPrototypeMT* n, unsigned thread)
		{

			unsigned startNodes = n->m_PaddingData.Nodes[(2*thread) + (2 * n->m_SlaveThreadCount * n->m_PositionData.Layer)];
			unsigned stopNodes = n->m_PaddingData.Nodes[(2*thread + 1) + (2 * n->m_SlaveThreadCount * n->m_PositionData.Layer)];


			const unsigned latterAPos = n->m_PositionData.A + n->m_LayerLayout[n->m_PositionData.Layer].Nodes + n->m_PaddingData.FloatPaddingPerLayer;
			const unsigned latterZPos = n->m_PositionData.Z + n->m_LayerLayout[n->m_PositionData.Layer].Nodes + n->m_PaddingData.FloatPaddingPerLayer;

			const unsigned latterBiases = n->m_PositionData.Biases + n->m_LayerLayout[n->m_PositionData.Layer].Biases + n->m_PaddingData.FloatPaddingPerLayer;
			const unsigned latterWeights = n->m_PositionData.Weights + n->m_LayerLayout[n->m_PositionData.Layer].Weights + n->m_PaddingData.FloatPaddingPerLayer;




			unsigned latterWeightPadding = 0;

			unsigned stopLatterWeightsPos =  1 + (2 * n->m_SlaveThreadCount * (n->m_PositionData.Layer - 1));
			unsigned stopLatterWeights = n->m_PaddingData.Weights[stopLatterWeightsPos];
			
			unsigned layerIndexNoPadding = startNodes - n->m_PaddingData.FloatPadding * thread;
			unsigned layerIndex = startNodes;
			while (layerIndex < stopNodes)
			{




				unsigned latterNodePadding = 0;

				unsigned stopLatterNodesPos =  1 + (2 * n->m_SlaveThreadCount * (n->m_PositionData.Layer - 1));
				unsigned stopLatterNodes = n->m_PaddingData.Nodes[stopLatterNodesPos];

				
				float errorSum = 0;
				unsigned latterLayerIndex = 0;
				while (latterLayerIndex < n->m_LayerLayout[n->m_PositionData.Layer + 1].Nodes)
				{
					if (latterLayerIndex + latterNodePadding >= stopLatterNodes)
					{

						latterNodePadding += n->m_PaddingData.FloatPadding;

						stopLatterNodesPos += 2;
						stopLatterNodes = n->m_PaddingData.Nodes[stopLatterNodesPos];
					}

					if (n->m_LayerLayout[n->m_PositionData.Layer].Nodes * latterLayerIndex + layerIndex + latterWeightPadding >= stopLatterWeights)
					{

						latterWeightPadding += n->m_PaddingData.FloatPadding;

						stopLatterWeightsPos += 2;
						stopLatterWeights = n->m_PaddingData.Weights[stopLatterWeightsPos];
					}

					errorSum += 
						n->m_Weights[latterWeights + n->m_LayerLayout[n->m_PositionData.Layer].Nodes * latterLayerIndex + layerIndexNoPadding + latterWeightPadding]
						* n->m_DeltaZ[latterZPos + latterLayerIndex + latterNodePadding ];
					latterLayerIndex++;
				}
				
				
				
				float dz = n->m_Functions.NeuronFunctionsDerivatives[n->m_PositionData.Layer - 1].f(n->m_Z[n->m_PositionData.Z + layerIndex]);
				n->m_DeltaZ[n->m_PositionData.Z + layerIndex] = errorSum * dz;
				

				layerIndexNoPadding++;
				layerIndex++;

			}

		}

		void FullyConnectedBackpropegateBW(NetworkPrototypeMT* n, unsigned thread)
		{
			unsigned startNodes = n->m_PaddingData.Nodes[(2 * thread) + (2 * n->m_SlaveThreadCount * n->m_PositionData.Layer)];
			unsigned stopNodes = n->m_PaddingData.Nodes[(2 * thread + 1) + (2 * n->m_SlaveThreadCount * n->m_PositionData.Layer)];

			
		

			const unsigned prevAPos = n->m_PositionData.A - (n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes + n->m_PaddingData.FloatPaddingPerLayer);


			unsigned layerIndexNoPadding = startNodes - n->m_PaddingData.FloatPadding * thread;
			unsigned layerIndex = startNodes;
			while (layerIndex < stopNodes)
			{


				const float dz = n->m_DeltaZ[n->m_PositionData.Z + layerIndex];
				
				n->m_DeltaBiases[n->m_PositionData.Biases + layerIndex] = dz;
				

				

				unsigned prevNodePadding = 0;

				unsigned stopPrevNodesPos = 1 + (2 * n->m_SlaveThreadCount * (n->m_PositionData.Layer - 1));
				unsigned stopPrevNodes = n->m_PaddingData.Nodes[stopPrevNodesPos];


				unsigned prevLayerIndex = 0;
				while (prevLayerIndex < n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes)
				{
					
					if (prevLayerIndex + prevNodePadding >= stopPrevNodes)
					{
						

						prevNodePadding += n->m_PaddingData.FloatPadding;

						stopPrevNodesPos += 2;
						stopPrevNodes = n->m_PaddingData.Nodes[stopPrevNodesPos];
					}

					float a = n->m_A[prevAPos + prevLayerIndex + prevNodePadding];
					float dw = a * dz;

					
					n->m_DeltaWeights[n->m_PositionData.Weights + n->m_LayerLayout[n->m_PositionData.Layer - 1].Nodes * layerIndexNoPadding + prevLayerIndex] = dw;
					
					

					prevLayerIndex++;
				}

				layerIndexNoPadding++;
				layerIndex++;
			}

		}


	}



	namespace CostFunctionsMT
	{
		void CrossEntropy(NetworkPrototypeMT* n, unsigned thread)
		{

		

			unsigned start = n->m_PaddingData.Nodes[2*thread  + (n->m_LayerLayoutCount - 1)*n->m_SlaveThreadCount];
			unsigned stop = n->m_PaddingData.Nodes[2*thread+1 + (n->m_LayerLayoutCount - 1)*n->m_SlaveThreadCount];

			unsigned layerIndex = start;
			while (layerIndex < stop)
			{
				
				float a = n->m_OutputBuffer[layerIndex];
				float y = n->m_TargetBuffer[layerIndex];


				float cost = Math::CrossEntropy(a, y);


				n->m_CostBuffer[thread] += cost; 
				


				layerIndex++;
			}
		}

		void CrossEntropyDerivative(NetworkPrototypeMT* n, unsigned thread)
		{

			unsigned start = n->m_PaddingData.Nodes[2 * thread + (n->m_LayerLayoutCount - 1) * n->m_SlaveThreadCount];
			unsigned stop = n->m_PaddingData.Nodes[2 * thread + 1 + (n->m_LayerLayoutCount - 1) * n->m_SlaveThreadCount];
		

			unsigned layerIndex = start;
			while (layerIndex < stop)
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


	namespace TrainingFunctionsMT
	{

		void L2Regularization(NetworkPrototypeMT* n, unsigned thread)
		{
			//Always remember that hte 0-th layer doesnt have any weights and biases.
			unsigned intitalStart = n->m_PaddingData.Weights[2*thread];
			unsigned laststop = n->m_PaddingData.Weights[(2*thread + 1) + 2 * n->m_SlaveThreadCount * (n->m_LayerLayoutCount - 2)];



			

			unsigned currentStartPos = (2 * thread );
			unsigned currentStopPos = (2 * thread + 1);
			unsigned currentStop = n->m_PaddingData.Weights[currentStopPos];

			unsigned index = intitalStart;
			while (index < laststop)
			{
				if (index >= currentStop)
				{

					currentStartPos += 2 * n->m_SlaveThreadCount;
					index = n->m_PaddingData.Weights[currentStartPos];

					currentStopPos += 2 * n->m_SlaveThreadCount;
					currentStop = n->m_PaddingData.Weights[currentStopPos];

					
				}

				n->m_WeightsBuffer[index] *= (1 - (n->m_HyperParameters.LearningRate * n->m_HyperParameters.RegularizationConstant / ((float)n->m_Data->TrainingCount)));
				
				index++;


			}

		}


		void GradientDecent(NetworkPrototypeMT* n, unsigned thread)
		{
			//Always remember that hte 0-th layer doesnt have any weights and biases.


			unsigned intitalStart = n->m_PaddingData.Weights[2 * thread];
			unsigned laststop = n->m_PaddingData.Weights[(2 * thread + 1) + 2 * n->m_SlaveThreadCount * (n->m_LayerLayoutCount - 2)];
			
		
			unsigned currentStartPos = (2 * thread);
			unsigned currentStopPos = (2 * thread + 1);
			unsigned currentStop = n->m_PaddingData.Weights[currentStopPos];

			unsigned index = intitalStart;
			while (index < laststop)
			{

				if (index >= currentStop)
				{

					currentStartPos += 2 * n->m_SlaveThreadCount;
					index = n->m_PaddingData.Weights[currentStartPos];

					currentStopPos += 2 * n->m_SlaveThreadCount;
					currentStop = n->m_PaddingData.Weights[currentStopPos];

				}
				
				n->m_WeightsBuffer[index] -= (n->m_HyperParameters.LearningRate / ((float)n->m_HyperParameters.BatchCount)) * n->m_DeltaWeights[index];
				
				index++;
			}






			intitalStart = n->m_PaddingData.Biases[2 * thread];
			laststop = n->m_PaddingData.Biases[(2 * thread + 1) + 2 * n->m_SlaveThreadCount * (n->m_LayerLayoutCount - 2)];

			currentStartPos = (2 * thread);
			currentStopPos = (2 * thread + 1);
			currentStop = n->m_PaddingData.Biases[currentStopPos];

			index = intitalStart;
			while (index < laststop)
			{
				
				if (index >= currentStop)
				{

					currentStartPos += 2 * n->m_SlaveThreadCount;
					index = n->m_PaddingData.Biases[currentStartPos];

					currentStopPos += 2 * n->m_SlaveThreadCount;
					currentStop = n->m_PaddingData.Biases[currentStopPos];

				}

				n->m_BiasesBuffer[index] -= (n->m_HyperParameters.LearningRate / ((float)n->m_HyperParameters.BatchCount)) * n->m_DeltaBiases[index];
				
				index++;
			}

		}


	}


}