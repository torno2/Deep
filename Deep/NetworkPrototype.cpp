#include "pch.h"
#include "NetworkPrototype.h"
#include "LayerFunctions.h"

namespace TNNT
{
	//Constructors And destructor

	NetworkPrototype::NetworkPrototype(LayerLayout* layerLayout, FunctionsLayout& functions, unsigned layoutCount, bool randomizeWeightsAndBiases)
		: m_LayerLayout(layerLayout), m_LayerLayoutCount(layoutCount), m_Functions(functions)
	{


		// 1 layer no network makes; need at least 2
		assert(m_LayerLayoutCount >= 2);

		// A layer of zero nodes would mean you have two separate networks (or a network with 1 less layer, if the input/output layer is missing), 
		// and a layer with a negative numbers of nodes is something I don't want to think about.

		unsigned nodesTotal = 0;

		//Keep in mind that their aren't supposed to be any biases or weights in the 0th layer, so their count for that layer should both be 0.
		unsigned biasTotal = 0;
		unsigned weightTotal = 0;

		unsigned layoutIndex = 0;
		while (layoutIndex < m_LayerLayoutCount)
		{
			// A layer of zero nodes would mean you have two separate networks (or a network with 1 less layer, if the input/output layer is missing), 
			// and a layer with a negative numbers of nodes is something I don't want to think about.
			assert(m_LayerLayout[layoutIndex].Nodes > 0);

			nodesTotal += m_LayerLayout[layoutIndex].Nodes;
			biasTotal += m_LayerLayout[layoutIndex].Biases;
			weightTotal += m_LayerLayout[layoutIndex].Weights;


			layoutIndex++;
		}
		//No Zs for the 0th layer;
		m_ZBufferCount = nodesTotal - m_LayerLayout[0].Nodes;
		m_ABufferCount = nodesTotal;
		m_BiasesCount = biasTotal;
		m_WeightsCount = weightTotal;
		

		m_ZBuffer = new float[m_ZBufferCount];
		m_DeltaZ = new float[m_ZBufferCount];

		m_ABuffer = new float[m_ABufferCount];

		m_Biases = new float[m_BiasesCount];
		m_DeltaBiases = new float[m_BiasesCount];
		m_BiasesBuffer = new float[m_BiasesCount];


		m_Weights = new float[weightTotal];
		m_WeightsTranspose = new float[weightTotal];
		m_DeltaWeights = new float[weightTotal];
		m_WeightsBuffer = new float[weightTotal];
		

		m_TargetBuffer = new float[m_LayerLayout[m_LayerLayoutCount - 1].Nodes];




		if (randomizeWeightsAndBiases)
		{
			//For randomly initializing the weights and biases
			std::default_random_engine generator;
			std::normal_distribution<float> distribution(0.0f, 1 / sqrt(m_LayerLayout[0].Nodes));

			//Randomly assigns weights and biases.
			unsigned i = 0;
			while (i < m_WeightsCount)
			{

				if (i < m_BiasesCount)
				{
					m_Biases[i] = distribution(generator);
				}
				m_Weights[i] = distribution(generator);
				i++;
			}
		}
		else {
			//Sets all weights and biases to zero
			unsigned i = 0;
			while (i < m_WeightsCount)
			{

				if (i < m_BiasesCount)
				{
					m_Biases[i] = 0;
				}
				m_Weights[i] = 0;
				i++;
			}
		}
	
		SetTempToBiases();

		SetTempToWeights();
		ResetWeightsTranspose();

	}

	NetworkPrototype::~NetworkPrototype()
	{
		delete[] m_LayerLayout;

		delete[] m_ZBuffer;
		delete[] m_DeltaZ;

		delete[] m_ABuffer;

		delete[] m_Biases;
		delete[] m_DeltaBiases;
		delete[] m_BiasesBuffer;

		delete[] m_Weights;
		delete[] m_WeightsTranspose;
		delete[] m_DeltaWeights;
		delete[] m_WeightsBuffer;

		delete[] m_TargetBuffer;


		delete[] m_InternalInputBuffer;
		delete[] m_InternalTargetBuffer;

		m_Functions.DestroyFunctionsLayout();

	}

	float NetworkPrototype::CheckSuccessRate()
	{
		return CheckSuccessRateMasterFunction( );
	}

	float NetworkPrototype::CheckCost()
	{
		return CheckCostMasterFunction();
	}

	void NetworkPrototype::Train(DataSet& data, HyperParameters& params)
	{
		SetData(&data);
		SetHyperParameters(params);
		TrainMasterFunction();
	}





	//Public functions


	// Private functions

	void NetworkPrototype::SetBiasesToTemp()
	{
		unsigned index = 0;
		while (index < m_BiasesCount)
		{
			m_Biases[index] = m_BiasesBuffer[index];
			index++;
		}
	}

	void NetworkPrototype::SetTempToBiases()
	{
		unsigned index = 0;
		while (index < m_BiasesCount)
		{
			m_BiasesBuffer[index] = m_Biases[index];
			index++;
		}
	}


void NetworkPrototype::SetWeightsToTemp()	
	{	
		unsigned currentOffset = 0;	
		unsigned currentCount = 0;	
		unsigned layoutIndex = 1;	
		while (layoutIndex < m_LayerLayoutCount)	
		{	
			unsigned rowSize = m_LayerLayout[layoutIndex].WeightsRowCount;	
			unsigned columnSize = m_LayerLayout[layoutIndex].Weights / rowSize;	
			currentOffset += currentCount;	
			currentCount = m_LayerLayout[layoutIndex].Weights;	
			unsigned index = 0;	
			while (index < currentCount)	
			{	
				m_Weights[currentOffset + index] = m_WeightsBuffer[currentOffset + index];	
				m_WeightsTranspose[currentOffset + (index / rowSize) + (index % rowSize) * columnSize] = m_WeightsBuffer[currentOffset + index];	
				index++;	
			}	
			layoutIndex++;	
		}	
	}

	void NetworkPrototype::SetTempToWeights()
	{
		unsigned index = 0;
		while (index < m_WeightsCount)
		{
			m_WeightsBuffer[index] = m_Weights[index];
			index++;
		}
	}

	void NetworkPrototype::ResetWeightsTranspose()
	{
		unsigned currentOffset = 0;
		unsigned currentCount = 0;
		unsigned layoutIndex = 1;
		while (layoutIndex < m_LayerLayoutCount)
		{
			unsigned rowSize = m_LayerLayout[layoutIndex].WeightsRowCount;
			unsigned columnSize = m_LayerLayout[layoutIndex].Weights / rowSize;

			currentOffset += currentCount;
			currentCount = m_LayerLayout[layoutIndex].Weights;
			unsigned index = 0;
			while (index < currentCount)
			{
				m_WeightsTranspose[currentOffset + (index / rowSize) + (index % rowSize) * columnSize] = m_Weights[currentOffset + index];
				index++;
			}

			layoutIndex++;
		}
	}


	void NetworkPrototype::SetData(DataSet* data)
	{

		m_Data = data;

		
	}



	void NetworkPrototype::SetHyperParameters(HyperParameters& params)
	{
		delete[] m_InternalInputBuffer;
		delete[] m_InternalTargetBuffer;

		m_HyperParameters = params;

		m_InternalInputBuffer = new float[m_HyperParameters.BatchCount * m_LayerLayout[0].Nodes];
		m_InternalTargetBuffer = new float[m_HyperParameters.BatchCount * m_LayerLayout[m_LayerLayoutCount-1].Nodes];


	}


	void NetworkPrototype::SetInput(const float* input)
	{

		unsigned index = 0;
		while (index < m_LayerLayout[0].Nodes)
		{
			m_ABuffer[index] = input[index];
			index++;
		}
	}

	void NetworkPrototype::SetTarget(const float* target)
	{






		unsigned index = 0;
		while (index < m_LayerLayout[m_LayerLayoutCount - 1].Nodes)
		{
			m_TargetBuffer[index] = target[index];
			index++;
		}
	}


	void NetworkPrototype::FeedForward()
	{
		
		m_PositionData.Layer = 1;
		m_PositionData.Z = 0;
		m_PositionData.A= m_LayerLayout[0].Nodes;
		m_PositionData.Biases= m_LayerLayout[0].Biases;
		m_PositionData.Weights = m_LayerLayout[0].Weights;

		unsigned layoutIndex = 1;

		while (layoutIndex < m_LayerLayoutCount)
		{

			//No function for the inputlayer, which means that the function corresponding to any other layer is located at layer - 1.
			m_Functions.FeedForwardCallBackFunctions[layoutIndex - 1].f(this);




			m_PositionData.Z += m_LayerLayout[m_PositionData.Layer].Nodes;

			m_PositionData.A += m_LayerLayout[m_PositionData.Layer].Nodes;

			m_PositionData.Biases += m_LayerLayout[m_PositionData.Layer].Biases;

			m_PositionData.Weights += m_LayerLayout[m_PositionData.Layer].Weights;

			m_PositionData.Layer++;

			layoutIndex++;
		}

	}

	void NetworkPrototype::Backpropegate()
	{



		unsigned lastLayer = m_LayerLayoutCount - 1;

		m_PositionData.Layer = lastLayer;

		m_PositionData.Z = m_ZBufferCount- m_LayerLayout[lastLayer].Nodes;
		m_PositionData.A = m_ABufferCount - m_LayerLayout[lastLayer].Nodes;

		m_PositionData.Biases = m_BiasesCount - m_LayerLayout[lastLayer].Biases;
		m_PositionData.Weights = m_WeightsCount - m_LayerLayout[lastLayer].Weights;



		{

			m_Functions.CostFunctionDerivative.f(this);

		}

		unsigned reveresLayoutIndex = 0;
		
		while (reveresLayoutIndex < lastLayer-1 )
		{
			m_Functions.BackPropegateCallBackFunctionsBW[reveresLayoutIndex].f(this);

			//Alters position data to be more in line with the layer in question
			{

				m_PositionData.Layer--;

				m_PositionData.Z -= m_LayerLayout[m_PositionData.Layer ].Nodes;

				m_PositionData.A -= m_LayerLayout[m_PositionData.Layer ].Nodes;

				m_PositionData.Biases -= m_LayerLayout[m_PositionData.Layer ].Biases;

				m_PositionData.Weights -= m_LayerLayout[m_PositionData.Layer ].Weights;
			}

			m_Functions.BackPropegateCallBackFunctionsZ[reveresLayoutIndex].f(this);
		
			reveresLayoutIndex++;
		}

		m_Functions.BackPropegateCallBackFunctionsBW[reveresLayoutIndex].f(this);
	}



	void NetworkPrototype::TrainOnSet(unsigned num)
	{


		m_Functions.RegularizationFunction.f(this);

		unsigned exampleIndex = 0;
		while (exampleIndex < num)
		{

			SetInput(&(m_InternalInputBuffer[exampleIndex * m_LayerLayout[0].Nodes]));
			SetTarget(&(m_InternalTargetBuffer[exampleIndex * m_LayerLayout[m_LayerLayoutCount - 1].Nodes]));


			FeedForward();
			Backpropegate();

			m_Functions.TrainingFunction.f(this);



			exampleIndex++;


		}


		SetBiasesToTemp();
		SetWeightsToTemp();
	}


	void NetworkPrototype::TrainMasterFunction()
	{


		const unsigned batchNum = m_Data->TrainingCount / m_HyperParameters.BatchCount;
		const unsigned remainingBatch = m_Data->TrainingCount % m_HyperParameters.BatchCount;

		std::mt19937 mt;

		unsigned* indicies = new unsigned[m_Data->TrainingCount];
		unsigned index = 0;
		while (index < m_Data->TrainingCount)
		{
			indicies[index] = index;
			index++;
		}

		unsigned epochNum = 0;
		while (epochNum < m_HyperParameters.Epochs)
		{
			unsigned randomIndexCount = m_Data->TrainingCount;


			unsigned batch = 0;
			while (batch < batchNum)
			{
				unsigned batchIndex = 0;
				while (batchIndex < m_HyperParameters.BatchCount)
				{

					unsigned randomIndex = mt() % randomIndexCount;

					unsigned epochRandomIndex = indicies[randomIndex];
					indicies[randomIndex] = indicies[randomIndexCount - 1];
					indicies[randomIndexCount - 1] = epochRandomIndex;

					unsigned inputIndex = 0;
					while (inputIndex < m_LayerLayout[0].Nodes)
					{
						m_InternalInputBuffer[batchIndex * m_LayerLayout[0].Nodes + inputIndex] = m_Data->TrainingInputs[epochRandomIndex * m_LayerLayout[0].Nodes + inputIndex];
						inputIndex++;
					}

					unsigned outputIndex = 0;
					while (outputIndex < m_LayerLayout[m_LayerLayoutCount - 1].Nodes)
					{
						m_InternalTargetBuffer[m_LayerLayout[m_LayerLayoutCount - 1].Nodes * batchIndex + outputIndex] = m_Data->TraningTargets[m_LayerLayout[m_LayerLayoutCount - 1].Nodes * epochRandomIndex + outputIndex];
						outputIndex++;
					}



					randomIndexCount--;
					batchIndex++;
				}

				TrainOnSet(m_HyperParameters.BatchCount);

				batch++;
			}


			if (remainingBatch > 0)
			{
				unsigned batchIndex = 0;
				while (batchIndex < remainingBatch)
				{

					unsigned randomIndex = mt() % randomIndexCount;

					unsigned epochRandomIndex = indicies[randomIndex];
					indicies[randomIndex] = indicies[randomIndexCount - 1];
					indicies[randomIndexCount - 1] = epochRandomIndex;

					unsigned inputIndex = 0;
					while (inputIndex < m_LayerLayout[0].Nodes)
					{
						m_InternalInputBuffer[batchIndex * m_LayerLayout[0].Nodes + inputIndex] = m_Data->TrainingInputs[epochRandomIndex * m_LayerLayout[0].Nodes + inputIndex];
						inputIndex++;
					}

					unsigned outputIndex = 0;
					while (outputIndex < m_LayerLayout[m_LayerLayoutCount - 1].Nodes)
					{
						m_InternalTargetBuffer[m_LayerLayout[m_LayerLayoutCount - 1].Nodes * batchIndex + outputIndex] = m_Data->TraningTargets[m_LayerLayout[m_LayerLayoutCount - 1].Nodes * epochRandomIndex + outputIndex];
						outputIndex++;
					}



					randomIndexCount--;
					batchIndex++;

				}
				TrainOnSet(remainingBatch);
			}


			epochNum++;
		}


		delete[] indicies;
	}

	
	float NetworkPrototype::CheckCostMasterFunction( )
	{

		
		m_PositionData.A = m_ABufferCount - m_LayerLayout[m_LayerLayoutCount - 1].Nodes;

		float cost = 0;

		unsigned checkIndex = 0;
		while (checkIndex < m_Data->TestCount)
		{
			SetInput(&m_Data->TestInputs[checkIndex * m_LayerLayout[0].Nodes]);
			SetTarget(&m_Data->TestTargets[checkIndex * m_LayerLayout[m_LayerLayoutCount - 1].Nodes]);
			FeedForward();
			m_PositionData.A = m_ABufferCount - m_LayerLayout[m_LayerLayoutCount - 1].Nodes;
			m_Functions.CostFunction.f(this);
			
			cost += m_CostBuffer;

			checkIndex++;
		}

		cost = cost / ((float)m_Data->TestCount);
		return cost;

	}

	float NetworkPrototype::CheckSuccessRateMasterFunction()
	{
		const unsigned Ap = m_ABufferCount - m_LayerLayout[m_LayerLayoutCount - 1].Nodes;

		float score = 0.0f;

		unsigned checkIndex = 0;
		while (checkIndex < m_Data->TestCount)
		{

			SetInput(&m_Data->TestInputs[checkIndex * m_LayerLayout[0].Nodes]);
			FeedForward();

			int championItterator = -1;
			float champion = 0;
			unsigned outputIndex = 0;
			while (outputIndex < m_LayerLayout[m_LayerLayoutCount - 1].Nodes)
			{
				
				if (m_ABuffer[Ap + outputIndex] >= champion)
				{
					champion = m_ABuffer[Ap + outputIndex];
					championItterator = outputIndex;
				}

				outputIndex++;
			}

			if (m_Data->TestTargets[m_LayerLayout[m_LayerLayoutCount - 1].Nodes * checkIndex + championItterator] == 1)
			{
				score += 1.0f;
			}
			checkIndex++;
		}

		float rate = score / ((float)m_Data->TestCount);

		return rate;

	}
}