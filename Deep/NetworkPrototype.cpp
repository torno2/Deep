#include "pch.h"
#include "NetworkPrototype.h"
#include "LayerFunctions.h"

namespace TNNT
{
	//Constructors And destructor

	NetworkPrototype::NetworkPrototype(LayerLayout* layerLayout, FunctionsLayout& functions, unsigned layoutCount, bool randomizeWeightsAndBiases)
		:  m_LayerLayoutCount(layoutCount), m_Functions(functions)
	{


		m_LayerLayout = new LayerLayout[m_LayerLayoutCount];


		m_Functions.NeuronFunctions = new FunctionsLayout::NeuronFunction[m_LayerLayoutCount - 1];
		m_Functions.NeuronFunctionsDerivatives = new FunctionsLayout::NeuronFunction[m_LayerLayoutCount - 1];

		m_Functions.FeedForwardCallBackFunctions = new FunctionsLayout::NetworkRelayFunction[m_LayerLayoutCount - 1];
		m_Functions.BackPropegateCallBackFunctionsZ = new FunctionsLayout::NetworkRelayFunction[m_LayerLayoutCount - 2];
		m_Functions.BackPropegateCallBackFunctionsBW = new FunctionsLayout::NetworkRelayFunction[m_LayerLayoutCount - 1];

		m_Functions.CostFunction = functions.CostFunction;
		m_Functions.CostFunctionDerivative = functions.CostFunctionDerivative;
	

		m_Functions.RegularizationFunctions = new FunctionsLayout::NetworkRelayFunction[m_LayerLayoutCount - 1];
		m_Functions.TrainingFunctions = new FunctionsLayout::NetworkRelayFunction[m_LayerLayoutCount - 1];




		unsigned layoutIndex = 0;
		while (layoutIndex < m_LayerLayoutCount)
		{
			m_LayerLayout[layoutIndex] = layerLayout[layoutIndex];

			if (layoutIndex < m_LayerLayoutCount - 1)
			{
				m_Functions.NeuronFunctions[layoutIndex] = functions.NeuronFunctions[layoutIndex];
				m_Functions.NeuronFunctionsDerivatives[layoutIndex] = functions.NeuronFunctionsDerivatives[layoutIndex];

				m_Functions.FeedForwardCallBackFunctions[layoutIndex] = functions.FeedForwardCallBackFunctions[layoutIndex];
				m_Functions.BackPropegateCallBackFunctionsBW[layoutIndex] = functions.BackPropegateCallBackFunctionsBW[layoutIndex];
				if (layoutIndex < m_LayerLayoutCount - 2)
				{
					m_Functions.BackPropegateCallBackFunctionsZ[layoutIndex] = functions.BackPropegateCallBackFunctionsZ[layoutIndex];
				}


				m_Functions.RegularizationFunctions[layoutIndex] = functions.RegularizationFunctions[layoutIndex];

				m_Functions.TrainingFunctions[layoutIndex] = functions.TrainingFunctions[layoutIndex];






			}

			layoutIndex++;
		}

		// 1 layer no network makes; need at least 2
		assert(m_LayerLayoutCount >= 2);

		// A layer of zero nodes would mean you have two separate networks (or a network with 1 less layer, if the input/output layer is missing), 
		// and a layer with a negative numbers of nodes is something I don't want to think about.

		unsigned nodesTotal = 0;

		//Keep in mind that their aren't supposed to be any biases or weights in the 0th layer, so their count for that layer should both be 0.
		unsigned biasTotal = 0;
		unsigned weightTotal = 0;

		layoutIndex = 0;
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
		m_ZCount = nodesTotal - m_LayerLayout[0].Nodes;
		m_ACount = nodesTotal;
		m_BiasesCount = biasTotal;
		m_WeightsCount = weightTotal;

		m_InputBufferCount = m_LayerLayout[0].Nodes;
		m_OutputBufferCount = m_LayerLayout[m_LayerLayoutCount - 1].Nodes;

		
		//Order: A, Weights, Biases, Z, dZ, dWeights, dBiases, WeightsBuffer, BiasesBuffer, Target
		m_NetworkFixedData = new float[m_ACount + 3 * m_WeightsCount + 3 * m_BiasesCount + 2 * m_ZCount + m_OutputBufferCount];

		m_A = m_NetworkFixedData;
		m_InputBuffer = m_A;
		m_OutputBuffer = &m_A[m_ACount - m_OutputBufferCount];

		m_Weights = &m_NetworkFixedData[m_ACount];
		m_Biases = &m_NetworkFixedData[m_ACount + m_WeightsCount];

		m_Z = &m_NetworkFixedData[m_ACount + m_WeightsCount + m_BiasesCount];
		m_DeltaZ = &m_NetworkFixedData[m_ACount + m_WeightsCount + m_BiasesCount + m_ZCount];

		m_DeltaWeights = &m_NetworkFixedData[m_ACount + m_WeightsCount + m_BiasesCount + 2 * m_ZCount];
		m_DeltaBiases = &m_NetworkFixedData[m_ACount + 2 * m_WeightsCount + m_BiasesCount + 2 * m_ZCount];

		m_WeightsBuffer = &m_NetworkFixedData[m_ACount + 2 * m_WeightsCount + 2 * m_BiasesCount + 2 * m_ZCount];
		m_BiasesBuffer = &m_NetworkFixedData[m_ACount + 3 * m_WeightsCount + 2 * m_BiasesCount + 2 * m_ZCount];
		
		m_TargetBuffer = &m_NetworkFixedData[m_ACount + 3 * m_WeightsCount + 3 * m_BiasesCount + 2 * m_ZCount];

	






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

	}

	NetworkPrototype::~NetworkPrototype()
	{
		delete[] m_LayerLayout;

		delete[] m_NetworkFixedData;


		delete[] m_Indices;
	}

	


	float NetworkPrototype::CheckSuccessRate()
	{
		return CheckSuccessRateMasterFunction( );
	}

	float NetworkPrototype::CheckCost()
	{
		return CheckCostMasterFunction();
	}

	void NetworkPrototype::Train(DataSet* data, HyperParameters& params)
	{
		SetData(data);
		SetHyperParameters(params);
		TrainMasterFunction();
	}

	unsigned NetworkPrototype::Check(float* input)
	{
		return(CheckMasterFunction(input));
	}





	//Public functions


	// Private functions

	void NetworkPrototype::SetBiasesToTemp()
	{

		memcpy(m_Biases, m_BiasesBuffer, sizeof(float) * m_BiasesCount);
	}

	void NetworkPrototype::SetTempToBiases()
	{

		memcpy(m_BiasesBuffer, m_Biases, sizeof(float) * m_BiasesCount);

	}


void NetworkPrototype::SetWeightsToTemp()	
	{	

		memcpy(m_Weights, m_WeightsBuffer, sizeof(float) * m_WeightsCount);
		
	}

	void NetworkPrototype::SetTempToWeights()
	{

		memcpy(m_WeightsBuffer, m_Weights, sizeof(float) * m_WeightsCount);
	}




	void NetworkPrototype::SetData(DataSet* data)
	{
		delete[] m_Indices;

		m_Data = data;

		m_Indices = new unsigned[m_Data->TrainingCount];



		unsigned index = 0;
		while (index < m_Data->TrainingCount)
		{
			m_Indices[index] = index;
			index++;
		}

	}



	void NetworkPrototype::SetHyperParameters(HyperParameters& params)
	{

		m_HyperParameters = params;


	}


	void NetworkPrototype::SetInput(const float* input)
	{

		memcpy(m_InputBuffer, input, sizeof(float) * m_InputBufferCount);

	}

	void NetworkPrototype::SetTarget(const float* target)
	{



		memcpy(m_TargetBuffer, target, sizeof(float) * m_OutputBufferCount);


	}


	void NetworkPrototype::FeedForward()
	{


		{
			m_PositionData.Layer = 1;
			m_PositionData.Z = 0;
			m_PositionData.A = m_LayerLayout[0].Nodes;
			m_PositionData.Biases = m_LayerLayout[0].Biases;
			m_PositionData.Weights = m_LayerLayout[0].Weights;
		}

		unsigned layoutIndex = 1;

		while (layoutIndex < m_LayerLayoutCount)
		{

			//No function for the inputlayer, which means that the function corresponding to any other layer is located at layer - 1.
			m_Functions.FeedForwardCallBackFunctions[layoutIndex - 1].f(this);




			{
				m_PositionData.Z += m_LayerLayout[m_PositionData.Layer].Nodes;

				m_PositionData.A += m_LayerLayout[m_PositionData.Layer].Nodes;

				m_PositionData.Biases += m_LayerLayout[m_PositionData.Layer].Biases;

				m_PositionData.Weights += m_LayerLayout[m_PositionData.Layer].Weights;

				m_PositionData.Layer++;

			}


			layoutIndex++;

		}

	}

	void NetworkPrototype::Backpropegate()
	{

		unsigned lastLayer = m_LayerLayoutCount - 1;

		
		{
			m_PositionData.Layer = lastLayer;

			m_PositionData.Z = m_ZCount - m_LayerLayout[lastLayer].Nodes;
			m_PositionData.A = m_ACount - m_LayerLayout[lastLayer].Nodes;

			m_PositionData.Biases = m_BiasesCount - m_LayerLayout[lastLayer].Biases;
			m_PositionData.Weights = m_WeightsCount - m_LayerLayout[lastLayer].Weights;
		}



		{

			m_Functions.CostFunctionDerivative.f(this);

		}

		unsigned reveresLayoutIndex = 0;
		
		while (reveresLayoutIndex < lastLayer - 1)
		{
			m_Functions.BackPropegateCallBackFunctionsBW[reveresLayoutIndex].f(this);


			//Alters position data to be more in line with the layer in question

			{
				m_PositionData.Layer--;

				m_PositionData.Z -= m_LayerLayout[m_PositionData.Layer].Nodes;

				m_PositionData.A -= m_LayerLayout[m_PositionData.Layer].Nodes;

				m_PositionData.Biases -= m_LayerLayout[m_PositionData.Layer].Biases;

				m_PositionData.Weights -= m_LayerLayout[m_PositionData.Layer].Weights;
			}



			m_Functions.BackPropegateCallBackFunctionsZ[reveresLayoutIndex].f(this);
		

			reveresLayoutIndex++;
		}

		m_Functions.BackPropegateCallBackFunctionsBW[reveresLayoutIndex].f(this);

	}

	void NetworkPrototype::Regularization()
	{

		{
			m_PositionData.Layer = 1;
			m_PositionData.Z = 0;
			m_PositionData.A = m_LayerLayout[0].Nodes;
			m_PositionData.Biases = m_LayerLayout[0].Biases;
			m_PositionData.Weights = m_LayerLayout[0].Weights;
		}

		unsigned layoutIndex = 1;

		while (layoutIndex < m_LayerLayoutCount)
		{

			
			m_Functions.RegularizationFunctions[layoutIndex - 1].f(this);




			{
				m_PositionData.Z += m_LayerLayout[m_PositionData.Layer].Nodes;

				m_PositionData.A += m_LayerLayout[m_PositionData.Layer].Nodes;

				m_PositionData.Biases += m_LayerLayout[m_PositionData.Layer].Biases;

				m_PositionData.Weights += m_LayerLayout[m_PositionData.Layer].Weights;

				m_PositionData.Layer++;

			}


			layoutIndex++;

		}
	}

	void NetworkPrototype::Train()
	{
		{
			m_PositionData.Layer = 1;
			m_PositionData.Z = 0;
			m_PositionData.A = m_LayerLayout[0].Nodes;
			m_PositionData.Biases = m_LayerLayout[0].Biases;
			m_PositionData.Weights = m_LayerLayout[0].Weights;
		}

		unsigned layoutIndex = 1;

		while (layoutIndex < m_LayerLayoutCount)
		{


			m_Functions.TrainingFunctions[layoutIndex - 1].f(this);




			{
				m_PositionData.Z += m_LayerLayout[m_PositionData.Layer].Nodes;

				m_PositionData.A += m_LayerLayout[m_PositionData.Layer].Nodes;

				m_PositionData.Biases += m_LayerLayout[m_PositionData.Layer].Biases;

				m_PositionData.Weights += m_LayerLayout[m_PositionData.Layer].Weights;

				m_PositionData.Layer++;

			}


			layoutIndex++;

		}
	}

 

	void NetworkPrototype::TrainOnSet(unsigned batchCount , unsigned batch)
	{


		Regularization();


		

	

		unsigned exampleIndex = 0;
		while (exampleIndex < batchCount)
		{

			unsigned index = m_Indices[exampleIndex + batch * m_HyperParameters.BatchCount];
			SetInput(&(m_Data->TrainingInputs[index * m_InputBufferCount]));
			SetTarget(&(m_Data->TraningTargets[index * m_OutputBufferCount]));


			FeedForward();
			Backpropegate();

			Train();


			


			exampleIndex++;


		}


		SetBiasesToTemp();
		SetWeightsToTemp();

	}


	void NetworkPrototype::TrainMasterFunction()
	{

		//Timer start
		auto start = std::chrono::high_resolution_clock::now();




		const unsigned batchNum = m_Data->TrainingCount / m_HyperParameters.BatchCount;
		const unsigned remainingBatch = m_Data->TrainingCount % m_HyperParameters.BatchCount;

		std::mt19937 mt;
		

		unsigned epochNum = 0;
		while (epochNum < m_HyperParameters.Epochs)
		{
			
			unsigned randomIndexPos = 0;
			unsigned randomIndexCount = m_Data->TrainingCount;


			unsigned batch = 0;
			while (batch < batchNum)
			{
				unsigned batchIndex = 0;
				while (batchIndex < m_HyperParameters.BatchCount)
				{

					unsigned randomIndex = (mt() % randomIndexCount)+ randomIndexPos;

					unsigned epochRandomIndex = m_Indices[randomIndex];
					m_Indices[randomIndex] = m_Indices[randomIndexPos];
					m_Indices[randomIndexPos] = epochRandomIndex;


					randomIndexPos++;
					randomIndexCount--;

					batchIndex++;
				}

				TrainOnSet(m_HyperParameters.BatchCount, batch);

				batch++;
			}


			if (remainingBatch > 0)
			{
				unsigned batchIndex = 0;
				while (batchIndex < remainingBatch)
				{


					unsigned randomIndex = (mt() % randomIndexCount) + randomIndexPos;

					unsigned epochRandomIndex = m_Indices[randomIndex];
					m_Indices[randomIndex] = m_Indices[randomIndexPos];
					m_Indices[randomIndexPos] = epochRandomIndex;



					randomIndexPos++;
					randomIndexCount--;
					batchIndex++;

				}
				unsigned tempBatchCount = m_HyperParameters.BatchCount;
				m_HyperParameters.BatchCount = remainingBatch;
				TrainOnSet(remainingBatch, batch);
				m_HyperParameters.BatchCount = tempBatchCount;
			}


			epochNum++;
		}
		
		//TODO Remove this:
		


		


		//Timer stop
		auto stop = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float>  time = stop - start;
		m_LastTime[0] = time.count();
	}

	
	float NetworkPrototype::CheckCostMasterFunction( )
	{

		auto start = std::chrono::high_resolution_clock::now();
		

		m_CostBuffer = 0;

		unsigned checkIndex = 0;
		while (checkIndex < m_Data->TestCount)
		{

			SetInput( &m_Data->TestInputs[checkIndex * m_InputBufferCount]  );
			SetTarget(&m_Data->TestTargets[checkIndex * m_OutputBufferCount]);
			
			
			FeedForward();
			
			m_Functions.CostFunction.f(this);
			

			checkIndex++;
		}



		auto stop = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float>  time = stop - start;
		m_LastTime[1] = time.count();


	
		return  m_CostBuffer / ((float)m_Data->TestCount);

	}

	float NetworkPrototype::CheckSuccessRateMasterFunction()
	{
		auto start = std::chrono::high_resolution_clock::now();

		

		float score = 0.0f;

		unsigned checkIndex = 0;
		while (checkIndex < m_Data->TestCount)
		{

			SetInput(&m_Data->TestInputs[checkIndex * m_InputBufferCount]);
			FeedForward();

			int championItterator = -1;
			float champion = 0;
			unsigned outputIndex = 0;
			while (outputIndex < m_OutputBufferCount)
			{
				
				if (m_OutputBuffer[outputIndex] >= champion)
				{
					champion = m_OutputBuffer[outputIndex];
					championItterator = outputIndex;
				}


				outputIndex++;
			}

			if (m_Data->TestTargets[m_OutputBufferCount * checkIndex + championItterator] == 1)
			{
				score += 1.0f;
			}
			checkIndex++;
		}

		float rate = score / ((float)m_Data->TestCount);

		auto stop = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float>  time = stop - start;
		m_LastTime[2] = time.count();

		return rate;

	}

	unsigned NetworkPrototype::CheckMasterFunction(float* input)
	{
		SetInput(input);
		FeedForward();

		int champIndex = -1;
		float champ = 0;

		unsigned index = 0;
		while (index < m_OutputBufferCount)
		{
			if (m_OutputBuffer[index] > champ)
			{
				champ = m_OutputBuffer[index];
				champIndex = index;
			}
			index++;
		}

		//This is not allowed.
		assert(champIndex != -1);


		return champIndex;
	}


}