#include "pch.h"
#include "NetworkPrototypeMT.h"


namespace TNNT
{

	NetworkPrototypeMT::NetworkPrototypeMT(LayerLayout* layerLayout, FunctionsLayoutMT& functions, unsigned layoutCount, unsigned slaveThreadCount , bool randomizeWeightsAndBiases)
		: m_LayerLayoutCount(layoutCount), m_SlaveThreadCount(slaveThreadCount)
	{


		// 1 layer no network makes; need at least 2
		assert(m_LayerLayoutCount >= 2);

		// A layer of zero nodes would mean you have two separate networks (or a network with 1 less layer, if the input/output layer is missing), 
		// and a layer with a negative numbers of nodes is something I don't want to think about.

		//These are gonna get reused a lot in the constructor.
		unsigned layoutIndex = 0;

	//Getting the LayerLayout ready START

		m_LayerLayout = new LayerLayout[m_LayerLayoutCount];

		layoutIndex = 0;
		while (layoutIndex < m_LayerLayoutCount)
		{
			m_LayerLayout[layoutIndex] = layerLayout[layoutIndex];

			layoutIndex++;
		}



	//Getting the LayerLayout ready STOP




	// DETERMENING THE COUNT OF MOST ARRAYS AND CALCULATING OTHER IMPORTANT INTEGERS START

		

		//Keep in mind that there aren't supposed to be any biases or weights in the 0th layer, so their count for that layer should both be 0.
		unsigned nodesTotal = 0;
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
		m_ZCount = nodesTotal - m_LayerLayout[0].Nodes;
		m_ACount = nodesTotal;
		m_BiasesCount = biasTotal;
		m_WeightsCount = weightTotal;

		m_InputBufferCount = m_LayerLayout[0].Nodes;
		m_OutputBufferCount = m_LayerLayout[m_LayerLayoutCount - 1].Nodes;

		m_PaddingData.FloatPaddingPerLayer = m_SlaveThreadCount * m_PaddingData.FloatPadding;

		unsigned APadding = m_PaddingData.FloatPaddingPerLayer * m_LayerLayoutCount;
		unsigned BiasesPadding = m_PaddingData.FloatPaddingPerLayer * (m_LayerLayoutCount - 1);
		unsigned WeightsPadding = m_PaddingData.FloatPaddingPerLayer * (m_LayerLayoutCount - 1);
		unsigned ZPadding = m_PaddingData.FloatPaddingPerLayer * (m_LayerLayoutCount - 1);

		unsigned OutputbufferPadding = m_PaddingData.FloatPaddingPerLayer;

	// DETERMENING THE COUNT OF MOST ARRAYS AND CALCULATING OTHER IMPORTANT INTEGERS START



	//MEMORY ALLOCATION AND POINTER SETUP START
		
		//Order: A, Weights, Biases, Z, dZ, WeightsBuffer, BiasesBuffer, dWeights, dBiases,  Target
		m_NetworkFixedData = new float[(m_ACount+APadding) + 3 * (m_WeightsCount+WeightsPadding) + 3 * (m_BiasesCount+BiasesPadding) + 2 * (m_ZCount+ZPadding) + (m_OutputBufferCount + OutputbufferPadding)];

		//NETWORK STRUCTURE
		m_A = m_NetworkFixedData;
		m_InputBuffer = m_A;
		m_OutputBuffer = &m_A[(m_ACount- m_OutputBufferCount) + (APadding- m_PaddingData.FloatPaddingPerLayer)];

		m_Weights = &m_NetworkFixedData[(m_ACount+APadding)];
		m_Biases = &m_NetworkFixedData[(m_ACount+APadding) + (m_WeightsCount+WeightsPadding)];

		m_Z = &m_NetworkFixedData[(m_ACount+APadding) + (m_WeightsCount+WeightsPadding) + (m_BiasesCount+BiasesPadding)];
		m_DeltaZ = &m_NetworkFixedData[(m_ACount+APadding) + (m_WeightsCount+WeightsPadding) + (m_BiasesCount+BiasesPadding) + (m_ZCount+ZPadding)];

		m_WeightsBuffer = &m_NetworkFixedData[(m_ACount+APadding) + (m_WeightsCount+WeightsPadding) + (m_BiasesCount+BiasesPadding) + 2 * (m_ZCount+ZPadding)];
		m_BiasesBuffer = &m_NetworkFixedData[(m_ACount+APadding) + 2 * (m_WeightsCount+WeightsPadding) + (m_BiasesCount+BiasesPadding) + 2 * (m_ZCount+ZPadding)];
		
		m_DeltaWeights = &m_NetworkFixedData[(m_ACount+APadding) + 2 * (m_WeightsCount+WeightsPadding) + 2 * (m_BiasesCount+BiasesPadding) + 2 * (m_ZCount+ZPadding)];
		m_DeltaBiases = &m_NetworkFixedData[(m_ACount+APadding) + 3 * (m_WeightsCount+WeightsPadding) + 2 * (m_BiasesCount+BiasesPadding) + 2 * (m_ZCount+ZPadding)];

		//EVALUATION BUFFERS
		m_TargetBuffer = &m_NetworkFixedData[(m_ACount + APadding) + 3 * (m_WeightsCount + WeightsPadding) + 3 * (m_BiasesCount + BiasesPadding) + 2 * (m_ZCount + ZPadding)];

		m_CostBuffer = new float[m_SlaveThreadCount];
		m_GuessBuffer = new unsigned[m_SlaveThreadCount];

		//NETWORK STRUCTURE (Function layout)

		m_Functions.NeuronFunctions = new FunctionsLayoutMT::NeuronFunction[m_LayerLayoutCount - 1];
		m_Functions.NeuronFunctionsDerivatives = new FunctionsLayoutMT::NeuronFunction[m_LayerLayoutCount - 1];

		m_Functions.FeedForwardCallBackFunctions = new FunctionsLayoutMT::NetworkRelayFunctionMT[m_LayerLayoutCount - 1];
		m_Functions.BackPropegateCallBackFunctionsZ = new FunctionsLayoutMT::NetworkRelayFunctionMT[m_LayerLayoutCount - 2];
		m_Functions.BackPropegateCallBackFunctionsBW = new FunctionsLayoutMT::NetworkRelayFunctionMT[m_LayerLayoutCount - 1];

		m_Functions.CostFunction = functions.CostFunction;
		m_Functions.CostFunctionDerivative = functions.CostFunctionDerivative;

		m_Functions.TrainingFunction = functions.TrainingFunction;
		m_Functions.RegularizationFunction = functions.RegularizationFunction;

		//MULTITHREAD MANAGEMENT
		m_SlaveThreads = new std::thread[m_SlaveThreadCount];
		m_Locks = new bool[m_SlaveThreadCount * 2];
		m_SlaveFlags = new bool[m_SlaveThreadCount];


		//PADDING
		m_PaddingData.Nodes = new unsigned[2 * m_SlaveThreadCount * m_LayerLayoutCount];
		m_PaddingData.Weights = new unsigned[2 * m_SlaveThreadCount * (m_LayerLayoutCount - 1)];
		m_PaddingData.Biases = new unsigned[2 * m_SlaveThreadCount * (m_LayerLayoutCount - 1)];

	//MEMORY ALLOCATION AND POINTER SETUP STOP



	//FUNCTION LAYOUT SETUP START

		layoutIndex = 0;
		while (layoutIndex < m_LayerLayoutCount)
		{

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

			}

			layoutIndex++;
		}

	//FUNCTION LAYOUT STOP



	//ENSURING THAT CERTAIN INTEGER AND FLOAT ARRAYS HAVE ACCEPTABLE INITIAL VALUES START


		//PADDING DATA SETUP SECTION START
		layoutIndex = 0;
		while (layoutIndex < m_LayerLayoutCount)
		{
			unsigned NStart, NStop, WStart, WStop, BStart, BStop;

			unsigned threadIndex = 0;
			while (threadIndex < m_SlaveThreadCount)
			{
				ThreadWorkloadDividerWithPadding(NStart, NStop, m_LayerLayout[layoutIndex].Nodes, threadIndex);

				m_PaddingData.Nodes[layoutIndex * (2 * m_SlaveThreadCount) + 2 * threadIndex] = NStart;
				m_PaddingData.Nodes[layoutIndex * (2 * m_SlaveThreadCount) + 2 * threadIndex + 1] = NStop;

				if (layoutIndex != 0)
				{
					ThreadWorkloadDividerWithPadding(WStart, WStop, m_LayerLayout[layoutIndex].Weights, threadIndex);
					ThreadWorkloadDividerWithPadding(BStart, BStop, m_LayerLayout[layoutIndex].Biases, threadIndex);

					m_PaddingData.Weights[(layoutIndex - 1) * (2 * m_SlaveThreadCount) + 2 * threadIndex] = WStart;
					m_PaddingData.Weights[(layoutIndex - 1) * (2 * m_SlaveThreadCount) + 2 * threadIndex + 1] = WStop;

					m_PaddingData.Biases[(layoutIndex - 1) * (2 * m_SlaveThreadCount) + 2 * threadIndex] = BStart;
					m_PaddingData.Biases[(layoutIndex - 1) * (2 * m_SlaveThreadCount) + 2 * threadIndex + 1] = BStop;
				}


				threadIndex++;
			}


			layoutIndex++;
		}
		//PADDING SETUP SECTION STOP


		//SPINLOCK SETUP START
		unsigned flagIndex = 0;
		while (flagIndex < m_SlaveThreadCount)
		{
			m_Locks[flagIndex * 2] = false;
			m_Locks[flagIndex * 2 + 1] = false;
			m_SlaveFlags[flagIndex] = false;
			flagIndex++;
		}
		//SPINLOCK SETUP STOP


		//WEIGHTS AND BIASES SETUP START
		if (randomizeWeightsAndBiases)
		{
			//For randomly initializing the weights and biases
			std::default_random_engine generator;
			std::normal_distribution<float> distribution(0.0f, 1 / sqrt(m_LayerLayout[0].Nodes));

			unsigned weightStart, weightStop;
			unsigned biasStart, biasStop;

			float champ = 0.0f;

			//Randomly assigns weights and biases.
			layoutIndex = 1;
			while (layoutIndex < m_LayerLayoutCount)
			{
				

				unsigned slaveThread = 0;
				while (slaveThread < m_SlaveThreadCount)
				{
					weightStart = m_PaddingData.Weights[2 * slaveThread + 2*(layoutIndex - 1) * m_SlaveThreadCount];
					weightStop = m_PaddingData.Weights[1 + 2 * slaveThread + 2*(layoutIndex - 1) * m_SlaveThreadCount];
					biasStart = m_PaddingData.Biases[2 * slaveThread + 2*(layoutIndex - 1) * m_SlaveThreadCount];
					biasStop = m_PaddingData.Biases[1 + 2 * slaveThread + 2*(layoutIndex - 1) * m_SlaveThreadCount];
						

					unsigned i = weightStart;
					while (i < weightStop)
					{

						m_Weights[i] = distribution(generator);
						
						if (abs(m_Weights[i]) > champ)
						{
							champ = abs(m_Weights[i]);
						}

						i++;
					}

					i = biasStart;
					while (i < biasStop)
					{

						m_Biases[i] = distribution(generator);

						i++;
					}
					slaveThread++;
				}				

				layoutIndex++;
			}

		}
		else {
			//Sets all weights and biases to zero

			unsigned weightStart, weightStop;
			unsigned biasStart, biasStop;

			layoutIndex = 1;
			while (layoutIndex < m_LayerLayoutCount)
			{
				unsigned slaveThread = 0;
				while (slaveThread < m_SlaveThreadCount)
				{


					weightStart = m_PaddingData.Weights[2 * slaveThread + 2 * (layoutIndex - 1) * m_SlaveThreadCount];
					weightStop = m_PaddingData.Weights[1 + 2 * slaveThread + 2 * (layoutIndex - 1) * m_SlaveThreadCount];
					biasStart = m_PaddingData.Biases[2 * slaveThread + 2 * (layoutIndex - 1) * m_SlaveThreadCount];
					biasStop = m_PaddingData.Biases[1 + 2 * slaveThread + 2 * (layoutIndex - 1) * m_SlaveThreadCount];


					unsigned i = weightStart;
					while (i < weightStop)
					{

						m_Weights[i] = 0;
						i++;
					}

					i = biasStart;
					while (i < biasStop)
					{

						m_Biases[i] = 0;
						i++;
					}
					slaveThread++;
				}
				

				layoutIndex++;
			}
		}
		
		{
			unsigned thread = 0;
			while (thread < m_SlaveThreadCount)
			{
				m_SlaveThreads[thread] = std::thread(&NetworkPrototypeMT::SetTempToBiasesAndWeights, this, thread);
				thread++;
			}

			thread = 0;
			while (thread < m_SlaveThreadCount)
			{
				m_SlaveThreads[thread].join();
				thread++;
			}
		}
		//WEIGHTS AND BIASES SETUP STOP



	//ENSURING THAT CERTAIN INTEGER AND FLOAT ARRAYS HAVE ACCEPTABLE INITIAL VALUES STOP

	}

	NetworkPrototypeMT::~NetworkPrototypeMT()
	{

		delete[] m_LayerLayout;

		delete[] m_NetworkFixedData;

		delete[] m_CostBuffer;
		delete[] m_GuessBuffer;


		delete[] m_SlaveThreads;
		delete[] m_Locks;
		delete[] m_SlaveFlags;

	}

	float NetworkPrototypeMT::CheckSuccessRate()
	{
		return CheckSuccessRateMasterFunction();
	}

	float NetworkPrototypeMT::CheckCost()
	{
		return CheckCostMasterFunction();
	}

	void NetworkPrototypeMT::Train(DataSet* data, HyperParameters& params)
	{
		SetData(data);
		SetHyperParameters(params);
		TrainMasterFunction();

	}

	unsigned NetworkPrototypeMT::Check(float* input)
	{
		return CheckMasterFunction(input);
	}





	void NetworkPrototypeMT::SetTempToBiases(unsigned thread)
	{
		

		unsigned localStart = 0;
		unsigned localStartPos = 2 * thread;
		unsigned localStop = 0;
		unsigned localStopPos = 2 * thread + 1;

		unsigned globalOffset = 0;

		unsigned layer = 0;
		while (layer < (m_LayerLayoutCount-1) )
		{
			
			localStart = m_PaddingData.Biases[localStartPos];
			localStop = m_PaddingData.Biases[localStopPos];


			unsigned dist = localStop - localStart;
			memcpy(&m_BiasesBuffer[localStart + globalOffset], &m_Biases[localStart + globalOffset], dist * sizeof(float));

			globalOffset += m_LayerLayout[layer + 1].Biases + m_PaddingData.FloatPaddingPerLayer;

			layer++;
			localStartPos += 2 * m_SlaveThreadCount;
			localStopPos += 2 * m_SlaveThreadCount;
		}
		

	}

	void NetworkPrototypeMT::SetTempToWeights(unsigned thread)
	{

		unsigned localStart = 0;
		unsigned localStartPos = 2 * thread;
		unsigned localStop = 0;
		unsigned localStopPos = 2 * thread + 1;

		unsigned globalOffset = 0;

		unsigned layer = 0;
		while (layer < (m_LayerLayoutCount - 1))
		{

			localStart = m_PaddingData.Weights[localStartPos];
			localStop = m_PaddingData.Weights[localStopPos];

			unsigned dist = localStop - localStart;
			memcpy(&m_WeightsBuffer[localStart + globalOffset], &m_Weights[localStart + globalOffset], dist * sizeof(float));

			globalOffset += m_LayerLayout[layer + 1].Weights + m_PaddingData.FloatPaddingPerLayer;

			layer++;
			localStartPos += 2 * m_SlaveThreadCount;
			localStopPos += 2 * m_SlaveThreadCount;
		}

	}

	void NetworkPrototypeMT::SetTempToBiasesAndWeights(unsigned thread)
	{
		SetTempToBiases(thread);
		SetTempToWeights(thread);
	}

	void NetworkPrototypeMT::SetBiasesToTemp(unsigned thread)
	{

		unsigned localStart = 0;
		unsigned localStartPos = 2 * thread;
		unsigned localStop = 0;
		unsigned localStopPos = 2 * thread + 1;

		unsigned globalOffset = 0;

		unsigned layer = 0;
		while (layer < (m_LayerLayoutCount - 1))
		{

			localStart = m_PaddingData.Biases[localStartPos];
			localStop = m_PaddingData.Biases[localStopPos];

			unsigned dist = localStop - localStart;
			memcpy(&m_Biases[localStart + globalOffset], &m_BiasesBuffer[localStart + globalOffset], dist * sizeof(float));

			globalOffset += m_LayerLayout[layer + 1].Biases + m_PaddingData.FloatPaddingPerLayer;

			layer++;
			localStartPos += 2 * m_SlaveThreadCount;
			localStopPos += 2 * m_SlaveThreadCount;
		}
	}

	void NetworkPrototypeMT::SetWeightsToTemp(unsigned thread)
	{

		unsigned localStart = 0;
		unsigned localStartPos = 2*thread;
		unsigned localStop = 0;
		unsigned localStopPos = 2*thread +1;

		unsigned globalOffset = 0;

		unsigned layer = 0;
		while (layer < (m_LayerLayoutCount - 1))
		{

			localStart = m_PaddingData.Weights[localStartPos];
			localStop = m_PaddingData.Weights[localStopPos];

			unsigned dist = localStop - localStart;
			memcpy(&m_Weights[localStart + globalOffset], &m_WeightsBuffer[localStart + globalOffset], dist * sizeof(float));

			globalOffset += m_LayerLayout[layer+1].Weights + m_PaddingData.FloatPaddingPerLayer;

			layer++;
			localStartPos += 2 * m_SlaveThreadCount;
			localStopPos +=  2 * m_SlaveThreadCount;
		}


	}

	void NetworkPrototypeMT::SetData(DataSet* data)
	{
		delete[] m_Indices;

		m_Data = data;

		m_Indices = new unsigned[m_Data->TrainingCount ];


		unsigned index = 0;
		while (index < m_Data->TrainingCount)
		{

			m_Indices[index] = index;

			index++;
		}

	}

	//TODO figure out whether you want to keep this or not.
	void NetworkPrototypeMT::ResetIndices(unsigned thread)
	{
		unsigned start;
		unsigned stop;
		ThreadWorkloadDivider(start, stop, m_Data->TrainingCount, thread);

		

		unsigned index = start;
		while (index < stop)
		{
			
			m_Indices[index] = index;
			
			index++;
		}
	}

	void NetworkPrototypeMT::SetHyperParameters(HyperParameters& params)
	{


		m_HyperParameters = params;

	}

	void NetworkPrototypeMT::SetInput(float* input, unsigned thread)
	{
		unsigned start = m_PaddingData.Nodes[2 * thread ];
		unsigned stop = m_PaddingData.Nodes[2 * thread + 1];

		unsigned dist = stop - start;
		
		unsigned paddingCorrection = thread * m_PaddingData.FloatPadding;


		memcpy(&m_InputBuffer[start], &input[start - paddingCorrection], dist * sizeof(float));


	}

	void NetworkPrototypeMT::SetTarget(float* target, unsigned thread)
	{
		unsigned start = m_PaddingData.Nodes[2 * thread + 2 * m_SlaveThreadCount * (m_LayerLayoutCount - 1)];
		unsigned stop = m_PaddingData.Nodes[2 * thread + 1 + 2 * m_SlaveThreadCount * (m_LayerLayoutCount - 1)];

		unsigned dist = stop - start;

		unsigned paddingCorrection = thread * m_PaddingData.FloatPadding;


		memcpy(&m_TargetBuffer[start], &target[start - paddingCorrection], dist * sizeof(float));
			

	}

	void NetworkPrototypeMT::ThreadWorkloadDivider(unsigned& start, unsigned& stop, unsigned workCount, unsigned thread)
	{
		start = 0;
		stop = 0;

		unsigned workloadCount = (workCount / m_SlaveThreadCount);
		unsigned workloadRemainder = (workCount % m_SlaveThreadCount);


		if (thread < workloadRemainder)
		{
			start = (workloadCount + 1) * thread;
			stop = start + (workloadCount + 1);
		}
		else
		{
			start = workloadCount * thread + workloadRemainder;

			stop = start + workloadCount;
		}
	}

	void NetworkPrototypeMT::ThreadWorkloadDividerWithPadding(unsigned& start, unsigned& stop, unsigned workCount, unsigned thread)
	{
		start = 0;
		stop = 0;

		unsigned workloadCount = (workCount / m_SlaveThreadCount);
		unsigned workloadRemainder = (workCount % m_SlaveThreadCount);


		if (thread < workloadRemainder)
		{
			start = (workloadCount + 1) * thread + m_PaddingData.FloatPadding * thread;
			stop = start + (workloadCount + 1);
		}
		else
		{
			start = workloadCount * thread + workloadRemainder + m_PaddingData.FloatPadding * thread;

			stop = start + workloadCount;
		}
	}

	void NetworkPrototypeMT::SpinLock(unsigned thread)
	{

		unsigned i = 0;
		while (i < m_SlaveThreadCount)
		{
			if (i == thread)
			{
				m_Locks[i + m_SlaveThreadCount] = false;
				m_Locks[i] = true;
			}
			if (!m_Locks[i])
			{
				continue;
			}
			i++;
		}
		while (i < 2 * m_SlaveThreadCount)
		{
			if (i == (thread + m_SlaveThreadCount))
			{
				m_Locks[i - m_SlaveThreadCount] = false;
				m_Locks[i] = true;
			}
			if (!m_Locks[i])
			{
				continue;
			}
			i++;
		}
	}

	void NetworkPrototypeMT::SlaveControlStation(unsigned position)
	{

		while (position > m_MasterControlPoint)
		{
			;//About as empty as my head.
		}
	}

	void NetworkPrototypeMT::WaitForSlaves()
	{
		unsigned i = 0;
		while (i < m_SlaveThreadCount)
		{
			if (m_SlaveFlags[i])
			{
				m_SlaveFlags[i] = false;
				i++;
			}
		}

	}

	void NetworkPrototypeMT::FeedForward(unsigned thread)
	{

		if (thread == 0)
		{
			m_PositionData.Layer = 1;
			m_PositionData.Z = 0;
			m_PositionData.A = m_LayerLayout[0].Nodes			+ m_PaddingData.FloatPaddingPerLayer;
			m_PositionData.Biases = 0;
			m_PositionData.Weights = 0;
		}
		SpinLock(thread);
		unsigned layoutIndex = 1;

		while (layoutIndex < m_LayerLayoutCount)
		{

			//No function for the inputlayer, which means that the function corresponding to any other layer is located at layer - 1.
			m_Functions.FeedForwardCallBackFunctions[layoutIndex - 1].f(this, thread);
			SpinLock(thread);

			//PositionData
			if (thread == 0)
			{
				m_PositionData.Z += m_LayerLayout[m_PositionData.Layer].Nodes			+ m_PaddingData.FloatPaddingPerLayer;

				m_PositionData.A += m_LayerLayout[m_PositionData.Layer].Nodes			+ m_PaddingData.FloatPaddingPerLayer;

				m_PositionData.Biases += m_LayerLayout[m_PositionData.Layer].Biases		+ m_PaddingData.FloatPaddingPerLayer;

				m_PositionData.Weights += m_LayerLayout[m_PositionData.Layer].Weights	+ m_PaddingData.FloatPaddingPerLayer;

				m_PositionData.Layer++;

			}
			
			SpinLock(thread);
			layoutIndex++;
			
		}



	}

	void NetworkPrototypeMT::Backpropegate(unsigned thread)
	{

		unsigned lastLayer = m_LayerLayoutCount - 1;

		if (thread == 0)
		{
			m_PositionData.Layer = lastLayer;

			m_PositionData.Z = (m_ZCount - m_LayerLayout[lastLayer].Nodes)					+ m_PaddingData.FloatPaddingPerLayer * (lastLayer-1);
			m_PositionData.A = (m_ACount - m_LayerLayout[lastLayer].Nodes)					+ m_PaddingData.FloatPaddingPerLayer * lastLayer;

			m_PositionData.Biases  = (m_BiasesCount - m_LayerLayout[lastLayer].Biases)		+ m_PaddingData.FloatPaddingPerLayer * (lastLayer - 1);
			m_PositionData.Weights = (m_WeightsCount - m_LayerLayout[lastLayer].Weights)	+ m_PaddingData.FloatPaddingPerLayer * (lastLayer - 1);
		}
		SpinLock(thread);


		{

			m_Functions.CostFunctionDerivative.f(this,thread);
			SpinLock(thread);
		}

		unsigned reveresLayoutIndex = 0;

		while (reveresLayoutIndex < lastLayer - 1)
		{
			m_Functions.BackPropegateCallBackFunctionsBW[reveresLayoutIndex].f(this,thread);
			SpinLock(thread);

			//Alters position data to be more in line with the layer in question
			if(thread == 0)
			{
				m_PositionData.Layer--;

				m_PositionData.Z -= (m_LayerLayout[m_PositionData.Layer].Nodes + m_PaddingData.FloatPaddingPerLayer);

				m_PositionData.A -= (m_LayerLayout[m_PositionData.Layer].Nodes + m_PaddingData.FloatPaddingPerLayer);

				m_PositionData.Biases -= (m_LayerLayout[m_PositionData.Layer].Biases + m_PaddingData.FloatPaddingPerLayer);

				m_PositionData.Weights -= (m_LayerLayout[m_PositionData.Layer].Weights + m_PaddingData.FloatPaddingPerLayer);
			}
			SpinLock(thread);
			

			m_Functions.BackPropegateCallBackFunctionsZ[reveresLayoutIndex].f(this,thread);
			SpinLock(thread);

			reveresLayoutIndex++;
		}

		m_Functions.BackPropegateCallBackFunctionsBW[reveresLayoutIndex].f(this, thread);
		SpinLock(thread);
	}

	void NetworkPrototypeMT::Regularization(unsigned thread)
	{
	}

	void NetworkPrototypeMT::Train(unsigned thread)
	{
	}


	void NetworkPrototypeMT::TrainOnSet(unsigned batchCount, unsigned batch, unsigned thread)
	{
		SpinLock(thread);
		m_Functions.RegularizationFunction.f(this,thread);
		
		
		

		unsigned exampleIndex = 0;
		while (exampleIndex < batchCount)
		{


			
			unsigned index = m_Indices[exampleIndex + batch * m_HyperParameters.BatchCount];
			SetInput(&(m_Data->TrainingInputs[index * m_InputBufferCount]), thread);
			SetTarget(&(m_Data->TraningTargets[index * m_OutputBufferCount]), thread);
			
			SpinLock(thread);

			FeedForward(thread);
			Backpropegate(thread);

			m_Functions.TrainingFunction.f(this, thread);



			exampleIndex++;


		}

		SpinLock(thread);
		SetBiasesToTemp(thread);
		SetWeightsToTemp(thread);
	
	}

	void NetworkPrototypeMT::TrainSlaveFunction( unsigned thread)
	{

		const unsigned batchTotal = m_Data->TrainingCount / m_HyperParameters.BatchCount;
		const unsigned remainingBatchCount = m_Data->TrainingCount % m_HyperParameters.BatchCount;

		unsigned position = 0;

		unsigned epoch = 0;
		while (epoch < m_HyperParameters.Epochs)
		{

			unsigned batch = 0;
			while (batch < batchTotal)
			{
				position++;

				SlaveControlStation(position);
				TrainOnSet(m_HyperParameters.BatchCount, batch, thread);

				batch++;
			}


			if(remainingBatchCount !=0)
			{
				m_SlaveFlags[thread] = true;

				position++;
				SlaveControlStation(position);

				TrainOnSet(remainingBatchCount, batch , thread);
			}

			epoch++;
		}

	}

	void NetworkPrototypeMT::TrainMasterFunction()
	{
		//Timer start
		auto start = std::chrono::high_resolution_clock::now();

		m_MasterControlPoint = 0;

		


		const unsigned batchNum = m_Data->TrainingCount / m_HyperParameters.BatchCount;
		const unsigned remainingBatch = m_Data->TrainingCount % m_HyperParameters.BatchCount;

		std::mt19937 mt;

		unsigned epoch = 0;
		while (epoch < m_HyperParameters.Epochs)
		{
			unsigned randomIndexPos = 0;
			unsigned randomIndexCount = m_Data->TrainingCount;


			unsigned batch = 0;
			while (batch < batchNum)
			{
				unsigned batchIndex = 0;
				while (batchIndex < m_HyperParameters.BatchCount)
				{

					unsigned randomIndex = (mt() % randomIndexCount) + randomIndexPos;
					
					unsigned epochRandomIndex = m_Indices[randomIndex];
					m_Indices[randomIndex] = m_Indices[randomIndexPos];
					m_Indices[randomIndexPos] = epochRandomIndex;
					
					randomIndexPos++;
					randomIndexCount--;
					
					batchIndex++;
				}
				

				m_MasterControlPoint++;
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

				
			}




			unsigned tempBatchCountStorage = m_HyperParameters.BatchCount;

			unsigned thread = 0;
			while (thread < m_SlaveThreadCount)
			{
				m_SlaveThreads[thread] = std::thread(&NetworkPrototypeMT::TrainSlaveFunction, this, thread);
				thread++;
			}

			if (remainingBatch > 0)
			{
				WaitForSlaves();
				m_HyperParameters.BatchCount = remainingBatch;

				m_MasterControlPoint++;
			
			}

			thread = 0;
			while (thread < m_SlaveThreadCount)
			{
				m_SlaveThreads[thread].join();
				thread++;
			}

			m_HyperParameters.BatchCount = tempBatchCountStorage;


			epoch++;

		}

		//Timer stop
		auto stop = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float>  time = stop - start;
		m_LastTime[0] = time.count();
	}

	void NetworkPrototypeMT::CheckCostSlaveFunction(unsigned thread)
	{
		unsigned checkIndex = 0;
		while (checkIndex < m_Data->TestCount)
		{

			SetInput(&m_Data->TestInputs[checkIndex * m_InputBufferCount], thread);
			SetTarget(&m_Data->TestTargets[checkIndex * m_OutputBufferCount], thread);
			SpinLock(thread);

			FeedForward(thread);

			m_Functions.CostFunction.f(this, thread);
			SpinLock(thread);
			
			checkIndex++;
		}

		
	}

	float NetworkPrototypeMT::CheckCostMasterFunction()
	{
		auto start = std::chrono::high_resolution_clock::now();

		float cost = 0;
		

		unsigned thread = 0;
		while (thread < m_SlaveThreadCount)
		{
			m_CostBuffer[thread] = 0;
			m_SlaveThreads[thread] = std::thread(&NetworkPrototypeMT::CheckCostSlaveFunction, this, thread);
			thread++;
		}

		thread = 0;
		while (thread < m_SlaveThreadCount)
		{
			m_SlaveThreads[thread].join();
			cost += m_CostBuffer[thread];

			thread++;
		}


		auto stop = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float>  time = stop - start;
		m_LastTime[1] = time.count();


		return  cost / ((float)m_Data->TestCount);

	}

	void NetworkPrototypeMT::CheckSuccessRateSlaveFunction(unsigned thread)
	{
		

		

		unsigned checkIndex = 0;
		while (checkIndex < m_Data->TestCount)
		{
			
			SetInput(&m_Data->TestInputs[checkIndex * m_InputBufferCount], thread);
			SpinLock(thread);
			
			FeedForward(thread);
			
			
			

			int championItterator = -1;
			float champion = 0;

			unsigned start = m_PaddingData.Nodes[2 * thread + 2 * m_SlaveThreadCount * (m_LayerLayoutCount - 1)];
			unsigned stop = m_PaddingData.Nodes[2 * thread + 1 + 2 * m_SlaveThreadCount * (m_LayerLayoutCount - 1)];

			unsigned outputIndex = start;
			while (outputIndex < stop)
			{

				if (m_OutputBuffer[outputIndex] >= champion)
				{
					champion = m_OutputBuffer[outputIndex];
					championItterator = outputIndex;
					if (champion != 0)
					{
						pr("HEY");
					}
				}


				outputIndex++;
			}
			if (championItterator == -1)
			{
				
				assert(false);
				
			}
			m_GuessBuffer[thread] = championItterator;

			
			m_SlaveFlags[thread] = true;
			

			
			checkIndex++;
			SlaveControlStation(checkIndex);
		}

		
	}

	float NetworkPrototypeMT::CheckSuccessRateMasterFunction()
	{
		auto start = std::chrono::high_resolution_clock::now();


		m_MasterControlPoint = 0;
		unsigned thread = 0;
		while (thread < m_SlaveThreadCount)
		{
			m_SlaveThreads[thread] = std::thread(&NetworkPrototypeMT::CheckSuccessRateSlaveFunction, this, thread);
			thread++;
		}

		
		float score = 0.0f;

		unsigned checkIndex = 0;
		while (checkIndex < m_Data->TestCount)
		{
			int championItterator = -1;
			float champion = 0;

			unsigned paddingCorrection = 0;

			WaitForSlaves();

			unsigned lim = (m_SlaveThreadCount < m_OutputBufferCount) ? m_SlaveThreadCount : m_OutputBufferCount;

			unsigned threadItt = 0;
			while (threadItt < lim)
			{
				unsigned itt = m_GuessBuffer[threadItt];
				if (m_OutputBuffer[itt] >= champion)
				{
					paddingCorrection = threadItt * m_PaddingData.FloatPadding;

					champion = m_OutputBuffer[itt];
					championItterator = itt;
				}


				threadItt++;
			}

			if (m_Data->TestTargets[m_OutputBufferCount * checkIndex + championItterator - paddingCorrection] == 1)
			{
				score += 1.0f;
			}

			m_MasterControlPoint++;

			checkIndex++;
		}

		float rate = score / ((float)m_Data->TestCount);

		

		thread = 0;
		while (thread < m_SlaveThreadCount)
		{
			m_SlaveThreads[thread].join();
			thread++;
		}

		auto stop = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float>  time = stop - start;
		m_LastTime[2] = time.count();

		return rate;

	}

	void NetworkPrototypeMT::CheckSlaveFunction(float* input, unsigned thread)
	{


		SetInput(input , thread);
		SpinLock(thread);
		FeedForward(thread);

		int championItterator = -1;
		float champion = 0;

		unsigned start = m_PaddingData.Nodes[2 * thread + 2 * m_SlaveThreadCount * (m_LayerLayoutCount - 1)];
		unsigned stop = m_PaddingData.Nodes[ 1 + 2 * thread + 2 * m_SlaveThreadCount * (m_LayerLayoutCount - 1)];
	

		unsigned outputIndex = start;
		while (outputIndex < stop)
		{

			if (m_OutputBuffer[outputIndex] >= champion)
			{
				champion = m_OutputBuffer[outputIndex];
				championItterator = outputIndex;
			}


			outputIndex++;
		}
		if (championItterator == -1)
		{
			assert(false);
		}

		m_GuessBuffer[thread] = championItterator;


		m_SlaveFlags[thread] = true;

		
		
		
	}

	unsigned NetworkPrototypeMT::CheckMasterFunction(float* input)
	{
		auto start = std::chrono::high_resolution_clock::now();


		

		unsigned thread = 0;
		while (thread < m_SlaveThreadCount)
		{
			m_SlaveThreads[thread] = std::thread(&NetworkPrototypeMT::CheckSlaveFunction, this, input, thread);
			thread++;
		}



		int championItterator = -1;
		float champion = 0;

		WaitForSlaves();

		unsigned lim = (m_SlaveThreadCount < m_OutputBufferCount) ? m_SlaveThreadCount : m_OutputBufferCount;

		unsigned paddingCorrection = 0;

		unsigned threadItt = 0;
		while (threadItt < lim)
		{
			unsigned itt = m_GuessBuffer[threadItt];
			if (m_OutputBuffer[itt] >= champion)
			{
				paddingCorrection = threadItt * m_PaddingData.FloatPadding;
				champion = m_OutputBuffer[itt];
				championItterator = itt;
			}

			threadItt++;
		}

		thread = 0;
		while (thread < m_SlaveThreadCount)
		{
			m_SlaveThreads[thread].join();
			thread++;
		}

		auto stop = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float>  time = stop - start;
		m_LastTime[2] = time.count();

		return championItterator - paddingCorrection;
	}


}