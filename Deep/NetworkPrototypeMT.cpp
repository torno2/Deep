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

		m_LayerLayout = new LayerLayout[m_LayerLayoutCount];

		
		m_Functions.NeuronFunctions = new FunctionsLayoutMT::NeuronFunction[m_LayerLayoutCount - 1];
		m_Functions.NeuronFunctionsDerivatives = new FunctionsLayoutMT::NeuronFunction[m_LayerLayoutCount - 1];

		m_Functions.FeedForwardCallBackFunctions = new FunctionsLayoutMT::NetworkRelayFunctionMT[m_LayerLayoutCount-1];
		m_Functions.BackPropegateCallBackFunctionsZ = new FunctionsLayoutMT::NetworkRelayFunctionMT[m_LayerLayoutCount-2];
		m_Functions.BackPropegateCallBackFunctionsBW = new FunctionsLayoutMT::NetworkRelayFunctionMT[m_LayerLayoutCount - 1];

		m_Functions.CostFunction = functions.CostFunction;
		m_Functions.CostFunctionDerivative = functions.CostFunctionDerivative;

		m_Functions.TrainingFunction = functions.TrainingFunction;
		m_Functions.RegularizationFunction = functions.RegularizationFunction;



		unsigned layoutIndex = 0;
		while (layoutIndex < m_LayerLayoutCount)
		{
			m_LayerLayout[layoutIndex] = layerLayout[layoutIndex];

			if (layoutIndex < m_LayerLayoutCount-1)
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
		m_DeltaWeights = new float[weightTotal];
		m_WeightsBuffer = new float[weightTotal];


		m_TargetBuffer = new float[m_LayerLayout[m_LayerLayoutCount - 1].Nodes];

		m_CostBuffer = new float[m_SlaveThreadCount];
		m_GuessBuffer = new unsigned[m_SlaveThreadCount];

		m_SlaveThreads = new std::thread[m_SlaveThreadCount];
		m_Locks = new bool[m_SlaveThreadCount * 2];
		m_SlaveFlags = new bool[m_SlaveThreadCount];

		unsigned flagIndex = 0;
		while (flagIndex < m_SlaveThreadCount)
		{
			m_Locks[flagIndex * 2] = false;
			m_Locks[flagIndex * 2 + 1] = false;
			m_SlaveFlags[flagIndex] = false;
			flagIndex++;
		}


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

	NetworkPrototypeMT::~NetworkPrototypeMT()
	{

		delete[] m_LayerLayout;

		delete[] m_ZBuffer;
		delete[] m_DeltaZ;

		delete[] m_ABuffer;

		delete[] m_Biases;
		delete[] m_DeltaBiases;
		delete[] m_BiasesBuffer;

		delete[] m_Weights;
		delete[] m_DeltaWeights;
		delete[] m_WeightsBuffer;

		delete[] m_TargetBuffer;

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





	void NetworkPrototypeMT::SetTempToBiases(unsigned thread)
	{
		unsigned localOffset = 0;
		unsigned localStop = 0;
		ThreadWorkloadDivider(localOffset, localStop, m_BiasesCount, thread);

		
		

		unsigned index = localOffset;
		while (index < localStop)
		{
			
			

			m_BiasesBuffer[index] =  m_Biases[index];

			

			index++;
		}
	}

	void NetworkPrototypeMT::SetTempToWeights(unsigned thread)
	{
		unsigned localOffset = 0;
		unsigned localStop = 0;
		ThreadWorkloadDivider(localOffset, localStop, m_WeightsCount, thread);

		

		unsigned index = localOffset;
		while (index < localStop)
		{
			

			m_WeightsBuffer[index] = m_Weights[index];

			

			index++;
		}
	}

	void NetworkPrototypeMT::SetTempToBiasesAndWeights(unsigned thread)
	{
		SetTempToBiases(thread);
		SetTempToWeights(thread);
	}

	void NetworkPrototypeMT::SetBiasesToTemp(unsigned thread)
	{
		unsigned localOffset = 0;
		unsigned localStop = 0;
		ThreadWorkloadDivider(localOffset, localStop, m_BiasesCount, thread);

		

		unsigned index = localOffset;
		while (index < localStop)
		{
			
			m_Biases[index] = m_BiasesBuffer[index];
			
			index++;
		}
	}

	void NetworkPrototypeMT::SetWeightsToTemp(unsigned thread)
	{
		unsigned localOffset = 0;
		unsigned localStop = 0;
		ThreadWorkloadDivider(localOffset, localStop, m_WeightsCount, thread);

		

		unsigned index = localOffset;
		while (index < localStop)
		{
			
			m_Weights[index] = m_WeightsBuffer[index];
			
			index++;
		}
	}

	void NetworkPrototypeMT::SetData(DataSet* data)
	{
		delete[] m_Indices;

		m_Data = data;

		m_Indices = new unsigned[m_Data->TrainingCount];

		unsigned thread = 0;
		while (thread < m_SlaveThreadCount)
		{
			m_SlaveThreads[thread] = std::thread(&NetworkPrototypeMT::ResetIndices, this, thread);
			thread++;
		}		
		thread = 0;
		while (thread < m_SlaveThreadCount)
		{
			m_SlaveThreads[thread].join();
			thread++;
		}


	}

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
		unsigned start;
		unsigned stop;
		ThreadWorkloadDivider(start, stop, m_LayerLayout[0].Nodes, thread);

		

		unsigned index = start;
		while (index < stop)
		{
			
			m_ABuffer[index] = input[index];
			
			index++;
		}
	}

	void NetworkPrototypeMT::SetTarget(float* target, unsigned thread)
	{
		unsigned start;
		unsigned stop;
		ThreadWorkloadDivider(start, stop, m_LayerLayout[m_LayerLayoutCount - 1].Nodes, thread);

		

		unsigned index = start;
		while (index < stop)
		{
			
			m_TargetBuffer[index] = target[index];
			
			index++;
		}
	}

	void NetworkPrototypeMT::ThreadWorkloadDivider(unsigned& start, unsigned& stop, unsigned workCount, unsigned thread)
	{
		start = 0;
		stop = 0;
		if (workCount < m_SlaveThreadCount)
		{
			if (thread < workCount)
			{
				start = thread;
				stop = thread + 1;
			}
		}
		else
		{
			if (thread < (workCount % m_SlaveThreadCount) )
			{
				start = ((workCount / m_SlaveThreadCount) + 1) * thread;
				stop = start + ((workCount / m_SlaveThreadCount) + 1);
			}
			else
			{
				start = (workCount / m_SlaveThreadCount) * thread + (workCount % m_SlaveThreadCount);

				stop = start + (workCount / m_SlaveThreadCount);
			}
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
			m_PositionData.A = m_LayerLayout[0].Nodes;
			m_PositionData.Biases = m_LayerLayout[0].Biases;
			m_PositionData.Weights = m_LayerLayout[0].Weights;
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
				m_PositionData.Z += m_LayerLayout[m_PositionData.Layer].Nodes;

				m_PositionData.A += m_LayerLayout[m_PositionData.Layer].Nodes;

				m_PositionData.Biases += m_LayerLayout[m_PositionData.Layer].Biases;

				m_PositionData.Weights += m_LayerLayout[m_PositionData.Layer].Weights;

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

			m_PositionData.Z = m_ZBufferCount - m_LayerLayout[lastLayer].Nodes;
			m_PositionData.A = m_ABufferCount - m_LayerLayout[lastLayer].Nodes;

			m_PositionData.Biases = m_BiasesCount - m_LayerLayout[lastLayer].Biases;
			m_PositionData.Weights = m_WeightsCount - m_LayerLayout[lastLayer].Weights;
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

				m_PositionData.Z -= m_LayerLayout[m_PositionData.Layer].Nodes;

				m_PositionData.A -= m_LayerLayout[m_PositionData.Layer].Nodes;

				m_PositionData.Biases -= m_LayerLayout[m_PositionData.Layer].Biases;

				m_PositionData.Weights -= m_LayerLayout[m_PositionData.Layer].Weights;
			}
			SpinLock(thread);
			

			m_Functions.BackPropegateCallBackFunctionsZ[reveresLayoutIndex].f(this,thread);
			SpinLock(thread);

			reveresLayoutIndex++;
		}

		m_Functions.BackPropegateCallBackFunctionsBW[reveresLayoutIndex].f(this, thread);
		SpinLock(thread);
	}


	void NetworkPrototypeMT::TrainOnSet(unsigned batchCount, unsigned batch, unsigned thread)
	{
		SpinLock(thread);
		m_Functions.RegularizationFunction.f(this,thread);
		
		
		

		unsigned exampleIndex = 0;
		while (exampleIndex < batchCount)
		{


			
			unsigned indedx = m_Indices[exampleIndex + batch * m_HyperParameters.BatchCount];
			SetInput(&(m_Data->TrainingInputs[indedx * m_LayerLayout[0].Nodes]), thread);
			SetTarget(&(m_Data->TraningTargets[indedx * m_LayerLayout[m_LayerLayoutCount - 1].Nodes]), thread);
			
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

				m_MasterControlPoint++;
			}


			epochNum++;
		}
		unsigned thread = 0;
		while (thread < m_SlaveThreadCount)
		{
			m_SlaveThreads[thread] = std::thread(&NetworkPrototypeMT::TrainSlaveFunction, this, thread);
			thread++;
		}

		thread = 0;
		while (thread < m_SlaveThreadCount)
		{
			m_SlaveThreads[thread].join();
			thread++;
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

			SetInput(&m_Data->TestInputs[checkIndex * m_LayerLayout[0].Nodes], thread);
			SetTarget(&m_Data->TestTargets[checkIndex * m_LayerLayout[m_LayerLayoutCount - 1].Nodes], thread);
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
		const unsigned Ap = m_ABufferCount - m_LayerLayout[m_LayerLayoutCount - 1].Nodes;

		

		unsigned checkIndex = 0;
		while (checkIndex < m_Data->TestCount)
		{
			
			SetInput(&m_Data->TestInputs[checkIndex * m_LayerLayout[0].Nodes], thread);
			SpinLock(thread);
			FeedForward(thread);

			int championItterator = -1;
			float champion = 0;

			unsigned start, stop;
			ThreadWorkloadDivider(start, stop, m_LayerLayout[m_LayerLayoutCount - 1].Nodes, thread);

			unsigned outputIndex = start;
			while (outputIndex < stop)
			{

				if (m_ABuffer[Ap + outputIndex] >= champion)
				{
					champion = m_ABuffer[Ap + outputIndex];
					championItterator = outputIndex;
				}


				outputIndex++;
			}
			if (championItterator != -1)
			{
				
				m_GuessBuffer[thread] = championItterator;
				
			}
			
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

		const unsigned Ap = m_ABufferCount - m_LayerLayout[m_LayerLayoutCount - 1].Nodes;
		float score = 0.0f;

		unsigned checkIndex = 0;
		while (checkIndex < m_Data->TestCount)
		{
			int championItterator = -1;
			float champion = 0;
			
			WaitForSlaves();

			unsigned lim = (m_SlaveThreadCount < m_LayerLayout[m_LayerLayoutCount - 1].Nodes) ? m_SlaveThreadCount : m_LayerLayout[m_LayerLayoutCount - 1].Nodes;

			unsigned threadItt = 0;
			while (threadItt < lim)
			{
				unsigned itt = m_GuessBuffer[threadItt];
				if (m_ABuffer[Ap + itt] >= champion)
				{
					champion = m_ABuffer[Ap + itt];
					championItterator = itt;
				}


				threadItt++;
			}

			if (m_Data->TestTargets[m_LayerLayout[m_LayerLayoutCount - 1].Nodes * checkIndex + championItterator] == 1)
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


}