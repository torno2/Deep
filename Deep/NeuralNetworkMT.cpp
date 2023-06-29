#include "pch.h"
#include "NeuralNetworkMT.h"

namespace TNNT
{
	//Constructors And destructor

	NeuralNetworkMT::NeuralNetworkMT(unsigned* layerLayout, unsigned  layoutSize, LayerFucntionsLayout functions, unsigned slaveThreadCount, bool randomizeWeightsAndBiases)
		: m_LayerLayout(layerLayout), m_LayerLayoutCount(layoutSize), m_Functions(functions), m_SlaveThreadCount(slaveThreadCount)
	{
		// 1 layer no network makes; need at least 2
		assert(m_LayerLayoutCount >= 2);

		unsigned biasTotal = 0;
		unsigned weightTotal = 0;

		// A layer of zero nodes would mean you have two separate networks (or a network with 1 less layer, if the input/output layer is missing), 
		// and a layer with a negative numbers of nodes is something I don't want to think about.
		assert(m_LayerLayout[0] > 0);
		unsigned layoutIndex = 1;
		while (layoutIndex < m_LayerLayoutCount)
		{
			// A layer of zero nodes would mean you have two separate networks (or a network with 1 less layer, if the input/output layer is missing), 
			// and a layer with a negative numbers of nodes is something I don't want to think about.
			assert(m_LayerLayout[layoutIndex] > 0);

			biasTotal += m_LayerLayout[layoutIndex];
			weightTotal += m_LayerLayout[layoutIndex] * m_LayerLayout[layoutIndex - 1];


			layoutIndex++;
		}
		m_BiasesCount = biasTotal;
		m_WeightsCount = weightTotal;
		m_ABufferCount = biasTotal + m_LayerLayout[0];

		m_Biases = new float[m_BiasesCount];
		m_DeltaBiases = new float[m_BiasesCount];
		m_BiasesBuffer = new float[m_BiasesCount];
		m_BiasesOld = new float[m_BiasesCount];
		m_ZBuffer = new float[m_BiasesCount];

		m_ABuffer = new float[m_ABufferCount];

		m_Weights = new float[weightTotal];
		m_WeightsTranspose = new float[weightTotal];
		m_DeltaWeights = new float[weightTotal];
		m_WeightsBuffer = new float[weightTotal];
		m_WeightsOld = new float[weightTotal];

		m_TargetBuffer = new float[m_LayerLayout[m_LayerLayoutCount - 1]];


	
		m_Locks = new bool[m_SlaveThreadCount * 2];
		m_SlaveFlags = new bool[m_SlaveThreadCount];

		unsigned flagIndex = 0;
		while (flagIndex < m_SlaveThreadCount)
		{
			m_Locks[flagIndex*2] = false;
			m_Locks[flagIndex*2 +1] = false;
			m_SlaveFlags[flagIndex] = false;
			flagIndex++;
		}

		if (randomizeWeightsAndBiases)
		{
			//For randomly initializing the weights and biases
			std::default_random_engine generator;
			std::normal_distribution<float> distribution(0.0f, 1 / sqrt(m_LayerLayout[0]));

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

	NeuralNetworkMT::NeuralNetworkMT(unsigned* layerLayout, unsigned layoutSize, float* biases, float* weights, LayerFucntionsLayout functions, unsigned slaveThreadCount)
		: m_LayerLayout(layerLayout), m_LayerLayoutCount(layoutSize),m_Biases(biases),m_Weights(weights), m_Functions(functions), m_SlaveThreadCount(slaveThreadCount)
	{
		// 1 layer no network makes; need at least 2
		assert(m_LayerLayoutCount >= 2);

		unsigned biasTotal = 0;
		unsigned weightTotal = 0;

		// A layer of zero nodes would mean you have two separate networks (or a network with 1 less layer, if the input/output layer is missing), 
		// and a layer with a negative numbers of nodes is something I don't want to think about.
		assert(m_LayerLayout[0] > 0);
		unsigned layoutIndex = 1;
		while (layoutIndex < m_LayerLayoutCount)
		{
			// A layer of zero nodes would mean you have two separate networks (or a network with 1 less layer, if the input/output layer is missing), 
			// and a layer with a negative numbers of nodes is something I don't want to think about.
			assert(m_LayerLayout[layoutIndex] > 0);

			biasTotal += m_LayerLayout[layoutIndex];
			weightTotal += m_LayerLayout[layoutIndex] * m_LayerLayout[layoutIndex - 1];


			layoutIndex++;
		}
		m_BiasesCount = biasTotal;
		m_WeightsCount = weightTotal;
		m_ABufferCount = biasTotal + m_LayerLayout[0];

		m_DeltaBiases = new float[m_BiasesCount];
		m_BiasesBuffer = new float[m_BiasesCount];
		m_BiasesOld = new float[m_BiasesCount];
		m_ZBuffer = new float[m_BiasesCount];

		m_ABuffer = new float[m_ABufferCount];

		m_WeightsTranspose = new float[weightTotal];
		m_DeltaWeights = new float[weightTotal];
		m_WeightsBuffer = new float[weightTotal];
		m_WeightsOld = new float[weightTotal];

		m_TargetBuffer = new float[m_LayerLayout[m_LayerLayoutCount - 1]];



		m_Locks = new bool[m_SlaveThreadCount * 2];
		m_SlaveFlags = new bool[m_SlaveThreadCount];

		unsigned flagIndex = 0;
		while (flagIndex < m_SlaveThreadCount)
		{
			m_Locks[flagIndex] = false;
			m_Locks[flagIndex + 1] = false;
			m_SlaveFlags[flagIndex] = false;
			flagIndex++;
		}

		SetTempToBiases();

		SetTempToWeights();
		ResetWeightsTranspose();
	}

	NeuralNetworkMT::~NeuralNetworkMT()
	{
		delete[] m_LayerLayout;

		delete[] m_Biases;
		delete[] m_DeltaBiases;
		delete[] m_BiasesBuffer;
		delete[] m_BiasesOld;
		delete[] m_ZBuffer;

		delete[] m_ABuffer;

		delete[] m_Weights;
		delete[] m_WeightsTranspose;
		delete[] m_DeltaWeights;
		delete[] m_WeightsBuffer;
		delete[] m_WeightsOld;

		delete[] m_TargetBuffer;

		delete[] m_Functions.NeuronFunction;
		delete[] m_Functions.NeuronFunctionDerivative;

		//Multi thread nonsense
		delete[] m_Locks;
		delete[] m_SlaveFlags;


	}



	float NeuralNetworkMT::CheckCost(DataSet& data)
	{
		return CheckCostMTMasterFunction(data.TestInputs, data.TestTargets, data.TestCount);
	}

	float NeuralNetworkMT::CheckSuccessRate(DataSet& data)
	{
		return CheckSuccessRateMTMasterFunction(data.TestInputs, data.TestTargets, data.TestCount);
	}



	//The function you should be calling to train the network.

	void NeuralNetworkMT::Train(DataSet& data, HyperParameters& params)
	{
	
		TrainMTMasterThreadFunction(
			data.TrainingInputs, data.TraningTargets, data.TrainingCount,
			params.Epochs, params.BatchCount, params.LearningRate, params.LearningRate
		);
	}

	void NeuralNetworkMT::TrainWCondition(DataSet& data, HyperParameters& params, ConditionFunctionPointer condFunc)
	{
		TrainWithCheckMTMasterFunction
		(
			data.TrainingInputs, data.TraningTargets, data.TrainingCount,
			params.Epochs, params.BatchCount, params.LearningRate,params.LearningRate,
			data.ValidationInputs,data.ValidationTargets,data.ValidationCount,
			condFunc
		);
	}

	void NeuralNetworkMT::SaveToFile(const char* filepath) const
	{
		std::ofstream save(filepath, std::ios::binary);


		char layoutSizeBuffer[sizeof(float)];
		memcpy(&layoutSizeBuffer, &m_LayerLayoutCount, sizeof(float));

		save.write(layoutSizeBuffer, sizeof(float));


		unsigned index = 0;
		while (index < m_LayerLayoutCount)
		{
			char layoutBuffer[sizeof(float)];
			memcpy(&layoutBuffer, &m_LayerLayout[index], sizeof(float));

			save.write(layoutBuffer, sizeof(float));

			index++;
		}

		index = 0;
		while (index < m_BiasesCount)
		{
			char biasesBuffer[sizeof(float)];
			memcpy(&biasesBuffer, &m_Biases[index], sizeof(float));
			save.write(biasesBuffer, sizeof(float));
			index++;
		}

		index = 0;
		while (index < m_WeightsCount)
		{

			char weightsBuffer[sizeof(float)];
			memcpy(&weightsBuffer, &m_Weights[index], sizeof(float));
			save.write(weightsBuffer, sizeof(float));
			index++;
		}

		save.close();
	}

	void NeuralNetworkMT::LoadFromFile(const char* filepath)
	{
		std::ifstream save(filepath, std::ios::binary);

		unsigned layoutSize;
		char layoutBuffer[sizeof(unsigned)];
		save.read(layoutBuffer, sizeof(unsigned));
		memcpy(&layoutSize, &layoutBuffer, sizeof(unsigned));


		//This function is only intended as a way to replace the weights and biases in the network, 
		//meaning that the bytes representing the layout are only used to verify that the layout of the saved network is the same as this.
		assert(layoutSize == m_LayerLayoutCount);

		unsigned layoutIndex = 0;
		while (layoutIndex < layoutSize)
		{
			unsigned layerCount;
			save.read(layoutBuffer, sizeof(unsigned));
			memcpy(&layerCount, &layoutBuffer, sizeof(unsigned));
			assert(layerCount == m_LayerLayout[layoutIndex]);

			layoutIndex++;
		}

		char biasesBuffer[sizeof(float)];
		unsigned biasesIndex = 0;
		while (biasesIndex < m_BiasesCount)
		{
			save.read(biasesBuffer, sizeof(float));
			memcpy(&(m_Biases[biasesIndex]), &biasesBuffer, sizeof(float));
			biasesIndex++;
		}

		char weightsBuffer[sizeof(float)];
		unsigned weightsIndex = 0;
		while (weightsIndex < m_WeightsCount)
		{
			save.read(weightsBuffer, sizeof(float));
			memcpy(&(m_Weights[weightsIndex]), &weightsBuffer, sizeof(float));
			weightsIndex++;
		}

		SetTempToBiases();
		SetTempToWeights();
		ResetWeightsTranspose();
		save.close();

	}

	NeuralNetworkMT* NeuralNetworkMT::CreateFromFile(const char* filepath, LayerFucntionsLayout functions, unsigned slaveThreadCount)
	{
		unsigned layoutSize;
		unsigned* LayerLayout;
	

		float* Biases;
		float* Weights;


		std::ifstream save(filepath, std::ios::binary);



		char layoutBuffer[sizeof(unsigned)];
		save.read(layoutBuffer, sizeof(unsigned));
		memcpy(&layoutSize, &layoutBuffer, sizeof(unsigned));

		LayerLayout = new unsigned[layoutSize];
		unsigned layoutIndex = 0;
		while (layoutIndex < layoutSize)
		{
			save.read(layoutBuffer, sizeof(unsigned));
			memcpy(&(LayerLayout[layoutIndex]), &layoutBuffer, sizeof(unsigned));

			layoutIndex++;
		}

		// 1 layer no network makes; need at least 2
		assert(layoutSize >= 2);

		unsigned biasTotal = 0;
		unsigned weightTotal = 0;

		// A layer of zero nodes would mean you have two separate networks (or a network with 1 less layer, if the input/output layer is missing), 
		// and a layer with a negative numbers of nodes is something I don't want to think about.
		assert(LayerLayout[0] > 0);
		layoutIndex = 1;
		while (layoutIndex < layoutSize)
		{
			// A layer of zero nodes would mean you have two separate networks (or a network with 1 less layer, if the input/output layer is missing), 
			// and a layer with a negative numbers of nodes is something I don't want to think about.
			assert(LayerLayout[layoutIndex] > 0);

			biasTotal += LayerLayout[layoutIndex];
			weightTotal += LayerLayout[layoutIndex] * LayerLayout[layoutIndex - 1];


			layoutIndex++;
		}
		Biases = new float[biasTotal];
		Weights = new float[weightTotal];



		char biasesBuffer[sizeof(float)];
		unsigned biasesIndex = 0;
		while (biasesIndex < biasTotal)
		{
			save.read(biasesBuffer, sizeof(float));
			memcpy(&(Biases[biasesIndex]), &biasesBuffer, sizeof(float));
			biasesIndex++;
		}

		char weightsBuffer[sizeof(float)];
		unsigned weightsIndex = 0;
		while (weightsIndex < weightTotal)
		{
			save.read(weightsBuffer, sizeof(float));
			memcpy(&(Weights[weightsIndex]), &weightsBuffer, sizeof(float));
			weightsIndex++;
		}
		save.close();

		NeuralNetworkMT* result = new NeuralNetworkMT(LayerLayout, layoutSize, Biases, Weights,functions,slaveThreadCount);
		return result;
	}


	//Debugging

	float NeuralNetworkMT::CheckCostD(const float* checkInputs, const float* checkTargets, unsigned num)
	{
		const auto costFunc = m_Functions.CostFunction;

		const unsigned Ap = m_ABufferCount - m_LayerLayout[m_LayerLayoutCount - 1];

		float cost = 0;

		unsigned checkIndex = 0;
		while (checkIndex < num)
		{
			SetInput(checkInputs, checkIndex);
			FeedForward();

			unsigned outputIndex = 0;
			while (outputIndex < m_LayerLayout[m_LayerLayoutCount - 1])
			{

				cost += costFunc(m_ABuffer[Ap + outputIndex], checkTargets[m_LayerLayout[m_LayerLayoutCount - 1] * checkIndex + outputIndex]);

				outputIndex++;
			}

			checkIndex++;
		}
		cost = cost / ((float)num);
		return cost;

	}

	float NeuralNetworkMT::CheckSuccessRateD(const float* checkInputs, const float* checkTargets, unsigned num)
	{
		const unsigned Ap = m_ABufferCount - m_LayerLayout[m_LayerLayoutCount - 1];

		float score = 0.0f;

		unsigned checkIndex = 0;
		while (checkIndex < num)
		{

			SetInput(checkInputs, checkIndex);
			FeedForward();

			int championItterator = -1;
			float champion = 0;
			unsigned outputIndex = 0;
			while (outputIndex < m_LayerLayout[m_LayerLayoutCount - 1])
			{

				if (m_ABuffer[Ap + outputIndex] >= champion)
				{
					champion = m_ABuffer[Ap + outputIndex];
					championItterator = outputIndex;
				}

				outputIndex++;
			}

			if (checkTargets[m_LayerLayout[m_LayerLayoutCount - 1] * checkIndex + championItterator] == 1)
			{
				score += 1.0f;
			}
			checkIndex++;
		}


		float rate = score / ((float)num);

		return rate;

	}


	void NeuralNetworkMT::SetInput(const float* input, unsigned startElement)
	{
		unsigned index = 0;
		while (index < m_LayerLayout[0])
		{
			m_ABuffer[index] = input[m_LayerLayout[0] * startElement + index];
			index++;
		}

	}


	void NeuralNetworkMT::FeedForward()
	{


		unsigned layerBiasesStart = 0;
		unsigned layerWeightsStart = 0;
		unsigned intakePos = 0;

		unsigned layoutIndex = 1;
		while (layoutIndex < m_LayerLayoutCount)
		{

			auto neuronFunc = m_Functions.NeuronFunction[layoutIndex].Function;

			unsigned layerIndex = 0;
			while (layerIndex < m_LayerLayout[layoutIndex])
			{
				float weightedSum = 0;

				unsigned prevIndex = 0;
				while (prevIndex < m_LayerLayout[layoutIndex - 1])
				{
					weightedSum += m_ABuffer[intakePos + prevIndex] * m_Weights[layerWeightsStart + m_LayerLayout[layoutIndex - 1] * layerIndex + prevIndex];
					prevIndex++;
				}

				/* We use "layerBiasesStart" here because there are just as many biases as there are totals of "z"s, and "a"s minus the amount of inputs*/
				m_ZBuffer[layerBiasesStart + layerIndex] = weightedSum + m_Biases[layerBiasesStart + layerIndex];
				m_ABuffer[m_LayerLayout[0] + layerBiasesStart + layerIndex] = neuronFunc(m_ZBuffer[layerBiasesStart + layerIndex]);
				layerIndex++;
			}

			layerBiasesStart += m_LayerLayout[layoutIndex];
			layerWeightsStart += m_LayerLayout[layoutIndex] * m_LayerLayout[layoutIndex - 1];
			intakePos += m_LayerLayout[layoutIndex - 1];


			layoutIndex++;
		}
	}



	//Private helper-functions for the constructor.

	void NeuralNetworkMT::SetTempToBiases()
	{
		unsigned index = 0;
		while (index < m_BiasesCount)
		{
			m_BiasesBuffer[index] = m_Biases[index];
			index++;
		}
	} 


	void NeuralNetworkMT::SetTempToWeights()
	{
		unsigned index = 0;
		while (index < m_WeightsCount)
		{
			m_WeightsBuffer[index] = m_Weights[index];
			index++;
		}
	}

	void NeuralNetworkMT::ResetWeightsTranspose()
	{
		unsigned currentOffset = 0;
		unsigned currentSize = 0;
		unsigned layoutIndex = 1;
		while (layoutIndex < m_LayerLayoutCount)
		{
			currentOffset += currentSize;
			currentSize = m_LayerLayout[layoutIndex] * m_LayerLayout[layoutIndex - 1];
			unsigned index = 0;
			while (index < currentSize)
			{
				m_WeightsTranspose[currentOffset + (index / m_LayerLayout[layoutIndex - 1]) + (index % m_LayerLayout[layoutIndex - 1]) * m_LayerLayout[layoutIndex]] = m_Weights[currentOffset + index];
				index++;
			}

			layoutIndex++;
		}
	}


	// Multithread private functions

	void NeuralNetworkMT::SetBiasesToTempMT(const unsigned thread)
	{
		//Sorts out some index stuff for thread workload distribution
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

	void NeuralNetworkMT::SetWeightsToTempMT(const unsigned thread)

	{
		unsigned localOffset = 0;
		unsigned localStop = 0;

		unsigned currentOffset = 0;
		unsigned currentSize = 0;
		unsigned layoutIndex = 1;
		while (layoutIndex < m_LayerLayoutCount)
		{

			currentOffset += currentSize;
			currentSize = m_LayerLayout[layoutIndex] * m_LayerLayout[layoutIndex - 1];
			ThreadWorkloadDivider(localOffset, localStop, currentSize, thread);


			unsigned index = localOffset;
			while (index < localStop)
			{
				m_Weights[currentOffset + index] = m_WeightsBuffer[currentOffset + index];
				m_WeightsTranspose[currentOffset + (index / m_LayerLayout[layoutIndex - 1]) + (index % m_LayerLayout[layoutIndex - 1]) * m_LayerLayout[layoutIndex]] = m_WeightsBuffer[currentOffset + index];
				index++;
			}

			layoutIndex++;
		}

	}

	void NeuralNetworkMT::SetOldToBiasesMT(const unsigned thread)
	{
		//Sorts out some index stuff for thread workload distribution
		unsigned localOffset = 0;
		unsigned localStop = 0;
		ThreadWorkloadDivider(localOffset, localStop, m_BiasesCount, thread);


		unsigned index = localOffset;
		while (index < localStop)
		{
			m_BiasesOld[index] = m_Biases[index];
			index++;
		}
	}

	void NeuralNetworkMT::SetOldToWeightsMT(const unsigned thread)
	{
		//Sorts out some index stuff for thread workload distribution
		unsigned localOffset = 0;
		unsigned localStop = 0;
		ThreadWorkloadDivider(localOffset, localStop, m_WeightsCount, thread);


		unsigned index = localOffset;
		while (index < localStop)
		{
			m_WeightsOld[index] = m_Weights[index];
			index++;
		}
	}

	void NeuralNetworkMT::SetBiasesToOldMT(const unsigned thread)
	{
		//Sorts out some index stuff for thread workload distribution
		unsigned localOffset = 0;
		unsigned localStop = 0;
		ThreadWorkloadDivider(localOffset, localStop, m_BiasesCount, thread);


		unsigned index = localOffset;
		while (index < localStop)
		{
			m_Biases[index] = m_BiasesOld[index];
			index++;
		}
	}

	void NeuralNetworkMT::SetWeightsToOldMT(const unsigned thread)
	{
		unsigned localOffset = 0;
		unsigned localStop = 0;

		unsigned currentOffset = 0;
		unsigned currentSize = 0;
		unsigned layoutIndex = 1;
		while (layoutIndex < m_LayerLayoutCount)
		{

			currentOffset += currentSize;
			currentSize = m_LayerLayout[layoutIndex] * m_LayerLayout[layoutIndex - 1];
			ThreadWorkloadDivider(localOffset, localStop, currentSize, thread);


			unsigned index = localOffset;
			while (index < localStop)
			{
				m_Weights[currentOffset + index] = m_WeightsOld[currentOffset + index];
				m_WeightsTranspose[currentOffset + (index / m_LayerLayout[layoutIndex - 1]) + (index % m_LayerLayout[layoutIndex - 1]) * m_LayerLayout[layoutIndex]] = m_WeightsOld[currentOffset + index];
				index++;
			}

			layoutIndex++;
		}
	}



	void NeuralNetworkMT::SetInputMT(const float* input, const unsigned thread)
	{
		//Sorts out some index stuff for thread workload distribution
		unsigned localOffset = 0;
		unsigned localStop = 0;
		ThreadWorkloadDivider(localOffset, localStop, m_LayerLayout[0], thread);

		unsigned index = localOffset;
		while (index < localStop)
		{
			m_ABuffer[index] = input[index];
			index++;
		}
	}

	void NeuralNetworkMT::SetTargetMT(const float* target, const unsigned thread)
	{

		//Sorts out some index stuff for thread workload distribution
		unsigned localOffset = 0;
		unsigned localStop = 0;
		ThreadWorkloadDivider(localOffset, localStop, m_LayerLayout[m_LayerLayoutCount - 1], thread);

		unsigned index = localOffset;
		while (index < localStop)
		{
			m_TargetBuffer[index] = target[index];
			index++;
		}
	}


	void NeuralNetworkMT::FeedForwardMT(const unsigned thread)
	{

		unsigned biasesStart = 0;
		unsigned weightsStart = 0;
		unsigned intakePos = 0;

		unsigned layoutIndex = 1;
		while ( layoutIndex < m_LayerLayoutCount)
		{
			auto neuronFunc = m_Functions.NeuronFunction[layoutIndex-1].Function;

			//Sorts out some index stuff for thread workload distribution
			unsigned localOffset = 0;
			unsigned localStop = 0;
			ThreadWorkloadDivider(localOffset, localStop, m_LayerLayout[layoutIndex], thread);

			unsigned layerIndex = localOffset;
			while ( layerIndex < localStop)
			{
			
				float weightedSum = 0;

				unsigned prevIndex = 0;
				while ( prevIndex < m_LayerLayout[layoutIndex - 1])
				{
					weightedSum += m_ABuffer[intakePos + prevIndex] * m_Weights[weightsStart + m_LayerLayout[layoutIndex - 1] * layerIndex + prevIndex];

					prevIndex++;
				}
				/* We use "layerBiasesStart" here because there are just as many biases as there are totals of "z"s, and "a"s minus the amount of inputs*/
				m_ZBuffer[biasesStart + layerIndex] = weightedSum + m_Biases[biasesStart + layerIndex];
				m_ABuffer[m_LayerLayout[0] + biasesStart + layerIndex] = neuronFunc(m_ZBuffer[biasesStart + layerIndex]);
				layerIndex++;
			}

			biasesStart += m_LayerLayout[layoutIndex];
			weightsStart += m_LayerLayout[layoutIndex] * m_LayerLayout[layoutIndex - 1];
			intakePos += m_LayerLayout[layoutIndex - 1];


			layoutIndex++;

			SpinLockMT(thread);
		}

	}

	void NeuralNetworkMT::BackpropegateMT(const unsigned thread)
	{

		//Always assumes that the network has fed forward first.

		unsigned layoutIndex = m_LayerLayoutCount - 1;

		unsigned latterBiases = m_BiasesCount;
		unsigned latterWeights = m_WeightsCount;

		unsigned biasesStart = latterBiases - m_LayerLayout[layoutIndex];
		unsigned weightsStart = latterWeights - m_LayerLayout[layoutIndex] * m_LayerLayout[layoutIndex - 1];


		//Used to fetch the "a" belonging to the layer that comes before the one we're focusing on.
		unsigned apos = biasesStart + m_LayerLayout[0] - m_LayerLayout[layoutIndex - 1];

		//Sorts out some index stuff for thread workload distribution
		unsigned localOffset = 0;
		unsigned localStop = 0;
		ThreadWorkloadDivider(localOffset, localStop, m_LayerLayout[layoutIndex], thread);
		const auto costDerivative = m_Functions.CostFunctionDerivative;

		unsigned layerIndex = localOffset;
		while (layerIndex < localStop)
		{

			//Right here we need the a from this layer, and we therfore have to add the length of the previous layer to get to the start of this one.
			float z = m_ZBuffer[biasesStart + layerIndex];
			float a = m_ABuffer[apos + m_LayerLayout[layoutIndex - 1] + layerIndex];
			float y = m_TargetBuffer[layerIndex];

			float db = costDerivative(z, a, y);


			m_DeltaBiases[biasesStart + layerIndex] = db;


			unsigned formerLayerIndex = 0;
			while (formerLayerIndex < m_LayerLayout[layoutIndex - 1])
			{
				float a2 = m_ABuffer[apos + formerLayerIndex];
				m_DeltaWeights[weightsStart + m_LayerLayout[layoutIndex - 1] * layerIndex + formerLayerIndex] = a2 * db;

				formerLayerIndex++;
			}

			layerIndex++;
		}

		SpinLockMT(thread);

		while (layoutIndex > 1)
		{
			layoutIndex--;

			const auto neuronFuncDeriv = m_Functions.NeuronFunctionDerivative[layoutIndex - 1].Function;

			latterBiases -= m_LayerLayout[layoutIndex + 1];
			latterWeights -= m_LayerLayout[layoutIndex + 1] * m_LayerLayout[layoutIndex];

			biasesStart -= m_LayerLayout[layoutIndex];
			weightsStart -= m_LayerLayout[layoutIndex] * m_LayerLayout[layoutIndex - 1];

			apos -= m_LayerLayout[layoutIndex - 1];

		
			ThreadWorkloadDivider(localOffset, localStop, m_LayerLayout[layoutIndex], thread);

			unsigned layerIndex = localOffset;
			while (layerIndex < localStop)
			{
				float errorSum = 0;
				unsigned latterLayerIndex = 0;
				while (latterLayerIndex < m_LayerLayout[layoutIndex + 1])
				{
					errorSum += m_WeightsTranspose[latterWeights + m_LayerLayout[layoutIndex + 1] * layerIndex + latterLayerIndex] * m_DeltaBiases[latterBiases + latterLayerIndex];
					latterLayerIndex++;
				}

				float z = m_ZBuffer[biasesStart + layerIndex];
				float db = errorSum * neuronFuncDeriv(z);


				m_DeltaBiases[biasesStart + layerIndex] = db;

				unsigned formerLayerIndex = 0;
				while (formerLayerIndex < m_LayerLayout[layoutIndex - 1])
				{
					float a = m_ABuffer[apos + formerLayerIndex];
					m_DeltaWeights[weightsStart + m_LayerLayout[layoutIndex - 1] * layerIndex + formerLayerIndex] = a * db;

					formerLayerIndex++;
				}

				layerIndex++;
			}
			SpinLockMT(thread);
		}
	}


	void NeuralNetworkMT::RegWeightsL2MT(unsigned trainingSetSize, float learingRate, float L2regConst, const unsigned thread)
	{
		//Sorts out some index stuff for thread workload distribution
		unsigned localOffset = 0;
		unsigned localStop = 0;
		ThreadWorkloadDivider(localOffset, localStop,m_WeightsCount, thread);

		unsigned index = localOffset;
		while (index < localStop)
		{
			m_WeightsBuffer[index] *= (1 - (learingRate * L2regConst / (float)trainingSetSize));
			index++;
		}
	}

	void NeuralNetworkMT::GradientDecentMT(unsigned batchSize, float learingRate, const unsigned thread)
	{
		//Sorts out some index stuff for thread workload distribution
		unsigned localOffset1 = 0;
		unsigned localStop1 = 0;
		unsigned localOffset2 = 0;
		unsigned localStop2 = 0;

		ThreadWorkloadDivider(localOffset1, localStop1, m_WeightsCount, thread);


		ThreadWorkloadDivider(localOffset2, localStop2, m_BiasesCount, thread);


		unsigned i = localOffset1;
		while (i < localStop1)
		{
			m_WeightsBuffer[i] -= (learingRate / (float)batchSize) * m_DeltaWeights[i];
			i++;
		}

		i = localOffset2;
		while (i < localStop2)
		{
			m_BiasesBuffer[i] -=  (learingRate / (float)batchSize) * m_DeltaBiases[i];
			i++;
		}
	}


	void NeuralNetworkMT::SamplePrepp(const float* trainingInputs, const float* trainingTargets, float* inputBuffer, float* targetBuffer, unsigned* indicies, unsigned num, unsigned step, const unsigned thread)
	{


		unsigned localOffset1 = 0;
		unsigned localStop1 = 0;
		unsigned localOffset2 = 0;
		unsigned localStop2 = 0;

		ThreadWorkloadDivider(localOffset1, localStop1,  m_LayerLayout[0]*num, thread);


		ThreadWorkloadDivider(localOffset2, localStop2, m_LayerLayout[m_LayerLayoutCount - 1]*num, thread);


		unsigned inputIndex = localOffset1;
		while (inputIndex < localStop1)
		{
			unsigned indiciesPos = inputIndex / m_LayerLayout[0] + step * num;
			inputBuffer[inputIndex] = trainingInputs[indicies[indiciesPos] * m_LayerLayout[0] + inputIndex % m_LayerLayout[0]];
			inputIndex++;
		
		}

		unsigned outputIndex = localOffset2;
		while (outputIndex < localStop2)
		{
			unsigned indiciesPos =outputIndex / m_LayerLayout[m_LayerLayoutCount - 1]+ step * num;
			targetBuffer[outputIndex] = trainingTargets[indicies[indiciesPos] * m_LayerLayout[m_LayerLayoutCount - 1] + outputIndex % m_LayerLayout[m_LayerLayoutCount - 1]];
			outputIndex++;
		}
	


	}

	void NeuralNetworkMT::TrainOnSetMT(const float* inputBuffer, float* targetBuffer,unsigned num, float learingRate, float regConst, unsigned trainingSetSize, const unsigned thread )
	{
		SpinLockMT(thread);
		RegWeightsL2MT(trainingSetSize, learingRate, regConst, thread);
		unsigned exampleIndex = 0;
		while (exampleIndex < num)
		{

			SetInputMT(&(inputBuffer[exampleIndex * m_LayerLayout[0]]), thread);
			SetTargetMT(&(targetBuffer[exampleIndex * m_LayerLayout[m_LayerLayoutCount - 1]]), thread);


			SpinLockMT(thread);

			// Start: These two have a lot of spinlocks in them; one for each jump between layers
			FeedForwardMT(thread);
			BackpropegateMT(thread);
			// End.

			GradientDecentMT(num, learingRate, thread);


			exampleIndex++;
		}

		SpinLockMT(thread);
		SetBiasesToTempMT(thread);
		SetWeightsToTempMT(thread);
	
	}


	void NeuralNetworkMT::ThreadWorkloadDivider(unsigned& start, unsigned& stop, unsigned workCount, unsigned thread)
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

	void NeuralNetworkMT::SpinLockMT(const unsigned thread)
	{

		unsigned i = 0;
		while (i < m_SlaveThreadCount)
		{
			if (i == thread)
			{
				m_Locks[i+ m_SlaveThreadCount] = false;
				m_Locks[i] = true;
			}
			if (!m_Locks[i])
			{
				continue;
			}
			i++;
		}
		while (i < 2*m_SlaveThreadCount )
		{
			if (i == (thread + m_SlaveThreadCount))
			{
				m_Locks[i- m_SlaveThreadCount] = false;
				m_Locks[i] = true;
			}
			if (!m_Locks[i])
			{
				continue;
			}
			i++;
		}

	}

	void NeuralNetworkMT::SlaveControlStationMT(unsigned standpoint) 
	{


		while (standpoint > m_MasterControlPoint) 
		{
			;
		}

	}

	void NeuralNetworkMT::WaitForSlavesMT()
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


	void NeuralNetworkMT::TrainMTSlaveThreadFunction(const float* traningInputs, const float* traningTargets, float* inputBuffer, float* targetBuffer, unsigned* indicies, unsigned num, unsigned epochs, unsigned batchSize,  float learningRate, float regConst, const unsigned thread)
	{

		const unsigned batchCount = num / batchSize;
		const unsigned remainingBatch = num % batchSize;

		unsigned standpoint = 0;
		unsigned epochCount = 0;
		while (epochCount < epochs)
		{

			unsigned batchNum = 0;
			while (batchNum < batchCount)
			{
				standpoint++;
				SlaveControlStationMT(standpoint);

				SamplePrepp(traningInputs, traningTargets, inputBuffer, targetBuffer, indicies, batchSize, batchNum, thread);
				TrainOnSetMT(inputBuffer, targetBuffer,batchSize, learningRate, regConst, num, thread);

				batchNum++;
			}


			if (remainingBatch > 0)
			{
				standpoint++;
				SlaveControlStationMT(standpoint);

				SamplePrepp(traningInputs, traningTargets, inputBuffer, targetBuffer, indicies, remainingBatch, batchNum, thread);
				TrainOnSetMT(inputBuffer, targetBuffer, remainingBatch, learningRate, regConst, num,thread);
			}

		
			epochCount++;
		}
	}

	void NeuralNetworkMT::TrainMTMasterThreadFunction(const float* traningInputs, const float* traningTargets, unsigned num, unsigned epochs, unsigned batchSize,  float learningRate, float regConst)
	{

		float* sharedInputBuffer	= new float[m_LayerLayout[0] * batchSize];
		float* sharedTargetBuffer	= new float[m_LayerLayout[m_LayerLayoutCount - 1] * batchSize];
		unsigned* indices			= new unsigned[num];

		m_MasterControlPoint = 0;



	

		const unsigned batchCount = num / batchSize;
		const unsigned remainingBatch = num % batchSize;

		unsigned index = 0;
		while (index < num)
		{
			indices[index] = index;
			index++;
		}

		std::mt19937 mt;

		unsigned epochCount = 0;
		while (epochCount < epochs)
		{
			unsigned randomIndexCount = num;
			

			unsigned batchNum = 0;
			while (batchNum < batchCount)
			{
				unsigned batchIndex = 0;
				while (batchIndex < batchSize)
				{

					unsigned randomIndex = mt() % randomIndexCount;

					unsigned epochRandomIndex = indices[randomIndex];
					indices[randomIndex] = indices[randomIndexCount - 1];
					indices[randomIndexCount - 1] = epochRandomIndex;

					
					randomIndexCount--;	

					batchIndex++;
				}
				//Used to prevent the slave threads from getting too far ahead
				
				m_MasterControlPoint++;
				
				batchNum++;
			}


			if (remainingBatch > 0)
			{
				unsigned batchIndex = 0;
				while (batchIndex < remainingBatch)
				{

					unsigned randomIndex = mt() % randomIndexCount;

					unsigned epochRandomIndex = indices[randomIndex];
					indices[randomIndex] = indices[randomIndexCount - 1];
					indices[randomIndexCount - 1] = epochRandomIndex;

					randomIndexCount--;
					batchIndex++;

				}
				//Used to prevent the slave threads from getting too far ahead
				m_MasterControlPoint++;
			}



			epochCount++;
		}

		std::thread* slaves = new std::thread[m_SlaveThreadCount];
		unsigned slaveThread = 0;
		while (slaveThread < m_SlaveThreadCount)
		{
			slaves[slaveThread] = std::thread(&NeuralNetworkMT::TrainMTSlaveThreadFunction, this, traningInputs, traningTargets, sharedInputBuffer, sharedTargetBuffer, indices, num, epochs, batchSize, learningRate, regConst, slaveThread);
			slaveThread++;
		}

		slaveThread = 0;
		while (slaveThread < m_SlaveThreadCount)
		{
			slaves[slaveThread].join();
			slaveThread++;
		}
		delete[] slaves;

		delete[] sharedInputBuffer;
		delete[] sharedTargetBuffer;
		delete[] indices;
	}


	void NeuralNetworkMT::TrainWithCheckMTSlaveFunction(const float* traningInputs, const float* traningTargets, float* inputBuffer, float* targetBuffer, unsigned* indicies, unsigned trainingNum, unsigned* epochs, unsigned* batchSize, float* learningRate, float* regConst, bool* updateOrRevert, const float* checkInputs, const float* checkTargets, unsigned checkNum, float* resultBuffer, const unsigned thread)
	{
		const unsigned batchCount = trainingNum / *batchSize;
		const unsigned remainingBatch = trainingNum % *batchSize;
		const unsigned Ap = m_ABufferCount - m_LayerLayout[m_LayerLayoutCount - 1];

		auto costFunc = m_Functions.CostFunction;

		unsigned standpoint = 0;
		unsigned epochCount = 0;
		while (epochCount < *epochs)
		{

			unsigned batchNum = 0;
			while (batchNum < batchCount)
			{
				standpoint++;
				SlaveControlStationMT(standpoint);

				SamplePrepp(traningInputs, traningTargets, inputBuffer, targetBuffer, indicies, *batchSize, batchNum, thread);
				TrainOnSetMT(inputBuffer, targetBuffer, *batchSize, *learningRate, *regConst, trainingNum, thread);

				batchNum++;
			}


			if (remainingBatch > 0)
			{
				standpoint++;
				SlaveControlStationMT(standpoint);

				SamplePrepp(traningInputs, traningTargets, inputBuffer, targetBuffer, indicies, remainingBatch, batchNum, thread);
				TrainOnSetMT(inputBuffer, targetBuffer, remainingBatch, *learningRate, *regConst, trainingNum, thread);
			}





			resultBuffer[thread] = 0;


			unsigned checkIndex = 0;
			while (checkIndex < checkNum)
			{
				SpinLockMT(thread);
				SetInputMT(&checkInputs[checkIndex * m_LayerLayout[0]], thread);
				SpinLockMT(thread);
				FeedForwardMT(thread);

				unsigned localOffset = 0;
				unsigned localStop = 0;

				ThreadWorkloadDivider(localOffset, localStop, m_LayerLayout[m_LayerLayoutCount - 1], thread);

				unsigned outputIndex = localOffset;
				while (outputIndex < localStop)
				{

					resultBuffer[thread] += costFunc(m_ABuffer[Ap + outputIndex], checkTargets[m_LayerLayout[m_LayerLayoutCount - 1] * checkIndex + outputIndex]);
					outputIndex++;
				}
				checkIndex++;
			}
			m_SlaveFlags[thread] = true;
			standpoint++;
			SlaveControlStationMT(standpoint);

			if (updateOrRevert[0])
			{
				SetOldToBiasesMT(thread);
				SetOldToWeightsMT(thread);
			}
			if (updateOrRevert[1])
			{
				SpinLockMT(thread);
				SetBiasesToOldMT(thread);
				SetWeightsToOldMT(thread);
			}


			m_SlaveFlags[thread] = true;
			standpoint++;
			SlaveControlStationMT(standpoint);

			epochCount++;
		}
	}

	void NeuralNetworkMT::TrainWithCheckMTMasterFunction(const float* traningInputs, const float* traningTargets, unsigned trainingNum, unsigned epochs, unsigned batchSize,  float learningRate, float regConst,  const float* checkInputs, const float* checkTargets, unsigned checkNum, ConditionFunctionPointer condFunc)
	{
		float* sharedInputBuffer = new float[m_LayerLayout[0] * batchSize];
		float* sharedTargetBuffer = new float[m_LayerLayout[m_LayerLayoutCount - 1] * batchSize];
		unsigned* indices = new unsigned[trainingNum];
		float* resultBuffer = new float[m_SlaveThreadCount];
		float* cost = new float[epochs];
		bool updateOrRevertToOld[2] = { false,false };

		m_MasterControlPoint = 0;
		
		std::thread* slaves = new std::thread[m_SlaveThreadCount];
		unsigned slaveThread = 0;
		while (slaveThread < m_SlaveThreadCount)
		{
			slaves[slaveThread] = std::thread(
				&NeuralNetworkMT::TrainWithCheckMTSlaveFunction, this,

				traningInputs, traningTargets, sharedInputBuffer, sharedTargetBuffer, indices, trainingNum,

				&epochs, &batchSize, &learningRate, &regConst, updateOrRevertToOld,

				checkInputs, checkTargets,checkNum, 

				resultBuffer,

				slaveThread
			);

			slaveThread++;
		}



		std::mt19937 mt;

		unsigned index = 0;
		while (index < trainingNum)
		{
			indices[index] = index;
			index++;
		}





		unsigned epochIndex = 0;
		while (epochIndex < epochs)
		{
			unsigned batchCount = trainingNum / batchSize;
			unsigned remainingBatch = trainingNum % batchSize;


			unsigned randomIndexCount = trainingNum;
			unsigned randomIndexStep = 0;

			unsigned batchNum = 0;
			while (batchNum < batchCount)
			{
				unsigned batchIndex = 0;
				while (batchIndex < batchSize)
				{

					unsigned randomIndex = mt() % randomIndexCount;

					unsigned epochRandomIndex = indices[randomIndex + randomIndexStep];
					indices[randomIndex + randomIndexStep] = indices[randomIndexStep];
					indices[randomIndexStep] = epochRandomIndex;

					randomIndexStep++;
					randomIndexCount--;
					batchIndex++;
				}
				//Used to prevent the slave threads from getting too far ahead
				m_MasterControlPoint++;
				batchNum++;
			}


			if (remainingBatch > 0)
			{
				unsigned batchIndex = 0;
				while (batchIndex < remainingBatch)
				{

					unsigned randomIndex = mt() % randomIndexCount;

					unsigned epochRandomIndex = indices[randomIndex + randomIndexStep];
					indices[randomIndex + randomIndexStep] = indices[randomIndexStep];
					indices[randomIndexStep] = epochRandomIndex;

					randomIndexStep++;
					randomIndexCount--;
					batchIndex++;

				}
				//Used to prevent the slave threads from getting too far ahead
				m_MasterControlPoint++;
			}


			WaitForSlavesMT();

			float tempCost = 0;
			slaveThread = 0;
			while (slaveThread < m_SlaveThreadCount)
			{
				tempCost += resultBuffer[slaveThread];
				slaveThread++;
			}
			cost[epochIndex] = tempCost;

			condFunc.Function(cost, epochIndex, epochs,batchSize,learningRate,regConst, updateOrRevertToOld);

			m_MasterControlPoint++;
			WaitForSlavesMT();

			updateOrRevertToOld[0] = false;
			updateOrRevertToOld[1] = false;

			m_MasterControlPoint++;

			epochIndex++;
		}

		slaveThread = 0;
		while (slaveThread < m_SlaveThreadCount)
		{
			slaves[slaveThread].join();
			slaveThread++;
		}
		
		delete[] slaves;
		delete[] resultBuffer;
		delete[] cost;
		delete[] sharedInputBuffer;
		delete[] sharedTargetBuffer;
		delete[] indices;
	}



	void NeuralNetworkMT::CheckCostMTSlaveFunction(const float* checkInputs, const float* checkTargets, unsigned num, float* resultBuffer, const unsigned thread)
	{
		const unsigned Ap = m_ABufferCount - m_LayerLayout[m_LayerLayoutCount - 1];

		const auto costFunc = m_Functions.CostFunction;

		resultBuffer[thread] = 0;
	

		unsigned checkIndex = 0;
		while (checkIndex < num)
		{
			SpinLockMT(thread);
			SetInputMT(&checkInputs[checkIndex * m_LayerLayout[0]], thread);
			SpinLockMT(thread);
			FeedForwardMT(thread);

			unsigned localOffset = 0;
			unsigned localStop = 0;

			ThreadWorkloadDivider(localOffset, localStop, m_LayerLayout[m_LayerLayoutCount-1], thread);
		
			unsigned outputIndex = localOffset;
			while (outputIndex < localStop)
			{
			
				resultBuffer[thread] += costFunc(m_ABuffer[Ap + outputIndex], checkTargets[m_LayerLayout[m_LayerLayoutCount - 1] * checkIndex + outputIndex]);
				outputIndex++;
			}
			checkIndex++;
		}
	
	}

	float NeuralNetworkMT::CheckCostMTMasterFunction(const float* checkInputs, const float* checkTargets, unsigned num)
	{
		float* resultBuffer = new float[m_SlaveThreadCount];
		std::thread* slaves = new std::thread[m_SlaveThreadCount];
		unsigned thread = 0;
		while (thread < m_SlaveThreadCount)
		{

			slaves[thread] = std::thread(&NeuralNetworkMT::CheckCostMTSlaveFunction, this, checkInputs, checkTargets, num, resultBuffer,  thread);

			thread++;
		}
		float cost = 0;
		thread = 0;
		while (thread < m_SlaveThreadCount)
		{
			slaves[thread].join();

			cost += resultBuffer[thread];
			thread++;
		}

		delete[] resultBuffer;
		delete[] slaves;
		return (cost / ((float)num));
	}


	void NeuralNetworkMT::CheckSuccessRateMTSlaveFunction(const float* checkInputs, const float* checkTargets, unsigned num, unsigned* resultBuffer, const unsigned thread)
	{

		const unsigned targetSize = num * m_LayerLayout[m_LayerLayoutCount - 1];
		const unsigned Ap = m_ABufferCount - m_LayerLayout[m_LayerLayoutCount - 1];

		unsigned checkIndex = 0;
		while (checkIndex < num)
		{
			SpinLockMT(thread);
			SetInputMT(&checkInputs[checkIndex * m_LayerLayout[0]], thread);
			SpinLockMT(thread);
			FeedForwardMT(thread);

			int championItterator = -1;
			float champion = 0;
			unsigned localOffset = 0;
			unsigned localStop = 0;

			ThreadWorkloadDivider(localOffset, localStop, m_LayerLayout[m_LayerLayoutCount - 1], thread);


			unsigned outputIndex = localOffset;
			while (outputIndex < localStop)
			{

				if (m_ABuffer[Ap + outputIndex] >= champion)
				{
					champion = m_ABuffer[Ap + outputIndex];
					championItterator = outputIndex;
				}

				outputIndex++;
			}

			resultBuffer[thread] = championItterator;
			m_SlaveFlags[thread] = true;
			checkIndex++;
			SlaveControlStationMT(checkIndex);
		
		}

	}

	float NeuralNetworkMT::CheckSuccessRateMTMasterFunction(const float* checkInputs, const float* checkTargets, unsigned num)
	{
	
		unsigned* resultBuffer = new unsigned[m_SlaveThreadCount];
		m_MasterControlPoint = 0;

		std::thread* slaves = new std::thread[m_SlaveThreadCount];
		unsigned thread = 0;
		while (thread < m_SlaveThreadCount)
		{

			slaves[thread] = std::thread(&NeuralNetworkMT::CheckSuccessRateMTSlaveFunction, this, checkInputs, checkTargets, num, resultBuffer, thread);

			thread++;
		}

		float score = 0;
		const unsigned Ap = m_ABufferCount - m_LayerLayout[m_LayerLayoutCount - 1];

		unsigned checkIndex = 0;
		while (checkIndex < num)
		{
			

			float champion = 0;
			int championItterator = -1;

			WaitForSlavesMT();

			unsigned i = 0;
			while (i < m_SlaveThreadCount)
			{
				unsigned itterator = resultBuffer[i];
				if (m_ABuffer[Ap + itterator] >= champion)
				{
					championItterator = itterator;
					champion = m_ABuffer[Ap + itterator];
				}
				i++;
			}
		
			if (checkTargets[m_LayerLayout[m_LayerLayoutCount - 1] * checkIndex + championItterator] == 1)
			{
				score += 1.0f;
			}

			m_MasterControlPoint++;


			checkIndex++;
		}


		thread = 0;
		while (thread < m_SlaveThreadCount)
		{
			slaves[thread].join();
			thread++;
		}
		delete[] slaves;
		delete[] resultBuffer;
		m_MasterControlPoint = 0;

		return (score / ((float)num));
	}

}