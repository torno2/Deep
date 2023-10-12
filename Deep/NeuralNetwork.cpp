 #include "pch.h"
#include "NeuralNetwork.h"


namespace TNNT 
{
	//Constructors And destructor

	NeuralNetwork::NeuralNetwork(unsigned* layerLayout, unsigned  layoutSize, LayerFucntionsLayout functions, bool randomizeWeightsAndBiases)
		: m_LayerLayout(layerLayout), m_LayerLayoutCount(layoutSize), m_Functions(functions)
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

	NeuralNetwork::NeuralNetwork(unsigned* layerLayout, unsigned  layoutSize, float* biases, float* weights, LayerFucntionsLayout functions)
		: m_LayerLayout(layerLayout), m_LayerLayoutCount(layoutSize),
		m_Biases(biases),
		m_Weights(weights),
		m_Functions(functions)
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

		SetTempToBiases();

		SetTempToWeights();
		ResetWeightsTranspose();

	}

	NeuralNetwork::~NeuralNetwork()
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

	}

	float NeuralNetwork::CheckCost(DataSet& data)
	{
		return CheckCostMasterFunction(data.TestInputs,data.TestTargets,data.TestCount);
	}

	float NeuralNetwork::CheckSuccessRate(DataSet& data)
	{
		return CheckSuccessRateMasterFunction(data.TestInputs, data.TestTargets, data.TestCount);
	}

	void NeuralNetwork::Train(DataSet& data, HyperParameters& params)
	{
		TrainMasterFunction(data.TrainingInputs, data.TraningTargets, data.TrainingCount,  params.Epochs, params.BatchCount, params.LearningRate, params.RegularizationConstant);
	}

	void NeuralNetwork::TrainWCondition(DataSet& data, HyperParameters& params, ConditionFunctionPointer condFunc)
	{
		TrainWConditionMasterFunction(data.TrainingInputs, data.TraningTargets, data.TrainingCount,  params.Epochs, params.BatchCount, params.LearningRate, params.RegularizationConstant,data.ValidationInputs,data.TraningTargets,data.ValidationCount,condFunc);
	}


	//Public functions



	void NeuralNetwork::SaveToFile(const char* filepath) const
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

	void NeuralNetwork::LoadFromFile(const char* filepath)
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


	NeuralNetwork* NeuralNetwork::CreateFromFile(const char* filepath, LayerFucntionsLayout functions)
	{

		unsigned* LayerLayout;
		unsigned layoutSize;

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
		NeuralNetwork* result = new NeuralNetwork(LayerLayout, layoutSize, Biases, Weights, functions);
		return result;

	}


	// Private functions

	void NeuralNetwork::SetBiasesToTemp()
	{

		
		memcpy(m_Biases, m_BiasesBuffer, sizeof(float) * m_BiasesCount);
	}

	void NeuralNetwork::SetTempToBiases()
	{
		memcpy(m_BiasesBuffer, m_Biases, sizeof(float) * m_BiasesCount);
	}


	void NeuralNetwork::SetWeightsToTemp()
	{

		memcpy(m_Weights, m_WeightsBuffer, sizeof(float) * m_WeightsCount);

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
				m_WeightsTranspose[currentOffset + (index / m_LayerLayout[layoutIndex - 1]) + (index % m_LayerLayout[layoutIndex - 1]) * m_LayerLayout[layoutIndex]] = m_WeightsBuffer[currentOffset + index];
				index++;
			}

			layoutIndex++;
		}

	}

	void NeuralNetwork::SetTempToWeights()
	{
		memcpy(m_WeightsBuffer, m_Weights, sizeof(float) * m_WeightsCount);
	}

	void NeuralNetwork::ResetWeightsTranspose()
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

	void NeuralNetwork::SetBiasesToOld()
	{
		unsigned index = 0;
		while (index < m_BiasesCount)
		{
			m_Biases[index] = m_BiasesOld[index];
			index++;
		}
	}

	void NeuralNetwork::SetWeightsToOld()
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
				m_Weights[currentOffset + index] = m_WeightsOld[currentOffset + index];
				m_WeightsTranspose[currentOffset + (index / m_LayerLayout[layoutIndex - 1]) + (index % m_LayerLayout[layoutIndex - 1]) * m_LayerLayout[layoutIndex]] = m_WeightsOld[currentOffset + index];
				
				index++;
			}

			layoutIndex++;
		}
	}

	void NeuralNetwork::SetOldToBiases()
	{
		unsigned index = 0;
		while (index < m_BiasesCount)
		{
			m_BiasesOld[index] = m_Biases[index];
			index++;
		}
	}

	void NeuralNetwork::SetOldToWeights()
	{
		unsigned index = 0;
		while (index < m_WeightsCount)
		{
			m_WeightsOld[index] = m_Weights[index];
			index++;
		}
	}


	void NeuralNetwork::SetInput(const float* input)
	{





		unsigned index = 0;
		while (index < m_LayerLayout[0])
		{
			m_ABuffer[index] = input[index];
			index++;
		}
	}

	void NeuralNetwork::SetTarget(const float* target)
	{






		unsigned index = 0;
		while (index < m_LayerLayout[m_LayerLayoutCount - 1])
		{
			m_TargetBuffer[index] = target[index];
			index++;
		}
	}


	void NeuralNetwork::FeedForward()
	{

		unsigned biasesStart = 0;
		unsigned weightsStart = 0;
		unsigned intakePos = 0;

		unsigned layoutIndex = 1;
		while (layoutIndex < m_LayerLayoutCount)
		{

			auto neuronFunc = m_Functions.NeuronFunction[layoutIndex - 1].Function;



			unsigned layerIndex = 0;
			while (layerIndex < m_LayerLayout[layoutIndex])
			{
				float weightedSum = 0;

				unsigned prevIndex = 0;
				while (prevIndex < m_LayerLayout[layoutIndex - 1])
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


		}

	}

	void NeuralNetwork::Backpropegate()
	{

		//Always assumes that the network has fed forward first.

		unsigned layoutIndex = m_LayerLayoutCount - 1;

		unsigned latterBiases = m_BiasesCount;
		unsigned latterWeights = m_WeightsCount;

		unsigned biasesStart = latterBiases - m_LayerLayout[layoutIndex];
		unsigned weightsStart = latterWeights - m_LayerLayout[layoutIndex] * m_LayerLayout[layoutIndex - 1];


		//Used to fetch the "a" belonging to the layer that comes before the one we're focusing on.
		unsigned apos = biasesStart + m_LayerLayout[0] - m_LayerLayout[layoutIndex - 1];


		const auto costDerivative = m_Functions.CostFunctionDerivative;



		unsigned layerIndex = 0;
		while (layerIndex < m_LayerLayout[layoutIndex])
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



		while (layoutIndex > 1)
		{
			layoutIndex--;

			const auto neuronFuncDeriv = m_Functions.NeuronFunctionDerivative[layoutIndex - 1].Function;

			latterBiases -= m_LayerLayout[layoutIndex + 1];
			latterWeights -= m_LayerLayout[layoutIndex + 1] * m_LayerLayout[layoutIndex];

			biasesStart -= m_LayerLayout[layoutIndex];
			weightsStart -= m_LayerLayout[layoutIndex] * m_LayerLayout[layoutIndex - 1];

			apos -= m_LayerLayout[layoutIndex - 1];




			unsigned layerIndex = 0;
			while (layerIndex < m_LayerLayout[layoutIndex])
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

		}
	}


	void NeuralNetwork::RegWeightsL2(unsigned trainingSetSize, float learingRate, float L2regConst)
	{





		unsigned index = 0;
		while (index < m_WeightsCount)
		{
			m_WeightsBuffer[index] *= (1 - (learingRate * L2regConst / (float)trainingSetSize));
			



			index++;
		}
	}

	void NeuralNetwork::GradientDecent(unsigned batchSize, float learingRate)
	{












		unsigned index = 0;
		while (index < m_WeightsCount)
		{
	
			m_WeightsBuffer[index] -= (learingRate / (float)batchSize) * m_DeltaWeights[index];
			

			index++;
		}

		index = 0;
		while (index < m_BiasesCount)
		{
			m_BiasesBuffer[index] -= (learingRate / (float)batchSize) * m_DeltaBiases[index];
			


			index++;
		}


	}


	void NeuralNetwork::TrainOnSet(const float* inputs, const float* targets, unsigned num, float learingRate, float regConst, unsigned trainingSetSize)
	{

		RegWeightsL2(trainingSetSize, learingRate, regConst);
		

		unsigned exampleIndex = 0;
		while (exampleIndex < num)
		{

			SetInput(&(inputs[exampleIndex * m_LayerLayout[0]]));
			SetTarget(&(targets[exampleIndex * m_LayerLayout[m_LayerLayoutCount - 1]]));


			FeedForward();
			Backpropegate();


			GradientDecent(num, learingRate);
			

			exampleIndex++;
		}


		SetBiasesToTemp();
		SetWeightsToTemp();

	}


	void NeuralNetwork::TrainMasterFunction(const float* traningInputs, const float* traningTargets, unsigned num,  unsigned epochs, unsigned batchSize, float learningRate, float regConst)
	{

		const unsigned inputBufferSize = m_LayerLayout[0] * batchSize;
		const unsigned outputBufferSize = m_LayerLayout[m_LayerLayoutCount - 1] * batchSize;
		float* inputBuffer = new float[inputBufferSize];
		float* outputBuffer = new float[outputBufferSize];

		const unsigned batchCount = num / batchSize;
		const unsigned remainingBatch = num % batchSize;

		

		unsigned* indices = new unsigned[num];
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
			unsigned randomIndexPos = 0;
			unsigned randomIndexCount = num;
	
			

			unsigned batchNum = 0;
			while (batchNum < batchCount)
			{
				unsigned batchIndex = 0;
				while (batchIndex < batchSize)
				{

					unsigned randomIndex = mt() % randomIndexCount + randomIndexPos;

					unsigned epochRandomIndex = indices[randomIndex];
					indices[randomIndex] = indices[randomIndexPos];
					indices[randomIndexPos] = epochRandomIndex;


					unsigned inputIndex = 0;
					while (inputIndex < m_LayerLayout[0])
					{
						inputBuffer[ m_LayerLayout[0] * batchIndex + inputIndex] = traningInputs[epochRandomIndex * m_LayerLayout[0] + inputIndex];
						inputIndex++;
					}

					unsigned outputIndex = 0;
					while (outputIndex < m_LayerLayout[m_LayerLayoutCount - 1])
					{
						outputBuffer[m_LayerLayout[m_LayerLayoutCount - 1] * batchIndex + outputIndex] = traningTargets[m_LayerLayout[m_LayerLayoutCount - 1] * epochRandomIndex + outputIndex];
						outputIndex++;
					}


					randomIndexPos++;
					randomIndexCount--;

					batchIndex++;
				}

				TrainOnSet(inputBuffer, outputBuffer, batchSize, learningRate, regConst, num);

				batchNum++;
			}


			if (remainingBatch > 0)
			{
				unsigned batchIndex = 0;
				while (batchIndex < remainingBatch)
				{

					unsigned randomIndex = mt() % randomIndexCount + randomIndexPos;

					unsigned epochRandomIndex = indices[randomIndex];
					indices[randomIndex] = indices[randomIndexPos];
					indices[randomIndexPos] = epochRandomIndex;

					unsigned inputIndex = 0;
					while (inputIndex < m_LayerLayout[0])
					{
						inputBuffer[batchIndex * m_LayerLayout[0] + inputIndex] = traningInputs[epochRandomIndex * m_LayerLayout[0] + inputIndex];
						inputIndex++;
					}

					unsigned outputIndex = 0;
					while (outputIndex < m_LayerLayout[m_LayerLayoutCount - 1])
					{
						outputBuffer[m_LayerLayout[m_LayerLayoutCount - 1] * batchIndex + outputIndex] = traningTargets[m_LayerLayout[m_LayerLayoutCount - 1] * epochRandomIndex + outputIndex];
						outputIndex++;
					}


					randomIndexPos++;
					randomIndexCount--;
					
					batchIndex++;

				}
				TrainOnSet(inputBuffer, outputBuffer, remainingBatch, learningRate, regConst, num);
			}



			epochCount++;
		}

		

		


		delete[] inputBuffer;
		delete[] outputBuffer;
		delete[] indices;
	}

	void NeuralNetwork::TrainWConditionMasterFunction(const float* traningInputs, const float* traningTargets, unsigned trainingNum,unsigned epochs, unsigned batchSize, float learningRate, float regConst, const float* checkInputs, const float* checkTargets, unsigned checkNum, ConditionFunctionPointer condFunc)
	{
		const unsigned inputBufferSize = m_LayerLayout[0] * batchSize;
		const unsigned outputBufferSize = m_LayerLayout[m_LayerLayoutCount - 1] * batchSize;
		float* inputBuffer = new float[inputBufferSize];
		float* outputBuffer = new float[outputBufferSize];
		float* cost = new float[epochs];
		bool updateOrRevertToOld[2] = { false,false };

		std::mt19937 mt;

		unsigned* indicies = new unsigned[trainingNum];
		unsigned index = 0;
		while (index < trainingNum)
		{
			indicies[index] = index;
			index++;
		}

		unsigned epochIndex = 0;
		while (epochIndex < epochs)
		{
			unsigned remainingBatch = trainingNum % batchSize;
			unsigned batchCount = trainingNum / batchSize;
			
			
			unsigned randomIndexCount = trainingNum;


			unsigned batchNum = 0;
			while (batchNum < batchCount)
			{
				unsigned batchIndex = 0;
				while (batchIndex < batchSize)
				{

					unsigned randomIndex = mt() % randomIndexCount;

					unsigned epochRandomIndex = indicies[randomIndex];
					indicies[randomIndex] = indicies[randomIndexCount - 1];
					indicies[randomIndexCount - 1] = epochRandomIndex;

					unsigned inputIndex = 0;
					while (inputIndex < m_LayerLayout[0])
					{
						inputBuffer[batchIndex * m_LayerLayout[0] + inputIndex] = traningInputs[epochRandomIndex * m_LayerLayout[0] + inputIndex];
						inputIndex++;
					}

					unsigned outputIndex = 0;
					while (outputIndex < m_LayerLayout[m_LayerLayoutCount - 1])
					{
						outputBuffer[m_LayerLayout[m_LayerLayoutCount - 1] * batchIndex + outputIndex] = traningTargets[m_LayerLayout[m_LayerLayoutCount - 1] * epochRandomIndex + outputIndex];
						outputIndex++;
					}



					randomIndexCount--;
					batchIndex++;
				}

				TrainOnSet(inputBuffer, outputBuffer, batchSize, learningRate, regConst, trainingNum);

				batchNum++;
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
					while (inputIndex < m_LayerLayout[0])
					{
						inputBuffer[batchIndex * m_LayerLayout[0] + inputIndex] = traningInputs[epochRandomIndex * m_LayerLayout[0] + inputIndex];
						inputIndex++;
					}

					unsigned outputIndex = 0;
					while (outputIndex < m_LayerLayout[m_LayerLayoutCount - 1])
					{
						outputBuffer[m_LayerLayout[m_LayerLayoutCount - 1] * batchIndex + outputIndex] = traningTargets[m_LayerLayout[m_LayerLayoutCount - 1] * epochRandomIndex + outputIndex];
						outputIndex++;
					}



					randomIndexCount--;
					batchIndex++;

				}
				TrainOnSet(inputBuffer, outputBuffer, remainingBatch, learningRate, regConst, trainingNum);
			}

			
			
			cost[epochIndex] = CheckCostMasterFunction(checkInputs, checkTargets, checkNum);


			condFunc.Function( cost, epochIndex, epochs,batchSize,learningRate,regConst, updateOrRevertToOld);

			if (updateOrRevertToOld[0])
			{
				SetOldToBiases();
				SetOldToWeights();
				updateOrRevertToOld[0] = false;
			}
			if (updateOrRevertToOld[1])
			{
				SetBiasesToOld();
				SetWeightsToOld();
				updateOrRevertToOld[1] = false;
			}

			epochIndex++;
		}

		delete[] cost;
		delete[] inputBuffer;
		delete[] outputBuffer;
		delete[] indicies;
	}


	float NeuralNetwork::CheckCostMasterFunction(const float* checkInputs, const float* checkTargets, unsigned num)
	{


		const auto costFunc = m_Functions.CostFunction;

		const unsigned Ap = m_ABufferCount - m_LayerLayout[m_LayerLayoutCount - 1];

		float cost = 0;

		unsigned checkIndex = 0;
		while (checkIndex < num)
		{
			SetInput(&checkInputs[checkIndex * m_LayerLayout[0]]);
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

	float NeuralNetwork::CheckSuccessRateMasterFunction(const float* checkInputs, const float* checkTargets, unsigned num)
	{
		const unsigned Ap = m_ABufferCount - m_LayerLayout[m_LayerLayoutCount - 1];

		float score = 0.0f;

		unsigned checkIndex = 0;
		while (checkIndex < num)
		{

			SetInput(&checkInputs[checkIndex * m_LayerLayout[0]]);
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
}