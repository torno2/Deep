#pragma once
#include "NeuralNetworkCustomVariables.h"
namespace TNNT
{
	class NeuralNetwork
	{
	public:

		//Note: All pointers decleared here untill "End of mention." are all arrays. Each array has their size 
		//equal to the first unsigned declared bellow (or the one described in the comment above) them.
		//For example: "m_ZBuffer" has a size equal to "m_BiasesCount", whilst the size of "m_ABuffer" is equal to "m_ABufferCount"

		//"m_LayerLayout" is supposed to have the number of nodes in a layer, under the index corresponding to that layers postion in the network (zero being the first layer) 
		// In addition to being the size of "m_LAyerLayout", "m_LayerLayoutCount" is also equal to the number of layers in the network. 
		unsigned* m_LayerLayout;
		unsigned m_LayerLayoutCount;

		//Is supposed to have size equal to the sum of all the elements in m_LayerLayout, minus the first element.
		float* m_Biases;
		float* m_DeltaBiases;
		float* m_BiasesBuffer;
		float* m_BiasesOld;
		float* m_ZBuffer;
		unsigned m_BiasesCount;

		//Is supposed to have size equal to the sum of all the elements in m_LayerLayout.
		float* m_ABuffer;
		unsigned m_ABufferCount;

		//Is supposed to have size equal to the sum of each element in m_LayerLayout (starting at index 1) multiplied by the previous element.
		float* m_Weights;
		float* m_WeightsTranspose;
		float* m_DeltaWeights;
		float* m_WeightsBuffer;
		float* m_WeightsOld;
		unsigned m_WeightsCount;


		//Is supposed to have size equal to the last element in m_LayerLayout.
		float* m_TargetBuffer;
		//End of mention.

		LayerFucntionsLayout m_Functions;

	public:

		NeuralNetwork(unsigned* layerLayout, unsigned layoutSize, LayerFucntionsLayout functions, bool randomizeWeightsAndBiases = true);
		~NeuralNetwork();


		float CheckCost(DataSet& data);
		float CheckSuccessRate(DataSet& data);

		void Train(DataSet& data, HyperParameters& params);
		//Provide a function that manipulates the hyper-parameters based on the cost of the network in each epoch. Setting the "epoch" variable will cause the traning function to exit.
		void TrainWCondition(DataSet& data, HyperParameters& params, ConditionFunctionPointer condFunc);


		void SaveToFile(const char* filepath) const;
		void LoadFromFile(const char* filepath);

		static NeuralNetwork* CreateFromFile(const char* filepath, LayerFucntionsLayout functions);


		

	public:

		NeuralNetwork(unsigned* layerLayout, unsigned const layoutSize,
			float* biases,
			float* weights, LayerFucntionsLayout functions
		);


		void SetBiasesToTemp();
		void SetTempToBiases();

		void SetWeightsToTemp();
		void SetTempToWeights();
		void ResetWeightsTranspose();

		void SetBiasesToOld();
		void SetWeightsToOld();

		void SetOldToBiases();
		void SetOldToWeights();

		void SetInput(const float* input);
		void SetTarget(const float* target);

		void FeedForward();
		void Backpropegate();

		void RegWeightsL2(unsigned trainingSetSize, float learingRate, float L2regConst);
		void GradientDecent(unsigned batchSize, float learingRate);

		void TrainOnSet(const float* inputs, const float* targets, unsigned num, float learingRate, float regConst, unsigned trainingSetSize);

		void TrainMasterFunction(const float* traningInputs, const float* traningTargets, unsigned num, unsigned epochs, unsigned batchSize, float learningRate, float regConst);
		void TrainWConditionMasterFunction(const float* traningInputs, const float* traningTargets, unsigned trainingNum, unsigned epochs, unsigned batchSize,  float learningRate, float regConst, const float* checkInputs, const float* checkTargets, unsigned checkNum, ConditionFunctionPointer condFunc);

		float CheckCostMasterFunction(const float* checkInputs, const float* checkTargets, unsigned num);
		float CheckSuccessRateMasterFunction(const float* checkInputs, const float* checkTargets, unsigned num);

		

	};
}


