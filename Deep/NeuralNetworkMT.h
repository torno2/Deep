#pragma once
#include "NeuralNetworkCustomVariables.h"
namespace TNNT
{
	class NeuralNetworkMT
	{
	public:


	private:
		//Publicly available for easy access

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
		float* deltaZBuffer;
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

		//Multi Threading
		unsigned m_SlaveThreadCount;

		//Synching
		bool* m_Locks;
		bool* m_SlaveFlags;
		unsigned m_MasterControlPoint = 0;


	public:



		NeuralNetworkMT(unsigned* layerLayout, unsigned layoutSize, LayerFucntionsLayout functions, unsigned slaveThreadCount, bool randomizeWeightsAndBiases = true);
		~NeuralNetworkMT();

		float CheckCost(DataSet& data);
		float CheckSuccessRate(DataSet& data);

		void Train(DataSet& data, HyperParameters& params);
		//Provide a function that manipulates the hyper-parameters based on the cost of the network in each epoch. Setting the "epoch" variable will cause the traning function to exit.
		void TrainWCondition(DataSet& data, HyperParameters& params, ConditionFunctionPointer condFunc);

		void SaveToFile(const char* filepath) const;
		void LoadFromFile(const char* filepath);

		static NeuralNetworkMT* CreateFromFile(const char* filepath, LayerFucntionsLayout functions, unsigned slaveThreadCount);



	private:

		NeuralNetworkMT(unsigned* layerLayout, unsigned layoutSize, float* biases, float* weights, LayerFucntionsLayout functions, unsigned slaveThreadCount);


		//Used by the constructors

		void SetTempToBiases();

		void SetTempToWeights();
		void ResetWeightsTranspose();


		//Debugging

		float CheckCostD(const float* checkInputs, const float* checkTargets, unsigned num);
		float CheckSuccessRateD(const float* checkInputs, const float* checkTargets, unsigned num);

		void FeedForward();
		void SetInput(const float* input, unsigned startElement = 0);


		//Multithreading

		void SetBiasesToTempMT(const unsigned thread);
		void SetWeightsToTempMT(const unsigned thread);

		void SetOldToBiasesMT(const unsigned thread);
		void SetOldToWeightsMT(const unsigned thread);

		void SetBiasesToOldMT(const unsigned thread);
		void SetWeightsToOldMT(const unsigned thread);

		void SetInputMT(const float* input, const unsigned thread);
		void SetTargetMT(const float* target, const unsigned thread);

		void FeedForwardMT(const unsigned thread);
		void BackpropegateMT(const unsigned thread);

		void GradientDecentMT(unsigned batchSize, float learingRate, const unsigned thread);
		void RegWeightsL2MT(unsigned trainingSetSize, float learingRate, float L2regConst, const unsigned thread);

		void SamplePrepp(const float* trainingInputs, const float* trainingTargets, float* inputBuffer, float* targetBuffer, unsigned* indicies, unsigned num, unsigned step, const unsigned thread);
		void TrainOnSetMT(const float* inputBuffer, float* targetBuffer, unsigned num, float learingRate, float regConst, unsigned trainingSetSize, const unsigned thread);

		void ThreadWorkloadDivider(unsigned& start, unsigned& stop, unsigned workCount, unsigned thread);
		void SpinLockMT(const unsigned thread);
		void SlaveControlStationMT(unsigned standpoint);
		void WaitForSlavesMT();

		void TrainMTSlaveThreadFunction(const float* traningInputs, const float* traningTargets, float* inputBuffer, float* targetBuffer, unsigned* indicies, unsigned num, unsigned epochs, unsigned batchSize,  float learningRate, float regConst, const unsigned thread);
		void TrainMTMasterThreadFunction(const float* traningInputs, const float* traningTargets, unsigned num, unsigned epochs, unsigned batchSize,  float learningRate, float regConst);

		void TrainWithCheckMTSlaveFunction(const float* traningInputs, const float* traningTargets, float* inputBuffer, float* targetBuffer, unsigned* indicies, unsigned trainingNum, unsigned* epochs, unsigned* batchSize, float* learningRate, float* regConst, bool* updateOrRevert, const float* checkInputs, const float* checkTargets, unsigned checkNum, float* resultBuffer, const unsigned thread);
		void TrainWithCheckMTMasterFunction(const float* traningInputs, const float* traningTargets, unsigned trainingNum, unsigned epochs, unsigned batchSize,  float startingLearningRate, float regConst, const float* checkInputs, const float* checkTargets, unsigned checkNum, ConditionFunctionPointer condFunc);


		void CheckCostMTSlaveFunction(const float* checkInputs, const float* checkTargets, unsigned num, float* resultBuffer, const unsigned thread);
		float CheckCostMTMasterFunction(const float* checkInputs, const float* checkTargets, unsigned num);

		void CheckSuccessRateMTSlaveFunction(const float* checkInputs, const float* checkTargets, unsigned num, unsigned* resultBuffer, const unsigned thread);
		float CheckSuccessRateMTMasterFunction(const float* checkInputs, const float* checkTargets, unsigned num);


		};
}