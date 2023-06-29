#pragma once
#include "NeuralNetworkCustomVariables.h"
#include <thread>



namespace TNNT
{
	class NetworkPrototypeMT
	{

	public:

		LayerLayout* m_LayerLayout;
		unsigned m_LayerLayoutCount;

		FunctionsLayoutMT m_Functions;

		float* m_ZBuffer;
		float* m_DeltaZ;
		unsigned m_ZBufferCount;


		float* m_ABuffer;
		unsigned m_ABufferCount;


		float* m_Biases;
		float* m_DeltaBiases;
		float* m_BiasesBuffer;
		unsigned m_BiasesCount;


		float* m_Weights;
		float* m_DeltaWeights;
		float* m_WeightsBuffer;
		unsigned m_WeightsCount;

		float* m_TargetBuffer;
		float* m_CostBuffer;
		unsigned* m_GuessBuffer;

		//Note to self: Handling m_PositionData with multiple threads is tricky. Current fix is having thread 0 take care of it.
		PositionData m_PositionData;
		unsigned* m_Indices = nullptr;
		float* m_InternalInputBuffer = nullptr;
		float* m_InternalTargetBuffer = nullptr;

		//Temporary ownership. The network is not responsible for this pointer.
		DataSet* m_Data;

		HyperParameters m_HyperParameters;

		// 0: Training, 1: Cost, 2: Success rate
		float m_LastTime[3];


		//Multi Threading
		std::thread* m_SlaveThreads;
		unsigned m_SlaveThreadCount;
		

		//Synching
		bool* m_Locks;
		bool* m_SlaveFlags;
		unsigned m_MasterControlPoint = 0;

		


	public:

		NetworkPrototypeMT(LayerLayout* layerLayout, FunctionsLayoutMT& functions, unsigned layoutCount, unsigned slaveThreadCount, bool randomizeWeightsAndBiases = true);
		~NetworkPrototypeMT();

		float CheckSuccessRate();
		float CheckCost();

		void Train(DataSet* data, HyperParameters& params);





	public:


		//Network helpers:

		void SetTempToBiases(unsigned thread);
		void SetTempToWeights(unsigned thread);
		void SetTempToBiasesAndWeights(unsigned thread);


		void SetBiasesToTemp(unsigned thread);
		void SetWeightsToTemp(unsigned thread);
		

		void SetData(DataSet* data);
		void ResetIndices(unsigned thread);

		void SetHyperParameters(HyperParameters& params);

		void SetInput(float* input, unsigned thread);
		void SetTarget(float* target, unsigned thread);

		void ThreadWorkloadDivider(unsigned& start, unsigned& stop, unsigned workCount, unsigned thread);
		void SpinLock(unsigned thread);
		void SlaveControlStation(unsigned position);
		void WaitForSlaves();

		//Actual network mechanisms

		void FeedForward(unsigned thread);
		void Backpropegate(unsigned thread);

		void SamplePrepp(unsigned batchCount, unsigned step, unsigned thread);
		void TrainOnSet(unsigned batchCount, unsigned thread);

		void TrainSlaveFunction(unsigned thread);
		void TrainMasterFunction();

		// Performance evaluation

		void CheckCostSlaveFunction(unsigned thread);
		float CheckCostMasterFunction();

		void CheckSuccessRateSlaveFunction(unsigned thread);
		float CheckSuccessRateMasterFunction();




	};
}


