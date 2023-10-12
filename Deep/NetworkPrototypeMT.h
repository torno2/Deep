#pragma once
#include "NeuralNetworkCustomVariables.h"
#include <thread>



namespace TNNT
{
	class NetworkPrototypeMT
	{

	public:

		//Order A, Weights, Biases, Z, dZ, dWeights, dBiases, WeightsBuffer, BiasesBuffer 
		float* m_NetworkFixedData;

		LayerLayout* m_LayerLayout;
		unsigned m_LayerLayoutCount;

		FunctionsLayoutMT m_Functions;

		float* m_A;
		unsigned m_ACount;


		float* m_Z;
		float* m_DeltaZ;
		unsigned m_ZCount;


		float* m_InputBuffer;
		unsigned m_InputBufferCount;

		float* m_OutputBuffer;
		float* m_TargetBuffer;
		unsigned m_OutputBufferCount;


		float* m_Weights;
		float* m_Biases;

		float* m_DeltaWeights;
		float* m_DeltaBiases;

		float* m_WeightsBuffer;
		float* m_BiasesBuffer;

		unsigned m_WeightsCount;
		unsigned m_BiasesCount;


		float* m_CostBuffer;
		unsigned* m_GuessBuffer;

		//Note to self: Handling m_PositionData with multiple threads is tricky. Current fix is having thread 0 take care of it.
		PositionData m_PositionData;
		unsigned* m_Indices = nullptr;

		//Temporary ownership. The network is not responsible for this pointer.
		DataSet* m_Data;

		HyperParameters m_HyperParameters;

		// 0: Training, 1: Cost, 2: Success rate
		float m_LastTime[3];


		//Multi Threading
		std::thread* m_SlaveThreads;
		unsigned m_SlaveThreadCount;

		//comming soon
		unsigned m_CachePaddingFloat = CacheLineSize / sizeof (float);
		

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

		unsigned Check(float* input);



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
		
		void Regularization(unsigned thread);
		void Train(unsigned thread);
		
		void TrainOnSet(unsigned batchCount, unsigned batch, unsigned thread);

		void TrainSlaveFunction(unsigned thread);
		void TrainMasterFunction();

		// Performance evaluation

		void CheckCostSlaveFunction(unsigned thread);
		float CheckCostMasterFunction();

		void CheckSuccessRateSlaveFunction(unsigned thread);
		float CheckSuccessRateMasterFunction();

		void CheckSlaveFunction(float* input, unsigned thread);
		unsigned CheckMasterFunction(float* input);
		


	};
}


