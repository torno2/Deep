#pragma once
#include "NeuralNetworkCustomVariables.h"



namespace TNNT {

	class NetworkPrototype
	{
	public:
		//Order: A, Weights, Biases, Z, dZ, dWeights, dBiases, WeightsBuffer, BiasesBuffer 
		float* m_NetworkFixedData;

		FunctionsLayout m_Functions;

		PositionData m_PositionData;

		LayerLayout* m_LayerLayout;
		unsigned m_LayerLayoutCount;


		float* m_A;
		unsigned m_ACount;


		float* m_Z;
		float* m_DeltaZ;
		unsigned m_ZCount;


		float* m_InputBuffer;
		unsigned m_InputBufferCount;

		float* m_TargetBuffer;
		unsigned m_TargetBufferCount;
		

		float* m_Weights;
		float* m_Biases;

		float* m_DeltaWeights;
		float* m_DeltaBiases;

		float* m_WeightsBuffer;
		float* m_BiasesBuffer;
		
		unsigned m_WeightsCount;
		unsigned m_BiasesCount;

		
		float m_CostBuffer;
		
		
		HyperParameters m_HyperParameters;
		
		unsigned* m_Indices = nullptr;
		
		
		DataSet* m_Data = nullptr;

		// 0: Training, 1: Cost, 2: Success rate
		float m_LastTime[3];


	public:

		NetworkPrototype(LayerLayout* layerLayout, FunctionsLayout& functions, unsigned layoutCount , bool randomizeWeightsAndBiases = true);
		~NetworkPrototype();

		float CheckSuccessRate();
		float CheckCost();

		void Train(DataSet* data, HyperParameters& params);





	public:


		//Network helpers:

		void SetBiasesToTemp();
		void SetTempToBiases();

		void SetWeightsToTemp();
		void SetTempToWeights();

		void SetData(DataSet* data);
		void SetHyperParameters(HyperParameters& params);

		void SetInput(const float* input);
		void SetTarget(const float* target);

		//Actual network mechanisms

		void FeedForward();
		void Backpropegate();

		void TrainOnSet(unsigned batchCount, unsigned batch);

		void TrainMasterFunction();
		
		// Performance evaluation

		float CheckCostMasterFunction();
		float CheckSuccessRateMasterFunction();



	};

}