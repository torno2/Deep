#pragma once
#include "NeuralNetworkCustomVariables.h"

namespace TNNT {

	class NetworkPrototype
	{
	public:

		FunctionsLayout m_Functions;

		PositionData m_PositionData;

		LayerLayout* m_LayerLayout;
		unsigned m_LayerLayoutCount;

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
		float* m_WeightsTranspose;
		float* m_DeltaWeights;
		float* m_WeightsBuffer;
		unsigned m_WeightsCount;

		float* m_TargetBuffer;
		float m_CostBuffer;
		
		
		HyperParameters m_HyperParameters;
		
		float* m_InternalInputBuffer = nullptr;
		float* m_InternalTargetBuffer = nullptr;
		
		

		DataSet* m_Data;


	public:

		NetworkPrototype(LayerLayout* layerLayout, FunctionsLayout& functions, unsigned layoutCount , bool randomizeWeightsAndBiases = true);
		~NetworkPrototype();

		float CheckSuccessRate();
		float CheckCost();

		void Train(DataSet& data, HyperParameters& params);





	public:


		void SetBiasesToTemp();
		void SetTempToBiases();

		void SetWeightsToTemp();
		void SetTempToWeights();
		void ResetWeightsTranspose();

		void SetData(DataSet* data);
		void SetHyperParameters(HyperParameters& params);

		void SetInput(const float* input);
		void SetTarget(const float* target);

		void FeedForward();
		void Backpropegate();

		void TrainOnSet(unsigned num);

		void TrainMasterFunction();
		
		float CheckCostMasterFunction();
		float CheckSuccessRateMasterFunction();



	};

}