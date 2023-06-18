#pragma once

namespace TNNT
{

	struct LayerLayout
	{
		unsigned Nodes;
		unsigned Biases;
		unsigned Weights;

		unsigned SubLayers = 0;
		
	};


	struct PositionData
	{
		unsigned Layer		= 0;
		unsigned Z			= 0;
		unsigned A			= 0;
		unsigned Biases		= 0;
		unsigned Weights	= 0;
	};



	struct HyperParameters
	{
		//General
		unsigned Epochs;


		//Stochastic Gradient Decent
		unsigned BatchCount;
		float LearningRate;
		float RegularizationConstant;
	};


	//ForwardDeclaration
	class NetworkPrototype;

	struct FunctionsLayout
	{
		
		struct NeuronFunction
		{
			float (*f)(float z);
		};
		struct NetworkRelayFunction
		{
			void (*f)(NetworkPrototype* n);
		};




		//NeuronFunctions and their derivatives
		NeuronFunction* NeuronFunctions;
		//Derivatives here are not in reverse order compared to layers.
		NeuronFunction* NeuronFunctionsDerivatives;



		//Functions that descripe how layers are connected and what operations are preformed on their activations.
		//Count of this array is m_LayerLayoutCount-1, where the first element applies to the zs of the second layer (no zs in the inputlayer) and the last applies to the zs of the outputlayer.
		NetworkRelayFunction* FeedForwardCallBackFunctions;

		//The output layert has its derivative (with repect to z) calculated in the CostFunctionDerivative function, so the Count of this array is m_LayerLayoutCount-2. Function for the last layer is supposed to be first in the array.
		NetworkRelayFunction* BackPropegateCallBackFunctionsZ; //Meant for calculating the entries of m_DeltaZ
		//This one has a count of m_LayerLayoutCount-1
		NetworkRelayFunction* BackPropegateCallBackFunctionsBW; //Meant for calculating the entries of m_DeltaBiases and m_DeltaWeights
		
	
		//Cost function. Its "derivative" is used as the beackpropegation function for the last layer.
		NetworkRelayFunction CostFunction;
		//Is supposed to be the derivative with respects to z in the outputlayer, effectively meaning that this function is: dC/da * da/dz.
		NetworkRelayFunction CostFunctionDerivative;




		//For the trainingprocess
		NetworkRelayFunction TrainingFunction;

		NetworkRelayFunction RegularizationFunction;



		
		void DestroyFunctionsLayout()
		{
			delete[] NeuronFunctions;
			delete[] NeuronFunctionsDerivatives;

			delete[] FeedForwardCallBackFunctions;
			delete[] BackPropegateCallBackFunctionsZ;
			delete[] BackPropegateCallBackFunctionsBW;
		}

	};







	struct DataSet
	{
		float* TrainingInputs;
		float* TraningTargets;
		unsigned TrainingCount;

		float* ValidationInputs;
		float* ValidationTargets;
		unsigned ValidationCount;

		float* TestInputs;
		float* TestTargets;
		unsigned TestCount;



		~DataSet()
		{
			delete[] TrainingInputs;
			delete[] TraningTargets;

			delete[] ValidationInputs;
			delete[] ValidationTargets;

			delete[] TestInputs;
			delete[] TestTargets;
		}
	};

	struct ConditionFunctionPointer
	{
		void (*Function)(float* cost, unsigned step, unsigned& epochs, unsigned& batchSize, float& learningRate, float& regConst, bool* updateOrRevert);
	};



	//TODO: Temporary, remove later.
	struct LayerFucntionsLayout
	{

		//Wrapper for easier array makin'. The neural network is responsible for deleting this.
		struct NeuronFunctionPointer
		{
			float (*Function)(float z);
		};

		float (*CostFunction)(float a, float y);
		//Is supposed to be the derivative with respects to z in the outputlayer, effectively meaning that this function is: dC/da * da/dz.
		float (*CostFunctionDerivative)(float z, float a, float y);

		//size of these arrays are assumed to be of size equal to m_LayerLayoutSize-1, where the first element applies to the zs of the second layer (no zs in the inputlayer) and the last applies to the zs of the outputlayer
		NeuronFunctionPointer* NeuronFunction;
		//The output layert has its derivative used in the cost function derivative, so the size of this array is m_LayerLayoutSize-2;
		NeuronFunctionPointer* NeuronFunctionDerivative;


	};
	

	

}
