#include "pch.h"
#include "NeuralNetwork.h"
#include "NeuralNetworkMT.h"
#include "NeuralNetworkMT.cpp"
#include "DataProcessing.h"
#include "NetworkPrototype.h"
#include "LayerFunctions.h"


int main() {


#if 2 < 3
	{
		using namespace TNNT;


		//DataFormating start
		auto start = std::chrono::high_resolution_clock::now();

		DataSet data;
		{
			constexpr unsigned labelSize = 10;
			constexpr unsigned inputSize = 28 * 28;

			data.TrainingCount = 50000;
			data.TrainingInputs = new float[data.TrainingCount * inputSize];
			data.TraningTargets = new float[data.TrainingCount * labelSize];

			data.ValidationCount = 10000;
			data.ValidationInputs = new float[data.ValidationCount * inputSize];
			data.ValidationTargets = new float[data.ValidationCount * labelSize];

			data.TestCount = 10000;
			data.TestInputs = new float[data.TestCount * inputSize];
			data.TestTargets = new float[data.TestCount * labelSize];



			//Data Formating start

			ProcessMNISTDataMT(10, data.TrainingInputs, data.TraningTargets, "trainLabel.idx1-ubyte", "trainIm.idx3-ubyte", data.TrainingCount);
			ProcessMNISTDataMT(10, data.ValidationInputs, data.ValidationTargets, "trainLabel.idx1-ubyte", "trainIm.idx3-ubyte", data.ValidationCount, 50000);
			ProcessMNISTDataMT(10, data.TestInputs, data.TestTargets, "testLabel.idx1-ubyte", "testIm.idx3-ubyte", data.TestCount);


		}


		auto stop = std::chrono::high_resolution_clock::now();
		std::chrono::duration<float>  time = stop - start;
		pr("Data formating time:" << time.count() << "s");
		//Data Formating end






		//Network setup start

		start = std::chrono::high_resolution_clock::now();

		unsigned int layoutCount = 3;
		unsigned int* layout = new unsigned int[layoutCount];
		{
			layout[0] = 784;
			layout[1] = 300;
			layout[2] = 10;
		}

		LayerFucntionsLayout funcLayout;
		{
			funcLayout.CostFunction = Math::CrossEntropy;
			funcLayout.CostFunctionDerivative = Math::CrossEntropyCostDerivative;
			funcLayout.NeuronFunction = new LayerFucntionsLayout::NeuronFunctionPointer[layoutCount - 1];
			for (int i = 0; i < layoutCount - 1; i++)
			{
				funcLayout.NeuronFunction[i].Function = Math::Sigmoid;
			}

			funcLayout.NeuronFunctionDerivative = new LayerFucntionsLayout::NeuronFunctionPointer[layoutCount - 2];
			for (int i = 0; i < layoutCount - 2; i++)
			{
				funcLayout.NeuronFunctionDerivative[i].Function = Math::SigmoidDerivative;
			}
		}


		NeuralNetwork n(layout,layoutCount,funcLayout, true);

		//Network setup stop




		

		//Prototype Network setup start

		unsigned int playoutCount = 3;
		LayerLayout* pLayout = new LayerLayout[playoutCount];
		{
			pLayout[0].Nodes = 28 * 28;
			pLayout[0].Biases = 0;
			pLayout[0].Weights = 0;
			pLayout[0].WeightsRowCount = 0;

			pLayout[1].Nodes = 300;
			pLayout[1].Biases = pLayout[1].Nodes;
			pLayout[1].Weights = pLayout[1].Nodes* pLayout[0].Nodes;
			pLayout[1].WeightsRowCount = pLayout[0].Nodes;

			pLayout[2].Nodes = 10;
			pLayout[2].Biases = pLayout[2].Nodes;
			pLayout[2].Weights = pLayout[2].Nodes* pLayout[1].Nodes;
			pLayout[2].WeightsRowCount = pLayout[1].Nodes;


		}

		FunctionsLayout pFuncLayout;
		{

			pFuncLayout.NeuronFunctions = new FunctionsLayout::NeuronFunction[playoutCount -1];
			{
				pFuncLayout.NeuronFunctions[0].f = Math::Sigmoid;
				pFuncLayout.NeuronFunctions[1].f = Math::Sigmoid;
			;
			}
			pFuncLayout.NeuronFunctionsDerivatives = new FunctionsLayout::NeuronFunction[playoutCount - 1];
			{
				pFuncLayout.NeuronFunctionsDerivatives[0].f = Math::SigmoidDerivative;
				pFuncLayout.NeuronFunctionsDerivatives[1].f = Math::SigmoidDerivative;
				
			}

			pFuncLayout.FeedForwardCallBackFunctions = new FunctionsLayout::NetworkRelayFunction[playoutCount - 1];
			{
				pFuncLayout.FeedForwardCallBackFunctions[0].f = LayerFunctions::FullyConnectedFeedForward;
				pFuncLayout.FeedForwardCallBackFunctions[1].f = LayerFunctions::FullyConnectedFeedForward;
				
			}

			pFuncLayout.BackPropegateCallBackFunctionsZ = new FunctionsLayout::NetworkRelayFunction[playoutCount - 2];
			{
				pFuncLayout.BackPropegateCallBackFunctionsZ[0].f = LayerFunctions::FullyConnectedBackpropegateZ;
				
				
			}

			pFuncLayout.BackPropegateCallBackFunctionsBW = new FunctionsLayout::NetworkRelayFunction[playoutCount - 1];
			{
				pFuncLayout.BackPropegateCallBackFunctionsBW[0].f = LayerFunctions::FullyConnectedBackpropegateBW;
				pFuncLayout.BackPropegateCallBackFunctionsBW[1].f = LayerFunctions::FullyConnectedBackpropegateBW;
			}



			pFuncLayout.CostFunction.f = CostFunctions::CrossEntropy;
			pFuncLayout.CostFunctionDerivative.f = CostFunctions::CrossEntropyDerivative;


			pFuncLayout.RegularizationFunction.f = TrainingFunctions::L2Regularization;
			pFuncLayout.TrainingFunction.f = TrainingFunctions::GradientDecent;
		}

		NetworkPrototype pN(pLayout, pFuncLayout , playoutCount, true);

		//Prototype Network setup stop



		stop = std::chrono::high_resolution_clock::now();


		HyperParameters params;
		{
			params.Epochs = 1;

			params.BatchCount = 10;
			params.LearningRate = 0.05f;
			params.RegularizationConstant = 0.01f;
		}


		time = stop - start;
		pr("Network setup time:" << time.count() << "s");
		//Network setup end

		
		pr("New");
		pr("Prototype: ");
		{
			
			//Training start
			start = std::chrono::high_resolution_clock::now();

			pN.Train(data, params);

			stop = std::chrono::high_resolution_clock::now();
			time = stop - start;
			pr("Train time: " << time.count() << "s");
			//Traning End


			//Check Start
			start = std::chrono::high_resolution_clock::now();

			pr("Cost: " << pN.CheckCost());
			pr("Guessrate: " << pN.CheckSuccessRate());

			stop = std::chrono::high_resolution_clock::now();
			time = stop - start;
			pr("Check time: " << time.count() << "s");
			//Check Stop
		}

		pr("Old: ");
		{
			//Training start
			start = std::chrono::high_resolution_clock::now();

			n.Train(data, params);

			stop = std::chrono::high_resolution_clock::now();
			time = stop - start;
			pr("Train time: " << time.count() << "s");
			//Traning End


			//Check Start
			start = std::chrono::high_resolution_clock::now();

			pr("Cost: " << n.CheckCost(data));
			pr("Guessrate: " << n.CheckSuccessRate(data));

			stop = std::chrono::high_resolution_clock::now();
			time = stop - start;
			pr("Check time: " << time.count() << "s");
			//Check Stop
		}



		
	}

#endif



	std::cin.get();
}



