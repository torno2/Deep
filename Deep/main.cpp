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
		Timer t;

		//DataFormating start
		t.Start();

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
		pr("Data formating time:" << t.Stop() << "s");

		//Data Formating end





		std::ofstream efile;
		efile.open("Experiment.txt");


	#if 2<3
		unsigned int testlayoutCount = 3;
		FunctionsLayout testFuncLayout;
		{

			testFuncLayout.NeuronFunctions = new FunctionsLayout::NeuronFunction[testlayoutCount - 1];
			{
				testFuncLayout.NeuronFunctions[0].f = Math::Sigmoid;
				testFuncLayout.NeuronFunctions[1].f = Math::Sigmoid;
				;
			}
			testFuncLayout.NeuronFunctionsDerivatives = new FunctionsLayout::NeuronFunction[testlayoutCount - 1];
			{
				testFuncLayout.NeuronFunctionsDerivatives[0].f = Math::SigmoidDerivative;
				testFuncLayout.NeuronFunctionsDerivatives[1].f = Math::SigmoidDerivative;

			}

			testFuncLayout.FeedForwardCallBackFunctions = new FunctionsLayout::NetworkRelayFunction[testlayoutCount - 1];
			{
				testFuncLayout.FeedForwardCallBackFunctions[0].f = LayerFunctions::FullyConnectedFeedForward;
				testFuncLayout.FeedForwardCallBackFunctions[1].f = LayerFunctions::FullyConnectedFeedForward;

			}

			testFuncLayout.BackPropegateCallBackFunctionsZ = new FunctionsLayout::NetworkRelayFunction[testlayoutCount - 2];
			{
				testFuncLayout.BackPropegateCallBackFunctionsZ[0].f = LayerFunctions::FullyConnectedBackpropegateZ;


			}

			testFuncLayout.BackPropegateCallBackFunctionsBW = new FunctionsLayout::NetworkRelayFunction[testlayoutCount - 1];
			{
				testFuncLayout.BackPropegateCallBackFunctionsBW[0].f = LayerFunctions::FullyConnectedBackpropegateBW;
				testFuncLayout.BackPropegateCallBackFunctionsBW[1].f = LayerFunctions::FullyConnectedBackpropegateBW;
			}



			testFuncLayout.CostFunction.f = CostFunctions::CrossEntropy;
			testFuncLayout.CostFunctionDerivative.f = CostFunctions::CrossEntropyDerivative;


			testFuncLayout.RegularizationFunction.f = TrainingFunctions::L2Regularization;
			testFuncLayout.TrainingFunction.f = TrainingFunctions::GradientDecent;
		}


		t.Start();
		unsigned experiments = 0;

		unsigned i = 1;
		while (i < 21)
		{
			
			LayerLayout* testLayout = new LayerLayout[testlayoutCount];
			{
				testLayout[0].Nodes = 28 * 28;
				testLayout[0].Biases = 0;
				testLayout[0].Weights = 0;

				testLayout[1].Nodes = 5*i;
				testLayout[1].Biases = testLayout[1].Nodes;
				testLayout[1].Weights = testLayout[1].Nodes * testLayout[0].Nodes;

				testLayout[2].Nodes = 10;
				testLayout[2].Biases = testLayout[2].Nodes;
				testLayout[2].Weights = testLayout[2].Nodes * testLayout[1].Nodes;


			}

			experiments++;
			efile << "Experiment: " << experiments << '\n' << '\n';
			

			
			
			efile << "Nodes in hidden layer: " << testLayout[1].Nodes << '\n' << '\n';




			efile << "Testing network using different learning rates: " << '\n' << '\n';
			unsigned j = 1;
			while (j < 11)
			{
				NetworkPrototype testN(testLayout, testFuncLayout, testlayoutCount);

				HyperParameters testparams;
				{
					testparams.Epochs = 1;

					testparams.BatchCount = 10;
					testparams.LearningRate = 0.002f*j;
					testparams.RegularizationConstant = 0.01f;
				}

				
				
				

				testN.Train(data, testparams);

				efile << j << " : " << testparams.LearningRate << '\n' << "Results: | Cost: " << testN.CheckCost() << " |  Success Rate: " << testN.CheckSuccessRate() << '\n' << '\n';

				j++;


				efile << "Training duration: " << testN.m_LastTime << '\n' << '\n';
			}
			
			



			delete[] testLayout;
			

			i++;
			
			
		}

		efile.close();

		testFuncLayout.DestroyFunctionsLayout();
		pr("Experiments completed. Duration: " << t.Stop() << "s");

	#endif



		
	}

#endif



	std::cin.get();
}



