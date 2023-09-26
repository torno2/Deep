#include "pch.h"
#include "NeuralNetwork.h"
#include "NeuralNetworkMT.h"
#include "NeuralNetworkMT.cpp"
#include "DataProcessing.h"
#include "NetworkPrototype.h"
#include "NetworkPrototypeMT.h"
#include "LayerFunctions.h"
#include "LayerFunctionsMT.h"

#define clean false
#define setup true
#define threadN 10

#define TestData false
#define MNIST true

#define train true


#define FullyConnected false
#define Convolution !FullyConnected && false 
#define Test true


#define testPerformance true
#define testPrototype testPerformance && true
#define testMultithread testPerformance && false
#define testOld testPerformance && true
#define testOldMT testOld && false






int main() {





#if clean == false
	{






#if setup  
		using namespace TNNT;
		Timer t;

		//DataFormating start
		t.Start();

#if TestData
		DataSet data;
		{
			constexpr unsigned dummyCount = 3;
			constexpr unsigned dummyRepeat = 700;
			constexpr unsigned testNum = dummyCount* dummyRepeat;
			constexpr unsigned labelSize = 2;
			constexpr unsigned inputSize = 5*5;

			float dummyLabels[labelSize * dummyCount] =
			{
				0,1,

				1,0,

				0,1
			};

			float dummyInpiuts[inputSize * dummyCount] =
			{
					0, 1, 0, 0, 0,
					0, 1, 0, 1, 0,
					1, 0, 0, 0, 0,
					0, 1, 0, 0, 0,
					0, 1, 0, 0, 0,

					0, 0, 0, 0, 0,
					0, 1, 0, 0, 0,
					0, 0, 0, 0, 0,
					0, 1, 0, 0, 0,
					0, 0, 1, 0, 0,

					0, 0, 0, 0, 0,
					0, 0, 0, 1, 0,
					0, 0, 0, 0, 0,
					0, 0, 0, 0, 1,
					0, 0, 0, 1, 0
			};


			data.TrainingCount = testNum;
			data.TrainingInputs = new float[data.TrainingCount * inputSize];
			data.TraningTargets = new float[data.TrainingCount * labelSize];
			

			data.TestCount = testNum;
			data.TestInputs = new float[data.TestCount * inputSize];
			data.TestTargets = new float[data.TestCount * labelSize];

			
			for (int i = 0; i < dummyRepeat; i++)
			{
				memcpy(&data.TrainingInputs[i * inputSize * dummyCount], dummyInpiuts, sizeof(float) * inputSize * dummyCount);
				memcpy(&data.TraningTargets[i * labelSize * dummyCount], dummyLabels, sizeof(float) * labelSize * dummyCount);

				memcpy(&data.TestInputs[i * inputSize * dummyCount], dummyInpiuts, sizeof(float) * inputSize * dummyCount);
				memcpy(&data.TestTargets[i * labelSize * dummyCount], dummyLabels, sizeof(float) * labelSize * dummyCount);
			}

			pr(data.TestInputs[inputSize * testNum - 2]);



			data.ValidationCount = 1;
			data.ValidationInputs = new float[data.ValidationCount * inputSize];
			data.ValidationTargets = new float[data.ValidationCount * labelSize];
	
		}
#endif

#if MNIST

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
#endif 


		pr("Data formating time:" << t.Stop() << "s");

		//Data Formating end



		//Network setup start

		unsigned int layoutCount = 3;
		unsigned int* layout = new unsigned int[layoutCount];
		{
			layout[0] = 28*28;
			layout[1] = 10;
			layout[2] = 10;
		}		
		unsigned int* layoutMT = new unsigned int[layoutCount];
		{
			layoutMT[0] = 784;
			layoutMT[1] = 10;
			layoutMT[2] = 10;
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
		LayerFucntionsLayout funcLayoutMT;
		{
			funcLayoutMT.CostFunction = Math::CrossEntropy;
			funcLayoutMT.CostFunctionDerivative = Math::CrossEntropyCostDerivative;
			funcLayoutMT.NeuronFunction = new LayerFucntionsLayout::NeuronFunctionPointer[layoutCount - 1];
			for (int i = 0; i < layoutCount - 1; i++)
			{
				funcLayoutMT.NeuronFunction[i].Function = Math::Sigmoid;
			}

			funcLayoutMT.NeuronFunctionDerivative = new LayerFucntionsLayout::NeuronFunctionPointer[layoutCount - 2];
			for (int i = 0; i < layoutCount - 2; i++)
			{
				funcLayoutMT.NeuronFunctionDerivative[i].Function = Math::SigmoidDerivative;
			}
		}


		NeuralNetwork nOld(layout, layoutCount, funcLayout, true);

		NeuralNetworkMT nOldMT(layoutMT, layoutCount, funcLayoutMT, threadN);

		//Network setup stop


		//Prototype Network setup start


	#if FullyConnected
		
		unsigned int playoutCount = 3;
		LayerLayout* pLayout = new LayerLayout[playoutCount];
		{
			pLayout[0].Nodes = 28 * 28;
			pLayout[0].Biases = 0;
			pLayout[0].Weights = 0;

			pLayout[1].Nodes = 10;
			pLayout[1].Biases = pLayout[1].Nodes;
			pLayout[1].Weights = pLayout[1].Nodes * pLayout[1-1].Nodes;

			pLayout[2].Nodes = 10;
			pLayout[2].Biases = pLayout[2].Nodes;
			pLayout[2].Weights = pLayout[2].Nodes * pLayout[2-1].Nodes;


		}

		FunctionsLayout pFuncLayout;
		{

			pFuncLayout.NeuronFunctions = new FunctionsLayout::NeuronFunction[playoutCount - 1];
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


			pFuncLayout.RegularizationFunctions.f = TrainingFunctions::L2Regularization;
			pFuncLayout.TrainingFunctions.f = TrainingFunctions::GradientDecent;
		}
		
	#endif

	#if Convolution
		unsigned int playoutCount = 4;
		LayerLayout* pLayout = new LayerLayout[playoutCount];
		{
			pLayout[0].Nodes = 28 * 28;
			pLayout[0].Biases = 0;
			pLayout[0].Weights = 0;

			pLayout[1].Nodes = 24*24*3;
			pLayout[1].Biases = 1*3;
			pLayout[1].Weights = 5*3;

			pLayout[2].Nodes = 12*12*3;
			pLayout[2].Biases =0;
			pLayout[2].Weights = 0;			
			
			
			pLayout[3].Nodes = 10;
			pLayout[3].Biases = pLayout[3].Nodes;
			pLayout[3].Weights = pLayout[3].Nodes * pLayout[2].Nodes;			


		}

		FunctionsLayout pFuncLayout;
		{

			pFuncLayout.NeuronFunctions = new FunctionsLayout::NeuronFunction[playoutCount - 1];
			{
				pFuncLayout.NeuronFunctions[0].f = Math::Sigmoid;
				pFuncLayout.NeuronFunctions[1].f = Math::Identity;
				pFuncLayout.NeuronFunctions[2].f = Math::Sigmoid;
				
			}
			pFuncLayout.NeuronFunctionsDerivatives = new FunctionsLayout::NeuronFunction[playoutCount - 1];
			{
				pFuncLayout.NeuronFunctionsDerivatives[0].f = Math::SigmoidDerivative;
				pFuncLayout.NeuronFunctionsDerivatives[1].f = Math::IdentityDerivative;
				pFuncLayout.NeuronFunctionsDerivatives[2].f = Math::SigmoidDerivative;

			}

			pFuncLayout.FeedForwardCallBackFunctions = new FunctionsLayout::NetworkRelayFunction[playoutCount - 1];
			{
				pFuncLayout.FeedForwardCallBackFunctions[0].f = LayerFunctions::ConvolutionLayerFeedForward;
				pFuncLayout.FeedForwardCallBackFunctions[1].f = LayerFunctions::PoolingLayerFeedForward;
				pFuncLayout.FeedForwardCallBackFunctions[2].f = LayerFunctions::FullyConnectedFeedForward;

			}

			pFuncLayout.BackPropegateCallBackFunctionsZ = new FunctionsLayout::NetworkRelayFunction[playoutCount - 2];
			{
				pFuncLayout.BackPropegateCallBackFunctionsZ[1].f = LayerFunctions::PoolingLayerBackpropegateZ;
				pFuncLayout.BackPropegateCallBackFunctionsZ[0].f = LayerFunctions::FullyConnectedBackpropegateZ;


			}

			pFuncLayout.BackPropegateCallBackFunctionsBW = new FunctionsLayout::NetworkRelayFunction[playoutCount - 1];
			{
				pFuncLayout.BackPropegateCallBackFunctionsBW[2].f = LayerFunctions::ConvolutionLayerBackpropegateBW;
				pFuncLayout.BackPropegateCallBackFunctionsBW[1].f = LayerFunctions::PoolingLayerBackpropegateBW;
				pFuncLayout.BackPropegateCallBackFunctionsBW[0].f = LayerFunctions::FullyConnectedBackpropegateBW;
			}



			pFuncLayout.CostFunction.f = CostFunctions::CrossEntropy;
			pFuncLayout.CostFunctionDerivative.f = CostFunctions::CrossEntropyDerivative;


			pFuncLayout.RegularizationFunction.f = TrainingFunctions::L2Regularization;
			pFuncLayout.TrainingFunction.f = TrainingFunctions::GradientDecent;
		}
	#endif


#if Test
		unsigned int playoutCount = 3;
		LayerLayout* pLayout = new LayerLayout[playoutCount];
		{
			pLayout[0].Nodes = 28*28;
			pLayout[0].Biases = 0;
			pLayout[0].Weights = 0;

			pLayout[1].Nodes = 10;
			pLayout[1].Biases = pLayout[1].Nodes;
			pLayout[1].Weights = pLayout[1].Nodes * pLayout[1-1].Nodes;

			pLayout[2].Nodes = 10;
			pLayout[2].Biases = pLayout[2].Nodes;
			pLayout[2].Weights = pLayout[2].Nodes * pLayout[2-1].Nodes;



		}

		FunctionsLayout pFuncLayout;
		{

			pFuncLayout.NeuronFunctions = new FunctionsLayout::NeuronFunction[playoutCount - 1];
			{
				pFuncLayout.NeuronFunctions[0].f = Math::Sigmoid;
				pFuncLayout.NeuronFunctions[1].f = Math::Sigmoid;




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


				pFuncLayout.BackPropegateCallBackFunctionsBW[1].f = LayerFunctions::FullyConnectedBackpropegateBW;
				pFuncLayout.BackPropegateCallBackFunctionsBW[0].f = LayerFunctions::FullyConnectedBackpropegateBW;
			}



			pFuncLayout.CostFunction.f = CostFunctions::CrossEntropy;
			pFuncLayout.CostFunctionDerivative.f = CostFunctions::CrossEntropyDerivative;


#if OLD
			pFuncLayout.RegularizationFunctions.f= TrainingFunctions::L2Regularization;
			pFuncLayout.TrainingFunctions.f = TrainingFunctions::GradientDecent;
#endif

#if NEW
			pFuncLayout.RegularizationFunctions = new FunctionsLayout::NetworkRelayFunction[playoutCount - 1];
			{
				pFuncLayout.RegularizationFunctions[0].f = TrainingFunctions::L2Regularization;
				pFuncLayout.RegularizationFunctions[1].f = TrainingFunctions::L2Regularization;

			}

				
			pFuncLayout.TrainingFunctions = new FunctionsLayout::NetworkRelayFunction[playoutCount - 1];
			{
				pFuncLayout.TrainingFunctions[0].f = TrainingFunctions::GradientDecent;
				pFuncLayout.TrainingFunctions[1].f = TrainingFunctions::GradientDecent;

			}
#endif

		}
#endif




		NetworkPrototype nP(pLayout, pFuncLayout, playoutCount, true);

		//Prototype Network setup stop



		// MT prototype setup start
#if FullyConnected
		unsigned int MTlayoutCount = 3;
		LayerLayout* MTlayout = new LayerLayout[playoutCount];
		{
			MTlayout[0].Nodes = 28 * 28;
			MTlayout[0].Biases = 0;
			MTlayout[0].Weights = 0;

			MTlayout[1].Nodes = 10;
			MTlayout[1].Biases = MTlayout[1].Nodes;
			MTlayout[1].Weights = MTlayout[1].Nodes * MTlayout[0].Nodes;

			MTlayout[2].Nodes = 10;
			MTlayout[2].Biases = MTlayout[2].Nodes;
			MTlayout[2].Weights = MTlayout[2].Nodes* MTlayout[1].Nodes;


	


		}

		FunctionsLayoutMT mtFuncLayout;
		{

			mtFuncLayout.NeuronFunctions = new FunctionsLayoutMT::NeuronFunction[playoutCount - 1];
			{
				mtFuncLayout.NeuronFunctions[0].f = Math::Sigmoid;
				mtFuncLayout.NeuronFunctions[1].f = Math::Sigmoid;
				;
			}
			mtFuncLayout.NeuronFunctionsDerivatives = new FunctionsLayoutMT::NeuronFunction[playoutCount - 1];
			{
				mtFuncLayout.NeuronFunctionsDerivatives[0].f = Math::SigmoidDerivative;
				mtFuncLayout.NeuronFunctionsDerivatives[1].f = Math::SigmoidDerivative;

			}

			mtFuncLayout.FeedForwardCallBackFunctions = new FunctionsLayoutMT::NetworkRelayFunctionMT [playoutCount - 1] ;
			{
				mtFuncLayout.FeedForwardCallBackFunctions[0].f = LayerFunctionsMT::FullyConnectedFeedForward;
				mtFuncLayout.FeedForwardCallBackFunctions[1].f = LayerFunctionsMT::FullyConnectedFeedForward;

			}

			mtFuncLayout.BackPropegateCallBackFunctionsZ = new FunctionsLayoutMT::NetworkRelayFunctionMT[playoutCount - 2];
			{
				mtFuncLayout.BackPropegateCallBackFunctionsZ[0].f = LayerFunctionsMT::FullyConnectedBackpropegateZ;


			}

			mtFuncLayout.BackPropegateCallBackFunctionsBW = new FunctionsLayoutMT::NetworkRelayFunctionMT[playoutCount - 1];
			{
				mtFuncLayout.BackPropegateCallBackFunctionsBW[1].f = LayerFunctionsMT::FullyConnectedBackpropegateBW;
				mtFuncLayout.BackPropegateCallBackFunctionsBW[0].f = LayerFunctionsMT::FullyConnectedBackpropegateBW;
			}



			mtFuncLayout.CostFunction.f = CostFunctionsMT::CrossEntropy;
			mtFuncLayout.CostFunctionDerivative.f = CostFunctionsMT::CrossEntropyDerivative;


			mtFuncLayout.RegularizationFunction.f = TrainingFunctionsMT::L2Regularization;
			mtFuncLayout.TrainingFunction.f = TrainingFunctionsMT::GradientDecent;
		}

		NetworkPrototypeMT nMT(MTlayout,mtFuncLayout, MTlayoutCount, threadN);
		// MT prototype setup stop

#endif


		HyperParameters params;
		{
			params.Epochs = 1;

			params.BatchCount = 10;
			params.LearningRate = 0.01f;
			params.RegularizationConstant = 0.01f;
		}


		HyperParameters paramsMT;
		{
			paramsMT.Epochs = 1;

			paramsMT.BatchCount = 10;
			paramsMT.LearningRate = 0.01f;
			paramsMT.RegularizationConstant = 0.01f;
		}
#endif

		unsigned missmatch = 0;
		for (int i = 0; i < nOld.m_WeightsCount; i++)
		{
			
			if (nOld.m_Weights[i] != nP.m_Weights[i])
			{
				missmatch++;
			}

		}
		pr("Missmatches: " << missmatch);


#if testPerformance
#if testOld
		pr("Old: ");
		{
			//Training start
			t.Start();


			nOld.Train(data, params);
			pr("Train time: " << t.Stop() << "s");
	
			//Traning End


			//Check Start

			t.Start();
			pr("Cost: " << nOld.CheckCost(data));
			pr("Guessrate: " << nOld.CheckSuccessRate(data));

			pr("Check time: " << t.Stop() << "s");

			//Check Stop
		}
#endif
#if testOldMT && testMultithread
		pr("OldMT: ");
		{
			//Training start
			t.Start();

			nOldMT.Train(data, paramsMT);

			pr("Train time: " << t.Stop() << "s");

			//Traning End


			//Check Start

			t.Start();
			pr("Cost: " << nOldMT.CheckCost(data));
			pr("Guessrate: " << nOldMT.CheckSuccessRate(data));

			pr("Check time: " << t.Stop() << "s");

			//Check Stop
		}
#endif

#if testPrototype
		pr("New");
		pr("Prototype: ");
		{

			//Training start
			t.Start();
			nP.SetData(&data);


	#if train
			nP.Train(&data, params);
				
			pr("Train time: " << t.Stop() << "s");
	#endif
			//Traning End

			
			//Check Start
			t.Start();

			pr("Cost: " << nP.CheckCost());
			pr("Guessrate: " << nP.CheckSuccessRate());

			//PArr<float>(nP.m_OutputBuffer, nP.m_OutputBufferCount);

			pr("Check time: " << t.Stop() << "s");
			//Check Stop
		}
#endif

#if testMultithread

		pr("MT");
		pr("Prototype: ");
		{

			//Training start
			t.Start();
			nMT.SetData(&data);
	#if train
			nMT.Train(&data, paramsMT);
	
		
			pr("Train time: " << t.Stop() << "s");
	#endif
			//Traning End
			

			t.Start();

			pr("Cost: " << nMT.CheckCost());
			pr("Guessrate: " << nMT.CheckSuccessRate());


			pr("Check time: " << t.Stop() << "s");
			//Check Stop
		}
#endif

#endif

		


#if false
		std::ofstream efile;
		efile.open("Experiment.txt");


	
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
		while (i < 3)
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
			while (j < 4)
			{
				NetworkPrototype testN(testLayout, testFuncLayout, testlayoutCount);

				HyperParameters testparams;
				{
					testparams.Epochs = 1;

					testparams.BatchCount = 10;
					testparams.LearningRate = 0.002f*j;
					testparams.RegularizationConstant = 0.01f;
				}

				unsigned i = 0;
				
				

				testN.Train(&data, testparams);

				efile << j << " : " << testparams.LearningRate << '\n' << "Results: | Cost: " << testN.CheckCost() << " |  Success Rate: " << testN.CheckSuccessRate() << '\n' << '\n';

				j++;


				efile << "Training duration: " << testN.m_LastTime[0] << '\n' << '\n';
			}
			
			

			delete[] testLayout;
			

			i++;
			
			
		}

		efile.close();

		
		pr("Experiments completed. Duration: " << t.Stop() << "s");

	#endif

		unsigned* checkThis = new unsigned[nOld.m_WeightsCount];

		missmatch = 0;
		for (int i = 0; i < nOld.m_WeightsCount; i++)
		{

			if (nOld.m_Weights[i] != nP.m_Weights[i])
			{
				checkThis[missmatch] = i;
				missmatch++;
				
			}

		}
		pr("Missmatches: " << missmatch);

		unsigned champ = 0;
		for (int i = 0; i < missmatch; i++)
		{
			if (checkThis[i] > champ + 1)
			{
				champ = checkThis[i];
				pr(checkThis[i]);
			}
			else
			{
				champ = checkThis[i];
			}
		}

	}

#endif





	std::cin.get();
}



