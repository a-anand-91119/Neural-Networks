package com.anand.neuralnetwork.driver;

import com.anand.neuralnetwork.Perceptron;

public class DriverClass {

	public static void main(String[] args) {
		
		int[][][] data = Perceptron.orData;
		double[] weights = Perceptron.INITIAL_WEIGHTS;
		
		DriverClass driverClass = new DriverClass();
		Perceptron perceptron = new Perceptron();

		boolean errorFlag = true;
		int epochNumber = 0;
		double error = 0;
		double[] adjustedWeights = null;
		while (errorFlag) {
			
			driverClass.printHeading(epochNumber++);
			
			errorFlag = false;
			error = 0;
			
			for (int i = 0; i < data.length; i++) {
				double weightedSum = perceptron.calculateWeightedSum(data[i][0], weights);
				int result = perceptron.applyActivationFunction(weightedSum);
				
				error = data[i][1][0] - result;
				
				if (error != 0)
					errorFlag = true;
				
				adjustedWeights = perceptron.adjustWeights(data[i][0], weights, error);
				
				driverClass.printVector(data[i], weights, result, error, weightedSum, adjustedWeights);
				
				weights = adjustedWeights;
			}
		}
	}
	
	public void printHeading(int epochNumber) {
		System.out.println("\n=============================================================================="
				+ "======================== Epoch # "+epochNumber+" =================================="
						+ "==================================================================");
		System.out.println("	w1	|	w2	|	w3	|	x1	|	x2	|	x3	|Target Result	"
				+ "|	Result	|	Error	|Weighted Sum	|adjusted w1	|adjusted w2	|adjusted w3	|");
		System.out.println("--------------------------------------------------"
				+ "------------------------------------------------------------------------------"
				+ "--------------------------------------------------------------------------------------");
	}
	
	public void printVector(int[][] data, double[] weights, int result, double error, double weightedSum, double[] adjustedWeights) {
		System.out.println("	" + 
				String.format("%.2f", weights[0]) + "	|	" + 
				String.format("%.2f", weights[1]) + "	|	" +
				String.format("%.2f", weights[2]) + "	|	" +
				data[0][0] + "	|	" + 
				data[0][1] + "	|	" + 
				data[0][2] + "	|	" + 
				data[1][0] + "	|	" + 
				result + "	|	" + 
				error + "	|	" + 
				String.format("%.2f", weightedSum) + "	|	" + 
				String.format("%.2f", adjustedWeights[0]) + "	|	" + 
				String.format("%.2f", adjustedWeights[1]) + "	|	" +
				String.format("%.2f", adjustedWeights[2]) + "	|	");
	}
}
