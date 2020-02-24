package com.anand.network.driver;

import java.util.stream.IntStream;

import com.anand.network.Perceptron;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.ScatterChart;
import javafx.scene.chart.XYChart;
import javafx.scene.chart.XYChart.Series;
import javafx.scene.layout.Pane;
import javafx.stage.Stage;

public class DriverClass extends Application{
	
	static double m = 0.0;
	static double c = 0.0;
	// y = mx + b
	public static void main(String[] args) {
		int[][][] trainingData = Perceptron.andData;
		double[] weights = Perceptron.INITITAL_WEIGHTS;
		
		DriverClass driverClass = new DriverClass();
		Perceptron perceptron = new Perceptron();
		int epochNumber = 0;
		boolean errorFlag = true;
		double error = 0.0;
		double[] adjustedWeights = null;
		
		while(errorFlag) {
			driverClass.printHeading(epochNumber++);
			error = 0.0;
			errorFlag = false;
			for (int i = 0; i < trainingData.length; i++) {
				double weightedSum = perceptron.calulateWeightedSum(trainingData[i][0], weights);
				int result = perceptron.applyActivationFunction(weightedSum);
				error = trainingData[i][1][0] - result;
				if(error != 0)
					errorFlag = true;
				adjustedWeights = perceptron.calculateAdjustedWeights(trainingData[i][0], weights, error);
				driverClass.printVector(trainingData[i], weights, result, error, weightedSum, adjustedWeights);
				weights = adjustedWeights;
			}
		}
		m = (-weights[0] / weights[1]);
		c = (Perceptron.THRESHOLD) / weights[1];
		System.out.println("\n y = " + String.format("%.2f", m) + "x + " + String.format("%.2f", c));
		launch(args);
	}
	
	@Override
	public void start(Stage stage) throws Exception {
		stage.setTitle("Perceptron Visualization");
		
		XYChart.Series<Number, Number> series01 = new XYChart.Series<Number, Number>();
		XYChart.Series<Number, Number> series02 = new XYChart.Series<Number, Number>();
		series01.setName("Zero");
		series02.setName("One");
		
		IntStream.range(0, Perceptron.andData.length).forEach(i -> {
			int x = Perceptron.andData[i][0][0];
			int y = Perceptron.andData[i][0][1];
			if(Perceptron.andData[i][1][0] == 0)
				series01.getData().add(new XYChart.Data<Number, Number>(x, y));
			else
				series02.getData().add(new XYChart.Data<Number, Number>(x, y));
		});
		
		double x1 = 0;
		double y1 = c;
		double x2 = -(c/ m);
		double y2 = 0;
		
		String title = new String("y = " + String.format("%.2f", m) + "x + " + String.format("%.2f", c) + " | (0, " + 
				String.format("%.2f", c) + ") | (" + String.format("%.2f", (-c/m))+ ", 0)");
		
		NumberAxis xAxis = new NumberAxis(0, 4, 1);
		NumberAxis yAxis = new NumberAxis(0, 4, 1);
		
		ScatterChart<Number, Number> scatterChart = new ScatterChart<>(xAxis, yAxis);
		scatterChart.setTitle(title);
		scatterChart.getData().add(series01);
		scatterChart.getData().add(series02);
		
		XYChart.Series<Number, Number> series03 = new XYChart.Series<Number, Number>();
		series03.getData().add(new XYChart.Data<Number, Number>(x1, y1));
		series03.getData().add(new XYChart.Data<Number, Number>(x2, y2));
		
		LineChart<Number, Number> lineChart = new LineChart<Number, Number>(xAxis, yAxis);
		lineChart.getData().add(series03);
		lineChart.setOpacity(0.1);
		
		Pane pane = new Pane();
		pane.getChildren().addAll(scatterChart, lineChart);
		
		stage.setScene(new Scene(pane, 500, 500));
		stage.show();
	}

	public void printHeading(int epochNumber) {
		System.out.println("\n=================================================="
				+ "======================== Epoch # "+epochNumber+" =================================="
						+ "======================================");
		System.out.println("	w1	|	w2	|	x1	|	x2	|Target Result	"
				+ "|	Result	|	Error	|Weighted Sum	|adjusted w1	|adjusted w2	|");
		System.out.println("--------------------------------------------------"
				+ "------------------------------------------------------------------------------"
				+ "---------------------------------");
	}
	
	public void printVector(int[][] data, double[] weights, int result, double error, double weightedSum, double[] adjustedWeights) {
		System.out.println("	" + 
				String.format("%.2f", weights[0]) + "	|	" + 
				String.format("%.2f", weights[1]) + "	|	" +
				data[0][0] + "	|	" + 
				data[0][1] + "	|	" + 
				data[1][0] + "	|	" + 
				result + "	|	" + 
				error + "	|	" + 
				String.format("%.2f", weightedSum) + "	|	" + 
				String.format("%.2f", adjustedWeights[0]) + "	|	" + 
				String.format("%.2f", adjustedWeights[1]) + "	|	");
	}
}
