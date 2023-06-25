import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;

public class NeuralNet {
    static Layer[] layers;
    static TrainingData[] tDataSet;
	public static Scanner s = new Scanner(System.in);

    public static void main(String[] args) {
		//Neuron.setRangeBias(-1,1);
        Neuron.setRangeWeight(0.1f,0.5f);

		int amount_of_layers = 5;
		int[] amount_of_neurons = new int[amount_of_layers];
		amount_of_neurons[0] = 784;
		amount_of_neurons[1] = 80;
		amount_of_neurons[2] = 64;
        amount_of_neurons[3] = 12;
        amount_of_neurons[4] = 10;

		layers = new Layer[amount_of_layers];
		layers[0] = null; // Input Layer
		for (int i = 1; i < amount_of_layers; i++) {
			layers[i] = new Layer(amount_of_neurons[i - 1], amount_of_neurons[i]);
		}
        
    	CreateTrainingData();

		int check = (int)(Math.random() * tDataSet.length);

        System.out.println("Output before training");
		forward(tDataSet[check].data);
		for (int j = 0; j < layers[amount_of_layers - 1].neurons.length; j++)
			System.out.print(layers[amount_of_layers - 1].neurons[j].value + " ");
		System.out.println();

        train(4000, 0.5f);

        System.out.println("Output after training");
		forward(tDataSet[check].data);
		for (int j = 0; j < layers[amount_of_layers - 1].neurons.length; j++) {
			if (layers[amount_of_layers - 1].neurons[j].value > 0.5f)
				System.out.println(j + " " + "\u001B[31m" + layers[amount_of_layers - 1].neurons[j].value + "\u001B[0m" + " ");
			else System.out.println(j + " " + layers[amount_of_layers - 1].neurons[j].value + " ");
		}
		for (int i = 0; i < 28; i++) {
			for (int j = 0; j < 28; j++) System.out.print(tDataSet[check].data[i * 28 + j] == 0 ? "0" : "1");
			System.out.println();
		}
		System.out.println( " expected output: " + Arrays.toString(tDataSet[check].expectedOutput));

		int c = 0;
		while (c < 40) { c++; System.out.println(c + " accuracy: " + train(2000, 0.2f)); }
    }

    public static void CreateTrainingData() {
		List<float[]> images = new ArrayList<>();
        List<float[]> labels = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader("mnist_train.csv"))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                float[] imagePixels = new float[784];
                for (int i = 1; i < 784; i++) imagePixels[i - 1] = (float) (Integer.parseInt(values[i]) / 255.0f);
                images.add(imagePixels);

                float[] label = new float[10];
                label[values[0].charAt(0) - '0'] = 1.0f;
                labels.add(label);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

		tDataSet = new TrainingData[images.size()];
		for(int i = 0; i < images.size(); i++)
			tDataSet[i] = new TrainingData(images.get(i),labels.get(i));
    }
    
    public static void forward(float[] inputs) {
    	layers[0] = new Layer(inputs); // First bring the inputs into the input layer layers[0]
    	
        for(int i = 1; i < layers.length; i++) {
        	for(int j = 0; j < layers[i].neurons.length; j++) {
        		float sum = 0;
        		for(int k = 0; k < layers[i-1].neurons.length; k++) {
        			sum += layers[i-1].neurons[k].value * layers[i].neurons[j].weights[k];
        		}
        		//sum += layers[i].neurons[j].bias;
        		layers[i].neurons[j].value = StatUtil.Sigmoid(sum);
        	}
        } 	
    }
    
    // Calculate the output layer weights, calculate the hidden layer weight then update all the weights
    public static void backward(float learning_rate,TrainingData tData) {
    	int out_index = layers.length-1;
    	
    	for(int i = 0; i < layers[out_index].neurons.length; i++) { // Update the output layers For each output
    		float output = layers[out_index].neurons[i].value; // and for each of their weights
    		float target = tData.expectedOutput[i];
    		float derivative = output-target;
    		float delta = derivative*(output*(1-output));
    		layers[out_index].neurons[i].gradient = delta;
    		for(int j = 0; j < layers[out_index].neurons[i].weights.length;j++) { 
    			float previous_output = layers[out_index-1].neurons[j].value;
    			float error = delta*previous_output;
    			layers[out_index].neurons[i].cache_weights[j] = layers[out_index].neurons[i].weights[j] - learning_rate*error;
    		}
    	}
    	
    	for(int i = out_index-1; i > 0; i--) { //Update all the subsequent hidden layers
    		for(int j = 0; j < layers[i].neurons.length; j++) { // For all neurons in that layers
    			float output = layers[i].neurons[j].value;
    			float gradient_sum = sumGradient(j,i+1);
    			float delta = (gradient_sum)*(output*(1-output));
    			layers[i].neurons[j].gradient = delta;
    			for(int k = 0; k < layers[i].neurons[j].weights.length; k++) { // And for all their weights
    				float previous_output = layers[i-1].neurons[k].value;
    				float error = delta*previous_output;
    				layers[i].neurons[j].cache_weights[k] = layers[i].neurons[j].weights[k] - learning_rate*error;
    			}
    		}
    	}

    	for(int i = 0; i< layers.length;i++) for(int j = 0; j < layers[i].neurons.length;j++) layers[i].neurons[j].update_weight();
    }

    public static float sumGradient(int n_index,int l_index) { // This function sums up all the gradient connecting a given neuron in a given layer
    	float gradient_sum = 0;
    	Layer current_layer = layers[l_index];
    	for(int i = 0; i < current_layer.neurons.length; i++) {
    		Neuron current_neuron = current_layer.neurons[i];
    		gradient_sum += current_neuron.weights[n_index]*current_neuron.gradient;
    	}
    	return gradient_sum;
    }

	public static int atImage = 0;

    public static float train(int training_iterations,float learning_rate) {
		int correct = 0;
		for(int i = 0; i < training_iterations; i++) {
    		atImage++;
			if (atImage >= 44000) atImage = 0;
    		forward(tDataSet[atImage].data);
			int output = 0;
			float max = 0;
			for (int j = 0; j < layers[layers.length - 1].neurons.length; j++) {
				if (layers[layers.length - 1].neurons[j].value > max) {
					max = layers[layers.length - 1].neurons[j].value;
					output = j;
				}
			}
			if (tDataSet[atImage].expectedOutput[output] == 1) correct++;
			backward(learning_rate,tDataSet[atImage]);
    	}
		return (float)correct / (float)training_iterations;
    }

	public static void initialize (int[] amount_of_neurons) {
		Neuron.setRangeWeight(-1,1);
		layers = new Layer[amount_of_neurons.length];
		layers[0] = null; // Input Layer
		for (int i = 1; i < amount_of_neurons.length; i++) {
			layers[i] = new Layer(amount_of_neurons[i - 1], amount_of_neurons[i]);
		}
        
    	CreateTrainingData();
	}
}