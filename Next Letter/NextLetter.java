import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;


public class NextLetter {
    static Layer[] layers;
    static TrainingData[] tDataSet;
	public static Scanner s = new Scanner(System.in);
    public static void main(String[] args) {
        int[] amountOfNeurons = new int[] { 26, 26 };
        Neuron.setRangeWeight(-1f,1f);

		layers = new Layer[amountOfNeurons.length];
		layers[0] = null;
		for (int i = 1; i < amountOfNeurons.length; i++)
			layers[i] = new Layer(amountOfNeurons[i - 1], amountOfNeurons[i]);
    	
        //create training data
        List<float[]> in = new ArrayList<float[]>();
        List<float[]> out = new ArrayList<float[]>();
        
        List<String> lines = new ArrayList<String>();
        try (BufferedReader reader = new BufferedReader(new FileReader("brown.txt"))) {
            String line;
            while ((line = reader.readLine()) != null) lines.add(line);
        } catch (IOException e) { e.printStackTrace(); }

        for (int i = 0; i < lines.size(); i++) {
            String line = lines.get(i).toLowerCase();
            for (int j = 0; j < line.length() - 1; j++) {
                char c = line.charAt(j);
                char c2 = line.charAt(j + 1);
                if (Character.isLetter(c) && Character.isLetter(c2)) {
                    float[] input = new float[26];
                    float[] output = new float[26];
                    input[c - 'a'] = 1;
                    output[c2 - 'a'] = 1;
                    in.add(input);
                    out.add(output);
                }
            }
        }

        tDataSet = new TrainingData[in.size()];
        for (int i = 0; i < in.size(); i++) tDataSet[i] = new TrainingData(in.get(i), out.get(i));

        train(50000, 0.5f); //5000 iterations, learning rate of 0.5

        //test
        System.out.println("Enter a letter: ");
        String input = s.nextLine();
        char c = input.charAt(0);
        float[] inputArray = new float[26];
        inputArray[c - 'a'] = 1;
        System.out.println("Next letter: " + getLetter(forward(inputArray)));
        //print for each letter
        for (int i = 0; i < 26; i++) {
            inputArray = new float[26];
            inputArray[i] = 1;
            System.out.println("Next letter for " + (char) (i + 'a') + ": " + getLetter(forward(inputArray)));
        }
    }

    public static float sumGradient(int n_index, int l_index) {
        // This function sums up all the gradient connecting a given neuron in a given layer
    	float gradient_sum = 0;
    	Layer current_layer = layers[l_index];
    	for(int i = 0; i < current_layer.neurons.length; i++) {
    		Neuron current_neuron = current_layer.neurons[i];
    		gradient_sum += current_neuron.weights[n_index] * current_neuron.gradient;
    	}
    	return gradient_sum;
    }

    public static float[] forward (float[] inputs) {
        layers[0] = new Layer(inputs);

        for (int i = 1; i < layers.length; i++) {
            for (int j = 0; j < layers[i].neurons.length; j++) {
                float sum = 0;
                for (int k = 0; k < layers[i - 1].neurons.length; k++) {
                    sum += layers[i - 1].neurons[k].value * layers[i].neurons[j].weights[k];
                }
                sum += layers[i].neurons[j].bias;

                layers[i].neurons[j].value = StatUtil.Sigmoid(sum);
            }
        }

        //print loss/cost
        float cost = 0;
        for (int i = 0; i < layers[layers.length - 1].neurons.length; i++) {
            float output = layers[layers.length - 1].neurons[i].value;
            float expected = tDataSet[0].expectedOutput[i];
            cost += (output - expected) * (output - expected);
        }
        cost /= layers[layers.length - 1].neurons.length;
        if (Math.random() < 0.1f) System.out.println("Cost: " + cost);

        float[] output = new float[layers[layers.length - 1].neurons.length];
        for (int i = 0; i < output.length; i++) output[i] = layers[layers.length - 1].neurons[i].value;
        return output;
    }

    public static void backprop (float learning_Rate, TrainingData tData) {
        for (int i = 0; i < layers[layers.length - 1].neurons.length; i++) {
            float output = layers[layers.length - 1].neurons[i].value;
            float derivative = output - tData.expectedOutput[i];
            float delta = derivative * output * (1 - output);
            layers[layers.length - 1].neurons[i].gradient = delta;
            for (int j = 0; j < layers[layers.length - 1].neurons[i].weights.length; j++) {
                float previous_output = layers[layers.length - 2].neurons[j].value;
                float error = delta * previous_output;
                layers[layers.length - 1].neurons[i].cache_weights[j] -= learning_Rate * error;
            }
        }

        for(int i = layers.length - 1-1; i > 0; i--) {
    		for(int j = 0; j < layers[i].neurons.length; j++) {
    			float output = layers[i].neurons[j].value;
    			float gradient_sum = sumGradient(j, i + 1);
    			float delta = gradient_sum * output * (1 - output);
				layers[i].neurons[j].gradient = delta;
    			for(int k = 0; k < layers[i].neurons[j].weights.length; k++) {
    				float previous_output = layers[i - 1].neurons[k].value;
    				float error = delta * previous_output;
    				layers[i].neurons[j].cache_weights[k] -= learning_Rate * error;
    			}
				layers[i].neurons[j].bias += learning_Rate * delta;
    		}
    	}

    	for(int i = 0; i< layers.length;i++) for(int j = 0; j < layers[i].neurons.length;j++) layers[i].neurons[j].update_weight();
    }

    public static void train (int iterations, float learning_Rate) {
        for (int i = 0; i < iterations; i++) {
            forward(tDataSet[i].data);
            backprop(learning_Rate, tDataSet[i]);
        }
    }

    public static String getLetter (float[] out) {
        int maxIndex = 0;
        for (int i = 0; i < out.length; i++) if (out[i] > out[maxIndex]) maxIndex = i;
        return Character.toString((char) (maxIndex + 'a'));
    }
}