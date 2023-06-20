import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NeuralNet {
    static Layer[] layers;
    static TrainingData[] tDataSet;
    public static void main(String[] args) {
        Neuron.setRangeWeight(-1,1);
    	layers = new Layer[3];
    	layers[0] = null; // Input Layer 0,2
    	layers[1] = new Layer(2,6); // Hidden Layer 2,6
    	layers[2] = new Layer(6,1); // Output Layer 6,1
        
    	CreateTrainingData();
    	
        System.out.println("Output before training");
        for(int i = 0; i < tDataSet.length; i++) {
            forward(tDataSet[i].data);
            System.out.println(layers[2].neurons[0].value);
        }

        train(40, 0.05f);

        System.out.println("Output after training");
        for(int i = 0; i < tDataSet.length; i++) {
            forward(tDataSet[i].data);
            System.out.println(layers[2].neurons[0].value);
        }
    }

    public static void CreateTrainingData() {
        DataReader dataReader = new DataReader();
        List<float[]> in = dataReader.images;
        List<float[]> out = dataReader.labels;

		tDataSet = new TrainingData[in.size()];
		for(int i = 0; i < in.size(); i++)
			tDataSet[i] = new TrainingData(in.get(i),out.get(i));
    }
    
    public static void forward(float[] inputs) {
    	// First bring the inputs into the input layer layers[0]
    	layers[0] = new Layer(inputs);
    	
        for(int i = 1; i < layers.length; i++) {
        	for(int j = 0; j < layers[i].neurons.length; j++) {
        		float sum = 0;
        		for(int k = 0; k < layers[i-1].neurons.length; k++) {
        			sum += layers[i-1].neurons[k].value * layers[i].neurons[j].weights[k];
        		}
        		//sum += layers[i].neurons[j].bias; // TODO add in the bias 
        		layers[i].neurons[j].value = StatUtil.Sigmoid(sum);
        	}
        } 	
    }
    
    // Calculate the output layer weights, calculate the hidden layer weight then update all the weights
    public static void backward(float learning_rate,TrainingData tData) {
    	
    	int number_layers = layers.length;
    	int out_index = number_layers-1;
    	
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
					layers[i].neurons[j].update_weight();
    			}
    		}
    	}
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

    public static void train(int training_iterations,float learning_rate) {
    	for(int i = 0; i < training_iterations; i++) {
    		for(int j = 0; j < tDataSet.length; j++) {
    			forward(tDataSet[j].data);
    			backward(learning_rate,tDataSet[j]);
    		}
    	}
    }
}