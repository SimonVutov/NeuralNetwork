public class StatUtil {
	
	// Get a random numbers between min and max
    public static float RandomFloat(float min, float max) {
        float num = min + (float) Math.random() * (max - min);
        if(Math.random() < 0.5)
            return num;
        else
            return -num;
    }
    
    // Sigmoid function
    public static float Sigmoid(float x) {
        return (float) (1/(1+Math.pow(Math.E, -x)));
    }
    
    // Derivative of the sigmoid function
    public static float SigmoidDerivative(float x) {
        return Sigmoid(x)*(1-Sigmoid(x));
    }

    // ReLU function
    public static float ReLU(float x) {
        return (float) Math.max(0,x);
    }

    // Derivative of the ReLU function
    public static float ReLUDerivative(float x) {
        if(x>0)
            return 1;
        else
            return 0;
    }
    
    // Used for the backpropagation
    public static float squaredError(float output,float target) {
    	return (float) (0.5*Math.pow(2,(target-output)));
    }
    
    // Used to calculate the overall error rate (not yet used)
    public static float sumSquaredError(float[] outputs,float[] targets) {
    	float sum = 0;
    	for(int i=0;i<outputs.length;i++) {
    		sum += squaredError(outputs[i],targets[i]);
    	}
    	return sum;
    }
}
