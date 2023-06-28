public class TrainingData {
 
    float[] data; //data inputted into input nodes
    float[] expectedOutput; //data expected on output nodes
   
    public TrainingData(float[] data, float[] expectedOutput) {
        this.data = data;
        this.expectedOutput = expectedOutput;
    }
}