import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DataReader {

    public static List<float[]> images = new ArrayList<>();
    public static List<float[]> labels = new ArrayList<>();
    public static void main(String[] args) {
        //boolean hasPrintedOne = true;

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

/*                if (!hasPrintedOne) {
                    for (int i = 0; i < 28; i++) {
                        for (int j = 0; j < 28; j++) System.out.print(imagePixels[i * 28 + j] + " ");
                        System.out.println();
                    }
                    System.out.println("This is " + values[0] + "\n");
                    hasPrintedOne = true;
                } */
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
