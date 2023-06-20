import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

public class UI extends JFrame {
    
    private JLabel imageLabel;

    public UI() {
        imageLabel = new JLabel();
        imageLabel.setHorizontalAlignment(SwingConstants.CENTER);

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        getContentPane().add(imageLabel, BorderLayout.CENTER);
        pack();
        setLocationRelativeTo(null);
        setVisible(true);
    }

    public void displayImage(BufferedImage image) {
        imageLabel.setIcon(new ImageIcon(image));
        pack();
    }

    public static void main(String[] args) { 
        List<int[]> images = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader("mnist_train.csv"))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                int[] imagePixels = new int[values.length - 1];
                for (int i = 1; i < values.length; i++) {
                    imagePixels[i - 1] = Integer.parseInt(values[i]);
                }
                images.add(imagePixels);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        Random random = new Random();
        int randomIndex = random.nextInt(images.size());
        int[] randomImagePixels = images.get(randomIndex);
        BufferedImage randomImage = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);

        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                int pixelValue = randomImagePixels[i * 28 + j];
                System.out.print(pixelValue + " ");
                randomImage.setRGB(j, i, new Color(pixelValue, pixelValue, pixelValue).getRGB());
            }
            System.out.println();
        }

        SwingUtilities.invokeLater(() -> {
            UI imageDisplay = new UI();
            imageDisplay.displayImage(randomImage);
        });
    }
}
