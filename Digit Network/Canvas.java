import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import java.awt.image.BufferedImage;

public class Canvas {
    JButton clearBtn;
    JTextArea textArea;
    DrawArea drawArea;
    public static NeuralNet nn = new NeuralNet();
    ActionListener actionListener = new ActionListener() {
        public void actionPerformed(ActionEvent e) {
            if (e.getSource() == clearBtn) {
                drawArea.clear();
            }
        }
    };

    public static void main(String[] args) {
        nn.train();
        new Canvas().show();
    }

    public void show() { // create main frame
        JFrame frame = new JFrame("Swing Paint");
        Container content = frame.getContentPane();
        content.setLayout(new BorderLayout());// set layout on content pane
        drawArea = new DrawArea(); // create draw area
        content.add(drawArea, BorderLayout.CENTER);
        JPanel controls = new JPanel();

        clearBtn = new JButton("Clear");
        clearBtn.addActionListener(actionListener);
        controls.add(clearBtn); // add to panel

        textArea = new JTextArea(1, 10);
        controls.add(textArea); // add to panel

        content.add(controls, BorderLayout.NORTH);
        frame.setSize(360, 360);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }

    public class DrawArea extends JComponent {
        private BufferedImage image;
        private Graphics2D g2;
        private int currentX, currentY, oldX, oldY;

        public DrawArea() {
            setDoubleBuffered(false);
            addMouseListener(new MouseAdapter() {
                public void mousePressed(MouseEvent e) {
                    oldX = e.getX(); // save coord x,y when mouse is pressed
                    oldY = e.getY();
                }
            });

            addMouseMotionListener(new MouseMotionAdapter() {
                public void mouseDragged(MouseEvent e) {
                    int strokeSize = 23;
                    currentX = e.getX();
                    currentY = e.getY();

                    if (g2 != null) { // draw line if g2 context not null
                        g2.setStroke(new BasicStroke(strokeSize));
                        g2.drawLine(oldX, oldY, currentX, currentY);
                        repaint(); // refresh draw area to repaint
                        oldX = currentX; // store current coords x,y as olds x,y
                        oldY = currentY;
                    }

                    nn.forward(drawArea.getPixelValues()); //forward pass, then find output
                    int output = 0;
                    float max = 0;
                    for (int i = 0; i < nn.layers[nn.layers.length - 1].neurons.length; i++) {
                        if (nn.layers[nn.layers.length - 1].neurons[i].value > max) {
                            max = nn.layers[nn.layers.length - 1].neurons[i].value;
                            output = i;
                        }
                    }
                    textArea.setText(output + "");
                }
            });
        }

        protected void paintComponent(Graphics g) {
            if (image == null) {
                image = new BufferedImage(280, 280, BufferedImage.TYPE_INT_RGB);
                g2 = (Graphics2D) image.getGraphics();
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                clear();
            }
            g.drawImage(image, 0, 0, null);
        }

        public void clear() { // now we create exposed methods
            g2.setPaint(Color.white);
            g2.fillRect(0, 0, getSize().width, getSize().height);
            g2.setPaint(Color.black);
            repaint();
        }

        public float[] getPixelValues() {
            int canvasSize = 280;
            if (image == null) return new float[canvasSize * canvasSize]; // Return an empty array if image is null

            int[][] pixels = new int[canvasSize][canvasSize];
            for (int i = 0; i < canvasSize; i++) {
                for (int j = 0; j < canvasSize; j++) {
                    int rgb = image.getRGB(j, i);
                    int grayscale = (rgb >> 16) & 0xFF; // Extract the red component as grayscale value
                    pixels[i][j] = grayscale;
                }
            }

            int compressedSize = 28;
            int[][] compressedArray = new int[compressedSize][compressedSize];
            int blockSize = canvasSize / compressedSize;

            for (int i = 0; i < compressedSize; i++) {
                for (int j = 0; j < compressedSize; j++) {
                    int sum = 0;
                    for (int x = i * blockSize; x < (i + 1) * blockSize; x++) {
                        for (int y = j * blockSize; y < (j + 1) * blockSize; y++) {
                            sum += pixels[x][y];
                        }
                    }
                    int average = sum / (blockSize * blockSize);
                    compressedArray[i][j] = average;
                }
            }

            float[] floatPixels1D = new float[784];
            for (int i = 0; i < 28; i++) for (int j = 0; j < 28; j++) floatPixels1D[i * 28 + j] = 1.0f - compressedArray[i][j] / 255.0f;
            return floatPixels1D;
        }
    }
}
