import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.sql.Time;

public class Rocket {
    public static void main(String[] args) {
        float xPos = 0;
        float yPos = 100;
        float xVel = 10;
        float yVel = 100;

        while (true) {
            //wait 1.0/60.0 seconds
            try {
                Thread.sleep(1000/60);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            xPos += xVel/60.0f;
            yPos += yVel/60.0f;
            yVel -= 9.91f/60.0f;

            if (yPos < 0 && yVel < 0) {
                yPos = 0;
                yVel = 0;
            }

            System.out.println("xPos: " + xPos + " yPos: " + yPos);
        }
    }
}
