import javax.swing.*;

public class Visualize {
    public static void main(String[] args) {
        JFrame window = new JFrame("Visualzation");
        Ico ico = new Ico();
        window.add(ico);
        window.setSize(600,600);
        window.show();
    }
}
