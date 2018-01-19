public class City {
    private int id;
    private double x = 0.0;
    private double y = 0.0;
    private boolean isVisited = false;

    public City(int id, double x, double y) {
        this.id = id;
        this.x = x;
        this.y = y;
    }

    public City(int id) {
        this.id = id;
        this.x = Math.random();
        this.y = Math.random();
    }

    public void visit() {
        this.isVisited = true;
    }

    public double getX() {
        return this.x;
    }

    public double getY() {
        return this.y;
    }
}
